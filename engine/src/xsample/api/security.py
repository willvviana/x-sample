"""Host/Origin validation, CORS, security headers and bearer-token auth (ADR-0008)."""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse, Response

from xsample.config import EngineSettings

_UI_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'"
)
_CORS_METHODS = "GET, POST, OPTIONS"
_CORS_HEADERS = "Authorization, Content-Type"


def _deny(status: int, detail: str) -> JSONResponse:
    return JSONResponse(
        {"detail": detail}, status_code=status, headers={"Cache-Control": "no-store"}
    )


def _apply_common_headers(response: Response, path: str) -> None:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    if path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store"
    else:
        response.headers["Content-Security-Policy"] = _UI_CSP
        response.headers["X-Frame-Options"] = "DENY"


def make_guard(
    settings: EngineSettings,
) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    allowed_hosts = settings.allowed_hosts
    allowed_origins = settings.allowed_origins

    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        host = request.headers.get("host", "")
        if host not in allowed_hosts:
            return _deny(403, "host not allowed")

        origin = request.headers.get("origin")
        if origin is not None and origin not in allowed_origins:
            return _deny(403, "origin not allowed")
        if origin is None and request.headers.get("sec-fetch-site") in {"cross-site", "same-site"}:
            return _deny(403, "cross-site request without allowed origin")

        length = request.headers.get("content-length")
        if length is not None:
            try:
                limit = (
                    settings.max_upload_bytes
                    if path == "/api/project/import"
                    else settings.max_request_bytes
                )
                too_big = int(length) > limit
            except ValueError:
                return _deny(400, "invalid content-length")
            if too_big:
                return _deny(413, "request body too large")

        if request.method == "OPTIONS" and origin is not None:
            preflight = Response(status_code=204)
            preflight.headers["Access-Control-Allow-Origin"] = origin
            preflight.headers["Access-Control-Allow-Methods"] = _CORS_METHODS
            preflight.headers["Access-Control-Allow-Headers"] = _CORS_HEADERS
            preflight.headers["Access-Control-Max-Age"] = "600"
            preflight.headers["Vary"] = "Origin"
            _apply_common_headers(preflight, path)
            return preflight

        response = await call_next(request)
        if origin is not None:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        _apply_common_headers(response, path)
        return response

    return guard


def make_token_dependency(settings: EngineSettings) -> Callable[[Request], None]:
    expected = settings.token.encode("utf-8")

    def require_token(request: Request) -> None:
        header = request.headers.get("authorization", "")
        scheme, _, supplied = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            supplied.strip().encode("utf-8"), expected
        ):
            raise HTTPException(
                status_code=401,
                detail="missing or invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_token
