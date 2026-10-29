"""ADR-0008: token, Host and Origin validation, CORS and headers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import PORT, TOKEN

UI_ORIGIN = "http://127.0.0.1:5173"


def test_health_needs_no_token_but_reveals_nothing(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert set(response.json()) == {"status", "version"}


@pytest.mark.parametrize(
    "path,method",
    [
        ("/api/system", "GET"),
        ("/api/jobs", "GET"),
        ("/api/jobs", "POST"),
        ("/api/probes", "GET"),
        ("/api/models", "GET"),
        ("/api/backends/detect", "POST"),
        ("/api/jobs/abc/cancel", "POST"),
    ],
)
def test_protected_routes_require_token(client: TestClient, path: str, method: str) -> None:
    response = client.request(method, path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "header",
    [
        f"Bearer {TOKEN}x",
        f"Bearer {TOKEN[:-1]}",
        f"Basic {TOKEN}",
        TOKEN,
        "Bearer ",
    ],
)
def test_wrong_tokens_are_rejected(client: TestClient, header: str) -> None:
    assert client.get("/api/system", headers={"Authorization": header}).status_code == 401


def test_valid_token_is_accepted(client: TestClient, auth: dict[str, str]) -> None:
    assert client.get("/api/jobs", headers=auth).status_code == 200


@pytest.mark.parametrize(
    "host",
    ["evil.example", f"evil.example:{PORT}", "127.0.0.1", f"127.0.0.1:{PORT + 1}", "0.0.0.0:8765"],
)
def test_foreign_host_header_is_rejected(
    client: TestClient, auth: dict[str, str], host: str
) -> None:
    # DNS rebinding: a hostile name resolving to 127.0.0.1 still sends its own Host.
    response = client.get("/api/system", headers={**auth, "Host": host})
    assert response.status_code == 403
    assert client.get("/api/health", headers={"Host": host}).status_code == 403


def test_localhost_host_is_accepted(client: TestClient, auth: dict[str, str]) -> None:
    response = client.get("/api/jobs", headers={**auth, "Host": f"localhost:{PORT}"})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "origin",
    ["https://evil.example", "null", "http://127.0.0.1:5174", "http://localhost", "file://"],
)
def test_foreign_origin_is_rejected_even_with_token(
    client: TestClient, auth: dict[str, str], origin: str
) -> None:
    response = client.get("/api/system", headers={**auth, "Origin": origin})
    assert response.status_code == 403
    assert "access-control-allow-origin" not in response.headers


def test_foreign_origin_preflight_is_rejected(client: TestClient) -> None:
    response = client.options(
        "/api/jobs",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert response.status_code == 403
    assert "access-control-allow-origin" not in response.headers


def test_allowed_origin_preflight_and_request(client: TestClient, auth: dict[str, str]) -> None:
    preflight = client.options(
        "/api/jobs",
        headers={"Origin": UI_ORIGIN, "Access-Control-Request-Method": "POST"},
    )
    assert preflight.status_code == 204
    assert preflight.headers["access-control-allow-origin"] == UI_ORIGIN
    assert "authorization" in preflight.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in preflight.headers

    response = client.get("/api/jobs", headers={**auth, "Origin": UI_ORIGIN})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == UI_ORIGIN


def test_allowed_origin_still_needs_token(client: TestClient) -> None:
    assert client.get("/api/system", headers={"Origin": UI_ORIGIN}).status_code == 401


@pytest.mark.parametrize("site", ["cross-site", "same-site"])
def test_cross_site_fetch_without_origin_is_rejected(
    client: TestClient, auth: dict[str, str], site: str
) -> None:
    response = client.get("/api/jobs", headers={**auth, "Sec-Fetch-Site": site})
    assert response.status_code == 403


def test_non_browser_client_without_origin_uses_token(
    client: TestClient, auth: dict[str, str]
) -> None:
    assert client.get("/api/jobs", headers=auth).status_code == 200
    assert client.get("/api/jobs").status_code == 401


def test_security_headers_on_api(client: TestClient, auth: dict[str, str]) -> None:
    response = client.get("/api/jobs", headers=auth)
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["cross-origin-resource-policy"] == "same-origin"


def test_request_body_limit(client: TestClient, auth: dict[str, str]) -> None:
    big = {"kind": "test.steps", "params": {"pad": "x" * 70_000}}
    assert client.post("/api/jobs", headers=auth, json=big).status_code == 413


def test_docs_and_openapi_are_not_served(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_unknown_api_route_is_404_not_ui(client: TestClient, auth: dict[str, str]) -> None:
    assert client.get("/api/does-not-exist", headers=auth).status_code == 404


def test_internal_and_unknown_job_kinds_are_rejected(
    client: TestClient, auth: dict[str, str]
) -> None:
    for kind in ("probe", "detect_backends", "os.system", "../x"):
        response = client.post("/api/jobs", headers=auth, json={"kind": kind, "params": {}})
        assert response.status_code == 422, kind


def test_extra_params_are_rejected(client: TestClient, auth: dict[str, str]) -> None:
    response = client.post(
        "/api/jobs",
        headers=auth,
        json={"kind": "smoke_inference", "params": {"backend": "cpu", "cache_path": "C:/x"}},
    )
    assert response.status_code == 422
