"""Command line: `xsample serve` and `xsample export-openapi`."""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
import webbrowser
from pathlib import Path

from xsample import __version__
from xsample.config import (
    DEFAULT_PORT,
    DEV_UI_ORIGINS,
    EngineSettings,
    default_data_dir,
    default_projects_dir,
    default_ui_dist,
)


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from xsample.api.app import create_app

    token = os.environ.get("XSAMPLE_TOKEN") or secrets.token_urlsafe(32)
    if len(token) < 16:
        print("XSAMPLE_TOKEN must be at least 16 characters", file=sys.stderr)
        return 2
    ui_dist = Path(args.ui_dist) if args.ui_dist else default_ui_dist()
    settings = EngineSettings(
        token=token,
        port=args.port,
        data_dir=Path(args.data_dir) if args.data_dir else default_data_dir(),
        projects_dir=Path(args.projects_dir) if args.projects_dir else default_projects_dir(),
        ui_dist=ui_dist,
        extra_origins=tuple(args.dev_origin) if args.dev_origin else DEV_UI_ORIGINS,
        enable_test_tasks=args.enable_test_tasks,
        block_network_in_workers=not args.allow_worker_network,
    )
    app = create_app(settings)

    engine_url = f"http://127.0.0.1:{settings.port}"
    print(f"X-Sample engine {__version__} on {engine_url} (local only)")
    print(f"Data directory: {settings.data_dir}")
    if (ui_dist / "index.html").is_file():
        launch = f"{engine_url}/#token={token}"
        print(f"Open the UI:   {launch}")
        if args.open:
            webbrowser.open(launch)
    else:
        print("UI build not found (run `npm run build` in ui/), or use the Vite dev server:")
    for origin in settings.extra_origins:
        print(f"  dev UI:      {origin}/#token={token}")
    sys.stdout.flush()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=settings.port,
        log_level=args.log_level,
        server_header=False,
        proxy_headers=False,
        access_log=args.log_level == "debug",
    )
    return 0


def _export_openapi(args: argparse.Namespace) -> int:
    from xsample.api.app import openapi_schema

    # Never served: the app is only built to read its schema.
    settings = EngineSettings(token="schema-export-only", detect_on_startup=False)  # noqa: S106
    schema = openapi_schema(settings)
    Path(args.output).write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xsample")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the local engine")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--data-dir")
    serve.add_argument("--projects-dir", help="where new projects are created")
    serve.add_argument("--ui-dist", help="built UI directory (default: ../ui/dist)")
    serve.add_argument(
        "--dev-origin", action="append", help="extra allowed browser origin (repeatable)"
    )
    serve.add_argument("--open", action="store_true", help="open the UI in the browser")
    serve.add_argument("--enable-test-tasks", action="store_true", help=argparse.SUPPRESS)
    serve.add_argument("--allow-worker-network", action="store_true", help=argparse.SUPPRESS)
    serve.add_argument("--log-level", default="info", choices=["debug", "info", "warning"])
    serve.set_defaults(func=_serve)

    export = sub.add_parser("export-openapi", help="write the OpenAPI schema without serving")
    export.add_argument("output")
    export.set_defaults(func=_export_openapi)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
