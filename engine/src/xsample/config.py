"""Engine settings and data locations."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

GiB = 1024**3
DEFAULT_PORT = 8765
DEV_UI_ORIGINS = ("http://127.0.0.1:5173", "http://localhost:5173")


def default_data_dir() -> Path:
    override = os.environ.get("XSAMPLE_DATA_DIR")
    if override:
        return Path(override)
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    return base / "X-Sample"


def default_projects_dir() -> Path:
    override = os.environ.get("XSAMPLE_PROJECTS_DIR")
    if override:
        return Path(override)
    return Path.home() / "Documents" / "X-Sample Projects"


def default_ui_dist() -> Path:
    # engine/src/xsample/config.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3] / "ui" / "dist"


@dataclass(frozen=True)
class EngineSettings:
    token: str
    port: int = DEFAULT_PORT
    data_dir: Path = field(default_factory=default_data_dir)
    projects_dir: Path = field(default_factory=default_projects_dir)
    ui_dist: Path | None = None
    extra_origins: tuple[str, ...] = DEV_UI_ORIGINS
    enable_test_tasks: bool = False
    vram_budget_bytes: int = int(6.5 * GiB)
    worker_memory_limit_bytes: int = 4 * GiB
    max_concurrent_jobs: int = 2
    # OpenBLAS commits ~30 MB per thread at import; numpy work in workers is light.
    worker_blas_threads: int = 4
    job_timeout_s: float = 900.0
    cancel_grace_s: float = 3.0
    block_network_in_workers: bool = True
    max_request_bytes: int = 64 * 1024
    max_upload_bytes: int = 4 * GiB
    # Preview audio is rendered in the API process and decoded in the browser; keep it small.
    preview_max_seconds: float = 120.0
    max_recording_seconds: int = 4 * 3600
    detect_on_startup: bool = True

    @property
    def allowed_hosts(self) -> frozenset[str]:
        return frozenset({f"127.0.0.1:{self.port}", f"localhost:{self.port}"})

    @property
    def allowed_origins(self) -> frozenset[str]:
        own = (f"http://127.0.0.1:{self.port}", f"http://localhost:{self.port}")
        return frozenset(own + self.extra_origins)

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def probe_cache_path(self) -> Path:
        return self.data_dir / "cache" / "backend_probes.json"
