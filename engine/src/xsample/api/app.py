"""FastAPI application factory."""

from __future__ import annotations

import logging
import platform
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import psutil
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from xsample import __version__
from xsample.api import schemas
from xsample.api.routes_project import build_router
from xsample.api.security import make_guard, make_token_dependency
from xsample.config import EngineSettings
from xsample.jobs.manager import Job, JobManager, JobNotFoundError, JobRejectedError, JobState
from xsample.project.service import ProjectService
from xsample.project.store import ProjectError
from xsample.runtime import win_gpu
from xsample.runtime.probe_cache import ProbeCache
from xsample.runtime.registry import ModelRegistry

log = logging.getLogger("xsample.api")


class EngineState:
    """Shared engine state: job manager, detection results and probe cache."""

    def __init__(self, settings: EngineSettings) -> None:
        self.settings = settings
        self.probe_cache = ProbeCache(settings.probe_cache_path)
        self._lock = threading.Lock()
        self.detection: dict[str, Any] = {"state": "pending"}
        self.devices: dict[str, Any] = {"state": "pending"}
        self.project = ProjectService(settings)
        self.jobs = JobManager(settings, on_finished=self._on_job_finished)

    def start_device_listing(self) -> Job:
        """Enumerates audio devices in a worker; no stream is opened, so no mic access happens."""
        job = self.jobs.submit("list_audio_devices")
        with self._lock:
            self.devices = {**self.devices, "state": "running", "job_id": job.id}
        return job

    def devices_snapshot(self) -> dict[str, Any]:
        with self._lock:
            d = dict(self.devices)
        result = d.get("result") or {}
        return {
            "state": d.get("state", "pending"),
            "updated_at": d.get("updated_at"),
            "error": d.get("error"),
            "microphones": result.get("microphones", []),
            "loopbacks": result.get("loopbacks", []),
        }

    def _apply_project_job(self, job: Job) -> None:
        """Fold a finished audio job back into the open project (manifest is engine-owned)."""
        if job.state is not JobState.SUCCEEDED or not job.result:
            return
        try:
            if job.kind == "import_audio":
                self.project.apply_import(job.result, job.params)
            elif job.kind == "export_audio":
                self.project.apply_export(job.result, job.params)
            elif job.kind == "record_audio":
                # Recordings become ordinary sources through the same import path.
                self.jobs.submit(
                    "import_audio",
                    {
                        "project_dir": job.params["project_dir"],
                        "incoming_path": job.result["path"],
                        "original_name": Path(job.result["path"]).name,
                        "sample_rate": self.project.view_required().sample_rate,
                        "channels": "keep",
                        "origin": "recording",
                        "device": job.result.get("device"),
                    },
                )
        except ProjectError:
            log.exception("could not apply finished %s job", job.kind)

    def start_detection(self) -> Job:
        job = self.jobs.submit("detect_backends")
        with self._lock:
            self.detection = {
                "state": "running",
                "job_id": job.id,
                "result": self.detection.get("result"),
            }
        return job

    def _on_job_finished(self, job: Job) -> None:
        if job.kind == "detect_backends":
            with self._lock:
                if job.state is JobState.SUCCEEDED:
                    self.detection = {
                        "state": "ready",
                        "job_id": job.id,
                        "updated_at": job.finished_at,
                        "result": job.result,
                    }
                else:
                    self.detection = {
                        "state": "failed",
                        "job_id": job.id,
                        "updated_at": job.finished_at,
                        "error": job.error or str(job.state),
                    }
        elif job.kind == "list_audio_devices":
            with self._lock:
                ok = job.state is JobState.SUCCEEDED
                self.devices = {
                    "state": "ready" if ok else "failed",
                    "updated_at": job.finished_at,
                    "result": job.result if ok else self.devices.get("result"),
                    "error": None if ok else (job.error or str(job.state)),
                }
        elif job.kind in {"import_audio", "export_audio", "record_audio"}:
            self._apply_project_job(job)
        elif (
            job.kind == "probe"
            and job.state is JobState.SUCCEEDED
            and job.result
            and not job.result.get("cached")
        ):
            self.probe_cache.put(job.result["record"])

    def detection_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.detection)


def _system_info(state: EngineState) -> schemas.SystemInfo:
    settings = state.settings
    detection = state.detection_snapshot()
    result = detection.get("result") or {}
    detected = result.get("backends", {})
    passing = {r["backend"] for r in state.probe_cache.all() if r.get("status") == "pass"}

    backends = []
    for backend_id in ("cpu", "directml", "vulkan"):
        info = detected.get(backend_id, {})
        backends.append(
            schemas.BackendStatus(
                id=backend_id,
                state=info.get("state", "unknown"),
                detail=info.get("detail"),
                devices=info.get("devices", []),
                claims_model_support=backend_id in passing,
            )
        )

    memory = psutil.virtual_memory()
    return schemas.SystemInfo(
        engine_version=__version__,
        python_version=platform.python_version(),
        os=f"{platform.system()} {platform.release()} ({platform.version()})",
        cpu=schemas.CpuInfo(
            name=win_gpu.cpu_name(),
            cores=psutil.cpu_count(logical=False),
            threads=psutil.cpu_count(logical=True),
        ),
        memory=schemas.MemoryInfo(total_bytes=memory.total, available_bytes=memory.available),
        gpus=[
            schemas.GpuInfo(name=g.name, vram_bytes=g.vram_bytes, driver_version=g.driver_version)
            for g in win_gpu.registry_gpus()
        ],
        backends=backends,
        detection=schemas.DetectionInfo(
            state=detection.get("state", "pending"),
            job_id=detection.get("job_id"),
            updated_at=detection.get("updated_at"),
            error=detection.get("error"),
            runtime_version=result.get("runtime_version"),
        ),
        budgets=schemas.Budgets(
            vram_budget_bytes=settings.vram_budget_bytes,
            worker_memory_limit_bytes=settings.worker_memory_limit_bytes,
            max_concurrent_jobs=settings.max_concurrent_jobs,
        ),
        test_tasks_enabled=settings.enable_test_tasks,
    )


def create_app(settings: EngineSettings) -> FastAPI:
    state = EngineState(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if settings.detect_on_startup:
            state.start_detection()
            state.start_device_listing()
        yield
        state.jobs.shutdown()
        state.project.close()

    app = FastAPI(
        title="X-Sample Engine",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.engine = state

    @app.exception_handler(ProjectError)
    async def project_error(_: Request, exc: ProjectError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": str(exc)})

    app.middleware("http")(make_guard(settings))

    @app.get("/api/health", response_model=schemas.Health, tags=["system"])
    def health() -> schemas.Health:
        return schemas.Health(status="ok", version=__version__)

    api = APIRouter(prefix="/api", dependencies=[Depends(make_token_dependency(settings))])

    @api.get("/system", response_model=schemas.SystemInfo, tags=["system"])
    def system() -> schemas.SystemInfo:
        return _system_info(state)

    @api.post("/backends/detect", response_model=schemas.JobOut, status_code=202, tags=["system"])
    def detect() -> dict[str, Any]:
        return state.start_detection().snapshot()

    @api.get("/probes", response_model=list[schemas.ProbeRecord], tags=["backends"])
    def probes() -> list[dict[str, Any]]:
        return state.probe_cache.all()

    @api.post("/probes", response_model=schemas.JobOut, status_code=202, tags=["backends"])
    def run_probe(request: schemas.ProbeRequest) -> dict[str, Any]:
        job = state.jobs.submit(
            "probe",
            request.model_dump(),
            extra_params={"cache_path": str(settings.probe_cache_path)},
        )
        return job.snapshot()

    @api.get("/models", response_model=list[schemas.ModelInfo], tags=["models"])
    def models() -> list[schemas.ModelInfo]:
        registry = ModelRegistry.load(settings.models_dir)
        return [
            schemas.ModelInfo(
                id=e.id,
                name=e.name,
                version=e.version,
                license=e.license,
                size_bytes=e.size_bytes,
                candidate_backends=list(e.candidate_backends),
                status=registry.verify(e).value,
            )
            for e in registry.entries()
        ]

    @api.get("/jobs", response_model=list[schemas.JobOut], tags=["jobs"])
    def list_jobs() -> list[dict[str, Any]]:
        return state.jobs.list()

    @api.post("/jobs", response_model=schemas.JobOut, status_code=202, tags=["jobs"])
    def submit_job(request: schemas.JobRequest) -> dict[str, Any]:
        try:
            return state.jobs.submit(request.kind, request.params, public=True).snapshot()
        except JobRejectedError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @api.get("/jobs/{job_id}", response_model=schemas.JobOut, tags=["jobs"])
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return state.jobs.get(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @api.post("/jobs/{job_id}/cancel", response_model=schemas.JobOut, tags=["jobs"])
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return state.jobs.cancel(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @api.post("/jobs/{job_id}/stop", response_model=schemas.JobOut, tags=["jobs"])
    def stop_job(job_id: str) -> dict[str, Any]:
        try:
            return state.jobs.stop(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    api.include_router(build_router(state))
    app.include_router(api)

    @app.api_route("/api/{rest:path}", methods=["GET", "POST"], include_in_schema=False)
    def api_not_found(rest: str) -> None:
        raise HTTPException(status_code=404, detail="not found")

    if settings.ui_dist is not None and (settings.ui_dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=settings.ui_dist, html=True), name="ui")
    else:
        log.info("UI build not found; serving API only")

    app.state.started_at = time.time()
    return app


def openapi_schema(settings: EngineSettings) -> dict[str, Any]:
    from fastapi.openapi.utils import get_openapi

    app = create_app(settings)
    return get_openapi(title=app.title, version=app.version, routes=app.routes)
