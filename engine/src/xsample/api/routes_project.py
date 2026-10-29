"""Project, audio and recording routes (Phase 1). All require the API token."""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import soundfile as sf
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from xsample.api import schemas
from xsample.audio.peaks import SCALE, Pyramid, read_columns
from xsample.audio.render import preview_wav
from xsample.project import layout, ops
from xsample.project.model import Project
from xsample.project.store import ProjectError
from xsample.project.view import ProjectView

if TYPE_CHECKING:
    from xsample.api.app import EngineState

BUSY_KINDS = {"import_audio", "export_audio", "record_audio", "verify_project"}
_MEDIA_TYPES = {".wav": "audio/wav", ".flac": "audio/flac"}


class _PeakCache:
    """Small LRU of loaded pyramids keyed by (path, mtime, size); they are a few MB at most."""

    def __init__(self, capacity: int = 8) -> None:
        self._items: OrderedDict[tuple[str, int, int], Pyramid] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def get(self, path: Path) -> Pyramid:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        pyramid = Pyramid.load(path)
        with self._lock:
            self._items[key] = pyramid
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
        return pyramid


def build_router(state: EngineState) -> APIRouter:
    api = APIRouter()
    svc = state.project
    settings = state.settings
    peak_cache = _PeakCache()

    def ensure_idle() -> None:
        busy = state.jobs.active(BUSY_KINDS)
        if busy:
            raise ProjectError(
                f"{len(busy)} audio job(s) are still running; wait for them or cancel them first"
            )

    # --- projects ---------------------------------------------------------------------------

    @api.get("/projects", response_model=schemas.ProjectList, tags=["projects"])
    def list_projects() -> dict[str, Any]:
        return svc.list_projects()

    @api.post("/projects", response_model=ProjectView, tags=["projects"])
    def create_project(req: schemas.ProjectCreate) -> ProjectView:
        ensure_idle()
        return svc.create(req.name, req.sample_rate)

    @api.post("/projects/open", response_model=ProjectView, tags=["projects"])
    def open_project(req: schemas.ProjectOpen) -> ProjectView:
        ensure_idle()
        return svc.open(req.path)

    @api.get("/project", response_model=ProjectView | None, tags=["projects"])
    def current_project() -> ProjectView | None:
        return svc.view()

    @api.post("/project/close", status_code=204, tags=["projects"])
    def close_project() -> Response:
        ensure_idle()
        svc.close()
        return Response(status_code=204)

    @api.post("/project/save", response_model=ProjectView, tags=["projects"])
    def save_project() -> ProjectView:
        return svc.save()

    @api.patch("/project", response_model=ProjectView, tags=["projects"])
    def rename_project(req: schemas.ProjectPatch) -> ProjectView:
        return svc.rename(req.name)

    @api.post("/project/verify", response_model=schemas.JobOut, status_code=202, tags=["projects"])
    def verify_project() -> dict[str, Any]:
        root = svc.root()
        sources = svc.read(lambda _r, p: [s.model_dump() for s in p.sources])
        return state.jobs.submit(
            "verify_project", {"project_dir": str(root), "sources": sources}
        ).snapshot()

    # --- tracks, clips, samples -------------------------------------------------------------

    @api.post("/project/tracks", response_model=ProjectView, tags=["editing"])
    def add_track(req: schemas.TrackCreate) -> ProjectView:
        svc.edit(lambda p: ops.add_track(p, req.name))
        return svc.view_required()

    @api.patch("/project/tracks/{track_id}", response_model=ProjectView, tags=["editing"])
    def patch_track(track_id: str, req: schemas.TrackPatch) -> ProjectView:
        svc.edit(lambda p: ops.update_track(p, track_id, req.name, req.muted, req.solo, req.gain))
        return svc.view_required()

    @api.delete("/project/tracks/{track_id}", response_model=ProjectView, tags=["editing"])
    def delete_track(track_id: str) -> ProjectView:
        svc.edit(lambda p: ops.remove_track(p, track_id))
        return svc.view_required()

    @api.post("/project/clips/{clip_id}/trim", response_model=ProjectView, tags=["editing"])
    def trim_clip(clip_id: str, req: schemas.TrimRequest) -> ProjectView:
        svc.edit(lambda p: ops.trim_clip(p, clip_id, req.start, req.end))
        return svc.view_required()

    @api.post("/project/clips/{clip_id}/reset-trim", response_model=ProjectView, tags=["editing"])
    def reset_trim(clip_id: str) -> ProjectView:
        svc.edit(lambda p: ops.reset_trim(p, clip_id))
        return svc.view_required()

    @api.post("/project/clips/{clip_id}/move", response_model=ProjectView, tags=["editing"])
    def move_clip(clip_id: str, req: schemas.MoveRequest) -> ProjectView:
        svc.edit(lambda p: ops.move_clip(p, clip_id, req.timeline_start))
        return svc.view_required()

    @api.delete("/project/clips/{clip_id}", response_model=ProjectView, tags=["editing"])
    def delete_clip(clip_id: str) -> ProjectView:
        svc.edit(lambda p: ops.delete_clip(p, clip_id))
        return svc.view_required()

    @api.post("/project/clips/{clip_id}/slice", response_model=ProjectView, tags=["editing"])
    def slice_clip(clip_id: str, req: schemas.SliceRequest) -> ProjectView:
        svc.edit(lambda p: ops.slice_clip_to_sample(p, clip_id, req.start, req.end, req.name))
        return svc.view_required()

    @api.patch("/project/samples/{sample_id}", response_model=ProjectView, tags=["editing"])
    def rename_sample(sample_id: str, req: schemas.SamplePatch) -> ProjectView:
        svc.edit(lambda p: ops.rename_sample(p, sample_id, req.name))
        return svc.view_required()

    @api.delete("/project/samples/{sample_id}", response_model=ProjectView, tags=["editing"])
    def delete_sample(sample_id: str) -> ProjectView:
        svc.edit(lambda p: ops.delete_sample(p, sample_id))
        return svc.view_required()

    @api.post("/project/samples/{sample_id}/place", response_model=ProjectView, tags=["editing"])
    def place_sample(sample_id: str, req: schemas.PlaceRequest) -> ProjectView:
        svc.edit(lambda p: ops.place_sample(p, sample_id, req.track_id, req.at))
        return svc.view_required()

    # --- import (streamed upload) -------------------------------------------------------------

    @api.post("/project/import", response_model=schemas.JobOut, status_code=202, tags=["audio"])
    async def import_audio(
        request: Request,
        filename: str = Query(min_length=1, max_length=255),
        channels: Literal["keep", "mono", "stereo"] = "keep",
    ) -> dict[str, Any]:
        name = Path(filename.replace("\\", "/")).name
        suffix = Path(name).suffix.lower()
        if suffix not in layout.SOURCE_EXTENSIONS:
            raise ProjectError(
                f"Unsupported file type '{suffix or name}'. Supported: "
                + ", ".join(sorted(layout.SOURCE_EXTENSIONS))
            )
        root = svc.root()
        dest = svc.new_incoming_path(suffix)
        written = 0
        try:
            with dest.open("wb") as handle:
                async for chunk in request.stream():
                    written += len(chunk)
                    if written > settings.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="file too large")
                    handle.write(chunk)
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        if written == 0:
            dest.unlink(missing_ok=True)
            raise ProjectError("The file is empty")
        rate = svc.view_required().sample_rate
        job = state.jobs.submit(
            "import_audio",
            {
                "project_dir": str(root),
                "incoming_path": str(dest),
                "original_name": name,
                "sample_rate": rate,
                "channels": channels,
                "origin": "import",
            },
        )
        return job.snapshot()

    @api.post(
        "/project/sources/{source_id}/rebuild",
        response_model=schemas.JobOut,
        status_code=202,
        tags=["audio"],
    )
    def rebuild_source(source_id: str) -> dict[str, Any]:
        root = svc.root()
        source = svc.source_for_rebuild(source_id)
        mono = source.media is not None and source.media.channels == 1
        job = state.jobs.submit(
            "import_audio",
            {
                "project_dir": str(root),
                "original_name": source.original_name,
                "sample_rate": svc.view_required().sample_rate,
                "channels": "mono" if mono else "keep",
                "origin": source.origin,
                "device": source.device,
                "rebuild_source_id": source.id,
                "rebuild_stored_path": source.stored_path,
            },
        )
        return job.snapshot()

    # --- waveform peaks, preview, export --------------------------------------------------------

    @api.get("/project/sources/{source_id}/peaks", response_model=schemas.PeaksOut, tags=["audio"])
    def peaks(
        source_id: str,
        start: int = Query(ge=0),
        end: int = Query(gt=0),
        columns: int = Query(ge=1, le=8192),
    ) -> schemas.PeaksOut:
        def locate(root: Path, p: Project) -> tuple[Path, Path, int]:
            try:
                source = p.source(source_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="source not found") from exc
            if source.media is None:
                raise ProjectError("Waveform data is missing; rebuild this source")
            return root / source.media.peaks_path, root / source.media.path, source.media.frames

        peaks_path, media_path, frames = svc.read(locate)
        if end <= start:
            raise HTTPException(status_code=422, detail="end must be greater than start")
        end = min(end, frames)
        if start >= end:
            raise HTTPException(status_code=422, detail="range is outside the audio")
        try:
            pyramid = peak_cache.get(peaks_path)

            def read_raw(a: int, b: int) -> Any:
                with sf.SoundFile(media_path) as handle:
                    handle.seek(a)
                    return handle.read(b - a, dtype="float32", always_2d=True)

            mins, maxs = read_columns(pyramid, read_raw, start, end, columns)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ProjectError("Waveform data is unreadable; rebuild this source") from exc
        return schemas.PeaksOut(
            start=start,
            end=end,
            columns=columns,
            channels=pyramid.channels,
            scale=SCALE,
            min=[mins[:, c].tolist() for c in range(pyramid.channels)],
            max=[maxs[:, c].tolist() for c in range(pyramid.channels)],
        )

    def wav_response(plan: Any, start: int, end: int) -> Response:
        if end <= start:
            raise HTTPException(status_code=422, detail="end must be greater than start")
        limit = int(settings.preview_max_seconds * plan.sample_rate)
        if end - start > limit:
            raise HTTPException(
                status_code=422,
                detail=f"Previews are limited to {settings.preview_max_seconds:.0f} seconds",
            )
        return Response(content=preview_wav(plan, start, end), media_type="audio/wav")

    @api.get("/project/render", tags=["audio"], response_class=Response)
    def render_mix(start: int = Query(ge=0), end: int = Query(gt=0)) -> Response:
        return wav_response(svc.mix_plan(), start, end)

    @api.get("/project/samples/{sample_id}/render", tags=["audio"], response_class=Response)
    def render_sample(sample_id: str) -> Response:
        plan = svc.sample_plan(sample_id)
        duration = svc.read(lambda _r, p: ops.sample_by_id(p, sample_id).duration)
        return wav_response(plan, 0, duration)

    @api.post("/project/exports", response_model=schemas.JobOut, status_code=202, tags=["audio"])
    def create_export(req: schemas.ExportRequest) -> dict[str, Any]:
        root = svc.root()

        def build(r: Path, p: Project) -> tuple[Any, int, int, str]:
            if req.kind == "sample":
                if req.sample_id is None:
                    raise ProjectError("sample_id is required to export a sample")
                sample = ops.sample_by_id(p, req.sample_id)
                return ops.sample_plan(p, r, sample.id), 0, sample.duration, sample.name
            if req.start is None or req.end is None or req.end <= req.start:
                raise ProjectError("A non-empty start/end selection is required")
            if req.end > ops.timeline_length(p):
                raise ProjectError("The selection extends past the end of the audio")
            return ops.mix_plan(p, r), req.start, req.end, "selection"

        plan, start, end, default_name = svc.read(build)
        job = state.jobs.submit(
            "export_audio",
            {
                "project_dir": str(root),
                "kind": req.kind,
                "plan": plan.to_dict(),
                "start": start,
                "end": end,
                "name": req.name or default_name,
                "format": req.format,
                "subtype": req.subtype,
                "describe": {"kind": req.kind, "sample": req.sample_id},
            },
        )
        return job.snapshot()

    @api.get("/project/exports/{filename}", tags=["audio"], response_class=FileResponse)
    def download_export(filename: str) -> FileResponse:
        path = svc.export_path(filename)
        return FileResponse(
            path,
            media_type=_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
            filename=path.name,
        )

    # --- devices and recording -----------------------------------------------------------

    @api.get("/audio/devices", response_model=schemas.AudioDevices, tags=["recording"])
    def devices() -> dict[str, Any]:
        return state.devices_snapshot()

    @api.post(
        "/audio/devices/refresh", response_model=schemas.JobOut, status_code=202, tags=["recording"]
    )
    def refresh_devices() -> dict[str, Any]:
        return state.start_device_listing().snapshot()

    @api.post("/audio/record", response_model=schemas.JobOut, status_code=202, tags=["recording"])
    def start_recording(req: schemas.RecordRequest) -> dict[str, Any]:
        root = svc.root()
        if state.jobs.active({"record_audio"}):
            raise ProjectError("A recording is already in progress")
        job = state.jobs.submit(
            "record_audio",
            {
                "project_dir": str(root),
                "device_index": req.device_index,
                "device_name": req.device_name,
                "kind": req.kind,
                "max_seconds": min(req.max_seconds, settings.max_recording_seconds),
            },
        )
        return job.snapshot()

    return api
