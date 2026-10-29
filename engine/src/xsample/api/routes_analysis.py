"""Musical analysis routes (v0.2.0, ADR-0011). All require the API token.

Detected results are read from the versioned sidecar cache; corrections live in the manifest
and are only changed through `ProjectService.edit`.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError

from xsample.analysis.effective import EffectiveAnalysis, effective
from xsample.analysis.run import cache_key
from xsample.analysis.schema import ANALYZER_VERSION, AnalysisResult
from xsample.api import schemas
from xsample.project import layout, ops
from xsample.project.model import AnalysisCorrections, Project
from xsample.project.store import ProjectError, now_iso

if TYPE_CHECKING:
    from xsample.api.app import EngineState

AnalysisStatus = Literal["none", "queued", "running", "ready", "failed", "cancelled", "stale"]


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    force: bool = False


class AnalysisView(BaseModel):
    source_id: str
    status: AnalysisStatus
    analyzer_version: str
    cache_key: str | None
    job: schemas.JobOut | None
    error: str | None
    result: AnalysisResult | None
    corrections: AnalysisCorrections
    effective: EffectiveAnalysis | None


class _SidecarCache:
    """Parsed sidecars keyed by (path, mtime, size); a 10-minute result is a few hundred kB."""

    def __init__(self, capacity: int = 8) -> None:
        self._items: OrderedDict[tuple[str, int, int], AnalysisResult] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def get(self, path: Path) -> AnalysisResult | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        try:
            result = AnalysisResult.model_validate_json(path.read_bytes())
        except (OSError, ValidationError):
            return None  # damaged or from another schema: treated as not analysed
        with self._lock:
            self._items[key] = result
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
        return result


def build_analysis_router(state: EngineState) -> APIRouter:
    api = APIRouter()
    svc = state.project
    sidecars = _SidecarCache()

    def locate(source_id: str) -> tuple[Path, dict[str, Any], AnalysisCorrections]:
        def get(root: Path, p: Project) -> tuple[Path, dict[str, Any], AnalysisCorrections]:
            try:
                source = p.source(source_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="source not found") from exc
            if source.media is None:
                raise ProjectError("The working copy is missing; rebuild this source first")
            params = {
                "project_dir": str(root),
                "source_id": source.id,
                "media_path": source.media.path,
                "media_sha256": source.media.pcm_sha256,
                "sample_rate": source.media.sample_rate,
                "frames": source.media.frames,
            }
            corrections = p.analysis.get(source_id, AnalysisCorrections())
            return root, params, corrections.model_copy(deep=True)

        return svc.read(get)

    def latest_job(root: Path, source_id: str) -> dict[str, Any] | None:
        for job in state.jobs.list():  # newest first
            p = job["params"]
            if (
                job["kind"] == "analyze_audio"
                and p.get("source_id") == source_id
                and Path(p.get("project_dir", "")).resolve() == root
            ):
                return job
        return None

    def view(source_id: str) -> AnalysisView:
        root, params, corrections = locate(source_id)
        key = cache_key(params["media_sha256"])
        folder = layout.resolve_inside(root, f"{layout.ANALYSIS_RESULTS}/{source_id}")
        result = sidecars.get(folder / f"{key}.json")
        job = latest_job(root, source_id)
        status: AnalysisStatus
        error = None
        if job is not None and job["state"] in ("queued", "running", "cancelling"):
            status = "queued" if job["state"] == "queued" else "running"
        elif result is not None:
            status = "ready"
        elif job is not None and job["state"] in ("failed", "cancelled"):
            status = "failed" if job["state"] == "failed" else "cancelled"
            error = job["error"]
        elif folder.is_dir() and any(folder.glob("*.json")):
            status = "stale"
        else:
            status = "none"
        return AnalysisView(
            source_id=source_id,
            status=status,
            analyzer_version=ANALYZER_VERSION,
            cache_key=key,
            job=schemas.JobOut(**job) if job is not None else None,
            error=error,
            result=result,
            corrections=corrections,
            effective=effective(result, corrections) if result is not None else None,
        )

    @api.get(
        "/project/sources/{source_id}/analysis", response_model=AnalysisView, tags=["analysis"]
    )
    def get_analysis(source_id: str) -> AnalysisView:
        return view(source_id)

    @api.post(
        "/project/sources/{source_id}/analysis", response_model=AnalysisView, tags=["analysis"]
    )
    def run_analysis(source_id: str, req: AnalyzeRequest) -> AnalysisView:
        current = view(source_id)
        if current.status in ("queued", "running"):
            return current
        if current.status == "ready" and not req.force:
            return current
        _, params, _ = locate(source_id)
        state.jobs.submit("analyze_audio", params)
        return view(source_id)

    @api.put(
        "/project/sources/{source_id}/analysis/corrections",
        response_model=AnalysisView,
        tags=["analysis"],
    )
    def put_corrections(source_id: str, req: AnalysisCorrections) -> AnalysisView:
        _, params, _ = locate(source_id)
        frames = int(params["frames"])
        if any(c.end > frames for c in req.chords):
            raise ProjectError("A chord correction extends past the end of the audio")
        seen = set()
        for tag in req.tags:
            ident = (tag.kind, tag.label.strip().lower())
            if ident in seen:
                raise ProjectError(f"Duplicate tag: {tag.label}")
            seen.add(ident)
        fixed = req.model_copy(update={"updated_at": now_iso()})

        def change(p: Project) -> None:
            p.analysis[source_id] = fixed
            ops.log(p, "analysis.correct", source=source_id)

        svc.edit(change)
        return view(source_id)

    @api.delete(
        "/project/sources/{source_id}/analysis/corrections",
        response_model=AnalysisView,
        tags=["analysis"],
    )
    def reset_corrections(source_id: str) -> AnalysisView:
        locate(source_id)

        def change(p: Project) -> None:
            p.analysis.pop(source_id, None)
            ops.log(p, "analysis.reset_corrections", source=source_id)

        svc.edit(change)
        return view(source_id)

    return api
