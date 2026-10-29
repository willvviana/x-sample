"""What the API shows of a project: the manifest without history, plus derived status."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from xsample.project import layout
from xsample.project.model import Clip, ExportRecord, Project, Sample, Source, Track


class SourceView(Source):
    status: Literal["ok", "needs_rebuild", "missing_original"]


class ProjectView(BaseModel):
    path: str
    recovered_from_backup: bool
    format_version: int
    id: str
    name: str
    sample_rate: int
    created_at: str
    modified_at: str
    revision: int
    tracks: list[Track]
    clips: list[Clip]
    sources: list[SourceView]
    samples: list[Sample]
    exports: list[ExportRecord]
    length_samples: int
    issues: list[str]


def source_status(root: Path, source: Source) -> Literal["ok", "needs_rebuild", "missing_original"]:
    if not (root / source.stored_path).is_file():
        return "missing_original"
    media = source.media
    if (
        media is None
        or not (root / media.path).is_file()
        or not (root / media.peaks_path).is_file()
    ):
        return "needs_rebuild"
    return "ok"


def build_view(root: Path, project: Project, recovered: bool) -> ProjectView:
    sources = [SourceView(**s.model_dump(), status=source_status(root, s)) for s in project.sources]
    issues = []
    if recovered:
        issues.append("project.json was damaged; the last backup was loaded")
    for s in sources:
        if s.status == "missing_original":
            issues.append(f"The original file for '{s.original_name}' is missing")
        elif s.status == "needs_rebuild":
            issues.append(f"Waveform data for '{s.original_name}' must be rebuilt")
    return ProjectView(
        path=str(root),
        recovered_from_backup=recovered,
        format_version=project.format_version,
        id=project.id,
        name=project.name,
        sample_rate=project.sample_rate,
        created_at=project.created_at,
        modified_at=project.modified_at,
        revision=project.revision,
        tracks=project.tracks,
        clips=project.clips,
        sources=sources,
        samples=project.samples,
        exports=project.exports,
        length_samples=max((c.timeline_end for c in project.clips), default=0),
        issues=issues,
    )


__all__ = ["ProjectView", "SourceView", "build_view", "layout"]
