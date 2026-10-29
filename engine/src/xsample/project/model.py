"""Project manifest models (ADR-0006).

All positions are integer sample counts at `Project.sample_rate`. Original audio lives in
`sources/` and is never modified; clips reference a source range through a source-to-
timeline time map (piecewise-linear anchors). Phase 1 only creates unity (1:1) maps.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# 2: adds `analysis` (user corrections to detected analysis, ADR-0011). Format-1 manifests load
# unchanged (no corrections); they are written back as format 2.
FORMAT_VERSION = 2
MAX_CHORD_CORRECTIONS = 4000
MAX_TAGS = 100
MAX_TRACKS = 64


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProcessingRecord(_M):
    """Versioned provenance of a derived artefact (decode, export, ...)."""

    id: str
    operation: str
    engine_version: str
    created_at: str
    tool: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)


class Media(_M):
    """Re-creatable decoded working copy of a source at the project rate."""

    path: str
    peaks_path: str
    sample_rate: int = Field(gt=0)
    channels: int = Field(ge=1, le=2)
    frames: int = Field(gt=0)
    pcm_sha256: str
    peak: float
    rms: float
    processing: list[ProcessingRecord] = Field(default_factory=list)


class Source(_M):
    id: str = Field(pattern=r"^src_[0-9a-f]{12}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_name: str
    stored_path: str
    size_bytes: int = Field(gt=0)
    format_name: str
    codec: str
    duration_s: float | None = None
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    bit_depth: int | None = None
    bit_rate: int | None = None
    origin: Literal["import", "recording"] = "import"
    device: str | None = None
    imported_at: str
    media: Media | None = None


class Anchor(_M):
    source: int = Field(ge=0)
    timeline: int = Field(ge=0)


class Track(_M):
    id: str
    name: str = Field(min_length=1, max_length=120)
    muted: bool = False
    solo: bool = False
    gain: float = Field(default=1.0, ge=0.0, le=4.0)


class Clip(_M):
    id: str
    track_id: str
    source_id: str
    source_in: int = Field(ge=0)
    timeline_start: int = Field(ge=0)
    duration: int = Field(gt=0)
    time_map: list[Anchor]

    @property
    def timeline_end(self) -> int:
        return self.timeline_start + self.duration


class Sample(_M):
    """A named slice of a source (non-destructive); exported or placed on a track on demand."""

    id: str
    name: str = Field(min_length=1, max_length=120)
    source_id: str
    source_in: int = Field(ge=0)
    duration: int = Field(gt=0)
    created_at: str


class ExportRecord(_M):
    id: str
    filename: str
    kind: Literal["sample", "selection"]
    format: str
    subtype: str
    frames: int
    sample_rate: int
    channels: int
    size_bytes: int
    created_at: str
    processing: ProcessingRecord


class HistoryEntry(_M):
    id: str
    at: str
    operation: str
    params: dict[str, Any] = Field(default_factory=dict)


class BeatGridCorrection(_M):
    """Corrections to the detected beat grid, applied in this order: bpm or scale, offset,
    meter, downbeat shift."""

    bpm: float | None = Field(
        default=None, ge=20.0, le=300.0, description="replace the grid with a constant tempo"
    )
    scale: Literal["half", "normal", "double"] = "normal"
    half_phase: int = Field(default=0, ge=0, le=1, description="which beats 'half' keeps")
    offset: int = Field(default=0, ge=-192000, le=192000, description="shift in samples")
    meter: int | None = Field(default=None, ge=2, le=12)
    downbeat_shift: int = Field(default=0, ge=0, le=11, description="beats to move beat 1 by")

    def is_identity(self) -> bool:
        return self == BeatGridCorrection()


class KeyCorrection(_M):
    tonic: int = Field(ge=0, le=11)
    mode: Literal["major", "minor"]


class ChordCorrection(_M):
    """Replaces whatever was detected between start and end (source samples)."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    root: int | None = Field(default=None, ge=0, le=11, description="None = no chord")
    quality: Literal["maj", "min", "7", "maj7", "min7"] | None = None
    bass: int | None = Field(default=None, ge=0, le=11)

    @model_validator(mode="after")
    def _check(self) -> ChordCorrection:
        if self.end <= self.start:
            raise ValueError("chord end must be after its start")
        if (self.root is None) != (self.quality is None):
            raise ValueError("root and quality must both be set, or both be empty")
        return self


class TagCorrection(_M):
    kind: Literal["genre", "instrument"]
    label: str = Field(min_length=1, max_length=60)


class AnalysisCorrections(_M):
    """User corrections for one source. Kept in the manifest, never in the analysis cache, so
    re-running the analysis does not lose them."""

    beat_grid: BeatGridCorrection = Field(default_factory=BeatGridCorrection)
    key: KeyCorrection | None = None
    chords: list[ChordCorrection] = Field(default_factory=list, max_length=MAX_CHORD_CORRECTIONS)
    tags: list[TagCorrection] = Field(default_factory=list, max_length=MAX_TAGS)
    updated_at: str | None = None


class Project(_M):
    format_version: int = FORMAT_VERSION
    id: str
    name: str = Field(min_length=1, max_length=120)
    created_at: str
    modified_at: str
    revision: int = 0
    engine_version: str
    sample_rate: int = Field(gt=0)
    tracks: list[Track] = Field(default_factory=list)
    clips: list[Clip] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    samples: list[Sample] = Field(default_factory=list)
    exports: list[ExportRecord] = Field(default_factory=list)
    history: list[HistoryEntry] = Field(default_factory=list)
    analysis: dict[str, AnalysisCorrections] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_integrity(self) -> Project:
        problems = integrity_problems(self)
        if problems:
            raise ValueError("; ".join(problems[:5]))
        return self

    def source(self, source_id: str) -> Source:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(source_id)


# --- time map -------------------------------------------------------------------------


def unity_time_map(source_in: int, timeline_start: int, duration: int) -> list[Anchor]:
    return [
        Anchor(source=source_in, timeline=timeline_start),
        Anchor(source=source_in + duration, timeline=timeline_start + duration),
    ]


def _map(anchors: list[Anchor], value: int, frm: str, to: str) -> int:
    """Integer piecewise-linear interpolation (floor), extrapolating on the end segments."""
    pts = [(getattr(a, frm), getattr(a, to)) for a in anchors]
    if len(pts) < 2:
        raise ValueError("a time map needs at least two anchors")
    seg = 0
    for i in range(len(pts) - 1):
        seg = i
        if value < pts[i + 1][0]:
            break
    (x0, y0), (x1, y1) = pts[seg], pts[seg + 1]
    return int(y0 + ((value - x0) * (y1 - y0)) // (x1 - x0))


def timeline_to_source(clip: Clip, position: int) -> int:
    return _map(clip.time_map, position, "timeline", "source")


def source_to_timeline(clip: Clip, position: int) -> int:
    return _map(clip.time_map, position, "source", "timeline")


def is_unity(clip: Clip) -> bool:
    return all(
        (b.source - a.source) == (b.timeline - a.timeline)
        for a, b in zip(clip.time_map, clip.time_map[1:], strict=False)
    )


# --- integrity --------------------------------------------------------------------------


def integrity_problems(p: Project) -> list[str]:
    problems: list[str] = []

    def unique(kind: str, ids: list[str]) -> None:
        if len(ids) != len(set(ids)):
            problems.append(f"duplicate {kind} ids")

    unique("track", [t.id for t in p.tracks])
    unique("clip", [c.id for c in p.clips])
    unique("source", [s.id for s in p.sources])
    unique("sample", [s.id for s in p.samples])
    if len(p.tracks) > MAX_TRACKS:
        problems.append(f"more than {MAX_TRACKS} tracks")

    tracks = {t.id for t in p.tracks}
    sources = {s.id: s for s in p.sources}
    for s in p.sources:
        if s.media and s.media.sample_rate != p.sample_rate:
            problems.append(f"source {s.id}: media rate differs from the project rate")

    def frames_of(source_id: str) -> int | None:
        src = sources.get(source_id)
        return src.media.frames if src and src.media else None

    for c in p.clips:
        if c.track_id not in tracks:
            problems.append(f"clip {c.id}: unknown track")
        if c.source_id not in sources:
            problems.append(f"clip {c.id}: unknown source")
            continue
        if len(c.time_map) < 2:
            problems.append(f"clip {c.id}: time map needs two anchors")
        else:
            a, b = c.time_map[0], c.time_map[-1]
            if a.source != c.source_in or a.timeline != c.timeline_start:
                problems.append(f"clip {c.id}: time map does not start at the clip start")
            if b.timeline != c.timeline_end:
                problems.append(f"clip {c.id}: time map does not end at the clip end")
            for x, y in zip(c.time_map, c.time_map[1:], strict=False):
                if y.source <= x.source or y.timeline <= x.timeline:
                    problems.append(f"clip {c.id}: time map must be strictly increasing")
                    break
            if not is_unity(c):
                problems.append(f"clip {c.id}: only unity time maps are supported")
        total = frames_of(c.source_id)
        if total is not None and c.source_in + c.duration > total:
            problems.append(f"clip {c.id}: extends past the end of its source")
    for source_id in p.analysis:
        if source_id not in sources:
            problems.append(f"analysis corrections for unknown source {source_id}")
    for smp in p.samples:
        smp_total = frames_of(smp.source_id)
        if smp.source_id not in sources:
            problems.append(f"sample {smp.id}: unknown source")
        elif smp_total is not None and smp.source_in + smp.duration > smp_total:
            problems.append(f"sample {smp.id}: extends past the end of its source")

    by_track: dict[str, list[Clip]] = {}
    for c in p.clips:
        by_track.setdefault(c.track_id, []).append(c)
    for track_id, clips in by_track.items():
        clips.sort(key=lambda clip: clip.timeline_start)
        for left, right in itertools.pairwise(clips):
            if right.timeline_start < left.timeline_end:
                problems.append(f"track {track_id}: clips {left.id} and {right.id} overlap")
    return problems
