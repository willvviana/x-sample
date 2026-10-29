"""Non-destructive edit operations on a Project (no audio is read or written here).

Every operation validates integer sample ranges, mutates the manifest model and appends
a history entry. Callers persist with `store.save_project`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xsample.audio.render import RenderPlan, Segment
from xsample.project import layout
from xsample.project.model import (
    Clip,
    HistoryEntry,
    Project,
    Sample,
    Source,
    Track,
    source_to_timeline,  # noqa: F401  (re-exported for callers)
    timeline_to_source,
    unity_time_map,
)
from xsample.project.store import ProjectError, new_id, now_iso

MAX_HISTORY = 5000


def log(p: Project, operation: str, **params: Any) -> None:
    p.history.append(
        HistoryEntry(id=new_id("op"), at=now_iso(), operation=operation, params=params)
    )
    del p.history[:-MAX_HISTORY]


def track_by_id(p: Project, track_id: str) -> Track:
    for t in p.tracks:
        if t.id == track_id:
            return t
    raise ProjectError("Track not found")


def clip_by_id(p: Project, clip_id: str) -> Clip:
    for c in p.clips:
        if c.id == clip_id:
            return c
    raise ProjectError("Clip not found")


def sample_by_id(p: Project, sample_id: str) -> Sample:
    for s in p.samples:
        if s.id == sample_id:
            return s
    raise ProjectError("Sample not found")


def media_frames(p: Project, source_id: str) -> int:
    try:
        source = p.source(source_id)
    except KeyError as exc:
        raise ProjectError("Source not found") from exc
    if source.media is None:
        raise ProjectError("The source has no decoded audio; rebuild it first")
    return source.media.frames


def timeline_length(p: Project) -> int:
    return max((c.timeline_end for c in p.clips), default=0)


def _check_free(p: Project, track_id: str, start: int, end: int, ignore: str | None = None) -> None:
    for other in p.clips:
        if (
            other.track_id == track_id
            and other.id != ignore
            and (start < other.timeline_end and other.timeline_start < end)
        ):
            raise ProjectError("That would overlap another clip on the same track")


def _stem(name: str) -> str:
    return (Path(name).stem or name).strip() or "Audio"


# --- tracks ---------------------------------------------------------------------------


def add_track(p: Project, name: str | None = None) -> Track:
    track = Track(
        id=new_id("trk"), name=(name or f"Track {len(p.tracks) + 1}").strip()[:120] or "Track"
    )
    p.tracks.append(track)
    log(p, "track.add", track=track.id, name=track.name)
    return track


def update_track(
    p: Project,
    track_id: str,
    name: str | None = None,
    muted: bool | None = None,
    solo: bool | None = None,
    gain: float | None = None,
) -> Track:
    t = track_by_id(p, track_id)
    changes: dict[str, Any] = {}
    if name is not None:
        if not name.strip():
            raise ProjectError("Track name is empty")
        t.name = name.strip()[:120]
        changes["name"] = t.name
    if muted is not None:
        t.muted = muted
        changes["muted"] = muted
    if solo is not None:
        t.solo = solo
        changes["solo"] = solo
    if gain is not None:
        if not 0.0 <= gain <= 4.0:
            raise ProjectError("Track gain must be between 0 and 4")
        t.gain = gain
        changes["gain"] = gain
    log(p, "track.update", track=track_id, **changes)
    return t


def remove_track(p: Project, track_id: str) -> None:
    track_by_id(p, track_id)
    p.clips = [c for c in p.clips if c.track_id != track_id]
    p.tracks = [t for t in p.tracks if t.id != track_id]
    log(p, "track.remove", track=track_id)


# --- sources and clips ------------------------------------------------------------------


def add_imported_source(p: Project, source: Source) -> tuple[Source, Clip]:
    """Register a source (once per content hash) and put it on a new track at time 0."""
    existing = next((s for s in p.sources if s.id == source.id), None)
    if existing is None:
        p.sources.append(source)
        existing = source
    elif source.media is not None:
        existing.media = source.media
    if existing.media is None:
        raise ProjectError("The imported source has no decoded audio")
    track = add_track(p, _stem(existing.original_name))
    clip = Clip(
        id=new_id("clp"),
        track_id=track.id,
        source_id=existing.id,
        source_in=0,
        timeline_start=0,
        duration=existing.media.frames,
        time_map=unity_time_map(0, 0, existing.media.frames),
    )
    p.clips.append(clip)
    log(
        p, "source.import", source=existing.id, clip=clip.id, track=track.id, origin=existing.origin
    )
    return existing, clip


def trim_clip(p: Project, clip_id: str, start: int, end: int) -> Clip:
    """Keep only timeline range [start, end) of the clip. The source is untouched."""
    c = clip_by_id(p, clip_id)
    if not (c.timeline_start <= start < end <= c.timeline_end):
        raise ProjectError("The trim range must lie inside the clip")
    new_in = timeline_to_source(c, start)
    c.source_in, c.timeline_start, c.duration = new_in, start, end - start
    c.time_map = unity_time_map(new_in, start, end - start)
    log(p, "clip.trim", clip=clip_id, start=start, end=end)
    return c


def reset_trim(p: Project, clip_id: str) -> Clip:
    """Restore the clip to the full source, keeping the audible audio where it is when possible."""
    c = clip_by_id(p, clip_id)
    total = media_frames(p, c.source_id)
    start = max(0, c.timeline_start - c.source_in)
    _check_free(p, c.track_id, start, start + total, ignore=c.id)
    c.source_in, c.timeline_start, c.duration = 0, start, total
    c.time_map = unity_time_map(0, start, total)
    log(p, "clip.reset_trim", clip=clip_id)
    return c


def move_clip(p: Project, clip_id: str, timeline_start: int) -> Clip:
    c = clip_by_id(p, clip_id)
    if timeline_start < 0:
        raise ProjectError("A clip cannot start before time zero")
    _check_free(p, c.track_id, timeline_start, timeline_start + c.duration, ignore=c.id)
    c.timeline_start = timeline_start
    c.time_map = unity_time_map(c.source_in, timeline_start, c.duration)
    log(p, "clip.move", clip=clip_id, start=timeline_start)
    return c


def delete_clip(p: Project, clip_id: str) -> None:
    clip_by_id(p, clip_id)
    p.clips = [c for c in p.clips if c.id != clip_id]
    log(p, "clip.delete", clip=clip_id)


# --- samples --------------------------------------------------------------------------------


def slice_clip_to_sample(
    p: Project, clip_id: str, start: int, end: int, name: str | None
) -> Sample:
    c = clip_by_id(p, clip_id)
    if not (c.timeline_start <= start < end <= c.timeline_end):
        raise ProjectError("The selection must lie inside the clip")
    source_in = timeline_to_source(c, start)
    if source_in + (end - start) > media_frames(p, c.source_id):
        raise ProjectError("The selection extends past the end of the source")
    sample = Sample(
        id=new_id("smp"),
        name=(name or f"Sample {len(p.samples) + 1}").strip()[:120] or "Sample",
        source_id=c.source_id,
        source_in=source_in,
        duration=end - start,
        created_at=now_iso(),
    )
    p.samples.append(sample)
    log(
        p, "sample.slice", sample=sample.id, clip=clip_id, source_in=source_in, duration=end - start
    )
    return sample


def rename_sample(p: Project, sample_id: str, name: str) -> Sample:
    s = sample_by_id(p, sample_id)
    if not name.strip():
        raise ProjectError("Sample name is empty")
    s.name = name.strip()[:120]
    log(p, "sample.rename", sample=sample_id, name=s.name)
    return s


def delete_sample(p: Project, sample_id: str) -> None:
    sample_by_id(p, sample_id)
    p.samples = [s for s in p.samples if s.id != sample_id]
    log(p, "sample.delete", sample=sample_id)


def place_sample(p: Project, sample_id: str, track_id: str | None, at: int) -> Clip:
    s = sample_by_id(p, sample_id)
    if at < 0:
        raise ProjectError("A clip cannot start before time zero")
    if track_id is None:
        track_id = add_track(p, s.name).id
    else:
        track_by_id(p, track_id)
        _check_free(p, track_id, at, at + s.duration)
    clip = Clip(
        id=new_id("clp"),
        track_id=track_id,
        source_id=s.source_id,
        source_in=s.source_in,
        timeline_start=at,
        duration=s.duration,
        time_map=unity_time_map(s.source_in, at, s.duration),
    )
    p.clips.append(clip)
    log(p, "sample.place", sample=sample_id, clip=clip.id, track=track_id, at=at)
    return clip


# --- render plans -----------------------------------------------------------------------------


def _segment(
    root: Path, p: Project, source_id: str, source_in: int, start: int, dur: int, gain: float
) -> Segment:
    source = p.source(source_id)
    if source.media is None:
        raise ProjectError(f"Audio for '{source.original_name}' needs to be rebuilt first")
    path = layout.resolve_inside(root, source.media.path)
    if not path.is_file():
        raise ProjectError(f"Audio for '{source.original_name}' is missing; rebuild it first")
    return Segment(str(path), source.media.channels, source_in, start, dur, gain)


def mix_plan(p: Project, root: Path) -> RenderPlan:
    """All audible clips: honours mute/solo and track gain."""
    soloing = any(t.solo for t in p.tracks)
    gains = {t.id: t.gain for t in p.tracks if not t.muted and (t.solo or not soloing)}
    segments = tuple(
        _segment(root, p, c.source_id, c.source_in, c.timeline_start, c.duration, gains[c.track_id])
        for c in p.clips
        if c.track_id in gains
    )
    return RenderPlan(p.sample_rate, segments)


def sample_plan(p: Project, root: Path, sample_id: str) -> RenderPlan:
    s = sample_by_id(p, sample_id)
    return RenderPlan(
        p.sample_rate, (_segment(root, p, s.source_id, s.source_in, 0, s.duration, 1.0),)
    )
