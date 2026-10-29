"""Mix a render plan to float32 stereo blocks; encode previews and export files.

A plan is a flat list of segments (one per audible clip) resolved by the API from the
project manifest, so workers never read or write the manifest. Phase 1 supports only
unity time maps (no stretching), so a segment maps source frames 1:1 onto the timeline.
"""

from __future__ import annotations

import hashlib
import io
import os
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf

from xsample.audio.errors import AudioError, Cancelled
from xsample.audio.media import Metrics, processing_record

F32 = npt.NDArray[np.float32]
BLOCK_FRAMES = 65536
OUT_CHANNELS = 2

EXPORT_FORMATS: dict[str, tuple[str, str, set[str]]] = {
    "wav": ("WAV", ".wav", {"PCM_16", "PCM_24", "FLOAT"}),
    "flac": ("FLAC", ".flac", {"PCM_16", "PCM_24"}),
}


@dataclass(frozen=True)
class Segment:
    media_path: str
    media_channels: int
    source_in: int
    timeline_start: int
    duration: int
    gain: float

    @property
    def timeline_end(self) -> int:
        return self.timeline_start + self.duration


@dataclass(frozen=True)
class RenderPlan:
    sample_rate: int
    segments: tuple[Segment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "segments": [s.__dict__ for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RenderPlan:
        return cls(int(data["sample_rate"]), tuple(Segment(**s) for s in data["segments"]))

    @property
    def length(self) -> int:
        return max((s.timeline_end for s in self.segments), default=0)


def render_blocks(
    plan: RenderPlan,
    start: int,
    end: int,
    block_frames: int = BLOCK_FRAMES,
    cancelled: Callable[[], bool] | None = None,
) -> Iterator[F32]:
    """Yield float32 (n, 2) blocks covering [start, end) of the timeline."""
    if not 0 <= start <= end:
        raise ValueError("invalid render range")
    handles: dict[str, sf.SoundFile] = {}
    try:
        pos = start
        while pos < end:
            if cancelled and cancelled():
                raise Cancelled()
            stop = min(end, pos + block_frames)
            out = np.zeros((stop - pos, OUT_CHANNELS), dtype=np.float32)
            for seg in plan.segments:
                a, b = max(pos, seg.timeline_start), min(stop, seg.timeline_end)
                if a >= b:
                    continue
                handle = handles.get(seg.media_path)
                if handle is None:
                    handle = handles[seg.media_path] = sf.SoundFile(seg.media_path)
                handle.seek(seg.source_in + (a - seg.timeline_start))
                data = handle.read(b - a, dtype="float32", always_2d=True)
                if len(data) < b - a:  # media shorter than the manifest claims: pad silence
                    data = np.pad(data, ((0, b - a - len(data)), (0, 0)))
                if data.shape[1] == 1:
                    data = np.repeat(data, OUT_CHANNELS, axis=1)
                out[a - pos : b - pos] += data[:, :OUT_CHANNELS] * np.float32(seg.gain)
            yield out
            pos = stop
    finally:
        for handle in handles.values():
            handle.close()


def preview_wav(plan: RenderPlan, start: int, end: int) -> bytes:
    """16-bit stereo WAV of a short region for in-browser playback."""
    blocks = list(render_blocks(plan, start, end))
    audio = np.concatenate(blocks) if blocks else np.zeros((0, OUT_CHANNELS), dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(audio, -1.0, 1.0), plan.sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def export_audio(
    plan: RenderPlan,
    start: int,
    end: int,
    dest_dir: Path,
    filename_stem: str,
    fmt: str,
    subtype: str,
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    describe: dict[str, Any],
) -> dict[str, Any]:
    metrics = Metrics()
    if fmt not in EXPORT_FORMATS:
        raise AudioError(f"unsupported export format {fmt!r}")
    sf_format, ext, subtypes = EXPORT_FORMATS[fmt]
    if subtype not in subtypes:
        raise AudioError(f"{fmt.upper()} export supports {', '.join(sorted(subtypes))}")
    if end <= start:
        raise AudioError("Nothing to export: the selection is empty")
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / f"{filename_stem}{ext}"
    n = 1
    while final.exists():  # never overwrite an earlier export
        n += 1
        final = dest_dir / f"{filename_stem}-{n}{ext}"
    tmp = dest_dir / f".{uuid.uuid4().hex}.part"
    total = end - start
    done = 0
    peak = 0.0
    clipped = 0
    pcm = hashlib.sha256()
    try:
        with sf.SoundFile(
            tmp,
            mode="w",
            samplerate=plan.sample_rate,
            channels=OUT_CHANNELS,
            format=sf_format,
            subtype=subtype,
        ) as out:
            for block in render_blocks(plan, start, end, cancelled=cancelled):
                peak = max(peak, float(np.max(np.abs(block))))
                clipped += int(np.count_nonzero(np.abs(block) > 1.0))
                out.write(block if subtype == "FLOAT" else np.clip(block, -1.0, 1.0))
                pcm.update(block.tobytes())
                done += len(block)
                progress(done / total, "Rendering")
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    record = processing_record(
        "export.render",
        {"format": fmt, "subtype": subtype, "start": start, "end": end, **describe},
        [],
        {"file": final.name, "frames": total, "pcm_sha256": pcm.hexdigest()},
    )
    return {
        "filename": final.name,
        "frames": total,
        "sample_rate": plan.sample_rate,
        "channels": OUT_CHANNELS,
        "format": fmt,
        "subtype": subtype,
        "size_bytes": final.stat().st_size,
        "peak": peak,
        "clipped_samples": clipped,
        "processing": record,
        "metrics": metrics.snapshot(),
    }
