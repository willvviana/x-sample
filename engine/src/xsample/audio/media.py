"""Import pipeline (runs inside a worker process).

original file -> sources/<sha256>.<ext>   (immutable, read-only)
              -> cache/media/<key>.wav    (float32 working copy at the project rate/channels)
              -> analysis/peaks/<key>.npz (peak pyramid)

Everything is streamed in bounded chunks; memory does not depend on the file length.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import stat
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import soundfile as sf

from xsample import __version__
from xsample.audio import ffmpeg
from xsample.audio.errors import AudioError, Cancelled
from xsample.audio.peaks import PeakBuilder
from xsample.project import layout

CHUNK_FRAMES = 65536
HASH_BLOCK = 1024 * 1024
CHANNEL_MODES = ("keep", "mono", "stereo")


def sha256_file(
    path: Path,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[int], None] | None = None,
) -> str:
    digest = hashlib.sha256()
    done = 0
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK):
            if cancelled and cancelled():
                raise Cancelled()
            digest.update(block)
            done += len(block)
            if progress:
                progress(done)
    return digest.hexdigest()


def tool_versions() -> dict[str, str]:
    return {
        "ffmpeg": ffmpeg.tool_version("ffmpeg"),
        "libsndfile": str(sf.__libsndfile_version__),
        "python": platform.python_version(),
    }


def processing_record(
    operation: str, params: dict[str, Any], inputs: list[str], output: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex[:12],
        "operation": operation,
        "engine_version": __version__,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": tool_versions(),
        "params": params,
        "inputs": inputs,
        "output": output,
    }


class Metrics:
    """Wall/CPU time and peak memory of the current process, for performance records."""

    def __init__(self) -> None:
        self._proc = psutil.Process()
        self._t0 = time.perf_counter()
        self._cpu0 = self._proc.cpu_times()

    def snapshot(self) -> dict[str, float]:
        cpu = self._proc.cpu_times()
        mem = self._proc.memory_info()
        return {
            "wall_s": time.perf_counter() - self._t0,
            "cpu_user_s": cpu.user - self._cpu0.user,
            "cpu_system_s": cpu.system - self._cpu0.system,
            "peak_working_set_bytes": float(getattr(mem, "peak_wset", mem.rss)),
        }


def output_channels(native: int, mode: str) -> int:
    if mode not in CHANNEL_MODES:
        raise AudioError(f"unknown channel mode {mode!r}")
    if mode == "mono":
        return 1
    if mode == "stereo":
        return 2
    return min(native, 2)


def media_key(sha: str, sample_rate: int, channels: int) -> str:
    return f"{sha[:16]}-{sample_rate}-{channels}"


def _remove(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _store_original(incoming: Path, project: Path, sha: str) -> tuple[Path, bool]:
    """Move the incoming file into sources/ (content-addressed). Returns (path, was_new)."""
    ext = incoming.suffix.lower()
    if ext not in layout.SOURCE_EXTENSIONS:
        ext = ".bin"
    target = project / layout.SOURCES / f"{sha}{ext}"
    existing = next((p for p in (project / layout.SOURCES).glob(f"{sha}.*")), None)
    if existing is not None:
        _remove(incoming)  # the same bytes are already in the project
        return existing, False
    os.replace(incoming, target)
    target.chmod(stat.S_IREAD)  # originals are immutable
    return target, True


def import_audio(
    project_dir: str,
    incoming_path: str,
    original_name: str,
    sample_rate: int,
    channels: str,
    origin: str,
    device: str | None,
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    rebuild_stored_path: str | None = None,
) -> dict[str, Any]:
    """Import `incoming_path`, or (rebuild) regenerate media for an original already in sources/."""
    metrics = Metrics()
    project = Path(project_dir)
    rebuilding = rebuild_stored_path is not None
    incoming = project / rebuild_stored_path if rebuild_stored_path else Path(incoming_path)
    stored: Path | None = None
    was_new = False
    tmp = project / layout.TMP / f"{uuid.uuid4().hex}.wav"
    try:
        if not incoming.is_file():
            raise AudioError(
                "The original file is missing" if rebuilding else "The uploaded file is missing"
            )
        size = incoming.stat().st_size
        if size == 0:
            raise AudioError("The file is empty")

        progress(0.02, "Hashing original")
        sha = sha256_file(
            incoming, cancelled, lambda n: progress(0.02 + 0.08 * n / size, "Hashing original")
        )
        if rebuilding:
            stored = incoming
            if sha not in stored.name:
                raise AudioError("The stored original no longer matches its checksum")
        else:
            stored, was_new = _store_original(incoming, project, sha)

        progress(0.1, "Reading file information")
        info = ffmpeg.probe(stored)
        out_ch = output_channels(info.channels, channels)
        key = media_key(sha, sample_rate, out_ch)
        media_path = project / layout.MEDIA / f"{key}.wav"
        peaks_path = project / layout.PEAKS / f"{key}.npz"
        source = {
            "id": f"src_{sha[:12]}",
            "sha256": sha,
            "original_name": original_name,
            "stored_path": f"{layout.SOURCES}/{stored.name}",
            "size_bytes": stored.stat().st_size,
            "format_name": info.format_name,
            "codec": info.codec_name,
            "duration_s": info.duration_s,
            "sample_rate": info.sample_rate,
            "channels": info.channels,
            "bit_depth": info.bit_depth,
            "bit_rate": info.bit_rate,
            "origin": origin,
            "device": device,
        }
        params = {
            "sample_rate": sample_rate,
            "channels_mode": channels,
            "output_channels": out_ch,
            "resampled": info.sample_rate != sample_rate,
            "remixed": info.channels != out_ch,
        }

        if media_path.is_file() and peaks_path.is_file():
            # Same content and settings were imported before: describe the existing copy.
            progress(0.5, "Reusing existing decoded audio")
            digest, peak, sumsq, frames = hashlib.sha256(), 0.0, 0.0, 0
            with sf.SoundFile(media_path) as existing:
                for block in existing.blocks(
                    blocksize=CHUNK_FRAMES, dtype="float32", always_2d=True
                ):
                    if cancelled():
                        raise Cancelled()
                    digest.update(block.tobytes())
                    peak = max(peak, float(np.max(np.abs(block))))
                    sumsq += float(np.sum(block.astype(np.float64) ** 2))
                    frames += len(block)
            progress(1.0, "Done")
            return {
                "source": source,
                "media": {
                    "path": f"{layout.MEDIA}/{media_path.name}",
                    "peaks_path": f"{layout.PEAKS}/{peaks_path.name}",
                    "sample_rate": sample_rate,
                    "channels": out_ch,
                    "frames": frames,
                    "pcm_sha256": digest.hexdigest(),
                    "peak": peak,
                    "rms": float(np.sqrt(sumsq / max(1, frames * out_ch))),
                    "processing": [
                        processing_record(
                            "import.reuse", params, [sha], {"frames": frames, "reused": True}
                        )
                    ],
                },
                "reused_media": True,
                "metrics": metrics.snapshot(),
            }

        expected = max(1, int((info.duration_s or 0) * sample_rate))
        builder = PeakBuilder(sample_rate, out_ch)
        pcm = hashlib.sha256()
        frames = 0
        peak = 0.0
        sumsq = 0.0
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(
            tmp, mode="w", samplerate=sample_rate, channels=out_ch, format="WAV", subtype="FLOAT"
        ) as out:
            for chunk in ffmpeg.decode_f32(
                stored,
                out_channels=out_ch,
                sample_rate=sample_rate if info.sample_rate != sample_rate else None,
                chunk_frames=CHUNK_FRAMES,
                cancelled=cancelled,
            ):
                out.write(chunk)
                builder.add(chunk)
                pcm.update(chunk.tobytes())
                peak = max(peak, float(np.max(np.abs(chunk))))
                sumsq += float(np.sum(chunk.astype(np.float64) ** 2))
                frames += len(chunk)
                progress(0.1 + 0.85 * min(1.0, frames / expected), "Decoding")
        if frames == 0:
            raise AudioError("The file decoded to zero audio frames")

        progress(0.96, "Writing waveform data")
        pyramid = builder.finalize()
        media_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, media_path)
        pyramid.save(peaks_path)

        media = {
            "path": f"{layout.MEDIA}/{media_path.name}",
            "peaks_path": f"{layout.PEAKS}/{peaks_path.name}",
            "sample_rate": sample_rate,
            "channels": out_ch,
            "frames": frames,
            "pcm_sha256": pcm.hexdigest(),
            "peak": peak,
            "rms": float(np.sqrt(sumsq / (frames * out_ch))),
            "processing": [
                processing_record(
                    "import.decode",
                    params,
                    [sha],
                    {"frames": frames, "pcm_sha256": pcm.hexdigest(), "peak": peak},
                )
            ],
        }
        progress(1.0, "Done")
        return {
            "source": source,
            "media": media,
            "reused_media": False,
            "metrics": metrics.snapshot(),
        }
    except BaseException:
        _remove(tmp)
        # A cancelled/failed import must not leave the upload (or a brand-new original) behind.
        # A rebuild never touches the stored original.
        if not rebuilding:
            _remove(incoming)
            if was_new and stored is not None:
                with contextlib.suppress(OSError):
                    stored.chmod(stat.S_IWRITE | stat.S_IREAD)
                _remove(stored)
        raise
