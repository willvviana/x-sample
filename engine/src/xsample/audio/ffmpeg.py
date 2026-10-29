"""FFmpeg/ffprobe wrappers: probing and bounded-chunk decoding to float32 PCM."""

from __future__ import annotations

import collections
import functools
import json
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from xsample.audio.errors import AudioError, Cancelled

F32 = npt.NDArray[np.float32]
_NO_WINDOW = 0x08000000
PROBE_TIMEOUT_S = 60


@functools.cache
def find_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise AudioError(f"{name} was not found on PATH; install FFmpeg to import audio")
    return path


@functools.cache
def tool_version(name: str) -> str:
    try:
        out = subprocess.run(
            [find_tool(name), "-version"],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=_NO_WINDOW,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError, AudioError):
        return "unknown"
    return out.splitlines()[0].strip() if out else "unknown"


@dataclass(frozen=True)
class MediaInfo:
    format_name: str
    codec_name: str
    duration_s: float | None
    sample_rate: int
    channels: int
    bit_depth: int | None
    bit_rate: int | None


def _int(value: object) -> int | None:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def probe(path: Path) -> MediaInfo:
    try:
        proc = subprocess.run(
            [
                find_tool("ffprobe"),
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                "-select_streams",
                "a",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PROBE_TIMEOUT_S,
            creationflags=_NO_WINDOW,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioError("Reading the file timed out; it may be corrupt") from exc
    if proc.returncode != 0:
        detail = (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise AudioError(f"Not a readable audio file: {detail}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AudioError("Not a readable audio file: unexpected probe output") from exc
    streams = data.get("streams") or []
    if not streams:
        raise AudioError("The file contains no audio stream")
    stream = streams[0]
    fmt = data.get("format") or {}
    rate = _int(stream.get("sample_rate"))
    channels = _int(stream.get("channels"))
    if rate is None or channels is None:
        raise AudioError("The audio stream has no valid sample rate or channel count")
    try:
        duration: float | None = float(stream.get("duration") or fmt.get("duration") or "")
    except (TypeError, ValueError):
        duration = None
    return MediaInfo(
        format_name=str(fmt.get("format_name", "unknown")),
        codec_name=str(stream.get("codec_name", "unknown")),
        duration_s=duration,
        sample_rate=rate,
        channels=channels,
        bit_depth=_int(stream.get("bits_per_raw_sample")) or _int(stream.get("bits_per_sample")),
        bit_rate=_int(stream.get("bit_rate")) or _int(fmt.get("bit_rate")),
    )


def decode_f32(
    path: Path,
    *,
    out_channels: int,
    sample_rate: int | None = None,
    chunk_frames: int = 65536,
    cancelled: Callable[[], bool] | None = None,
) -> Iterator[F32]:
    """Yield (frames, out_channels) float32 chunks. Resamples/remixes inside FFmpeg.

    Memory is bounded by `chunk_frames`, independent of the file length.
    """
    cmd = [find_tool("ffmpeg"), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn"]
    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]
    cmd += ["-ac", str(out_channels), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=_NO_WINDOW,
    )
    assert proc.stdout is not None and proc.stderr is not None
    stderr_tail: collections.deque[str] = collections.deque(maxlen=10)
    stderr = proc.stderr

    def pump() -> None:
        for line in stderr:
            stderr_tail.append(line.decode("utf-8", "replace").rstrip())

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    frame_bytes = 4 * out_channels
    finished = False
    try:
        while True:
            if cancelled and cancelled():
                raise Cancelled()
            raw = proc.stdout.read(chunk_frames * frame_bytes)
            if not raw:
                break
            usable = len(raw) - len(raw) % frame_bytes
            if usable:
                yield np.frombuffer(raw, dtype="<f4", count=usable // 4).reshape(-1, out_channels)
        code = proc.wait(timeout=30)
        reader.join(timeout=5)
        if code != 0:
            detail = " ".join(stderr_tail) or f"FFmpeg exited with code {code}"
            raise AudioError(f"Could not decode the audio: {detail}")
        finished = True
    finally:
        if not finished and proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdout.close()
        proc.stderr.close()
