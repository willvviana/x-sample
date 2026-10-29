"""Real audio fixtures generated with FFmpeg/libsndfile (never downloaded)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

FFMPEG = "ffmpeg"


def run_ffmpeg(*args: str) -> None:
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr}")


def sine_array(seconds: float, rate: int, freqs: tuple[float, ...], amp: float = 0.5) -> np.ndarray:
    t = np.arange(round(seconds * rate)) / rate
    return np.stack([amp * np.sin(2 * np.pi * f * t) for f in freqs], axis=1).astype(np.float32)


def write_wav(path: Path, data: np.ndarray, rate: int, subtype: str = "PCM_16") -> Path:
    sf.write(path, data, rate, subtype=subtype)
    return path


def ramp_wav(path: Path, frames: int, rate: int = 44100) -> Path:
    """Mono ramp whose every sample is unique: position is recoverable from the value."""
    data = (np.arange(frames, dtype=np.float64) / frames * 1.8 - 0.9).astype(np.float32)
    return write_wav(path, data[:, None], rate, "FLOAT")


def encode(src_wav: Path, dest: Path, *codec_args: str) -> Path:
    run_ffmpeg("-i", str(src_wav), *codec_args, str(dest))
    return dest


def make_all_formats(directory: Path, seconds: float = 3.0, rate: int = 44100) -> dict[str, Path]:
    """Stereo 440/660 Hz tone in WAV, FLAC, OGG (Vorbis), MP3 and M4A (AAC)."""
    directory.mkdir(parents=True, exist_ok=True)
    wav = write_wav(directory / "tone.wav", sine_array(seconds, rate, (440.0, 660.0)), rate)
    return {
        "wav": wav,
        "flac": encode(wav, directory / "tone.flac", "-c:a", "flac"),
        "ogg": encode(wav, directory / "tone.ogg", "-c:a", "libvorbis", "-q:a", "6"),
        "mp3": encode(wav, directory / "tone.mp3", "-c:a", "libmp3lame", "-b:a", "192k"),
        "m4a": encode(wav, directory / "tone.m4a", "-c:a", "aac", "-b:a", "192k"),
    }


def make_long_wav(path: Path, seconds: int, rate: int = 44100) -> Path:
    """Stereo FFmpeg-generated tone of the requested length (e.g. 600 s = 10 minutes)."""
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate={rate}:duration={seconds}",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(path),
    )
    return path
