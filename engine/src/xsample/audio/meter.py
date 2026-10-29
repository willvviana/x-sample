"""Input level metering and clipping detection for recording."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
CLIP_THRESHOLD = 0.999
FLOOR_DB = -120.0


def to_db(value: float) -> float:
    return FLOOR_DB if value <= 1e-6 else max(FLOOR_DB, 20.0 * math.log10(value))


@dataclass
class LevelMeter:
    """Per-channel peak/RMS since the last `reset_window()`, plus session peak hold and clips."""

    channels: int
    peak_hold: list[float] = field(init=False)
    clip_samples: int = 0
    clip_events: int = 0
    session_peak: float = 0.0
    _win_peak: list[float] = field(init=False)
    _win_sumsq: list[float] = field(init=False)
    _win_n: int = 0

    def __post_init__(self) -> None:
        self.peak_hold = [0.0] * self.channels
        self._win_peak = [0.0] * self.channels
        self._win_sumsq = [0.0] * self.channels

    def feed(self, block: F32) -> int:
        """Account for a (frames, channels) block; returns the number of clipped samples."""
        if block.size == 0:
            return 0
        absolute = np.abs(block)
        peaks = absolute.max(axis=0)
        sumsq = np.sum(block.astype(np.float64) ** 2, axis=0)
        for ch in range(self.channels):
            self._win_peak[ch] = max(self._win_peak[ch], float(peaks[ch]))
            self._win_sumsq[ch] += float(sumsq[ch])
            self.peak_hold[ch] = max(self.peak_hold[ch], float(peaks[ch]))
        self._win_n += len(block)
        self.session_peak = max(self.session_peak, float(peaks.max()))
        clipped = int(np.count_nonzero(absolute >= CLIP_THRESHOLD))
        if clipped:
            self.clip_samples += clipped
            self.clip_events += 1
        return clipped

    def read_window(self) -> tuple[list[float], list[float]]:
        """(peak_db, rms_db) per channel for the window since the last call; resets the window."""
        n = max(1, self._win_n)
        peak = [to_db(p) for p in self._win_peak]
        rms = [to_db(math.sqrt(s / n)) for s in self._win_sumsq]
        self._win_peak = [0.0] * self.channels
        self._win_sumsq = [0.0] * self.channels
        self._win_n = 0
        return peak, rms
