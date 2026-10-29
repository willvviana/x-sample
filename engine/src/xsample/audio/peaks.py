"""Waveform peak pyramid: min/max per block at several zoom levels.

Level 0 holds one (min, max) pair per 64 frames; each further level groups 4 blocks of
the previous one. Values are int16-quantised (x * 32767). The pyramid is built
incrementally from decoded chunks, so building never needs the whole file in memory, and
it is ~1/16 of the audio size at the finest level.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
I16 = npt.NDArray[np.int16]

BASE_BLOCK = 64
FACTOR = 4
LEVELS = 6
SCALE = 32767
FORMAT = 1


def _quantise(x: F32) -> I16:
    return np.rint(np.clip(x, -1.0, 1.0) * SCALE).astype(np.int16)


@dataclass
class Pyramid:
    sample_rate: int
    channels: int
    frames: int
    levels: list[I16]  # each (blocks, channels, 2): [..., 0] = min, [..., 1] = max

    @property
    def block_sizes(self) -> list[int]:
        return [BASE_BLOCK * FACTOR**k for k in range(len(self.levels))]

    def save(self, path: Path) -> None:
        meta = {
            "format": FORMAT,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "block_sizes": self.block_sizes,
        }
        buffer = io.BytesIO()
        arrays: dict[str, Any] = {f"level{k}": lvl for k, lvl in enumerate(self.levels)}
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        np.savez(buffer, **arrays)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> Pyramid:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta"]).decode())
            if meta.get("format") != FORMAT:
                raise ValueError(f"unsupported peak file format {meta.get('format')!r}")
            levels = [data[f"level{k}"] for k in range(len(meta["block_sizes"]))]
        return cls(meta["sample_rate"], meta["channels"], meta["frames"], levels)


class PeakBuilder:
    def __init__(self, sample_rate: int, channels: int) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._carry: F32 = np.zeros((0, channels), dtype=np.float32)
        self._base: list[I16] = []
        self._frames = 0

    def _blocks(self, data: F32) -> I16:
        blocks = data.reshape(-1, BASE_BLOCK, self.channels)
        return np.stack(
            [_quantise(blocks.min(axis=1)), _quantise(blocks.max(axis=1))], axis=-1
        ).astype(np.int16)

    def add(self, frames: F32) -> None:
        if frames.ndim != 2 or frames.shape[1] != self.channels:
            raise ValueError("frames must be shaped (n, channels)")
        self._frames += len(frames)
        data = np.concatenate([self._carry, frames]) if len(self._carry) else frames
        usable = len(data) - len(data) % BASE_BLOCK
        if usable:
            self._base.append(self._blocks(np.ascontiguousarray(data[:usable])))
        self._carry = np.array(data[usable:], dtype=np.float32)

    def finalize(self) -> Pyramid:
        if len(self._carry):
            tail = self._carry
            self._base.append(
                np.stack(
                    [_quantise(tail.min(axis=0)), _quantise(tail.max(axis=0))], axis=-1
                ).astype(np.int16)[None]
            )
            self._carry = np.zeros((0, self.channels), dtype=np.float32)
        level0 = (
            np.concatenate(self._base)
            if self._base
            else np.zeros((0, self.channels, 2), dtype=np.int16)
        )
        levels = [level0]
        for _ in range(LEVELS - 1):
            prev = levels[-1]
            n = len(prev)
            if n == 0:
                levels.append(prev)
                continue
            starts = np.arange(0, n, FACTOR)
            mins = np.minimum.reduceat(prev[:, :, 0], starts, axis=0)
            maxs = np.maximum.reduceat(prev[:, :, 1], starts, axis=0)
            levels.append(np.stack([mins, maxs], axis=-1).astype(np.int16))
        return Pyramid(self.sample_rate, self.channels, self._frames, levels)


def _reduce(
    mins: npt.NDArray[np.generic], maxs: npt.NDArray[np.generic], ind: npt.NDArray[np.int64]
) -> tuple[I16, I16]:
    return (
        np.minimum.reduceat(mins, ind, axis=0).astype(np.int16),
        np.maximum.reduceat(maxs, ind, axis=0).astype(np.int16),
    )


def query_pyramid(pyr: Pyramid, start: int, end: int, columns: int) -> tuple[I16, I16] | None:
    """Min/max per column over [start, end). None if the span is finer than the base block."""
    span = (end - start) / columns
    k = -1
    for i, size in enumerate(pyr.block_sizes):
        if size <= span:
            k = i
    if k < 0:
        return None
    size = pyr.block_sizes[k]
    level = pyr.levels[k]
    nb = len(level)
    out_min = np.zeros((columns, pyr.channels), dtype=np.int16)
    out_max = np.zeros((columns, pyr.channels), dtype=np.int16)
    if nb == 0:
        return out_min, out_max
    edges = start + np.arange(columns + 1, dtype=np.float64) * span
    # Cover each column with floor(start)..ceil(end) blocks so extrema are never missed;
    # neighbouring columns therefore share a boundary block.
    starts = np.clip(np.floor(edges[:-1] / size).astype(np.int64), 0, nb)
    ends = np.clip(np.ceil(edges[1:] / size).astype(np.int64), 0, nb)
    ends = np.minimum(np.maximum(ends, starts + 1), nb)
    valid = starts < nb
    if not valid.any():
        return out_min, out_max
    lo = int(starts[valid].min())
    hi = int(ends[valid].max())
    # Neutral padding row so that an exclusive end index == len is a legal reduceat index.
    pad_min = np.full((1, pyr.channels), np.iinfo(np.int16).max, dtype=np.int16)
    pad_max = np.full((1, pyr.channels), np.iinfo(np.int16).min, dtype=np.int16)
    sub_min = np.concatenate([level[lo:hi, :, 0], pad_min])
    sub_max = np.concatenate([level[lo:hi, :, 1], pad_max])
    # reduceat over interleaved [start_0, end_0, start_1, end_1, ...]: even rows are the
    # wanted [start_i, end_i) reductions (odd rows are discarded), so overlaps are fine.
    ind = np.empty(2 * int(valid.sum()), dtype=np.int64)
    ind[0::2] = starts[valid] - lo
    ind[1::2] = ends[valid] - lo
    out_min[valid] = np.minimum.reduceat(sub_min, ind, axis=0)[0::2]
    out_max[valid] = np.maximum.reduceat(sub_max, ind, axis=0)[0::2]
    return out_min, out_max


def query_raw(samples: F32, start: int, end: int, columns: int) -> tuple[I16, I16]:
    """Min/max per column from raw samples (used when zoomed in past the base block)."""
    n = len(samples)
    channels = samples.shape[1]
    out_min = np.zeros((columns, channels), dtype=np.int16)
    out_max = np.zeros((columns, channels), dtype=np.int16)
    if n == 0:
        return out_min, out_max
    span = (end - start) / columns
    ind = np.floor(np.arange(columns, dtype=np.float64) * span).astype(np.int64)
    valid = ind < n
    if not valid.any():
        return out_min, out_max
    q = _quantise(samples)
    mins, maxs = _reduce(q, q, ind[valid])
    out_min[valid] = mins
    out_max[valid] = maxs
    return out_min, out_max


def read_columns(
    pyr: Pyramid,
    read_raw: Callable[[int, int], F32],
    start: int,
    end: int,
    columns: int,
) -> tuple[I16, I16]:
    """Peaks for [start, end) as `columns` min/max pairs; the pyramid is used when possible."""
    if not (0 <= start < end) or columns < 1:
        raise ValueError("invalid peak range")
    end = min(end, pyr.frames)
    if start >= end:
        z = np.zeros((columns, pyr.channels), dtype=np.int16)
        return z, z.copy()
    result = query_pyramid(pyr, start, end, columns)
    if result is not None:
        return result
    return query_raw(read_raw(start, end), start, end, columns)
