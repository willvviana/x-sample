"""Bar-pointer HMM decoder (beat_dbn.py) on synthetic beat/downbeat activations."""

from __future__ import annotations

import time

import numpy as np
import pytest

from xsample.analysis import beat_dbn


def logits(
    n: int, period: float, meter: int, jitter: float = 0.7, noise: int = 0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    beat = np.full(n, -6.0, np.float32)
    down = np.full(n, -8.0, np.float32)
    frames = []
    for i, f in enumerate(np.arange(10, n - 5, period)):
        f = int(np.clip(round(f + rng.normal(0, jitter)), 0, n - 1))
        beat[f] = 4.0
        if i % meter == 0:
            down[f] = 4.0
        frames.append(f)
    for f in rng.integers(0, n, noise):
        beat[f] = max(beat[f], 1.0)  # weak spurious peaks
    return beat, down, np.asarray(frames) / beat_dbn.FPS


@pytest.mark.parametrize(("bpm", "meter"), [(120.0, 4), (90.0, 3), (70.0, 4), (180.0, 4)])
def test_recovers_tempo_meter_and_downbeats(bpm: float, meter: int) -> None:
    period = 60.0 * beat_dbn.FPS / bpm
    beat, down, truth = logits(3000, period, meter, noise=60)
    result = beat_dbn.decode(beat, down)
    assert result.meter == meter
    est = 60.0 / float(np.median(np.diff(result.beats)))
    assert abs(est - bpm) / bpm < 0.02
    hits = sum(np.min(np.abs(result.beats - t)) < 0.07 for t in truth)
    assert hits / len(truth) > 0.95
    assert np.allclose(np.diff(result.downbeats), meter * 60.0 / bpm, rtol=0.05)


def test_tempo_outside_the_range_is_decoded_at_a_related_level() -> None:
    # 240 BPM is above MAX_BPM: the decoder cannot represent it and must halve it. This is
    # why the DBN is not the default (MusicNet references reach 225 BPM).
    beat, down, _ = logits(3000, 60.0 * beat_dbn.FPS / 240.0, 4)
    result = beat_dbn.decode(beat, down)
    est = 60.0 / float(np.median(np.diff(result.beats)))
    assert abs(est - 120.0) < 3.0


def test_empty_and_tiny_inputs() -> None:
    empty = beat_dbn.decode(np.zeros(0, np.float32), np.zeros(0, np.float32))
    assert len(empty.beats) == 0 and len(empty.downbeats) == 0
    tiny = beat_dbn.decode(np.zeros(1, np.float32), np.zeros(1, np.float32))
    assert len(tiny.beats) == 0


def test_activations_are_valid_probabilities() -> None:
    b = np.array([-50, 0, 50, 50], np.float32)
    d = np.array([-50, 0, 50, -50], np.float32)
    only_beat, down = beat_dbn.activations(b, d)
    assert np.all(only_beat > 0) and np.all(down > 0) and np.all(only_beat + down < 1)


def test_ten_minutes_decode_in_bounded_time() -> None:
    beat, down, _ = logits(30000, 25.0, 4)
    t0 = time.perf_counter()
    result = beat_dbn.decode(beat, down)
    assert time.perf_counter() - t0 < 20.0  # ~3 s on the reference CPU
    assert abs(len(result.beats) - 1200) <= 2
