"""Tempo- and meter-continuous decoding of beat/downbeat activations (bar-pointer HMM).

A NumPy implementation of the bar-pointer model with Viterbi decoding described by Krebs,
Böck and Widmer (ISMIR 2015, "An efficient state-space model for joint tempo and meter
tracking") and Böck, Krebs and Widmer (ISMIR 2016, "Joint beat and downbeat tracking with
recurrent neural networks"). Defaults follow the DBN option of beat_this 1.1.0 (55-215 BPM,
3 or 4 beats per bar, transition lambda 100, observation lambda 16, peak correction). No
code from madmom is used.

State: (beat k within the bar, beat interval tau in frames, phase 0..tau-1). The phase
advances by one frame per frame; at the end of a beat the next beat may change tempo with
probability proportional to exp(-lambda * |tau'/tau - 1|). Only the first 1/lambda_obs of a
beat observes the beat (k > 0) or downbeat (k = 0) activation; all other states observe
"no beat". Each meter is decoded separately and the more likely path wins.

Memory: backpointers are kept only for beat-start states (frames x beats x tempi bytes, about
5 MB for 10 minutes); the forward pass is O(frames x states).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

FPS = 50.0
MIN_BPM, MAX_BPM = 55.0, 215.0
BEATS_PER_BAR = (3, 4)
TRANSITION_LAMBDA = 100.0
OBSERVATION_LAMBDA = 16.0
_EPS = 1e-5


@dataclass(frozen=True)
class DbnResult:
    beats: F64  # seconds
    downbeats: F64
    meter: int
    log_prob: float


def _intervals(fps: float, min_bpm: float, max_bpm: float) -> npt.NDArray[np.int64]:
    lo = int(np.round(60.0 * fps / max_bpm))
    hi = int(np.round(60.0 * fps / min_bpm))
    return np.arange(lo, hi + 1, dtype=np.int64)


def _log_transitions(taus: npt.NDArray[np.int64], lam: float) -> F64:
    ratio = taus[None, :] / taus[:, None]
    prob = np.exp(-lam * np.abs(ratio - 1.0))
    prob /= prob.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        out: F64 = np.log(prob)
    return out  # [from tau, to tau]


def activations(beat_logits: F32, down_logits: F32) -> tuple[F64, F64]:
    """(beat-but-not-downbeat, downbeat) probabilities summing to < 1, as beat_this's DBN."""
    beat = 1.0 / (1.0 + np.exp(-beat_logits.astype(np.float64)))
    down = 1.0 / (1.0 + np.exp(-down_logits.astype(np.float64)))
    beat = beat * (1 - _EPS) + _EPS / 2
    down = down * (1 - _EPS) + _EPS / 2
    only_beat = np.maximum(beat - down, _EPS / 2)
    total = only_beat + down
    scale = np.where(total >= 1.0, (1.0 - _EPS) / total, 1.0)
    return only_beat * scale, down * scale


def _decode_meter(
    beat_act: F64, down_act: F64, meter: int, taus: npt.NDArray[np.int64], log_a: F64
) -> tuple[float, list[tuple[int, int, int]]]:
    """Viterbi for one meter. Returns (log prob, beat starts as (frame, k, tau index))."""
    n, nt = len(beat_act), len(taus)
    # Flat state layout: block (k, tau index) holds phases 0..tau-1.
    sizes = np.tile(taus, meter)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    total = int(sizes.sum())
    starts = offsets  # phase 0 of each (k, tau)
    ends = offsets + sizes - 1
    shift_dst = np.setdiff1d(np.arange(total), starts)
    shift_src = shift_dst - 1
    # Observation class per state: 0 no beat, 1 beat, 2 downbeat.
    obs_class = np.zeros(total, np.int8)
    for b in range(meter * nt):
        k, tau = b // nt, sizes[b]
        n_obs = max(1, int(np.ceil(tau / OBSERVATION_LAMBDA)))  # phase/tau < 1/lambda
        obs_class[offsets[b] : offsets[b] + n_obs] = 2 if k == 0 else 1
    log_none = np.log(np.maximum(1.0 - beat_act - down_act, _EPS) / (OBSERVATION_LAMBDA - 1))
    log_beat, log_down = np.log(beat_act), np.log(down_act)
    is_beat, is_down = obs_class == 1, obs_class == 2

    score = np.full(total, -np.log(total))
    back = np.zeros((n, meter, nt), np.uint8)
    prev_k = np.roll(np.arange(meter), 1)  # beat k follows beat k-1 (k=0 follows the last)
    for t in range(n):
        if t > 0:
            new = np.empty(total)
            new[shift_dst] = score[shift_src]
            end_scores = score[ends].reshape(meter, nt)[prev_k]  # [k, from tau]
            cand = end_scores[:, :, None] + log_a[None, :, :]  # [k, from, to]
            best = cand.argmax(axis=1)  # [k, to]
            back[t] = best
            new[starts] = np.take_along_axis(cand, best[:, None, :], axis=1)[:, 0, :].ravel()
            score = new
        obs = np.full(total, log_none[t])
        obs[is_beat] = log_beat[t]
        obs[is_down] = log_down[t]
        score = score + obs
    # Backtrack through beat starts only.
    state = int(score.argmax())
    block = int(np.searchsorted(offsets, state, side="right") - 1)
    k, ti = divmod(block, nt)
    t = n - 1 - (state - offsets[block])
    beats: list[tuple[int, int, int]] = []
    while t >= 0:
        beats.append((t, k, ti))
        if t == 0:
            break
        prev_ti = int(back[t, k, ti])
        k = int(prev_k[k])
        t = t - 1 - (int(taus[prev_ti]) - 1)
        ti = prev_ti
    beats.reverse()
    return float(score.max()), beats


def decode(
    beat_logits: F32,
    down_logits: F32,
    *,
    fps: float = FPS,
    min_bpm: float = MIN_BPM,
    max_bpm: float = MAX_BPM,
    beats_per_bar: tuple[int, ...] = BEATS_PER_BAR,
    transition_lambda: float = TRANSITION_LAMBDA,
) -> DbnResult:
    """Beats and downbeats (seconds) from per-frame logits of a beat/downbeat model."""
    beat_act, down_act = activations(beat_logits, down_logits)
    if len(beat_act) < 2:
        empty = np.zeros(0, np.float64)
        return DbnResult(empty, empty, beats_per_bar[-1], 0.0)
    taus = _intervals(fps, min_bpm, max_bpm)
    log_a = _log_transitions(taus, transition_lambda)
    best: tuple[float, int, list[tuple[int, int, int]]] | None = None
    for meter in beats_per_bar:
        lp, starts = _decode_meter(beat_act, down_act, meter, taus, log_a)
        if best is None or lp > best[0]:
            best = (lp, meter, starts)
    assert best is not None
    lp, meter, starts = best
    combined = beat_act + down_act
    beats, downs = [], []
    for t, k, ti in starts:
        # Peak correction: the beat is the strongest activation inside the observed window.
        width = max(1, int(np.ceil(int(taus[ti]) / OBSERVATION_LAMBDA)))
        window = combined[t : t + width]
        if len(window) == 0:
            continue
        frame = t + int(window.argmax())
        beats.append(frame / fps)
        if k == 0:
            downs.append(frame / fps)
    return DbnResult(np.asarray(beats), np.asarray(downs), meter, lp)
