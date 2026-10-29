"""Tempo, beats, downbeats, meter, subdivision, swing, syncopation and onsets (DSP).

Beats and downbeats can instead come from a beat-tracking model (apply_tracked_beats); tempo,
meter and the beat-relative descriptors are then derived from the model's beats.
Times are returned in seconds; the orchestrator converts them to integer project samples.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from xsample.analysis.dsp import ANALYSIS_RATE, Features

F64 = npt.NDArray[np.float64]

MIN_BPM, MAX_BPM = 50.0, 220.0
PRIOR_BPM, PRIOR_OCTAVES = 110.0, 0.6
LOCAL_WINDOW_S, LOCAL_HOP_S = 12.0, 3.0


@dataclass
class TempoCandidate:
    bpm: float
    relation: str  # "primary" | "half" | "double" | "other"
    score: float


@dataclass
class RhythmResult:
    bpm: float
    confidence: float
    candidates: list[TempoCandidate]
    tempo_curve: list[tuple[float, float]]  # (time s, bpm)
    variable_tempo: bool
    tempo_changes: list[tuple[float, float]]  # (time s, new bpm)
    beats: list[float]
    beat_strength: list[float]
    downbeats: list[float]
    meter: int | None
    meter_confidence: float
    subdivision: str | None  # "duple" | "triplet"
    subdivision_confidence: float
    swing_ratio: float | None
    swing_confidence: float
    syncopation: float | None
    microtiming_ms: float | None
    onsets: list[float]
    onset_strength: list[float]
    notes: list[str] = field(default_factory=list)


def _autocorr(x: F64) -> F64:
    n = len(x)
    if n < 4:
        return np.zeros(max(n, 1))
    y = x - x.mean()
    size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(y, size)
    ac = np.fft.irfft(spec * np.conj(spec), size)[:n]
    out: F64 = ac / max(float(ac[0]), 1e-12)
    return out


def _tempo_scores(
    odf: F64, fps: float, lo: float = MIN_BPM, hi: float = MAX_BPM
) -> tuple[F64, F64]:
    """Comb-weighted autocorrelation score per lag, with a perceptual tempo prior."""
    ac = _autocorr(odf)
    lags = np.arange(len(ac), dtype=np.float64)
    with np.errstate(divide="ignore"):
        bpm = np.where(lags > 0, 60.0 * fps / np.maximum(lags, 1e-9), np.inf)
    score = np.full(len(ac), -np.inf)
    valid = np.where((bpm >= lo) & (bpm <= hi))[0]
    for lag in valid:
        s = 0.0
        # Beat lag + bar-level repetition (2x) + subdivision (1/2) support each other.
        for k, w in ((1.0, 1.0), (2.0, 0.5), (0.5, 0.5)):
            j = round(lag * k)
            if 0 < j < len(ac):
                s += w * ac[j]
        score[lag] = s
    prior = np.exp(
        -0.5 * (np.log2(np.where(np.isfinite(bpm), bpm, 1.0) / PRIOR_BPM) / PRIOR_OCTAVES) ** 2
    )
    weighted = np.where(np.isfinite(score), score * prior, -np.inf)
    return weighted, bpm


def _refine(weighted: F64, best: int) -> float:
    if (
        0 < best < len(weighted) - 1
        and np.isfinite(weighted[best - 1])
        and np.isfinite(weighted[best + 1])
    ):
        a, b, c = weighted[best - 1], weighted[best], weighted[best + 1]
        denom = a - 2 * b + c
        if denom < 0:
            return best + float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    return float(best)


def estimate_tempo(odf: F64, fps: float) -> tuple[float, float, list[TempoCandidate]]:
    weighted, _ = _tempo_scores(odf, fps)
    if not np.any(np.isfinite(weighted)) or np.nanmax(weighted[np.isfinite(weighted)]) <= 0:
        return 0.0, 0.0, []
    best = int(np.argmax(weighted))
    bpm = 60.0 * fps / _refine(weighted, best)
    candidates, confidence = tempo_candidates(weighted, fps, bpm, float(weighted[best]))
    return bpm, confidence, candidates


def tempo_candidates(
    weighted: F64, fps: float, bpm: float, top: float
) -> tuple[list[TempoCandidate], float]:
    """Related tempi scored by the onset autocorrelation relative to `top` (the primary)."""

    def score_at(target: float) -> float:
        lag = 60.0 * fps / target
        j = round(lag)
        lo_j, hi_j = max(1, j - 2), min(len(weighted), j + 3)
        window = weighted[lo_j:hi_j]
        window = window[np.isfinite(window)]
        return float(window.max()) if len(window) else 0.0

    top = max(top, 1e-12)
    candidates = [TempoCandidate(round(bpm, 2), "primary", 1.0)]
    for factor, rel in ((0.5, "half"), (2.0, "double"), (2.0 / 3.0, "other"), (1.5, "other")):
        target = bpm * factor
        if MIN_BPM <= target <= MAX_BPM:
            candidates.append(
                TempoCandidate(round(target, 2), rel, round(max(0.0, score_at(target) / top), 3))
            )
    rivals = [c.score for c in candidates[1:]]
    confidence = float(np.clip(1.0 - (max(rivals) if rivals else 0.0), 0.0, 1.0))
    return candidates, confidence


def local_tempo(odf: F64, fps: float, global_bpm: float) -> list[tuple[float, float]]:
    """Windowed tempo constrained to [2/3, 3/2] of the global tempo (follows drift, not octaves)."""
    win, hop = int(LOCAL_WINDOW_S * fps), int(LOCAL_HOP_S * fps)
    if global_bpm <= 0 or len(odf) < win:
        return [(0.0, global_bpm)] if global_bpm > 0 else []
    curve = []
    for start in range(0, len(odf) - win + 1, hop):
        seg = odf[start : start + win]
        weighted, _ = _tempo_scores(seg, fps, global_bpm / 1.5, global_bpm * 1.5)
        if not np.any(np.isfinite(weighted)):
            continue
        best = int(np.argmax(weighted))
        curve.append(((start + win / 2) / fps, 60.0 * fps / _refine(weighted, best)))
    if not curve:
        return [(0.0, global_bpm)]
    # Median smoothing over three windows removes isolated octave slips.
    bpms = np.array([b for _, b in curve])
    smooth = np.array([np.median(bpms[max(0, i - 1) : i + 2]) for i in range(len(bpms))])
    return [(t, float(b)) for (t, _), b in zip(curve, smooth, strict=True)]


def track_beats(
    odf: F64, fps: float, curve: list[tuple[float, float]], tightness: float = 120.0
) -> F64:
    """Ellis (2007) dynamic programming with a time-varying period from the local tempo curve."""
    n = len(odf)
    if n == 0 or not curve:
        return np.zeros(0)
    times = np.array([t for t, _ in curve])
    bpms = np.array([b for _, b in curve])
    frame_t = np.arange(n) / fps
    period = 60.0 * fps / np.interp(frame_t, times, bpms)
    mean_period = float(np.mean(period))
    sigma = max(1.0, mean_period / 32)
    half = int(4 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    local = np.convolve(odf, kernel, mode="same")
    score = local.copy()
    back = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        p = period[i]
        lo, hi = i - round(2 * p), i - round(p / 2)
        if hi <= 0:
            continue
        lo = max(lo, 0)
        idx = np.arange(lo, hi)
        cost = -tightness * np.log((i - idx) / p) ** 2
        cand = score[idx] + cost
        j = int(np.argmax(cand))
        if cand[j] > 0:
            score[i] = local[i] + cand[j]
            back[i] = idx[j]
    tail = np.arange(max(0, n - round(period[-1])), n)
    i = int(tail[np.argmax(score[tail])])
    path = []
    while i >= 0:
        path.append(i)
        i = int(back[i])
    beats = np.array(path[::-1], dtype=np.float64)
    return beats / fps


def fix_offbeat_phase(beats: F64, odf_low: F64, odf: F64, fps: float) -> tuple[F64, bool]:
    """If the low-band (kick/bass) energy sits between the beats, the tracker locked onto
    off-beats: shift by half a period."""
    if len(beats) < 8:
        return beats, False
    mids = (beats[:-1] + beats[1:]) / 2

    def energy(times: F64, sig: F64) -> float:
        idx = np.clip(np.round(times * fps).astype(int), 0, len(sig) - 1)
        lo = np.clip(idx - 2, 0, len(sig) - 1)
        hi = np.clip(idx + 3, 0, len(sig))
        return float(
            np.mean([sig[a:b].max() if b > a else 0.0 for a, b in zip(lo, hi, strict=True)])
        )

    on = energy(beats, odf_low) + 0.25 * energy(beats, odf)
    off = energy(mids, odf_low) + 0.25 * energy(mids, odf)
    if off > 1.3 * on:
        return mids, True
    return beats, False


def beat_chroma_change(features: Features, beats: F64) -> F64:
    """Harmonic change at each beat (chords tend to change on downbeats)."""
    c = features.chroma
    if len(c) == 0 or len(beats) == 0:
        return np.zeros(len(beats))
    idx = np.clip(np.round(beats * features.chroma_fps).astype(int), 0, len(c) - 1)
    segs = np.concatenate([idx, [len(c)]])
    means = np.array([c[a : max(a + 1, b)].mean(axis=0) for a, b in itertools.pairwise(segs)])
    change = np.zeros(len(beats))
    change[1:] = np.linalg.norm(np.diff(means, axis=0), axis=1)
    return change


def meter_and_downbeats(beats: F64, features: Features) -> tuple[int | None, float, F64]:
    """Choose meter (3 or 4) and phase by low-band accent + harmonic change on the bar start."""
    if len(beats) < 12:
        return None, 0.0, beats[:0]
    fps = features.rhythm_fps
    idx = np.clip(np.round(beats * fps).astype(int), 0, len(features.onset_low) - 1)
    accent = np.array([features.onset_low[max(0, i - 2) : i + 3].max() for i in idx])
    accent = accent / max(float(accent.max()), 1e-9)
    change = beat_chroma_change(features, beats)
    change = change / max(float(change.max()), 1e-9)
    evidence = accent + 1.5 * change
    options = []
    for meter in (3, 4):
        for phase in range(meter):
            on = evidence[phase::meter]
            off = np.delete(evidence, np.arange(phase, len(evidence), meter))
            options.append((float(on.mean() - off.mean()), meter, phase))
    options.sort(reverse=True)
    best, meter, phase = options[0]
    # Confidence: margin over the best alternative with a *different meter*.
    rival = next((s for s, m, _ in options if m != meter), 0.0)
    confidence = float(np.clip((best - rival) / max(abs(best), 1e-9), 0.0, 1.0))
    return meter, confidence, beats[phase::meter]


def subdivision_and_swing(
    odf: F64, beats: F64, fps: float
) -> tuple[str | None, float, float | None, float]:
    if len(beats) < 8:
        return None, 0.0, None, 0.0
    duple, triplet, positions = [], [], []
    for a, b in itertools.pairwise(beats):
        seg_len = (b - a) * fps
        if seg_len < 6:
            continue

        def at(frac: float, a: float = a, seg_len: float = seg_len) -> float:
            i = round(a * fps + frac * seg_len)
            return float(odf[max(0, i - 1) : i + 2].max()) if 0 <= i < len(odf) else 0.0

        duple.append(at(0.5) + 0.5 * (at(0.25) + at(0.75)))
        triplet.append(at(1 / 3) + at(2 / 3))
        lo, hi = int(a * fps + 0.4 * seg_len), int(a * fps + 0.8 * seg_len)
        window = odf[lo:hi]
        if len(window) and window.max() > 0.2:
            positions.append((lo + int(np.argmax(window)) - a * fps) / seg_len)
    if not duple:
        return None, 0.0, None, 0.0
    d, t = float(np.mean(duple)), float(np.mean(triplet))
    kind = "triplet" if t > 1.15 * d else "duple"
    sub_conf = float(np.clip(abs(d - t) / max(d, t, 1e-9), 0.0, 1.0))
    swing = None
    swing_conf = 0.0
    if len(positions) >= 8:
        r = float(np.median(positions))
        swing = r / max(1e-6, 1.0 - r)
        spread = float(np.std(positions))
        swing_conf = float(np.clip(1.0 - spread / 0.1, 0.0, 1.0) * min(1.0, len(positions) / 32))
    return kind, sub_conf, swing, swing_conf


def pick_onsets(odf: F64, fps: float, delta: float = 0.12, wait_s: float = 0.03) -> tuple[F64, F64]:
    n = len(odf)
    if n < 3:
        return np.zeros(0), np.zeros(0)
    avg_w = int(0.1 * fps)
    local_mean = np.convolve(odf, np.ones(2 * avg_w + 1) / (2 * avg_w + 1), mode="same")
    is_peak = (
        (odf[1:-1] >= odf[:-2]) & (odf[1:-1] > odf[2:]) & (odf[1:-1] > local_mean[1:-1] + delta)
    )
    cand = np.where(is_peak)[0] + 1
    keep: list[int] = []
    wait = int(wait_s * fps)
    for c in cand:
        if not keep or c - keep[-1] > wait:
            keep.append(int(c))
        elif odf[c] > odf[keep[-1]]:
            keep[-1] = int(c)
    k = np.array(keep, dtype=np.int64)
    return k / fps, odf[k] if len(k) else np.zeros(0)


def syncopation_and_microtiming(
    onsets: F64, strengths: F64, beats: F64
) -> tuple[float | None, float | None]:
    """Share of onset strength on off-beat 16th positions, and mean onset deviation (ms) from
    the nearest 16th grid point — simple, documented groove descriptors."""
    if len(beats) < 8 or len(onsets) < 8:
        return None, None
    grid_strong, grid_weak, devs = 0.0, 0.0, []
    for t, s in zip(onsets, strengths, strict=True):
        i = int(np.searchsorted(beats, t)) - 1
        if i < 0 or i >= len(beats) - 1:
            continue
        period = beats[i + 1] - beats[i]
        pos = (t - beats[i]) / period * 4  # 16ths within the beat
        slot = round(pos) % 4
        devs.append(abs(pos - round(pos)) * period / 4 * 1000.0)
        if slot == 0:
            grid_strong += s
        else:
            grid_weak += s
    total = grid_strong + grid_weak
    if total <= 0:
        return None, None
    return float(grid_weak / total), float(np.mean(devs)) if devs else None


def complete_edges(beats: F64, rms: F64, fps: float, duration: float, tol: float = 0.08) -> F64:
    """The flux ODF has no frame before t=0, so a beat on the very first sample is never seen,
    which shifts every bar/phrase count by one. Extend the grid backwards by the local period
    while there is signal, and drop beats that land on the end boundary."""
    if len(beats) < 2:
        return beats
    period = float(np.median(np.diff(beats[: min(len(beats), 9)])))
    floor = 0.1 * float(np.max(rms)) if len(rms) else 0.0
    head: list[float] = []
    t = float(beats[0]) - period
    while t > -tol:
        t0 = max(t, 0.0)
        i = int(t0 * fps)
        if i >= len(rms) or rms[i : i + max(1, int(0.1 * fps))].max() <= floor:
            break
        head.append(t0)
        t -= period
    out = np.concatenate([np.array(head[::-1]), beats])
    out = out[out < duration - tol]
    return out


def tempo_changes(curve: list[tuple[float, float]]) -> tuple[bool, list[tuple[float, float]]]:
    bpms = np.array([b for _, b in curve])
    variable = bool(len(bpms) > 2 and np.std(bpms) / max(np.mean(bpms), 1e-9) > 0.02)
    changes: list[tuple[float, float]] = []
    if variable:
        ref = bpms[0]
        for t, b in curve:
            if abs(b - ref) / ref > 0.04:
                changes.append((t, b))
                ref = b
    return variable, changes


def beat_tempo_curve(beats: F64) -> list[tuple[float, float]]:
    """Windowed tempo from beat intervals, on the same window grid as local_tempo."""
    ibi = np.diff(beats)
    if len(ibi) == 0:
        return []
    mid = (beats[:-1] + beats[1:]) / 2
    end = float(beats[-1])
    if end - float(beats[0]) < LOCAL_WINDOW_S:
        return [(float(np.median(mid)), 60.0 / float(np.median(ibi)))]
    curve = []
    t = float(beats[0])
    while t + LOCAL_WINDOW_S <= end + 1e-9:
        sel = ibi[(mid >= t) & (mid < t + LOCAL_WINDOW_S)]
        if len(sel):
            curve.append((t + LOCAL_WINDOW_S / 2, 60.0 / float(np.median(sel))))
        t += LOCAL_HOP_S
    return curve


def meter_from_downbeats(beats: F64, downbeats: F64) -> tuple[int | None, float]:
    """Most common number of beats per bar, and the share of bars that have it."""
    if len(downbeats) < 3:
        return None, 0.0
    idx = np.searchsorted(beats, downbeats)
    counts = np.diff(idx)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return None, 0.0
    values, freq = np.unique(counts, return_counts=True)
    best = int(np.argmax(freq))
    return int(values[best]), float(freq[best] / len(counts))


def apply_tracked_beats(
    dsp: RhythmResult, features: Features, beats: F64, downbeats: F64, source: str
) -> RhythmResult:
    """Replace the DSP beat grid with beats and downbeats from a tracker (e.g. a model).

    Onsets stay DSP; tempo, tempo curve, meter, subdivision, swing, syncopation and
    microtiming are recomputed from the tracked beats. The tracker's output is not edited
    (no phase shift or edge completion).
    """
    if len(beats) < 3:
        dsp.notes.append(f"{source} found fewer than 3 beats; the DSP beat tracker was used.")
        return dsp
    fps = features.rhythm_fps
    odf = features.onset.astype(np.float64)
    bpm = 60.0 / float(np.median(np.diff(beats)))
    weighted, _ = _tempo_scores(odf, fps, 1.0, 1000.0)
    lag = round(60.0 * fps / bpm)
    top = float(weighted[lag]) if 0 < lag < len(weighted) and np.isfinite(weighted[lag]) else 0.0
    candidates, confidence = tempo_candidates(weighted, fps, bpm, top)
    if top <= 0:
        candidates, confidence = candidates[:1], 0.0
    curve = beat_tempo_curve(beats)
    variable, changes = tempo_changes(curve)
    idx = np.clip(np.round(beats * fps).astype(int), 0, len(odf) - 1)
    meter, meter_conf = meter_from_downbeats(beats, downbeats)
    sub, sub_conf, swing, swing_conf = subdivision_and_swing(odf, beats, fps)
    onsets = np.asarray(dsp.onsets, np.float64)
    sync, micro = syncopation_and_microtiming(onsets, np.asarray(dsp.onset_strength), beats)
    notes = [n for n in dsp.notes if "Beat phase" not in n and "No periodic" not in n]
    notes.append(f"Beats and downbeats from {source}; tempo and meter are derived from them.")
    return RhythmResult(
        bpm=bpm,
        confidence=confidence,
        candidates=candidates,
        tempo_curve=curve,
        variable_tempo=variable,
        tempo_changes=changes,
        beats=beats.tolist(),
        beat_strength=odf[idx].tolist(),
        downbeats=downbeats.tolist(),
        meter=meter,
        meter_confidence=meter_conf,
        subdivision=sub,
        subdivision_confidence=sub_conf,
        swing_ratio=swing,
        swing_confidence=swing_conf,
        syncopation=sync,
        microtiming_ms=micro,
        onsets=dsp.onsets,
        onset_strength=dsp.onset_strength,
        notes=notes,
    )


def analyze_rhythm(features: Features) -> RhythmResult:
    fps = features.rhythm_fps
    odf = features.onset.astype(np.float64)
    odf_low = features.onset_low.astype(np.float64)
    notes: list[str] = []
    bpm, conf, candidates = estimate_tempo(odf, fps)
    onsets, strengths = pick_onsets(odf, fps)
    if bpm <= 0:
        notes.append("No periodic rhythm was found.")
        return RhythmResult(
            0.0,
            0.0,
            [],
            [],
            False,
            [],
            [],
            [],
            [],
            None,
            0.0,
            None,
            0.0,
            None,
            0.0,
            None,
            None,
            onsets.tolist(),
            strengths.tolist(),
            notes,
        )
    curve = local_tempo(odf, fps, bpm)
    variable, changes = tempo_changes(curve)
    beats = track_beats(odf, fps, curve)
    beats, shifted = fix_offbeat_phase(beats, odf_low, odf, fps)
    if shifted:
        notes.append("Beat phase moved by half a beat: low-frequency accents fell between beats.")
    beats = complete_edges(
        beats, features.rms.astype(np.float64), fps, features.duration_samples / ANALYSIS_RATE
    )
    idx = np.clip(np.round(beats * fps).astype(int), 0, len(odf) - 1)
    beat_strength = odf[idx]
    meter, meter_conf, downbeats = meter_and_downbeats(beats, features)
    sub, sub_conf, swing, swing_conf = subdivision_and_swing(odf, beats, fps)
    sync, micro = syncopation_and_microtiming(onsets, strengths, beats)
    if conf < 0.3:
        notes.append("Tempo is ambiguous between related values (e.g. half/double time).")
    return RhythmResult(
        bpm=float(bpm),
        confidence=conf,
        candidates=candidates,
        tempo_curve=curve,
        variable_tempo=variable,
        tempo_changes=changes,
        beats=beats.tolist(),
        beat_strength=beat_strength.tolist(),
        downbeats=downbeats.tolist(),
        meter=meter,
        meter_confidence=meter_conf,
        subdivision=sub,
        subdivision_confidence=sub_conf,
        swing_ratio=swing,
        swing_confidence=swing_conf,
        syncopation=sync,
        microtiming_ms=micro,
        onsets=onsets.tolist(),
        onset_strength=strengths.tolist(),
        notes=notes,
    )
