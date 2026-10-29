"""Section boundaries and labels from a beat-synchronous self-similarity matrix (Foote novelty),
plus per-section loudness for dynamics."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from xsample.analysis.dsp import Features

F64 = npt.NDArray[np.float64]
MAX_UNITS = 2000  # beats are pooled above this so the matrix stays small


@dataclass
class Section:
    start: float
    end: float
    label: str
    confidence: float
    loudness_db: float


@dataclass
class StructureResult:
    sections: list[Section]
    loudness_curve: list[tuple[float, float]]  # (time s, dB RMS)
    dynamic_range_db: float | None


def _units(features: Features, beats: F64) -> tuple[F64, F64]:
    duration = features.duration_samples / 22050.0
    if len(beats) >= 16:
        bounds = np.concatenate([[0.0], beats, [duration]])
    else:
        bounds = np.arange(0.0, duration + 0.5, 0.5)
    if len(bounds) - 1 > MAX_UNITS:
        step = int(np.ceil((len(bounds) - 1) / MAX_UNITS))
        bounds = np.concatenate([bounds[::step], [duration]])
    return np.unique(bounds), np.array([duration])


def analyze_structure(
    features: Features, beats: list[float], downbeats: list[float]
) -> StructureResult:
    beats_arr = np.asarray(beats, dtype=np.float64)
    bounds, _ = _units(features, beats_arr)
    fps_r, fps_c = features.rhythm_fps, features.chroma_fps
    mel, chroma, rms = features.mel, features.chroma, features.rms
    loud_curve: list[tuple[float, float]] = []
    feats = []
    unit_db = []
    for a, z in itertools.pairwise(bounds):
        ra, rz = int(a * fps_r), max(int(a * fps_r) + 1, int(z * fps_r))
        ca, cz = int(a * fps_c), max(int(a * fps_c) + 1, int(z * fps_c))
        m = mel[ra:rz].mean(axis=0) if rz <= len(mel) and ra < len(mel) else np.zeros(mel.shape[1])
        c = chroma[ca:cz].mean(axis=0) if ca < len(chroma) else np.zeros(12)
        r = float(np.sqrt(np.mean(rms[ra:rz] ** 2))) if ra < len(rms) else 0.0
        db = 20 * np.log10(max(r, 1e-6))
        unit_db.append(db)
        loud_curve.append((float(a), round(db, 2)))
        feats.append(np.concatenate([m, 2.0 * c]))
    if len(feats) < 8:
        duration = float(bounds[-1])
        return StructureResult(
            [Section(0.0, duration, "A", 0.0, float(np.mean(unit_db)) if unit_db else -120.0)],
            loud_curve,
            None,
        )
    x = np.array(feats)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-6)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    ssm = x @ x.T
    n = len(x)
    half = min(16, n // 4)
    g = np.exp(-0.5 * (np.linspace(-2, 2, 2 * half) ** 2))
    kernel = np.outer(g, g) * np.outer(
        np.sign(np.arange(2 * half) - half + 0.5), np.sign(np.arange(2 * half) - half + 0.5)
    )
    padded = np.pad(ssm, half, mode="edge")
    novelty = np.array(
        [np.sum(padded[i : i + 2 * half, i : i + 2 * half] * kernel) for i in range(n)]
    )
    novelty = np.maximum(novelty, 0)
    if novelty.max() > 0:
        novelty /= novelty.max()
    peaks = [
        i
        for i in range(1, n - 1)
        if novelty[i] >= novelty[i - 1] and novelty[i] > novelty[i + 1] and novelty[i] > 0.3
    ]
    min_gap = max(4, half // 2)
    chosen: list[int] = []
    for p in sorted(peaks, key=lambda i: -novelty[i]):
        if all(abs(p - q) >= min_gap for q in chosen):
            chosen.append(p)
    chosen.sort()
    edges = [0, *chosen, n]
    down = np.asarray(downbeats, dtype=np.float64)
    sections: list[Section] = []
    seg_feats = []
    for a, z in itertools.pairwise(edges):
        start, end = float(bounds[a]), float(bounds[z])
        if len(down) and a > 0:
            nearest = down[np.argmin(np.abs(down - start))]
            if abs(nearest - start) <= 0.6 * float(np.median(np.diff(bounds))) * 2:
                start = float(nearest)
        seg_feats.append(x[a:z].mean(axis=0))
        conf = float(novelty[a]) if a > 0 else 1.0
        sections.append(
            Section(start, end, "", round(conf, 3), round(float(np.mean(unit_db[a:z])), 2))
        )
    for i in range(1, len(sections)):
        sections[i - 1].end = sections[i].start
    # Label repeated sections with the same letter (cosine similarity of their mean features).
    letters: list[int] = []
    reps: list[F64] = []
    for f in seg_feats:
        f = f / max(float(np.linalg.norm(f)), 1e-9)
        sims = [float(f @ r) for r in reps]
        if sims and max(sims) > 0.85:
            letters.append(int(np.argmax(sims)))
        else:
            reps.append(f)
            letters.append(len(reps) - 1)
    for s, k in zip(sections, letters, strict=True):
        s.label = chr(ord("A") + k) if k < 26 else f"S{k}"
    loud = np.array(unit_db)
    loud = loud[loud > -80]
    dyn = float(np.percentile(loud, 95) - np.percentile(loud, 10)) if len(loud) > 4 else None
    return StructureResult(sections, loud_curve, dyn)
