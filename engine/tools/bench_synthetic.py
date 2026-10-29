"""Synthetic benchmark of the v0.2.0 DSP analyzers on generated fixtures with exact ground truth
(tests/analysis_fixtures.py). Results go to test-reports/synthetic_bench.json; they describe
clean, generated material and are not a substitute for the real-audio evaluation.

    uv run python tools/bench_synthetic.py [case names...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from tests.analysis_fixtures import (  # noqa: E402
    Spec,
    chord_overlap,
    f_measure,
    melody_note_accuracy,
    mirex_key_score,
    render,
)

from xsample.analysis.dsp import FeatureExtractor  # noqa: E402
from xsample.analysis.harmony import analyze_harmony  # noqa: E402
from xsample.analysis.rhythm import analyze_rhythm  # noqa: E402
from xsample.analysis.structure import analyze_structure  # noqa: E402

REPORT = ENGINE / "test-reports" / "synthetic_bench.json"
CASES = {
    "pop 120 C maj": Spec(bpm=120, key_root=0),
    "pop 95 A min": Spec(bpm=95, key_root=9, minor=True),
    "noisy 128 E min": Spec(bpm=128, key_root=4, minor=True, noise=0.08),
    "slow 72 D maj": Spec(bpm=72, key_root=2, bars=10),
    "fast 174 G min": Spec(bpm=174, key_root=7, minor=True, bars=24),
    "drift 100->116 Bb": Spec(bpm=100, bpm_end=116, key_root=10, bars=20),
    "waltz 3/4 150 F": Spec(bpm=150, key_root=5, meter=3, bars=20),
    "no drums 90 C# min": Spec(bpm=90, key_root=1, minor=True, drums=False),
    "R&B 7ths 88 Eb + inversions": Spec(
        bpm=88, key_root=3, groove="rnb", sevenths=True, inversions=True
    ),
    "R&B 7ths 76 F min + sax": Spec(
        bpm=76, key_root=5, minor=True, groove="rnb", sevenths=True, lead="sax", bars=12
    ),
    "R&B 96 Ab + violin + sections": Spec(
        bpm=96, key_root=8, groove="rnb", sevenths=True, lead="violin", sections=True
    ),
    "ballad 68 G violin no drums": Spec(bpm=68, key_root=7, drums=False, lead="violin", bars=10),
    "sax 110 Bb min noisy": Spec(bpm=110, key_root=10, minor=True, lead="sax", noise=0.04),
}


def run(name: str, spec: Spec) -> dict[str, Any]:
    y, truth = render(spec)
    t0 = time.perf_counter()
    fx = FeatureExtractor()
    for i in range(0, len(y), 65536):
        fx.push(y[i : i + 65536])
    feats = fx.finish()
    r = analyze_rhythm(feats)
    h = analyze_harmony(feats, r.beats, r.downbeats, r.onsets)
    s = analyze_structure(feats, r.beats, r.downbeats)
    elapsed = time.perf_counter() - t0
    true_bpm = (truth.bpm_start + truth.bpm_end) / 2
    ratio = r.bpm / true_bpm if true_bpm else 0
    key = (
        (h.key_candidates[0].tonic, h.key_candidates[0].mode) if h.key_candidates else (0, "major")
    )
    chords = [(c.start, c.end, c.root, c.quality) for c in h.chords]
    mel = [(n.start, n.end, n.midi) for n in h.melody_notes]
    return {
        "case": name,
        "bpm": f"{r.bpm:.1f}/{true_bpm:.0f}",
        "acc1": abs(ratio - 1) < 0.04,
        "acc2": any(abs(ratio - f) < 0.04 * f for f in (0.5, 1, 2, 1 / 3, 3)),
        "beat_f": round(f_measure(r.beats, truth.beats), 3),
        "downbeat_f": round(f_measure(r.downbeats, truth.downbeats), 3),
        "meter": f"{r.meter}/{truth.meter}",
        "key": h.key_candidates[0].label if h.key_candidates else "-",
        "key_mirex": mirex_key_score(key, truth.key),
        "chord_majmin": round(chord_overlap(chords, truth.chords, True), 3),
        "chord_full": round(chord_overlap(chords, truth.chords, False), 3),
        "sections": len(s.sections),
        "melody_note_acc": round(melody_note_accuracy(mel, truth.melody), 3)
        if truth.melody
        else None,
        "analysis_s": round(elapsed, 2),
        "audio_s": round(len(y) / spec.rate, 1),
    }


if __name__ == "__main__":
    names = sys.argv[1:] or list(CASES)
    rows = [run(n, CASES[n]) for n in names]
    for row in rows:
        print(json.dumps(row))

    def mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if r[key] is not None]
        return round(float(np.mean(vals)), 3)

    summary = {
        k: mean(k)
        for k in (
            "acc1",
            "acc2",
            "beat_f",
            "downbeat_f",
            "key_mirex",
            "chord_majmin",
            "chord_full",
            "melody_note_acc",
        )
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {"kind": "synthetic (generated fixtures)", "summary": summary, "items": rows}, indent=1
        ),
        "utf-8",
    )
    print("SUMMARY", json.dumps(summary))
