"""Real-audio evaluation of the v0.2.0 DSP analyzers on locally fetched, licensed subsets.

Datasets live outside the repository in %LOCALAPPDATA%\\X-Sample\\eval-datasets (fetched with
per-file SHA-256 manifests; see docs/EVALUATION.md). Results go to
test-reports/real_audio_eval.json and are kept separate from the synthetic benchmark.

    uv run python tools/eval_real_audio.py [--limit N] [--beat-model [--backend B]]

With --beat-model, beats and downbeats from the installed Beat This! small exports (pinned
SHA-256, run through the engine's BeatTracker) are scored next to the DSP tracker as nn_*
fields. Beat This! was trained on GuitarSet, so its GuitarSet scores are not a held-out
measurement; MusicNet and Slakh were not in its training data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from tests.analysis_fixtures import (  # noqa: E402
    chord_overlap,
    f_measure,
    melody_note_accuracy,
    mirex_key_score,
)

from xsample.analysis import beat_model  # noqa: E402
from xsample.analysis.dsp import ANALYSIS_RATE, FeatureExtractor  # noqa: E402
from xsample.analysis.harmony import analyze_harmony  # noqa: E402
from xsample.analysis.rhythm import analyze_rhythm  # noqa: E402
from xsample.audio.ffmpeg import decode_f32  # noqa: E402

DATA = Path(os.environ["LOCALAPPDATA"]) / "X-Sample" / "eval-datasets"
REPORT = ENGINE / "test-reports" / "real_audio_eval.json"
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MELODY_MIN_MIDI = 48  # the DSP melody tracker searches from C3 upwards
BEAT_MODEL: Any = None  # beat_model.BeatTracker when --beat-model is given
# Whether the model's training data contains the dataset (beat_this_annotations repository).
TRAINING_OVERLAP = {"GuitarSet": True, "MusicNet": False, "BabySlakh": False}
NEURAL: dict[str, Any] = {}  # beats/downbeats/timing of the last analysed file


def pitch_class(name: str) -> int:
    pc = PC[name[0]]
    for ch in name[1:]:
        pc += 1 if ch == "#" else -1 if ch == "b" else 0
    return pc % 12


def harte(label: str) -> tuple[int, str] | None:
    """Harte chord label -> (root, one of our 5 qualities); None if outside maj/min classes."""
    if label in ("N", "X"):
        return None
    root, _, rest = label.partition(":")
    quality = re.split(r"[(/]", rest or "maj")[0]
    table = {
        "maj": "maj",
        "maj6": "maj",
        "": "maj",
        "7": "7",
        "9": "7",
        "11": "7",
        "13": "7",
        "maj7": "maj7",
        "maj9": "maj7",
        "min": "min",
        "min6": "min",
        "min7": "min7",
        "min9": "min7",
        "min11": "min7",
    }
    q = table.get(quality)
    return None if q is None else (pitch_class(root), q)


def analyse(path: Path) -> tuple[Any, Any, float, float]:
    t0 = time.perf_counter()
    fx = FeatureExtractor()
    mel = beat_model.LogMelStream()
    spect = []
    for chunk in decode_f32(path, out_channels=1, sample_rate=ANALYSIS_RATE):
        mono = np.ascontiguousarray(chunk[:, 0])
        fx.push(mono)
        if BEAT_MODEL is not None:
            spect.append(mel.push(mono))
    features = fx.finish()
    rhythm = analyze_rhythm(features)
    harmony = analyze_harmony(features, rhythm.beats, rhythm.downbeats, rhythm.onsets)
    secs = time.perf_counter() - t0
    NEURAL.clear()
    if BEAT_MODEL is not None:
        t1 = time.perf_counter()
        spect.append(mel.finish())
        beats, downs, report = BEAT_MODEL.track(np.concatenate(spect))
        NEURAL.update(
            beats=beats, downs=downs, secs=time.perf_counter() - t1, backend=report.backend
        )
    return rhythm, harmony, secs, features.duration_samples / ANALYSIS_RATE


def any_level_f(est: Any, ref: Any) -> float:
    """Best beat F over the reference at half, same and double metrical level.

    Reference beat levels follow the notation (MusicNet scores in eighths or halves), so a
    tracker at a neighbouring, equally valid level scores ~0 on plain F. Secondary metric.
    """
    ref = np.asarray(ref, np.float64)
    levels = [ref, ref[0::2], ref[1::2]]
    if len(ref) > 1:
        levels.append(np.sort(np.concatenate([ref, (ref[:-1] + ref[1:]) / 2])))
    return max(f_measure(list(est), list(lv)) for lv in levels)


def neural_fields(ref_beats: Any, ref_downs: Any, ref_bpm: float) -> dict[str, Any]:
    if not NEURAL:
        return {}
    beats, downs = NEURAL["beats"], NEURAL["downs"]
    bpm = 60.0 / float(np.median(np.diff(beats))) if len(beats) > 2 else 0.0
    out: dict[str, Any] = {
        "nn_bpm_est": round(bpm, 1),
        **{f"nn_{k}": v for k, v in tempo_scores(bpm, ref_bpm).items()},
        "nn_beat_f": round(f_measure(list(beats), list(ref_beats)), 3),
        "nn_beat_f_any": round(any_level_f(beats, ref_beats), 3),
    }
    if ref_downs is not None:
        out["nn_downbeat_f"] = round(f_measure(list(downs), list(ref_downs)), 3)
    out["nn_s"] = round(NEURAL["secs"], 2)
    out["nn_backend"] = NEURAL["backend"]
    return out


def tempo_scores(est: float, ref: float) -> dict[str, bool]:
    ratio = est / ref if ref else 0
    return {
        "acc1": abs(ratio - 1) < 0.04,
        "acc2": any(abs(ratio - f) < 0.04 * f for f in (0.5, 1, 2, 1 / 3, 3)),
    }


def est_chords(h: Any) -> list[tuple[float, float, int | None, str | None]]:
    return [(c.start, c.end, c.root, c.quality) for c in h.chords]


def est_notes(h: Any) -> list[tuple[float, float, float]]:
    return [(n.start, n.end, n.midi) for n in h.melody_notes]


# --- GuitarSet ----------------------------------------------------------------------------


def guitarset(limit: int | None) -> list[dict[str, Any]]:
    root = DATA / "guitarset"
    out = []
    for wav in sorted((root / "audio_mono-mic").glob("*.wav"))[:limit]:
        title = wav.stem.removesuffix("_mic")
        jams = json.loads((root / "annotation" / f"{title}.jams").read_text("utf-8"))
        ann: dict[str, list[Any]] = {}
        for a in jams["annotations"]:
            ann.setdefault(a["namespace"], []).append(a)
        beats = [d["time"] for d in ann["beat_position"][0]["data"]]
        downs = [d["time"] for d in ann["beat_position"][0]["data"] if d["value"]["position"] == 1]
        ref_bpm = float(ann["tempo"][0]["data"][0]["value"])
        tonic, mode = ann["key_mode"][0]["data"][0]["value"].split(":")
        ref_key = (pitch_class(tonic), mode)
        # The first chord annotation is the instructed lead sheet; the second is what was played.
        chords = []
        for d in ann["chord"][1]["data"]:
            h = harte(d["value"])
            if h is not None:
                chords.append((d["time"], d["time"] + d["duration"], h[0], h[1], h[0]))
        notes = sorted(
            (d["time"], d["time"] + d["duration"], round(d["value"]), "")
            for a in ann["note_midi"]
            for d in a["data"]
            if d["value"] >= MELODY_MIN_MIDI
        )
        rhythm, harmony, secs, dur = analyse(wav)
        key = harmony.key_candidates[0]
        row: dict[str, Any] = {
            "dataset": "GuitarSet",
            "item": title,
            "style": re.sub(r"\d.*", "", title.split("_")[1]),
            "part": title.rsplit("_", 1)[1],
            "duration_s": round(dur, 1),
            "bpm_ref": ref_bpm,
            "bpm_est": round(rhythm.bpm, 1),
            **tempo_scores(rhythm.bpm, ref_bpm),
            "beat_f": round(f_measure(rhythm.beats, beats), 3),
            "beat_f_any": round(any_level_f(rhythm.beats, beats), 3),
            "downbeat_f": round(f_measure(rhythm.downbeats, downs), 3),
            **neural_fields(beats, downs, ref_bpm),
            "key_ref": f"{tonic} {mode}",
            "key_est": key.label,
            "key_mirex": mirex_key_score((key.tonic, key.mode), ref_key),
            "analysis_s": round(secs, 2),
        }
        if row["part"] == "comp" and chords:
            row["chord_majmin"] = round(chord_overlap(est_chords(harmony), chords, True), 3)
            row["chord_full"] = round(chord_overlap(est_chords(harmony), chords, False), 3)
        if row["part"] == "solo" and notes:
            row["melody_note_acc"] = round(melody_note_accuracy(est_notes(harmony), notes), 3)
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


# --- MusicNet ---------------------------------------------------------------------------------

# Key of the movement (not only the work title); only movements whose key is unambiguous and
# stays in the title key are scored. Others are reported without a key score.
MUSICNET_KEYS = {
    "1759": ("C", "minor"),  # Schubert D.958 iii Menuetto
    "2191": ("E", "major"),  # Bach BWV 1006 Bourree
    "2210": ("C#", "minor"),  # WTC I Prelude 4
    "2303": ("D", "major"),  # WTC I Prelude 5
    "2390": ("D", "minor"),  # Beethoven Op. 31/2 iii
    "2633": ("G", "major"),  # Beethoven Op. 14/2 iii
    "2678": ("G", "major"),  # Beethoven Op. 14/2 i
}
VIOLIN = 41


def musicnet(limit: int | None) -> list[dict[str, Any]]:
    root = DATA / "musicnet"
    manifest = json.loads((root / "MANIFEST.json").read_text("utf-8"))
    out = []
    for rid, meta in list(sorted(manifest["recordings"].items()))[:limit]:
        wav = next(root.glob(f"musicnet/*_data/{rid}.wav"))
        labels = list(csv.DictReader(next(root.glob(f"musicnet/*_labels/{rid}.csv")).open()))
        sr = 44100
        t = np.array([int(r["start_time"]) / sr for r in labels])
        b = np.array([float(r["start_beat"]) for r in labels])
        order = np.argsort(b)
        b, t = b[order], t[order]
        b, idx = np.unique(b, return_index=True)
        t = t[idx]
        ref_beats = list(np.interp(np.arange(np.ceil(b[0]), np.floor(b[-1]) + 1), b, t))
        ref_bpm = 60.0 / float(np.median(np.diff(ref_beats))) if len(ref_beats) > 2 else 0.0
        rhythm, harmony, secs, dur = analyse(wav)
        key = harmony.key_candidates[0]
        row: dict[str, Any] = {
            "dataset": "MusicNet",
            "item": rid,
            "work": f"{meta['composer']}: {meta['composition']} - {meta['movement']}",
            "ensemble": meta["ensemble"],
            "duration_s": round(dur, 1),
            "bpm_ref_median": round(ref_bpm, 1),
            "bpm_est": round(rhythm.bpm, 1),
            **tempo_scores(rhythm.bpm, ref_bpm),
            "beat_f": round(f_measure(rhythm.beats, ref_beats), 3),
            "beat_f_any": round(any_level_f(rhythm.beats, ref_beats), 3),
            **neural_fields(ref_beats, None, ref_bpm),
            "key_est": key.label,
            "analysis_s": round(secs, 2),
        }
        if rid in MUSICNET_KEYS:
            tonic, mode = MUSICNET_KEYS[rid]
            row["key_ref"] = f"{tonic} {mode}"
            row["key_mirex"] = mirex_key_score((key.tonic, key.mode), (pitch_class(tonic), mode))
        violin = [
            (int(r["start_time"]) / sr, int(r["end_time"]) / sr, int(r["note"]))
            for r in labels
            if int(r["instrument"]) == VIOLIN
        ]
        if violin:
            # Skyline: where double stops overlap, the top note is the melody.
            ref = []
            for a, z, m in violin:
                mid = (a + z) / 2
                top = max(mm for aa, zz, mm in violin if aa <= mid < zz)
                if m == top and m >= MELODY_MIN_MIDI:
                    ref.append((a, z, m, ""))
            row["violin_melody_note_acc"] = round(melody_note_accuracy(est_notes(harmony), ref), 3)
            row["violin_notes"] = len(ref)
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


# --- BabySlakh (MIDI tempo map) ---------------------------------------------------------------


def _varlen(data: bytes, i: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[i]
        i += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, i


def midi_tempo_map(
    path: Path,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int, int]], int]:
    """(ticks per beat, [(tick, us per beat)], [(tick, numerator, denominator)], last tick)."""
    data = path.read_bytes()
    assert data[:4] == b"MThd"
    _, ntracks, division = struct.unpack(">HHH", data[8:14])
    i, tempos, sigs, last = 14, [], [], 0
    for _ in range(ntracks):
        assert data[i : i + 4] == b"MTrk"
        length = struct.unpack(">I", data[i + 4 : i + 8])[0]
        j, end, tick, status = i + 8, i + 8 + length, 0, 0
        while j < end:
            delta, j = _varlen(data, j)
            tick += delta
            if data[j] == 0xFF:
                kind = data[j + 1]
                size, j = _varlen(data, j + 2)
                if kind == 0x51:
                    tempos.append((tick, int.from_bytes(data[j : j + 3], "big")))
                elif kind == 0x58:
                    sigs.append((tick, data[j], 2 ** data[j + 1]))
                j += size
            elif data[j] in (0xF0, 0xF7):
                size, j = _varlen(data, j + 1)
                j += size
            else:
                if data[j] & 0x80:
                    status = data[j]
                    j += 1
                j += 1 if status & 0xF0 in (0xC0, 0xD0) else 2
            last = max(last, tick)
        i = end
    return division, sorted(tempos) or [(0, 500000)], sorted(sigs), last


def babyslakh(limit: int | None) -> list[dict[str, Any]]:
    root = DATA / "babyslakh" / "babyslakh_16k"
    out = []
    for track in sorted(p for p in root.iterdir() if (p / "mix.wav").exists())[:limit]:
        division, tempos, sigs, last = midi_tempo_map(track / "all_src.mid")

        def seconds(
            tick: float, tempos: list[tuple[int, int]] = tempos, division: int = division
        ) -> float:
            t, prev_tick, us = 0.0, 0, tempos[0][1]
            for tt, uu in tempos:
                if tt >= tick:
                    break
                t += (tt - prev_tick) * us / division / 1e6
                prev_tick, us = tt, uu
            return t + (tick - prev_tick) * us / division / 1e6

        beats = [seconds(k * division) for k in range(int(last / division) + 1)]
        num, den = (sigs[0][1], sigs[0][2]) if sigs else (4, 4)
        downs = beats[::num] if den == 4 else []
        meta = (track / "metadata.yaml").read_text("utf-8")
        instruments = sorted(set(re.findall(r"inst_class: (.+)", meta)))
        rhythm, _, secs, dur = analyse(track / "mix.wav")
        beats = [b for b in beats if b < dur]
        ref_bpm = 60e6 / tempos[0][1]
        row = {
            "dataset": "BabySlakh",
            "item": track.name,
            "duration_s": round(dur, 1),
            "meter_ref": f"{num}/{den}",
            "tempo_changes_ref": len({u for _, u in tempos}) - 1,
            "bpm_ref": round(ref_bpm, 1),
            "bpm_est": round(rhythm.bpm, 1),
            **tempo_scores(rhythm.bpm, ref_bpm),
            "beat_f": round(f_measure(rhythm.beats, beats), 3),
            "beat_f_any": round(any_level_f(rhythm.beats, beats), 3),
            "downbeat_f": round(f_measure(rhythm.downbeats, [d for d in downs if d < dur]), 3)
            if downs
            else None,
            **neural_fields(beats, [d for d in downs if d < dur] if downs else None, ref_bpm),
            "instruments_ref": instruments,
            "instrument_recognition": "not evaluated: needs a model (not yet approved)",
            "analysis_s": round(secs, 2),
        }
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str, subset: list[dict[str, Any]]) -> float | None:
        vals = [float(r[key]) for r in subset if r.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    out: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["dataset"], []).append(r)
        if r["dataset"] == "GuitarSet":
            groups.setdefault(f"GuitarSet {r['style']} {r['part']}", []).append(r)
        if r["dataset"] == "MusicNet":
            groups.setdefault(f"MusicNet {r['ensemble']}", []).append(r)
    keys = (
        "acc1",
        "acc2",
        "beat_f",
        "beat_f_any",
        "downbeat_f",
        "nn_acc1",
        "nn_acc2",
        "nn_beat_f",
        "nn_beat_f_any",
        "nn_downbeat_f",
        "key_mirex",
        "chord_majmin",
        "chord_full",
        "melody_note_acc",
        "violin_melody_note_acc",
    )
    for name, subset in sorted(groups.items()):
        out[name] = {
            "n": len(subset),
            **{k: mean(k, subset) for k in keys if mean(k, subset) is not None},
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--beat-model", action="store_true")
    ap.add_argument("--backend", default="directml", choices=("cpu", "directml"))
    args = ap.parse_args()
    if args.beat_model:
        from xsample.config import default_data_dir

        BEAT_MODEL = beat_model.BeatTracker(default_data_dir() / "models", args.backend)
    rows = guitarset(args.limit) + musicnet(args.limit) + babyslakh(args.limit)
    report = {
        "kind": "real audio (licensed subsets); not comparable with the synthetic benchmark",
        "analyzer": "xsample.analysis.dsp (CPU)",
        "beat_model": None
        if not args.beat_model
        else {
            "model": f"{beat_model.DISPLAY_NAME} (MIT), ONNX, minimal postprocessing",
            "sha256": [
                beat_model.PINNED[f][0] for f in (beat_model.FULL_FILE, beat_model.SHORT_FILE)
            ],
            "requested_backend": args.backend,
            "training_overlap": TRAINING_OVERLAP,
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summarise(rows),
        "items": rows,
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1), "utf-8")
    print("SUMMARY", json.dumps(report["summary"], indent=1))
