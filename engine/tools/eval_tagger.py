"""Instrument, vocal and genre tagging evaluation on the locally fetched, licensed subsets.

    uv run python tools/eval_tagger.py [--backend cpu|directml] [--limit N]

Runs the installed PANNs MobileNetV1 export through the engine's Tagger (same front end,
windows and thresholds as the analysis job) and scores every X-Sample label against ground
truth derived from the datasets' own annotations. Results go to
test-reports/tagger_eval.json. Nothing here changes a threshold or a mapping: those are fixed
in xsample.analysis.tagger before evaluation, and the rules below that decide which labels
count as evaluated (or, for genres, may be reported as detected) were also fixed beforehand.

Ground truth (none of these datasets is part of AudioSet; whether a MusicNet or GuitarSet
recording also appears in some YouTube upload cannot be ruled out):
- BabySlakh (20 synthesised Lakh MIDI renders, 16 kHz): an instrument label is present when
  a stem with a positive General MIDI program has at least one note; programs that only
  partly match a label (e.g. synth strings for Strings) make it "ambiguous" and the item or
  window is skipped for that label. No singing, speech or genre annotations: vocal labels
  are scored as negatives only, genres are not scored.
- MusicNet (10 classical recordings): instruments from the note labels (GM programs), genre
  Classical; no voices.
- GuitarSet (48 solo acoustic guitar recordings): Guitar present, every other instrument and
  voice absent; genre from the style code (Funk, Jazz, Rock, BN -> Latin; SS has no genre).

Windows are the tagger's 10 s windows (5 s hop); a window is positive for a label when a
note of a positive program starts inside it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "tools"))

from eval_real_audio import DATA, _varlen  # noqa: E402

from xsample.analysis import tagger  # noqa: E402
from xsample.audio.ffmpeg import decode_f32  # noqa: E402
from xsample.config import default_data_dir  # noqa: E402

REPORT = ENGINE / "test-reports" / "tagger_eval.json"
LABELS = [lb.name for lb in tagger.LABELS]
KIND = {lb.name: lb.kind for lb in tagger.LABELS}
INSTRUMENTS = [n for n in LABELS if KIND[n] == "instrument"]
VOCALS = [n for n in LABELS if KIND[n] == "vocal"]
GENRES = [n for n in LABELS if KIND[n] == "genre"]
# Decision rules, fixed before evaluation (docs/EVALUATION.md).
EVALUATED_MIN = 5  # pieces with and without the label, pooled over datasets
GENRE_MIN, GENRE_PRECISION, GENRE_RECALL = 10, 0.8, 0.5


def gm_labels(program: int, is_drum: bool = False) -> tuple[set[str], set[str]]:
    """(positive, ambiguous) labels for a 0-based General MIDI program."""
    if is_drum:
        return {"Drums"}, set()
    p = program
    table: list[tuple[range, set[str], set[str]]] = [
        (range(0, 6), {"Piano"}, set()),
        (range(6, 8), set(), {"Piano"}),
        (range(16, 21), {"Organ"}, set()),
        (range(21, 24), set(), {"Organ"}),
        (range(24, 32), {"Guitar"}, set()),
        (range(32, 38), {"Bass"}, set()),
        (range(38, 40), {"Bass"}, {"Synthesizer"}),
        (range(40, 41), {"Violin", "Strings"}, set()),
        (range(41, 42), {"Strings"}, {"Violin"}),
        (range(42, 43), {"Cello", "Strings"}, set()),
        (range(43, 44), {"Strings"}, {"Bass"}),
        (range(44, 46), {"Strings"}, {"Violin", "Cello"}),
        (range(47, 48), set(), {"Drums"}),
        (range(48, 50), {"Strings"}, {"Violin", "Cello"}),
        (range(50, 52), set(), {"Strings", "Violin", "Cello", "Synthesizer"}),
        (range(52, 55), set(), {"Singing", "Synthesizer"}),
        (range(56, 62), {"Brass"}, set()),
        (range(62, 64), set(), {"Brass", "Synthesizer"}),
        (range(64, 68), {"Saxophone"}, set()),
        (range(71, 72), {"Clarinet"}, set()),
        (range(72, 73), set(), {"Flute"}),
        (range(73, 74), {"Flute"}, set()),
        (range(74, 80), set(), {"Flute"}),
        (range(80, 96), {"Synthesizer"}, set()),
        (range(96, 104), set(), {"Synthesizer"}),
        (range(112, 120), set(), {"Drums"}),
    ]
    for r, pos, amb in table:
        if p in r:
            return pos, amb
    return set(), set()


def truth_from_notes(
    notes: list[tuple[float, int, bool]], starts: list[int], n_frames: int, scored: list[str]
) -> tuple[dict[str, bool | None], list[dict[str, bool | None]]]:
    """Piece and per-window truth from (onset s, program, is_drum) notes."""

    def judge(subset: list[tuple[float, int, bool]]) -> dict[str, bool | None]:
        pos: set[str] = set()
        amb: set[str] = set()
        for _, prog, drum in subset:
            p, a = gm_labels(prog, drum)
            pos |= p
            amb |= a
        return {n: True if n in pos else (None if n in amb else False) for n in scored}

    piece = judge(notes)
    windows = []
    for s in starts:
        a, z = s / tagger.FPS, min(s + tagger.WINDOW, n_frames) / tagger.FPS
        windows.append(judge([nt for nt in notes if a <= nt[0] < z]))
    return piece, windows


# --- datasets --------------------------------------------------------------------------------


def midi_notes(path: Path) -> list[tuple[float, int, bool]]:
    """Note onsets (seconds, program, is drum channel) of a type 0/1 MIDI file."""
    data = path.read_bytes()
    _, ntracks, division = struct.unpack(">HHH", data[8:14])
    i, tempos, events = 14, [(0, 500000)], []
    for _ in range(ntracks):
        length = struct.unpack(">I", data[i + 4 : i + 8])[0]
        j, end, tick, status = i + 8, i + 8 + length, 0, 0
        program = [0] * 16
        while j < end:
            delta, j = _varlen(data, j)
            tick += delta
            if data[j] == 0xFF:
                kind = data[j + 1]
                size, j = _varlen(data, j + 2)
                if kind == 0x51:
                    tempos.append((tick, int.from_bytes(data[j : j + 3], "big")))
                j += size
            elif data[j] in (0xF0, 0xF7):
                size, j = _varlen(data, j + 1)
                j += size
            else:
                if data[j] & 0x80:
                    status = data[j]
                    j += 1
                kind, ch = status & 0xF0, status & 0x0F
                if kind == 0xC0:
                    program[ch] = data[j]
                if kind == 0x90 and data[j + 1] > 0:
                    events.append((tick, program[ch], ch == 9))
                j += 1 if kind in (0xC0, 0xD0) else 2
        i = end
    tempos = sorted(tempos)
    if len(tempos) > 1 and tempos[1][0] == 0:
        tempos = tempos[1:]

    def seconds(tick: int) -> float:
        t, prev, us = 0.0, 0, tempos[0][1]
        for tt, uu in tempos:
            if tt >= tick:
                break
            t += (tt - prev) * us / division / 1e6
            prev, us = tt, uu
        return t + (tick - prev) * us / division / 1e6

    return [(seconds(t), p, d) for t, p, d in events]


def babyslakh(limit: int | None) -> list[dict[str, Any]]:
    root = DATA / "babyslakh" / "babyslakh_16k"
    items = []
    for track in sorted(p for p in root.iterdir() if (p / "mix.wav").exists())[:limit]:
        meta = (track / "metadata.yaml").read_text("utf-8")
        stems = re.findall(
            r"\n  (S\d\d):\n(?:    .*\n)*?    is_drum: (\w+)\n(?:    .*\n)*?    program_num: (\d+)",
            meta,
        )
        notes = []
        for stem, drum, prog in stems:
            midi = track / "MIDI" / f"{stem}.mid"
            if midi.exists():  # stems without notes have no MIDI file
                notes += [(t, int(prog), drum == "true") for t, _, _ in midi_notes(midi)]
        items.append(
            {
                "dataset": "BabySlakh",
                "item": track.name,
                "audio": track / "mix.wav",
                "notes": notes,
                "scored": INSTRUMENTS + VOCALS,
                "fixed": {"Rapping": False, "Speech": False},
            }
        )
    return items


def musicnet(limit: int | None) -> list[dict[str, Any]]:
    root = DATA / "musicnet"
    manifest = json.loads((root / "MANIFEST.json").read_text("utf-8"))
    items = []
    for rid, meta in list(sorted(manifest["recordings"].items()))[:limit]:
        labels = list(csv.DictReader(next(root.glob(f"musicnet/*_labels/{rid}.csv")).open()))
        notes = [(int(r["start_time"]) / 44100, int(r["instrument"]) - 1, False) for r in labels]
        genres = {g: g == "Classical" for g in GENRES}
        items.append(
            {
                "dataset": "MusicNet",
                "item": rid,
                "ensemble": meta["ensemble"],
                "audio": next(root.glob(f"musicnet/*_data/{rid}.wav")),
                "notes": notes,
                "scored": INSTRUMENTS + VOCALS,
                "fixed": {**genres, "Singing": False, "Rapping": False, "Speech": False},
            }
        )
    return items


GUITARSET_STYLE = {"Funk": "Funk", "Jazz": "Jazz", "Rock": "Rock", "BN": "Latin", "SS": None}
GUITARSET_GENRES = ("Funk", "Jazz", "Rock", "Latin", "Classical")


def guitarset(limit: int | None) -> list[dict[str, Any]]:
    root = DATA / "guitarset"
    items = []
    for wav in sorted((root / "audio_mono-mic").glob("*.wav"))[:limit]:
        style = re.match(r"\d+_([A-Za-z]+)\d", wav.stem)
        assert style is not None, wav.name
        genre = GUITARSET_STYLE[style.group(1)]
        fixed: dict[str, bool | None] = {n: n == "Guitar" for n in INSTRUMENTS + VOCALS}
        fixed.update({g: g == genre for g in GUITARSET_GENRES})
        items.append(
            {
                "dataset": "GuitarSet",
                "item": wav.stem,
                "style": style.group(1),
                "audio": wav,
                "notes": None,
                "scored": [],
                "fixed": fixed,
            }
        )
    return items


# --- scoring ---------------------------------------------------------------------------------


def logmel_of(path: Path) -> np.ndarray:
    stream = tagger.LogMelStream()
    parts = [
        stream.push(np.ascontiguousarray(c[:, 0]))
        for c in decode_f32(path, out_channels=1, sample_rate=tagger.SAMPLE_RATE)
    ]
    parts.append(stream.finish())
    return np.concatenate(parts)


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return round(float(((p > n) + 0.5 * (p == n)).mean()), 3)


def metrics(pairs: list[tuple[float, bool]], threshold: float) -> dict[str, Any]:
    pos = [s for s, t in pairs if t]
    neg = [s for s, t in pairs if not t]
    tp = sum(s >= threshold for s in pos)
    fp = sum(s >= threshold for s in neg)
    return {
        "n_pos": len(pos),
        "n_neg": len(neg),
        "tp": tp,
        "fp": fp,
        "fn": len(pos) - tp,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / len(pos), 3) if pos else None,
        "false_positive_rate": round(fp / len(neg), 3) if neg else None,
        "auc": auc(pos, neg),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="directml", choices=("cpu", "directml"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    models = default_data_dir() / "models"
    if not tagger.installed(models):
        sys.exit("the PANNs tagger export is not installed")
    model = tagger.Tagger(models, args.backend)
    rows = []
    # (dataset, label, level) -> [(score, truth)]
    pairs: dict[tuple[str, str, str], list[tuple[float, bool]]] = {}
    for item in babyslakh(args.limit) + musicnet(args.limit) + guitarset(args.limit):
        t0 = time.perf_counter()
        logmel = logmel_of(item["audio"])
        t1 = time.perf_counter()
        starts, scores, report = model.tag(logmel)
        t2 = time.perf_counter()
        piece_scores = {n: float(scores[:, j].max()) for j, n in enumerate(LABELS)}
        truth: dict[str, bool | None] = dict(item["fixed"])
        windows: list[dict[str, bool | None]] = []
        if item["notes"] is not None:
            piece, windows = truth_from_notes(item["notes"], starts, len(logmel), item["scored"])
            truth.update({k: v for k, v in piece.items() if k not in item["fixed"]})
            for w in windows:
                w.update({k: v for k, v in item["fixed"].items() if KIND[k] != "genre"})
        for name, value in truth.items():
            if value is not None:
                pairs.setdefault((item["dataset"], name, "piece"), []).append(
                    (piece_scores[name], value)
                )
        for w, row_scores in zip(windows, scores, strict=False):
            for name, value in w.items():
                if value is not None:
                    pairs.setdefault((item["dataset"], name, "window"), []).append(
                        (float(row_scores[LABELS.index(name)]), value)
                    )
        row = {
            "dataset": item["dataset"],
            "item": item["item"],
            **{k: item[k] for k in ("ensemble", "style") if k in item},
            "duration_s": round(len(logmel) / tagger.FPS, 1),
            "windows": len(starts),
            "backend": report.backend,
            "frontend_s": round(t1 - t0, 3),
            "tagger_s": round(t2 - t1, 3),
            "present": sorted(k for k, v in truth.items() if v),
            "detected": sorted(k for k, s in piece_scores.items() if s >= tagger.DETECT_THRESHOLD),
            "top": sorted(piece_scores.items(), key=lambda kv: -kv[1])[:6],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    model.close()

    per_class: dict[str, Any] = {}
    for (dataset, name, level), ps in sorted(pairs.items()):
        per_class.setdefault(name, {}).setdefault(level, {})[dataset] = {
            "detected": metrics(ps, tagger.DETECT_THRESHOLD),
            "detected_or_uncertain": metrics(ps, tagger.UNCERTAIN_THRESHOLD),
        }
    pooled: dict[str, Any] = {}
    for name in LABELS:
        for level in ("piece", "window"):
            ps = [p for (d, n, lv), v in pairs.items() if n == name and lv == level for p in v]
            if ps:
                pooled.setdefault(name, {})[level] = {
                    "detected": metrics(ps, tagger.DETECT_THRESHOLD),
                    "detected_or_uncertain": metrics(ps, tagger.UNCERTAIN_THRESHOLD),
                }
    evaluated = sorted(
        n
        for n, v in pooled.items()
        if "piece" in v
        and v["piece"]["detected"]["n_pos"] >= EVALUATED_MIN
        and v["piece"]["detected"]["n_neg"] >= EVALUATED_MIN
    )
    genre_evidence = sorted(
        n
        for n in GENRES
        if n in pooled
        and (m := pooled[n]["piece"]["detected"])["n_pos"] >= GENRE_MIN
        and m["n_neg"] >= GENRE_MIN
        and (m["precision"] or 0) >= GENRE_PRECISION
        and (m["recall"] or 0) >= GENRE_RECALL
    )
    times = [r["tagger_s"] / max(r["duration_s"], 1e-9) * 600 for r in rows]
    report_out = {
        "model": tagger.DISPLAY_NAME,
        "model_sha256": tagger.PINNED[0],
        "backend_requested": args.backend,
        "backends_used": sorted({r["backend"] for r in rows}),
        "thresholds": {
            "detected": tagger.DETECT_THRESHOLD,
            "uncertain": tagger.UNCERTAIN_THRESHOLD,
        },
        "rules": {
            "evaluated": f">= {EVALUATED_MIN} positive and negative pieces (pooled)",
            "genre_evidence": f">= {GENRE_MIN} positive and negative pieces, precision >= "
            f"{GENRE_PRECISION}, recall >= {GENRE_RECALL} at the detect threshold",
        },
        "evaluated_labels": evaluated,
        "genre_evidence": genre_evidence,
        "tagger_s_per_10min_median": round(float(np.median(times)), 2) if times else None,
        "per_class": per_class,
        "pooled": pooled,
        "items": rows,
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report_out, indent=1), "utf-8")
    print("evaluated:", evaluated)
    print("genre evidence:", genre_evidence)
    for name in LABELS:
        for level in ("piece", "window"):
            m = pooled.get(name, {}).get(level, {}).get("detected")
            if m:
                print(
                    f"{name:12s} {level:6s} pos {m['n_pos']:4d} neg {m['n_neg']:4d} "
                    f"P {m['precision']} R {m['recall']} FPR {m['false_positive_rate']} "
                    f"AUC {m['auc']}"
                )


if __name__ == "__main__":
    main()
