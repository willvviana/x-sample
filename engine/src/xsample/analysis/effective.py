"""Detected analysis + user corrections = what the editor uses (ADR-0011).

Pure functions over the sidecar and the manifest corrections; nothing is re-analysed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from xsample.analysis.harmony import PITCH_CLASSES
from xsample.analysis.schema import AnalysisResult, ChordOut, KeyOut
from xsample.project.model import AnalysisCorrections, BeatGridCorrection, ChordCorrection

QUALITY_SUFFIX = {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "min7"}


class EffectiveTag(BaseModel):
    kind: Literal["genre", "instrument"]
    label: str
    origin: Literal["user", "detected"]
    confidence: float | None


class EffectiveAnalysis(BaseModel):
    bpm: float
    meter: int | None
    beats: list[int]
    downbeats: list[int]
    key: KeyOut | None
    chords: list[ChordOut]
    tags: list[EffectiveTag]
    corrected: list[Literal["beat_grid", "key", "chords", "tags"]]


def chord_label(root: int | None, quality: str | None, bass: int | None) -> str:
    if root is None or quality is None:
        return "N"
    label = PITCH_CLASSES[root] + QUALITY_SUFFIX[quality]
    return label + (f"/{PITCH_CLASSES[bass]}" if bass is not None and bass != root else "")


def apply_beat_grid(
    beats: list[int],
    downbeats: list[int],
    meter: int | None,
    grid: BeatGridCorrection,
    sample_rate: int,
    frames: int,
) -> tuple[list[int], list[int], int | None]:
    if grid.is_identity():
        return beats, downbeats, meter
    b = np.asarray(beats, dtype=np.float64)
    if grid.bpm is not None:
        period = sample_rate * 60.0 / grid.bpm
        anchor = float(b[0]) if len(b) else 0.0
        anchor -= np.floor(anchor / period) * period  # earliest grid position >= 0
        b = anchor + period * np.arange(int(np.ceil((frames - anchor) / period)))
    elif grid.scale == "double" and len(b) > 1:
        mids = (b[:-1] + b[1:]) / 2
        b = np.sort(np.concatenate([b, mids]))
    elif grid.scale == "half":
        b = b[grid.half_phase :: 2]
    b = b + grid.offset
    b = b[(b >= 0) & (b < frames)]
    out = [round(x) for x in b]
    eff_meter = grid.meter or meter or 4
    if not out:
        return [], [], eff_meter
    # Keep the detected bar phase (nearest beat to the first detected downbeat), then shift.
    phase = 0
    if downbeats:
        phase = int(np.argmin(np.abs(np.asarray(out) - (downbeats[0] + grid.offset))))
    phase = (phase + grid.downbeat_shift) % eff_meter
    return out, out[phase::eff_meter], eff_meter


def apply_chords(detected: list[ChordOut], fixes: list[ChordCorrection]) -> list[ChordOut]:
    if not fixes:
        return detected
    spans = sorted((f.start, f.end) for f in fixes)
    out: list[ChordOut] = []
    for c in detected:
        pieces = [(c.start, c.end)]
        for a, z in spans:
            nxt = []
            for s, e in pieces:
                if z <= s or a >= e:
                    nxt.append((s, e))
                    continue
                if s < a:
                    nxt.append((s, a))
                if z < e:
                    nxt.append((z, e))
            pieces = nxt
        out += [c.model_copy(update={"start": s, "end": e}) for s, e in pieces if e > s]
    for f in fixes:
        out.append(
            ChordOut(
                start=f.start,
                end=f.end,
                root=f.root,
                quality=f.quality,
                bass=f.bass,
                label=chord_label(f.root, f.quality, f.bass),
                confidence=1.0,
            )
        )
    return sorted(out, key=lambda c: c.start)


def effective(result: AnalysisResult, corr: AnalysisCorrections) -> EffectiveAnalysis:
    r = result.rhythm
    beats, downbeats, meter = apply_beat_grid(
        r.beats, r.downbeats, r.meter, corr.beat_grid, result.sample_rate, result.frames
    )
    if corr.beat_grid.bpm is not None:
        bpm = corr.beat_grid.bpm
    elif len(beats) > 1:
        bpm = round(60.0 * result.sample_rate / float(np.median(np.diff(beats))), 2)
    else:
        bpm = r.bpm
    key = result.harmony.key_candidates[0] if result.harmony.key_candidates else None
    if corr.key is not None:
        key = KeyOut(
            tonic=corr.key.tonic,
            mode=corr.key.mode,
            label=f"{PITCH_CLASSES[corr.key.tonic]} {corr.key.mode}",
            score=1.0,
        )
    tags = [
        EffectiveTag(kind=t.kind, label=t.label, origin="user", confidence=None) for t in corr.tags
    ]
    corrected: list[Literal["beat_grid", "key", "chords", "tags"]] = []
    if not corr.beat_grid.is_identity():
        corrected.append("beat_grid")
    if corr.key is not None:
        corrected.append("key")
    if corr.chords:
        corrected.append("chords")
    if corr.tags:
        corrected.append("tags")
    return EffectiveAnalysis(
        bpm=bpm,
        meter=meter,
        beats=beats,
        downbeats=downbeats,
        key=key,
        chords=apply_chords(result.harmony.chords, corr.chords),
        tags=tags,
        corrected=corrected,
    )
