"""Synthesized music with exact ground truth for analysis tests and benchmarks.

Everything is generated here (no third-party audio). Timbres are simple models: additive
piano-like tones, a bowed-string (violin-like) sawtooth with vibrato, a reed (sax-like) tone
with breath noise and formant colouring, and synthesized drums. Scores on these fixtures are
an upper bound for real recordings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]

QUAL = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
}


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


@dataclass
class Truth:
    bpm_start: float
    bpm_end: float
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    meter: int = 4
    key: tuple[int, str] = (0, "major")
    chords: list[tuple[float, float, int, str, int]] = field(
        default_factory=list
    )  # start,end,root,qual,bass
    sections: list[tuple[float, str]] = field(default_factory=list)
    melody: list[tuple[float, float, int, str]] = field(
        default_factory=list
    )  # start,end,midi,artic
    lead: str | None = None  # "violin" | "sax"
    swing: float = 0.0  # fraction of a 16th the off-16ths are delayed (0 = straight)


@dataclass
class Spec:
    bpm: float = 120.0
    bpm_end: float | None = None
    key_root: int = 0
    minor: bool = False
    meter: int = 4
    bars: int = 16
    drums: bool = True
    groove: str = "straight"  # straight | rnb
    sevenths: bool = False
    inversions: bool = False
    lead: str | None = None  # None | "violin" | "sax"
    noise: float = 0.0
    sections: bool = False
    rate: int = 22050
    seed: int = 1


def piano_tone(m: float, dur: float, rate: int, amp: float) -> F32:
    n = max(1, int(dur * rate))
    t = np.arange(n) / rate
    f = midi_hz(m)
    out = np.zeros(n)
    for h in range(1, 9):
        # Slight inharmonicity and faster decay of upper partials, like a struck string.
        fh = f * h * np.sqrt(1 + 0.0004 * h * h)
        if fh > rate / 2 - 100:
            break
        out += (amp / h**1.1) * np.sin(2 * np.pi * fh * t) * np.exp(-t * (1.2 + 0.6 * h))
    return (out * np.minimum(1.0, t / 0.004)).astype(np.float32)


def _pitch_track(
    m: float, n: int, rate: int, glide_from: float | None, vib_hz: float, vib_semi: float
) -> npt.NDArray[np.float64]:
    t = np.arange(n) / rate
    base = np.full(n, float(m))
    if glide_from is not None:
        g = min(n, int(0.08 * rate))
        base[:g] = np.linspace(glide_from, m, g)
    vib = vib_semi * np.sin(2 * np.pi * vib_hz * t) * np.minimum(1.0, t / 0.25)
    out: npt.NDArray[np.float64] = 440.0 * 2 ** ((base + vib - 69) / 12)
    return out


def violin_tone(
    m: float, dur: float, rate: int, amp: float, glide_from: float | None, rng: np.random.Generator
) -> F32:
    """Bowed string ≈ sawtooth (all harmonics, 1/h) with ~5.8 Hz, ±25-cent vibrato."""
    n = max(1, int(dur * rate))
    t = np.arange(n) / rate
    f = _pitch_track(m, n, rate, glide_from, 5.8, 0.25)
    phase = 2 * np.pi * np.cumsum(f) / rate
    out = np.zeros(n)
    for h in range(1, 12):
        if f.max() * h > rate / 2 - 100:
            break
        out += (amp / h) * np.sin(h * phase)
    out *= 1 + 0.03 * rng.standard_normal(n)  # bow noise
    env = np.minimum(1.0, t / 0.05) * np.minimum(1.0, (dur - t) / 0.03)
    return (out * env).astype(np.float32)


def sax_tone(
    m: float, dur: float, rate: int, amp: float, glide_from: float | None, rng: np.random.Generator
) -> F32:
    """Reed (tenor-sax-like): conical bore → strong odd *and* even harmonics, brighter with
    loudness, a formant bump near 1.2 kHz, breath noise, slower/wider vibrato (~5 Hz, ±30 c)
    and a short 'growl' attack."""
    n = max(1, int(dur * rate))
    t = np.arange(n) / rate
    f = _pitch_track(m, n, rate, glide_from, 5.0, 0.30)
    phase = 2 * np.pi * np.cumsum(f) / rate
    out = np.zeros(n)
    f0 = float(np.median(f))
    for h in range(1, 16):
        fh = f0 * h
        if fh > rate / 2 - 200:
            break
        formant = 1.0 + 1.5 * np.exp(-0.5 * ((fh - 1200.0) / 450.0) ** 2)
        out += (amp / h**0.8) * formant * np.sin(h * phase)
    breath = np.convolve(rng.standard_normal(n), np.ones(6) / 6, mode="same")
    out += 0.04 * amp * 10 * breath
    growl = 1 + 0.15 * np.sin(2 * np.pi * 28 * t) * np.exp(-t * 12)
    env = np.minimum(1.0, t / 0.03) * np.minimum(1.0, (dur - t) / 0.04)
    return (out * env * growl).astype(np.float32)


def _kick(rate: int) -> F32:
    n = int(0.25 * rate)
    t = np.arange(n) / rate
    f = 50 + 110 * np.exp(-t * 30)
    return (0.9 * np.sin(2 * np.pi * np.cumsum(f) / rate) * np.exp(-t * 12)).astype(np.float32)


def _snare(rate: int, rng: np.random.Generator) -> F32:
    n = int(0.18 * rate)
    t = np.arange(n) / rate
    tone = 0.2 * np.sin(2 * np.pi * 190 * t)
    return ((0.45 * rng.standard_normal(n) + tone) * np.exp(-t * 22)).astype(np.float32)


def _hat(rate: int, rng: np.random.Generator, amp: float = 0.12) -> F32:
    n = int(0.05 * rate)
    t = np.arange(n) / rate
    return (amp * np.diff(rng.standard_normal(n + 1)) * np.exp(-t * 90)).astype(np.float32)


def _add(y: F32, x: F32, t: float, rate: int) -> None:
    s = round(t * rate)
    if s >= len(y) or s < 0:
        return
    e = min(len(y), s + len(x))
    y[s:e] += x[: e - s]


def progression(spec: Spec) -> list[tuple[int, str]]:
    if spec.sevenths:  # R&B-style ii-V-I-vi (major) or i-iv-VII-III (minor), with sevenths
        return (
            [(2, "min7"), (7, "7"), (0, "maj7"), (9, "min7")]
            if not spec.minor
            else [(0, "min7"), (5, "min7"), (10, "7"), (3, "maj7")]
        )
    return (
        [(0, "maj"), (9, "min"), (5, "maj"), (7, "maj")]
        if not spec.minor
        else [(0, "min"), (8, "maj"), (3, "maj"), (10, "maj")]
    )


def chorus_progression(spec: Spec) -> list[tuple[int, str]]:
    if spec.sevenths:
        return (
            [(5, "maj7"), (7, "7"), (4, "min7"), (9, "min7")]
            if not spec.minor
            else [(8, "maj7"), (10, "7"), (0, "min7"), (7, "min7")]
        )
    return (
        [(5, "maj"), (7, "maj"), (4, "min"), (9, "min")]
        if not spec.minor
        else [(8, "maj"), (10, "maj"), (0, "min"), (7, "min")]
    )


def render(spec: Spec) -> tuple[F32, Truth]:
    rng = np.random.default_rng(spec.seed)
    rate = spec.rate
    bpm_end = spec.bpm_end if spec.bpm_end is not None else spec.bpm
    total_beats = spec.bars * spec.meter
    beats, t = [], 0.0
    for i in range(total_beats + 1):
        beats.append(t)
        t += 60.0 / (spec.bpm + (bpm_end - spec.bpm) * i / total_beats)
    seconds = beats[-1]
    y = np.zeros(int(seconds * rate) + rate, np.float32)
    truth = Truth(
        spec.bpm,
        bpm_end,
        beats=beats[:-1],
        meter=spec.meter,
        key=(spec.key_root, "minor" if spec.minor else "major"),
        lead=spec.lead,
    )
    swing = 0.12 if spec.groove == "rnb" else 0.0
    truth.swing = swing
    prev_lead: float | None = None
    for bar in range(spec.bars):
        b0 = bar * spec.meter
        start, end = beats[b0], beats[b0 + spec.meter]
        truth.downbeats.append(start)
        in_chorus = spec.sections and bar >= spec.bars // 2
        if spec.sections and bar in (0, spec.bars // 2):
            truth.sections.append((start, "chorus" if in_chorus else "verse"))
        root_off, qual = (chorus_progression(spec) if in_chorus else progression(spec))[bar % 4]
        root = (spec.key_root + root_off) % 12
        tones = [(root + iv) % 12 for iv in QUAL[qual]]
        bass_pc = tones[1] if spec.inversions and bar % 2 == 1 else root
        truth.chords.append((start, end, root, qual, bass_pc))
        # Piano: close voicing between C4 and B4, struck on beats 1 and 3 (1 only in 3/4).
        voicing = sorted(60 + pc for pc in tones)
        for k in range(0, spec.meter, 2):
            t_hit = beats[b0 + k]
            dur = beats[min(b0 + k + 2, total_beats)] - t_hit
            for m in voicing:
                _add(y, piano_tone(m, dur, rate, 0.10 if not in_chorus else 0.13), t_hit, rate)
        # Bass: syncopated R&B pattern (positions in beats) or one note per bar.
        period = (end - start) / spec.meter
        pattern = (
            [(0.0, 0.75), (0.75, 0.5), (1.5, 0.5), (2.5, 0.75), (3.25, 0.5)]
            if spec.groove == "rnb"
            else [(0.0, float(spec.meter))]
        )
        for pos, length in pattern:
            if pos < spec.meter:
                _add(
                    y,
                    piano_tone(36 + bass_pc, length * period, rate, 0.35),
                    start + pos * period,
                    rate,
                )
        if spec.drums:
            for k in range(spec.meter):
                tb, p = beats[b0 + k], beats[b0 + k + 1] - beats[b0 + k]
                if spec.meter == 4:
                    hit = _kick(rate) if k in (0, 2) else _snare(rate, rng)
                else:
                    hit = _kick(rate) if k == 0 else _snare(rate, rng)
                _add(y, hit, tb, rate)
                if spec.groove == "rnb":
                    for s16 in range(4):  # 16th hats, off-16ths delayed (swing)
                        frac = s16 / 4 + (swing / 4 if s16 % 2 == 1 else 0.0)
                        _add(
                            y, _hat(rate, rng, 0.10 if s16 % 2 == 0 else 0.06), tb + frac * p, rate
                        )
                    if k == 1:
                        _add(y, 0.5 * _kick(rate), tb + 0.75 * p, rate)  # ghost kick (syncopation)
                else:
                    for frac in (0.0, 0.5):
                        _add(y, _hat(rate, rng), tb + frac * p, rate)
        if spec.lead is not None:
            # Two half-bar notes per bar (chord third, then a scale tone); cycle articulations.
            scale = [0, 2, 4, 5, 7, 9, 11] if not spec.minor else [0, 2, 3, 5, 7, 8, 10]
            octave = 72 if spec.lead == "violin" else 60  # tenor sax sits an octave lower
            notes = [octave + tones[1], octave + (spec.key_root + scale[(bar * 2 + 1) % 7]) % 12]
            half = (end - start) / 2
            for k, m in enumerate(notes):
                t_on = start + k * half
                artic = ["attack", "legato", "glide"][(bar * 2 + k) % 3]
                glide_from = prev_lead if artic == "glide" and prev_lead is not None else None
                length = half * (0.97 if artic == "attack" else 1.0)
                tone_fn = violin_tone if spec.lead == "violin" else sax_tone
                # Lead mixed above the accompaniment, as a featured melody usually is.
                _add(y, tone_fn(m, length, rate, 0.12, glide_from, rng), t_on, rate)
                truth.melody.append((t_on, t_on + half, int(m), artic))
                prev_lead = float(m)
    y = y[: int(seconds * rate)]
    if spec.noise > 0:
        y += spec.noise * rng.standard_normal(len(y)).astype(np.float32)
    y /= max(1e-9, float(np.max(np.abs(y)))) / 0.8
    return y.astype(np.float32), truth


# --- metrics -------------------------------------------------------------------------------


def f_measure(est: list[float], ref: list[float], tol: float = 0.07, skip: float = 5.0) -> float:
    r = np.array([x for x in ref if x >= skip])
    e = np.array([x for x in est if x >= skip])
    if len(r) == 0 or len(e) == 0:
        return 0.0
    used: set[int] = set()
    hits = 0
    for x in e:
        j = int(np.argmin(np.abs(r - x)))
        if abs(r[j] - x) <= tol and j not in used:
            used.add(j)
            hits += 1
    p, rc = hits / len(e), hits / len(r)
    return 0.0 if p + rc == 0 else 2 * p * rc / (p + rc)


def chord_overlap(
    est: list[tuple[float, float, int | None, str | None]],
    ref: list[tuple[float, float, int, str, int]],
    majmin: bool = True,
    step: float = 0.05,
) -> float:
    """Fraction of time where the estimated chord matches (root + maj/min class, or exact)."""

    def cls(q: str | None) -> str | None:
        if q is None:
            return None
        return ("min" if q in ("min", "min7") else "maj") if majmin else q

    hits = total = 0
    for t in np.arange(0.0, ref[-1][1], step):
        r = next(((root, q) for a, b, root, q, _ in ref if a <= t < b), None)
        if r is None:
            continue
        total += 1
        e = next(((root, q) for a, b, root, q in est if a <= t < b), None)
        if e is not None and e[0] == r[0] and cls(e[1]) == cls(r[1]):
            hits += 1
    return hits / max(total, 1)


def mirex_key_score(est: tuple[int, str], ref: tuple[int, str]) -> float:
    """MIREX key score: 1 exact, 0.5 fifth, 0.3 relative, 0.2 parallel, else 0."""
    (et, em), (rt, rm) = est, ref
    if (et, em) == (rt, rm):
        return 1.0
    if em == rm and (et - rt) % 12 in (5, 7):
        return 0.5
    if em != rm and (
        (rm == "major" and et == (rt + 9) % 12) or (rm == "minor" and et == (rt + 3) % 12)
    ):
        return 0.3
    if et == rt and em != rm:
        return 0.2
    return 0.0


def melody_note_accuracy(
    est: list[tuple[float, float, float]],
    ref: list[tuple[float, float, int, str]],
    tol_semitones: float = 0.5,
) -> float:
    """Fraction of reference notes whose middle has an estimated note within tolerance
    (octave errors count as misses)."""
    hits = 0
    for a, b, m, _ in ref:
        mid = (a + b) / 2
        e = next((em for ea, eb, em in est if ea <= mid < eb), None)
        if e is not None and abs(e - m) <= tol_semitones:
            hits += 1
    return hits / max(len(ref), 1)
