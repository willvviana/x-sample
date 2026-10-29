"""Key (global and local), chords with bass/inversions, harmonic rhythm and melody (DSP)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from xsample.analysis.dsp import ANALYSIS_RATE, PITCH_HOP, Features

F64 = npt.NDArray[np.float64]

PITCH_CLASSES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# Temperley-style major/minor key profiles (more robust than Krumhansl on pop/R&B harmony).
MAJOR_PROFILE = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
# The minor profile weights the flat and raised 7th alike: pop/R&B minor is mostly natural
# (Aeolian/Dorian), and Temperley's harmonic-minor bias pushes it to the relative major.
MINOR_PROFILE = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 3.5, 3.5])

# Chord vocabulary: triads plus the seventh chords common in piano/R&B harmony.
# Seventh chords must earn their extra note over the plain triad.
SEVENTH_PENALTY = 0.01
QUALITIES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
}
NO_CHORD = "N"
# Chord HMM. A change costs log(STAY / ((1 - STAY) / 60)) = 5.8 nats; with the emission
# scale that is a cosine margin of ~0.19 summed over the beats of the new chord. At 12 it
# was ~0.49, while the typical per-beat margin on real voicings is ~0.04, so the decoder held
# one chord through whole jazz/funk progressions. 30 was chosen on the synthetic benchmark
# and one GuitarSet player (results are flat from 20 to 45); the other player is held out.
CHORD_EMISSION_SCALE = 30.0
CHORD_STAY = 0.85


@dataclass
class KeyCandidate:
    tonic: int
    mode: str
    score: float  # normalised 0..1 (softmax over 24 keys)

    @property
    def label(self) -> str:
        return f"{PITCH_CLASSES[self.tonic]} {self.mode}"


@dataclass
class ChordSegment:
    start: float
    end: float
    root: int | None
    quality: str | None
    bass: int | None
    confidence: float

    @property
    def label(self) -> str:
        if self.root is None or self.quality is None:
            return NO_CHORD
        name = PITCH_CLASSES[self.root] + {"maj": "", "min": "m"}.get(self.quality, self.quality)
        if self.bass is not None and self.bass != self.root:
            name += "/" + PITCH_CLASSES[self.bass]
        return name


@dataclass
class KeySegment:
    start: float
    end: float
    key: KeyCandidate


@dataclass
class MelodyNote:
    start: float
    end: float
    midi: float
    articulation: str  # "attack" | "legato" | "glide"
    vibrato_hz: float | None
    vibrato_cents: float | None


@dataclass
class HarmonyResult:
    key_candidates: list[KeyCandidate]
    key_confidence: float
    local_keys: list[KeySegment]
    chords: list[ChordSegment]
    harmonic_rhythm_beats: float | None  # median chord length in beats
    changes_per_bar: float | None
    bass_notes: list[tuple[float, float, int]]  # (start, end, pitch class)
    melody_notes: list[MelodyNote]
    melody_contour: list[tuple[float, float | None]]  # (time s, midi) downsampled
    voiced_ratio: float
    notes: list[str] = field(default_factory=list)


def _key_profiles() -> tuple[list[tuple[int, str]], F64]:
    labels, rows = [], []
    for mode, prof in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        for tonic in range(12):
            labels.append((tonic, mode))
            p = np.roll(prof, tonic)
            rows.append((p - p.mean()) / p.std())
    return labels, np.array(rows)


KEY_LABELS, KEY_MATRIX = _key_profiles()


def _templates() -> tuple[list[tuple[int, str]], F64]:
    labels, rows = [], []
    for quality, intervals in QUALITIES.items():
        for root in range(12):
            v = np.zeros(12)
            for k, iv in enumerate(intervals):
                v[(root + iv) % 12] = 1.0 if k < 3 else 0.7
            labels.append((root, quality))
            rows.append(v / np.linalg.norm(v))
    return labels, np.array(rows)


CHORD_LABELS, CHORD_MATRIX = _templates()


def key_scores(chroma_profile: F64, bass_profile: F64, chord_roots: F64 | None) -> F64:
    """Correlation with key profiles plus tonic evidence from bass and chord roots, which
    resolves the relative major/minor confusion of profile matching alone."""
    p = chroma_profile - chroma_profile.mean()
    p = p / max(float(np.linalg.norm(p)), 1e-9)
    corr = KEY_MATRIX @ p / np.sqrt(12)
    tonic_bonus = np.zeros(24)
    for i, (tonic, mode) in enumerate(KEY_LABELS):
        b = bass_profile[tonic] / max(float(bass_profile.sum()), 1e-9)
        r = 0.0
        if chord_roots is not None:
            third = 3 if mode == "minor" else 4
            # chord_roots[root, 0]=major-ish, [root, 1]=minor-ish duration share
            r = chord_roots[tonic, 1 if third == 3 else 0]
        tonic_bonus[i] = 0.6 * b + 0.8 * r
    out: F64 = corr + tonic_bonus
    return out


def _softmax(x: F64, temp: float) -> F64:
    z = (x - x.max()) / temp
    e = np.exp(z)
    out: F64 = e / e.sum()
    return out


def recognise_chords(features: Features, beats: F64) -> list[ChordSegment]:
    """Beat-synchronous template matching (treble chroma + bass) with Viterbi smoothing."""
    c, b = features.chroma.astype(np.float64), features.bass_chroma.astype(np.float64)
    if len(c) == 0:
        return []
    fps = features.chroma_fps
    duration = len(c) / fps
    if len(beats) >= 4:
        bounds = np.concatenate([[0.0], beats[beats > 0], [duration]])
    else:
        bounds = np.arange(0.0, duration + 0.5, 0.5)
    bounds = np.unique(np.clip(bounds, 0.0, duration))
    frames = np.clip(np.round(bounds * fps).astype(int), 0, len(c))
    segs = [(a, max(a + 1, z)) for a, z in itertools.pairwise(frames) if a < len(c)]
    treble = np.array([c[a:z].mean(axis=0) for a, z in segs])
    bass = np.array([b[a:z].mean(axis=0) for a, z in segs])
    energy = np.linalg.norm(treble, axis=1)
    # Every pitch class carries a floor (harmonics, leakage, noise); against it a seventh
    # template always beats the plain triad. Only energy above the segment's median counts.
    t_clean = np.maximum(treble - np.median(treble, axis=1, keepdims=True), 0.0)
    b_clean = np.maximum(bass - np.median(bass, axis=1, keepdims=True), 0.0)
    # Combine: chord tones from treble and bass (root often in the bass).
    combined = t_clean + 0.5 * b_clean
    combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-9)
    sim = combined @ CHORD_MATRIX.T  # (n, K)
    penalty = np.array(
        [SEVENTH_PENALTY if q not in ("maj", "min") else 0.0 for _, q in CHORD_LABELS]
    )
    sim = sim - penalty
    n, k = sim.shape
    no_chord = np.where(energy < 0.05, 1.0, 0.0)  # silent segments
    obs = np.concatenate([sim, (0.55 + 0.5 * no_chord)[:, None]], axis=1)
    logp = CHORD_EMISSION_SCALE * obs
    states = k + 1
    trans = np.full((states, states), np.log((1.0 - CHORD_STAY) / (states - 1)))
    np.fill_diagonal(trans, np.log(CHORD_STAY))
    score = logp[0].copy()
    back = np.zeros((n, states), dtype=np.int64)
    for i in range(1, n):
        m = score[:, None] + trans
        back[i] = np.argmax(m, axis=0)
        score = m[back[i], np.arange(states)] + logp[i]
    path = [int(np.argmax(score))]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    path.reverse()
    probs = np.exp(logp - logp.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    out: list[ChordSegment] = []
    for i, state in enumerate(path):
        start, end = bounds[i], bounds[i + 1] if i + 1 < len(bounds) else duration
        conf = float(probs[i, state])
        if state == k:
            root = quality = bass_pc = None
        else:
            root, quality = CHORD_LABELS[state]
            bp = int(np.argmax(bass[i]))
            tones = {(root + iv) % 12 for iv in QUALITIES[quality]}
            bass_pc = bp if bass[i, bp] > 0.4 and bp in tones else root
        if out and out[-1].root == root and out[-1].quality == quality and out[-1].bass == bass_pc:
            prev = out[-1]
            span_prev = prev.end - prev.start
            span = end - start
            prev.confidence = (prev.confidence * span_prev + conf * span) / (span_prev + span)
            prev.end = end
        else:
            out.append(ChordSegment(float(start), float(end), root, quality, bass_pc, conf))
    return out


def _chord_root_shares(chords: list[ChordSegment]) -> F64:
    shares = np.zeros((12, 2))
    total = 0.0
    for ch in chords:
        if ch.root is None or ch.quality is None:
            continue
        span = ch.end - ch.start
        minorish = ch.quality in ("min", "min7")
        shares[ch.root, 1 if minorish else 0] += span
        total += span
    out: F64 = shares / max(total, 1e-9)
    return out


def phrase_start_bass(features: Features, downbeats: F64, bars_per_phrase: int = 4) -> F64:
    """Bass pitch-class histogram at phrase starts (every 4 bars from the first downbeat)."""
    b, fps = features.bass_chroma.astype(np.float64), features.chroma_fps
    out = np.zeros(12)
    if len(b) == 0 or len(downbeats) == 0:
        return out
    bar = float(np.median(np.diff(downbeats))) if len(downbeats) > 1 else 2.0
    for t in downbeats[::bars_per_phrase]:
        a, z = int(t * fps), int((t + bar / 2) * fps)
        if a < len(b):
            seg = b[a : max(a + 1, z)].sum(axis=0)
            if seg.max() > 0:
                out[int(np.argmax(seg))] += 1.0
    return out


def _relative_tiebreak(scores: F64, phrase_bass: F64, margin: float = 0.5) -> F64:
    """A key and its relative major/minor share every pitch class, so loops such as Am-F-C-G
    score close for both. Phrases usually start on the tonic, so the bass note at phrase starts
    decides between the two. It only reorders a relative pair, never introduces a third key."""
    if phrase_bass.sum() < 1:
        return scores
    best = int(np.argmax(scores))
    tonic, mode = KEY_LABELS[best]
    rel = KEY_LABELS.index(
        ((tonic + 9) % 12, "minor") if mode == "major" else ((tonic + 3) % 12, "major")
    )
    rel_tonic = KEY_LABELS[rel][0]
    share = phrase_bass / phrase_bass.sum()
    if (
        scores[best] - scores[rel] > margin
        or share[rel_tonic] < 0.5
        or share[rel_tonic] <= share[tonic]
    ):
        return scores
    out: F64 = scores.copy()
    out[rel], out[best] = scores[best], scores[rel]
    return out


def estimate_keys(
    features: Features,
    chords: list[ChordSegment],
    downbeats: F64 | None = None,
    window_s: float = 16.0,
    hop_s: float = 4.0,
) -> tuple[list[KeyCandidate], float, list[KeySegment]]:
    c, b = features.chroma.astype(np.float64), features.bass_chroma.astype(np.float64)
    if len(c) == 0:
        return [], 0.0, []
    weights = np.linalg.norm(c, axis=1)
    prof = (c * weights[:, None]).sum(axis=0)
    bprof = (b * weights[:, None]).sum(axis=0)
    phrase_bass = phrase_start_bass(features, downbeats if downbeats is not None else np.zeros(0))
    scores = _relative_tiebreak(key_scores(prof, bprof, _chord_root_shares(chords)), phrase_bass)
    probs = _softmax(scores, 0.05)
    order = np.argsort(-probs)
    cands = [
        KeyCandidate(KEY_LABELS[i][0], KEY_LABELS[i][1], round(float(probs[i]), 3))
        for i in order[:3]
    ]
    confidence = float(probs[order[0]])
    # Local keys: windowed scores smoothed by an HMM that strongly prefers staying in key.
    fps = features.chroma_fps
    win, hop = int(window_s * fps), max(1, int(hop_s * fps))
    duration = len(c) / fps
    if len(c) <= win:
        return cands, confidence, [KeySegment(0.0, duration, cands[0])]
    starts = list(range(0, len(c) - win + 1, hop))
    obs = []
    for s in starts:
        seg_c, seg_b, w = c[s : s + win], b[s : s + win], weights[s : s + win]
        t0, t1 = s / fps, (s + win) / fps
        seg_chords = [ch for ch in chords if ch.end > t0 and ch.start < t1]
        sc = key_scores(
            (seg_c * w[:, None]).sum(axis=0),
            (seg_b * w[:, None]).sum(axis=0),
            _chord_root_shares(seg_chords),
        )
        obs.append(np.log(_softmax(sc, 0.05) + 1e-12))
    obs_arr = np.array(obs)
    trans = np.full((24, 24), np.log(0.02 / 23))
    np.fill_diagonal(trans, np.log(0.98))
    score = obs_arr[0] + np.log(probs + 1e-12)
    back = np.zeros(obs_arr.shape, dtype=np.int64)
    for i in range(1, len(obs_arr)):
        m = score[:, None] + trans
        back[i] = np.argmax(m, axis=0)
        score = m[back[i], np.arange(24)] + obs_arr[i]
    path = [int(np.argmax(score))]
    for i in range(len(obs_arr) - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    path.reverse()
    segments: list[KeySegment] = []
    for i, state in enumerate(path):
        t_start = 0.0 if i == 0 else (starts[i] + win / 2) / fps - hop / fps / 2
        t_end = duration if i == len(path) - 1 else (starts[i] + win / 2) / fps + hop / fps / 2
        tonic, mode = KEY_LABELS[state]
        p = float(np.exp(obs_arr[i, state]))
        if segments and segments[-1].key.tonic == tonic and segments[-1].key.mode == mode:
            segments[-1].end = t_end
        else:
            segments.append(KeySegment(t_start, t_end, KeyCandidate(tonic, mode, round(p, 3))))
    return cands, confidence, segments


def bass_line(features: Features, beats: F64) -> list[tuple[float, float, int]]:
    b = features.bass_chroma
    if len(b) == 0 or len(beats) < 2:
        return []
    fps = features.chroma_fps
    out: list[tuple[float, float, int]] = []
    for a, z in itertools.pairwise(beats):
        fa, fz = int(a * fps), max(int(a * fps) + 1, int(z * fps))
        seg = b[fa:fz]
        if len(seg) == 0 or float(seg.max()) < 0.3:
            continue
        pc = int(np.argmax(seg.mean(axis=0)))
        if out and out[-1][2] == pc and abs(out[-1][1] - a) < 1e-6:
            out[-1] = (out[-1][0], float(z), pc)
        else:
            out.append((float(a), float(z), pc))
    return out


def melody(
    features: Features, onset_times: F64
) -> tuple[list[MelodyNote], list[tuple[float, float | None]], float]:
    """Predominant-pitch notes with articulation (attack/legato/glide) and vibrato estimates.

    This is the most uncertain analysis: in dense mixes the salient pitch may jump between
    instruments. Contour and notes are reported with that caveat.
    """
    pitch = features.pitch_midi.astype(np.float64)
    fps = ANALYSIS_RATE / PITCH_HOP
    n = len(pitch)
    voiced = ~np.isnan(pitch)
    voiced_ratio = float(voiced.mean()) if n else 0.0
    contour: list[tuple[float, float | None]] = []
    step = max(1, int(fps / 20))  # ~20 points per second
    for i in range(0, n, step):
        seg = pitch[i : i + step]
        seg = seg[~np.isnan(seg)]
        contour.append((i / fps, round(float(np.median(seg)), 2) if len(seg) else None))
    notes: list[MelodyNote] = []
    onset_frames = set(np.round(onset_times * fps).astype(int).tolist())
    i = 0
    min_len = int(0.08 * fps)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        ref = pitch[i]
        # Extend while the pitch stays within ±0.7 semitone of the running median.
        while j + 1 < n and voiced[j + 1] and abs(pitch[j + 1] - ref) < 0.7:
            j += 1
            ref = float(np.median(pitch[i : j + 1]))
        if j - i + 1 >= min_len:
            seg = pitch[i : j + 1]
            centre = float(np.median(seg))
            near_onset = any(k in onset_frames for k in range(i - 3, i + 4))
            prev = notes[-1] if notes else None
            if prev is not None and abs(i / fps - prev.end) < 0.03 and not near_onset:
                # Continuous pitch movement between notes = glide; else slurred (legato).
                gap = pitch[max(0, i - 4) : i + 1]
                articulation = "glide" if np.nanmax(gap) - np.nanmin(gap) > 0.5 else "legato"
            else:
                articulation = "attack"
            vib_hz = vib_cents = None
            if len(seg) >= int(0.3 * fps):
                dev = seg - np.convolve(seg, np.ones(15) / 15, mode="same")
                core = dev[7:-7]
                if len(core) > 10:
                    crossings = np.sum(np.diff(np.sign(core)) != 0)
                    rate = crossings / 2 / (len(core) / fps)
                    extent = float(np.std(core) * 100 * np.sqrt(2))
                    if 4.0 <= rate <= 8.5 and extent >= 15:
                        vib_hz, vib_cents = round(float(rate), 2), round(extent, 1)
            notes.append(
                MelodyNote(
                    i / fps, (j + 1) / fps, round(centre, 2), articulation, vib_hz, vib_cents
                )
            )
        i = j + 1
    return notes, contour, voiced_ratio


def analyze_harmony(
    features: Features, beats: list[float], downbeats: list[float], onsets: list[float]
) -> HarmonyResult:
    beats_arr = np.asarray(beats, dtype=np.float64)
    chords = recognise_chords(features, beats_arr)
    cands, conf, local = estimate_keys(features, chords, np.asarray(downbeats, dtype=np.float64))
    notes: list[str] = []
    harmonic_rhythm = changes_per_bar = None
    real = [ch for ch in chords if ch.root is not None]
    if len(beats_arr) >= 4 and real:
        period = float(np.median(np.diff(beats_arr)))
        harmonic_rhythm = float(np.median([(ch.end - ch.start) / period for ch in real]))
        if len(downbeats) >= 2:
            bars = len(downbeats)
            changes_per_bar = max(0, len(real) - 1) / max(1, bars)
    if len(local) > 1:
        notes.append(
            f"{len(local)} key regions detected; a single global key may not describe this audio."
        )
    if conf < 0.5:
        notes.append("Key is uncertain; consider the alternatives.")
    mel_notes, contour, voiced = melody(features, np.asarray(onsets, dtype=np.float64))
    return HarmonyResult(
        key_candidates=cands,
        key_confidence=conf,
        local_keys=local,
        chords=chords,
        harmonic_rhythm_beats=harmonic_rhythm,
        changes_per_bar=changes_per_bar,
        bass_notes=bass_line(features, beats_arr),
        melody_notes=mel_notes,
        melody_contour=contour,
        voiced_ratio=voiced,
        notes=notes,
    )
