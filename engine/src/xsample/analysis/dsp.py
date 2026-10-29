"""Streaming DSP feature extraction (numpy only).

Audio arrives in bounded chunks (mono, ANALYSIS_RATE); each feature stream keeps only its
window overlap, so memory for the audio itself is independent of duration. The per-frame
features kept for the analyzers are small (tens of floats per frame, ~10-20 MB per 10 min).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]

ANALYSIS_RATE = 22050
RHYTHM_FFT, RHYTHM_HOP = 2048, 256  # onset/tempo frames: 11.6 ms
PITCH_FFT, PITCH_HOP = 4096, 256  # melody salience: 5.4 Hz bins, same frame grid as rhythm
CHROMA_FFT, CHROMA_HOP = 8192, 2048  # harmony: 2.7 Hz bins so bass notes resolve
N_MELS = 64
EDGE_SAMPLES = ANALYSIS_RATE // 100  # 10 ms at each end of the audio
# Lead-melody range C3..~F#6: below C3 the bass line would dominate the salience.
MELODY_MIN_HZ, MELODY_MAX_HZ = 130.8, 1500.0
MELODY_BINS_PER_SEMITONE = 5  # 20-cent resolution
MELODY_CANDIDATES = 3
REF_MIDI = 69.0


def hz_to_midi(hz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out: npt.NDArray[np.float64] = REF_MIDI + 12.0 * np.log2(np.maximum(hz, 1e-6) / 440.0)
    return out


def mel_filterbank(n_fft: int, n_mels: int = N_MELS, fmin: float = 30.0) -> F32:
    def hz2mel(f: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out: npt.NDArray[np.float64] = 2595.0 * np.log10(1.0 + f / 700.0)
        return out

    def mel2hz(m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out: npt.NDArray[np.float64] = 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        return out

    freqs = np.fft.rfftfreq(n_fft, 1.0 / ANALYSIS_RATE)
    edges = mel2hz(
        np.linspace(hz2mel(np.array(fmin)), hz2mel(np.array(ANALYSIS_RATE / 2)), n_mels + 2)
    )
    fb = np.zeros((n_mels, len(freqs)), np.float32)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        rising = (freqs - lo) / max(mid - lo, 1e-9)
        falling = (hi - freqs) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


class Framer:
    """Centred, hop-spaced frames over a stream (frame k is centred on sample k*hop)."""

    def __init__(self, n_fft: int, hop: int) -> None:
        self.n_fft, self.hop = n_fft, hop
        self._buf = np.zeros(n_fft // 2, np.float32)  # centre padding
        self.window = np.hanning(n_fft).astype(np.float32)

    def push(self, samples: F32) -> F32:
        self._buf = np.concatenate([self._buf, samples.astype(np.float32, copy=False)])
        count = 0 if len(self._buf) < self.n_fft else 1 + (len(self._buf) - self.n_fft) // self.hop
        if count == 0:
            return np.zeros((0, self.n_fft), np.float32)
        idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(count)[:, None]
        frames: F32 = self._buf[idx]
        self._buf = self._buf[count * self.hop :]
        return frames

    def finish(self, total_samples: int, emitted: int) -> F32:
        """Flush with end padding so that ceil(total/hop) frames exist in total."""
        wanted = -(-total_samples // self.hop)
        tail = self.push(np.zeros(self.n_fft // 2 + self.hop, np.float32))
        return tail[: max(0, wanted - emitted)]


@dataclass
class Features:
    """Per-frame features consumed by the analyzers."""

    duration_samples: int  # at ANALYSIS_RATE
    onset: F32  # (n,) broadband spectral flux, rhythm frame rate
    onset_low: F32  # (n,) flux below ~200 Hz (kick/bass)
    mel: F32  # (n, N_MELS) log-mel energy, rhythm frame rate
    rms: F32  # (n,) frame RMS
    chroma: F32  # (m,) x 12 treble chroma, chroma frame rate
    bass_chroma: F32  # (m, 12)
    pitch_midi: F32  # (n,) predominant pitch (MIDI, fractional); NaN when unvoiced
    pitch_salience: F32  # (n,) salience of that pitch relative to frame energy (0..1)
    # (|boundary sample|, RMS of the boundary 10 ms) for "start" and "end"
    edges: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def rhythm_fps(self) -> float:
        return ANALYSIS_RATE / RHYTHM_HOP

    @property
    def chroma_fps(self) -> float:
        return ANALYSIS_RATE / CHROMA_HOP


@dataclass
class _Lists:
    onset_mel: list[F32] = field(default_factory=list)
    rms: list[F32] = field(default_factory=list)
    chroma: list[F32] = field(default_factory=list)
    bass: list[F32] = field(default_factory=list)
    pitch: list[F32] = field(default_factory=list)
    sal: list[F32] = field(default_factory=list)


class FeatureExtractor:
    def __init__(self) -> None:
        self._rhythm = Framer(RHYTHM_FFT, RHYTHM_HOP)
        self._pitch = Framer(PITCH_FFT, PITCH_HOP)
        self._chroma = Framer(CHROMA_FFT, CHROMA_HOP)
        self._mel_fb = mel_filterbank(RHYTHM_FFT)
        self._lists = _Lists()
        self._total = 0
        self._emitted = [0, 0, 0]
        self._head = np.zeros(0, np.float32)
        self._tail = np.zeros(0, np.float32)
        self._setup_chroma()
        self._setup_pitch()

    # --- configuration ---------------------------------------------------------------

    def _setup_chroma(self) -> None:
        freqs = np.fft.rfftfreq(CHROMA_FFT, 1.0 / ANALYSIS_RATE)
        midi = hz_to_midi(np.maximum(freqs, 1e-6))
        near = np.abs(midi - np.round(midi)) < 0.35
        pcs = np.round(midi).astype(int) % 12
        self._treble = [
            np.where((freqs >= 130) & (freqs <= 2500) & near & (pcs == k))[0] for k in range(12)
        ]
        self._bassb = [
            np.where((freqs >= 38) & (freqs < 200) & near & (pcs == k))[0] for k in range(12)
        ]

    def _setup_pitch(self) -> None:
        freqs = np.fft.rfftfreq(PITCH_FFT, 1.0 / ANALYSIS_RATE)
        lo = hz_to_midi(np.array(MELODY_MIN_HZ))
        hi = hz_to_midi(np.array(MELODY_MAX_HZ))
        self._cand_midi = np.arange(float(lo), float(hi), 1.0 / MELODY_BINS_PER_SEMITONE)
        cand_hz = 440.0 * 2 ** ((self._cand_midi - REF_MIDI) / 12)
        # Harmonic summation: salience(f0) = sum_h w^h * |X(h*f0)| (nearest bin, linear interp).
        harmonics = 6
        rows, cols, vals = [], [], []
        bin_hz = ANALYSIS_RATE / PITCH_FFT
        for h in range(1, harmonics + 1):
            pos = cand_hz * h / bin_hz
            ok = pos < len(freqs) - 1
            lo_bin = np.floor(pos[ok]).astype(int)
            frac = pos[ok] - lo_bin
            idx = np.where(ok)[0]
            weight = 0.8 ** (h - 1)
            rows += [idx, idx]
            cols += [lo_bin, lo_bin + 1]
            vals += [weight * (1 - frac), weight * frac]
        mat = np.zeros((len(self._cand_midi), len(freqs)), np.float32)
        np.add.at(mat, (np.concatenate(rows), np.concatenate(cols)), np.concatenate(vals))
        # Fundamental-only matrix (first two row/col/val blocks are h=1).
        fund = np.zeros_like(mat)
        np.add.at(
            fund, (np.concatenate(rows[:2]), np.concatenate(cols[:2])), np.concatenate(vals[:2])
        )
        self._fundamental_mat = fund.T
        self._salience_mat = mat.T  # (bins, candidates)
        self._height_weight = (1.0 + 0.012 * (self._cand_midi - self._cand_midi[0])).astype(
            np.float32
        )

    # --- streaming -------------------------------------------------------------------

    def push(self, mono: F32) -> None:
        self._total += len(mono)
        if len(self._head) < EDGE_SAMPLES:
            self._head = np.concatenate([self._head, mono[: EDGE_SAMPLES - len(self._head)]])
        self._tail = np.concatenate([self._tail, mono])[-EDGE_SAMPLES:]
        self._consume(0, self._rhythm.push(mono))
        self._consume(1, self._pitch.push(mono))
        self._consume(2, self._chroma.push(mono))

    def _consume(self, which: int, frames: F32) -> None:
        if len(frames) == 0:
            return
        self._emitted[which] += len(frames)
        if which == 0:
            spec = np.abs(np.fft.rfft(frames * self._rhythm.window, axis=1)).astype(np.float32)
            self._lists.onset_mel.append(np.log1p(100.0 * spec @ self._mel_fb.T).astype(np.float32))
            self._lists.rms.append(
                np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)).astype(np.float32)
            )
        elif which == 1:
            spec = np.abs(np.fft.rfft(frames * self._pitch.window, axis=1)).astype(np.float32)
            # Whitening (divide by the smoothed spectral envelope) evens out loud low notes vs.
            # the lead; compressed magnitude is fairer to quiet partials.
            mag = np.sqrt(spec)
            envelope = _smooth_rows(mag, 31) + 1e-6
            white = mag / envelope
            sal = white @ self._salience_mat
            # A real note has energy at its fundamental; this suppresses sub-harmonic
            # "virtual pitches" implied by chords.
            fundamental = white @ self._fundamental_mat
            sal = sal * np.sqrt(
                np.maximum(fundamental, 0.0) / (fundamental.mean(axis=1, keepdims=True) + 1e-9)
            )
            # The melody is usually the highest prominent voice: mild upward weighting.
            sal = sal * self._height_weight
            top = np.argsort(-sal, axis=1)[:, :MELODY_CANDIDATES]
            rows = np.arange(len(top))[:, None]
            peak = sal[rows, top]
            mean = sal.mean(axis=1, keepdims=True) + 1e-9
            self._lists.pitch.append(self._cand_midi[top].astype(np.float32))
            self._lists.sal.append((peak / mean).astype(np.float32))
        else:
            spec = np.abs(np.fft.rfft(frames * self._chroma.window, axis=1)) ** 2
            treble = np.stack([spec[:, b].sum(axis=1) for b in self._treble], axis=1)
            bass = np.stack([spec[:, b].sum(axis=1) for b in self._bassb], axis=1)
            self._lists.chroma.append(treble.astype(np.float32))
            self._lists.bass.append(bass.astype(np.float32))

    def finish(self) -> Features:
        self._consume(0, self._rhythm.finish(self._total, self._emitted[0]))
        self._consume(1, self._pitch.finish(self._total, self._emitted[1]))
        self._consume(2, self._chroma.finish(self._total, self._emitted[2]))
        lists = self._lists
        mel = (
            np.concatenate(lists.onset_mel)
            if lists.onset_mel
            else np.zeros((0, N_MELS), np.float32)
        )
        n = mel.shape[0]
        flux = np.maximum(0.0, np.diff(mel, axis=0, prepend=mel[:1] if n else mel))
        low_bands = mel_band_limit(200.0)
        onset = _normalise_odf(flux.sum(axis=1))
        onset_low = _normalise_odf(flux[:, :low_bands].sum(axis=1))
        chroma = _norm_chroma(
            np.concatenate(lists.chroma) if lists.chroma else np.zeros((0, 12), np.float32)
        )
        bass = _norm_chroma(
            np.concatenate(lists.bass) if lists.bass else np.zeros((0, 12), np.float32)
        )
        cands = (
            np.concatenate(lists.pitch)[:n]
            if lists.pitch
            else np.zeros((0, MELODY_CANDIDATES), np.float32)
        )
        cand_sal = (
            np.concatenate(lists.sal)[:n]
            if lists.sal
            else np.zeros((0, MELODY_CANDIDATES), np.float32)
        )
        rms = np.concatenate(lists.rms)[:n] if lists.rms else np.zeros(0, np.float32)
        pitch, sal = track_melody(cands, cand_sal)
        voiced = voicing_mask(sal, rms)
        pitch = np.where(voiced, pitch, np.nan).astype(np.float32)
        return Features(
            duration_samples=self._total,
            onset=onset,
            onset_low=onset_low,
            mel=mel,
            rms=rms,
            chroma=chroma,
            bass_chroma=bass,
            pitch_midi=pitch,
            pitch_salience=sal,
            edges={"start": _edge(self._head, 0), "end": _edge(self._tail, -1)},
        )


def _edge(x: F32, index: int) -> tuple[float, float]:
    if len(x) == 0:
        return 0.0, 0.0
    return float(abs(x[index])), float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _smooth_rows(x: F32, width: int) -> F32:
    kernel = np.ones(width, np.float32) / width
    pad = width // 2
    padded = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    c = np.cumsum(padded, axis=1, dtype=np.float64)
    c = np.concatenate([np.zeros((x.shape[0], 1)), c], axis=1)
    out = (c[:, width:] - c[:, :-width]) / width
    del kernel
    return out[:, : x.shape[1]].astype(np.float32)


def mel_band_limit(hz: float) -> int:
    """Number of mel bands whose centre lies below `hz`."""
    fb = mel_filterbank(RHYTHM_FFT)
    freqs = np.fft.rfftfreq(RHYTHM_FFT, 1.0 / ANALYSIS_RATE)
    centres = freqs[np.argmax(fb, axis=1)]
    return max(1, int(np.sum(centres < hz)))


def _normalise_odf(x: npt.NDArray[np.floating]) -> F32:
    if len(x) == 0:
        return np.zeros(0, np.float32)
    # Remove the slowly varying part so that loud passages do not dominate.
    width = 32
    kernel = np.ones(width) / width
    local = np.convolve(x, kernel, mode="same")
    out = np.maximum(x - local, 0.0)
    scale = np.percentile(out, 99) if np.any(out > 0) else 1.0
    result: F32 = (out / max(float(scale), 1e-9)).astype(np.float32)
    return result


def _norm_chroma(c: F32) -> F32:
    if len(c) == 0:
        return c
    peak = c.max(axis=1, keepdims=True)
    rel = c / np.maximum(peak, 1e-12)
    out = np.log1p(20.0 * rel)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    silent = peak[:, 0] < 1e-8
    out = out / np.maximum(norm, 1e-9)
    out[silent] = 0.0
    result: F32 = out.astype(np.float32)
    return result


VOICING_PEAKINESS = 2.5


def track_melody(cands: F32, sal: F32, jump_cost: float = 0.35) -> tuple[F32, F32]:
    """Viterbi over the top pitch candidates per frame: salience vs. pitch-jump cost, so the
    contour follows one voice instead of hopping between instruments."""
    n = len(cands)
    if n == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    k = cands.shape[1]
    logs = np.log(np.maximum(sal, 1e-6)).astype(np.float64)
    score = logs[0].copy()
    back = np.zeros((n, k), dtype=np.int64)
    for i in range(1, n):
        jump = np.abs(cands[i][None, :] - cands[i - 1][:, None])  # (prev, cur) semitones
        m = score[:, None] - jump_cost * np.minimum(jump, 12.0)
        back[i] = np.argmax(m, axis=0)
        score = m[back[i], np.arange(k)] + logs[i]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    rows = np.arange(n)
    return cands[rows, path].astype(np.float32), sal[rows, path].astype(np.float32)


def voicing_mask(salience: F32, rms: F32) -> npt.NDArray[np.bool_]:
    """Voiced frames: a pitch candidate well above the average candidate, with audible energy."""
    if len(salience) == 0:
        return np.zeros(0, bool)
    loud = rms > max(1e-4, 0.05 * float(np.percentile(rms, 95)))
    out: npt.NDArray[np.bool_] = (salience >= VOICING_PEAKINESS) & loud
    return out


def frame_to_sample(frames: npt.ArrayLike, hop: int, target_rate: int) -> npt.NDArray[np.int64]:
    """Frame index (centred at frame*hop at ANALYSIS_RATE) to an integer sample at target_rate."""
    t = np.asarray(frames, dtype=np.float64) * hop / ANALYSIS_RATE
    out: npt.NDArray[np.int64] = np.round(t * target_rate).astype(np.int64)
    return out
