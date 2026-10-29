"""Instrument, vocal and genre tagging with PANNs MobileNetV1 (AudioSet, 527 classes).

The log-mel front end is a NumPy re-implementation of the model's torchlibrosa layers
(audioset_tagging_cnn commit d2f4b8c: 32 kHz, Hann 1024, hop 320, centre/reflect padding,
power spectrum, librosa Slaney mel 64 bands 50-14000 Hz, 10*log10 with amin 1e-10). The ONNX
export (tools/export_panns.py) starts after that front end and ends with the sigmoid.
This module never imports onnxruntime, so importing it is safe in the API process.

The model was trained on 10 s AudioSet clips, so a piece is tagged in 10 s windows with a
5 s hop (the last window is aligned to the end; a piece shorter than 10 s is one window of
its own length). A label's piece score is its highest window score, and its activity is
the union of the windows at or above the uncertain threshold. Scores are AudioSet sigmoid
outputs, not calibrated probabilities. The thresholds were fixed before any evaluation.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

SAMPLE_RATE = 32000
N_FFT, HOP = 1024, 320  # 100 frames per second
FPS = SAMPLE_RATE / HOP
N_MELS = 64
F_MIN, F_MAX = 50.0, 14000.0
AMIN = 1e-10
N_CLASSES = 527
WINDOW, WINDOW_HOP = 1001, 500  # 10 s (an AudioSet clip at this hop), 5 s
MIN_FRAMES = 100  # five stride-2 pooling stages need >= 32 frames; below 1 s is not tagged
DETECT_THRESHOLD = 0.5
UNCERTAIN_THRESHOLD = 0.2
_BATCH_FRAMES = 2048

Kind = Literal["genre", "instrument", "vocal"]


@dataclass(frozen=True)
class Label:
    kind: Kind
    name: str
    classes: tuple[int, ...]  # AudioSet indices (class_labels_indices.csv)
    class_names: tuple[str, ...]


def _label(kind: Kind, name: str, *classes: tuple[int, str]) -> Label:
    return Label(kind, name, tuple(c[0] for c in classes), tuple(c[1] for c in classes))


# X-Sample labels and the AudioSet classes each aggregates (max over classes). Fixed before
# evaluation; kept close to the ontology so a label means what AudioSet annotators heard.
LABELS: tuple[Label, ...] = (
    _label("instrument", "Piano", (153, "Piano"), (154, "Electric piano")),
    _label(
        "instrument", "Organ", (155, "Organ"), (156, "Electronic organ"), (157, "Hammond organ")
    ),
    _label("instrument", "Synthesizer", (158, "Synthesizer")),
    _label(
        "instrument",
        "Guitar",
        (140, "Guitar"),
        (141, "Electric guitar"),
        (143, "Acoustic guitar"),
        (144, "Steel guitar, slide guitar"),
    ),
    _label("instrument", "Bass", (142, "Bass guitar"), (194, "Double bass")),
    _label(
        "instrument",
        "Drums",
        (162, "Drum kit"),
        (163, "Drum machine"),
        (164, "Drum"),
        (165, "Snare drum"),
        (168, "Bass drum"),
        (172, "Hi-hat"),
    ),
    _label("instrument", "Violin", (191, "Violin, fiddle")),
    _label("instrument", "Cello", (193, "Cello")),
    _label(
        "instrument",
        "Strings",
        (189, "Bowed string instrument"),
        (190, "String section"),
        (191, "Violin, fiddle"),
        (193, "Cello"),
    ),
    _label("instrument", "Saxophone", (197, "Saxophone")),
    _label(
        "instrument",
        "Brass",
        (185, "Brass instrument"),
        (186, "French horn"),
        (187, "Trumpet"),
        (188, "Trombone"),
    ),
    _label("instrument", "Flute", (196, "Flute")),
    _label("instrument", "Clarinet", (198, "Clarinet")),
    _label(
        "vocal",
        "Singing",
        (27, "Singing"),
        (28, "Choir"),
        (32, "Male singing"),
        (33, "Female singing"),
        (34, "Child singing"),
    ),
    _label("vocal", "Rapping", (36, "Rapping")),
    _label("vocal", "Speech", (0, "Speech")),
    _label("genre", "R&B", (226, "Rhythm and blues")),
    _label("genre", "Soul", (227, "Soul music")),
    _label("genre", "Funk", (232, "Funk")),
    _label("genre", "Jazz", (235, "Jazz")),
    _label("genre", "Blues", (251, "Blues")),
    _label("genre", "Gospel", (259, "Gospel music")),
    _label("genre", "Pop", (216, "Pop music")),
    _label("genre", "Hip hop", (217, "Hip hop music")),
    _label("genre", "Rock", (219, "Rock music")),
    _label("genre", "Electronic", (239, "Electronic music"), (245, "Electronic dance music")),
    _label("genre", "Classical", (237, "Classical music")),
    _label("genre", "Country", (229, "Country")),
    _label("genre", "Reggae", (228, "Reggae")),
    _label("genre", "Latin", (248, "Music of Latin America")),
)

# Genre labels that may be reported as "detected": only those meeting the evidence rule of
# tools/eval_tagger.py on labelled held-out audio (>= 10 pieces with and without the genre,
# precision >= 0.8 and recall >= 0.5 at DETECT_THRESHOLD; docs/EVALUATION.md). None does yet,
# so every genre is at most "uncertain". All labels report whether they were evaluated.
GENRE_EVIDENCE: frozenset[str] = frozenset()
EVALUATED: frozenset[str] = frozenset(
    # tools/eval_tagger.py rule: >= 5 labelled pieces with and >= 5 without the label.
    {"Bass", "Classical", "Drums", "Flute", "Funk", "Guitar", "Jazz", "Organ", "Piano"}
    | {"Strings", "Synthesizer"}
)


def _hz_to_mel(f: F64) -> F64:
    logstep = np.log(6.4) / 27.0
    out: F64 = np.where(
        f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / logstep, f / (200.0 / 3)
    )
    return out


def _mel_to_hz(m: F64) -> F64:
    logstep = np.log(6.4) / 27.0
    out: F64 = np.where(m >= 15.0, 1000.0 * np.exp(logstep * (m - 15.0)), m * 200.0 / 3)
    return out


def mel_filterbank() -> F64:
    """librosa.filters.mel(sr=32000, n_fft=1024, n_mels=64, fmin=50, fmax=14000).T (Slaney
    scale, Slaney area normalisation), as torchlibrosa's LogmelFilterBank."""
    fft_freqs = np.linspace(0.0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    mel_f = _mel_to_hz(
        np.linspace(_hz_to_mel(np.array(F_MIN)), _hz_to_mel(np.array(F_MAX)), N_MELS + 2)
    )
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : N_MELS + 2] - mel_f[:N_MELS]))[:, None]
    out: F64 = weights.T
    return out  # (513, 64)


class LogMelStream:
    """Streaming log-mel at 32 kHz (centre=True, reflect padding), bounded memory.

    Frame k is centred on sample k*HOP; a piece of N samples yields 1 + N // HOP frames.
    """

    def __init__(self) -> None:
        self._buf = np.zeros(0, np.float64)
        self._started = False
        self._total = 0
        self._frames = 0
        self._window = np.hanning(N_FFT + 1)[:-1]  # periodic Hann (librosa fftbins=True)
        self._fb = mel_filterbank()

    def _emit(self) -> F32:
        count = 0 if len(self._buf) < N_FFT else 1 + (len(self._buf) - N_FFT) // HOP
        out = [np.zeros((0, N_MELS), np.float32)]
        for first in range(0, count, _BATCH_FRAMES):
            n = min(_BATCH_FRAMES, count - first)
            idx = np.arange(N_FFT)[None, :] + HOP * (first + np.arange(n))[:, None]
            power = np.abs(np.fft.rfft(self._buf[idx] * self._window, axis=1)) ** 2
            out.append((10.0 * np.log10(np.maximum(power @ self._fb, AMIN))).astype(np.float32))
        self._buf = self._buf[count * HOP :]
        self._frames += count
        return np.concatenate(out)

    def push(self, samples: F32) -> F32:
        self._total += len(samples)
        self._buf = np.concatenate([self._buf, samples.astype(np.float64)])
        if not self._started:
            if len(self._buf) <= N_FFT // 2:
                return np.zeros((0, N_MELS), np.float32)
            self._buf = np.concatenate([self._buf[N_FFT // 2 : 0 : -1], self._buf])
            self._started = True
        return self._emit()

    def finish(self) -> F32:
        if not self._started:
            return np.zeros((0, N_MELS), np.float32)  # too short to reflect-pad
        tail = self._buf[-2 : -N_FFT // 2 - 2 : -1]
        self._buf = np.concatenate([self._buf, tail])
        out = self._emit()
        wanted = 1 + self._total // HOP
        return out[: max(0, wanted - (self._frames - len(out)))]


def window_starts(n_frames: int) -> list[int]:
    """Start frames of the 10 s windows; the last one ends at the end of the piece."""
    if n_frames <= WINDOW:
        return [0]
    starts = list(range(0, n_frames - WINDOW + 1, WINDOW_HOP))
    if starts[-1] + WINDOW < n_frames:
        starts.append(n_frames - WINDOW)
    return starts


def predict_windows(
    logmel: F32,
    session: Any,
    *,
    progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[list[int], F32]:
    """Per-window label scores (windows x len(LABELS)); only mapped classes are kept."""
    starts = window_starts(len(logmel))
    scores = np.zeros((len(starts), len(LABELS)), np.float32)
    for i, start in enumerate(starts):
        if cancelled is not None and cancelled():
            raise InterruptedError("cancelled")
        window = np.ascontiguousarray(logmel[start : start + WINDOW])[None]
        probs = np.asarray(session.run({"logmel": window})[0], np.float32)[0]
        scores[i] = [probs[list(label.classes)].max() for label in LABELS]
        if progress is not None:
            progress((i + 1) / len(starts))
    return starts, scores


@dataclass(frozen=True)
class TagSummary:
    label: Label
    score: float
    status: Literal["detected", "uncertain", "not_detected"]
    evaluated: bool
    activity: list[tuple[float, float, float]]  # (start s, end s, score)


def summarise(starts: list[int], scores: F32, n_frames: int) -> list[TagSummary]:
    out = []
    for j, label in enumerate(LABELS):
        col = scores[:, j]
        score = float(col.max()) if len(col) else 0.0
        status: Literal["detected", "uncertain", "not_detected"]
        if score >= DETECT_THRESHOLD:
            status = "detected"
        elif score >= UNCERTAIN_THRESHOLD:
            status = "uncertain"
        else:
            status = "not_detected"
        if status == "detected" and label.kind == "genre" and label.name not in GENRE_EVIDENCE:
            status = "uncertain"
        segments: list[tuple[float, float, float]] = []
        for start, s in zip(starts, col, strict=True):
            if s < UNCERTAIN_THRESHOLD:
                continue
            a, z = start / FPS, min(start + WINDOW, n_frames) / FPS
            if segments and a <= segments[-1][1]:
                prev = segments[-1]
                segments[-1] = (prev[0], max(prev[1], z), max(prev[2], float(s)))
            else:
                segments.append((a, z, float(s)))
        out.append(TagSummary(label, score, status, label.name in EVALUATED, segments))
    return out


# --- Installed model ------------------------------------------------------------------------

MODEL_ID = "panns-mobilenetv1"
REGISTRY_ID = "panns-mobilenetv1-audioset"
DISPLAY_NAME = "PANNs MobileNetV1 (AudioSet)"
FILE = "panns_mobilenetv1.onnx"  # dynamic time axis; runs on CPU and DirectML
LICENSE = (
    "CC BY 4.0 (weights, Zenodo 3987831); code MIT (qiuqiangkong/audioset_tagging_cnn). "
    "Trained on AudioSet (YouTube audio); AudioSet labels CC BY 4.0 (Google)."
)
ATTRIBUTION = (
    'PANNs: Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, M. D. Plumbley, "PANNs: Large-Scale '
    'Pretrained Audio Neural Networks for Audio Pattern Recognition", IEEE/ACM TASLP 28, '
    "2020. Weights: MobileNetV1_mAP=0.389.pth, https://zenodo.org/records/3987831 (CC BY "
    "4.0); converted to ONNX without changes to the weights."
)
SOURCE_URL = "https://zenodo.org/records/3987831/files/MobileNetV1_mAP=0.389.pth"
CHECKPOINT_SHA256 = "5f1d094310e439620b63d377e89e69bc663b782f1c242c0695ceb5db6db4271f"
CHECKPOINT_BYTES = 23639473
# Pin of the ONNX export (tools/export_panns.py); the trust root, not registry.json.
PINNED: tuple[str, int] = (
    "0dfb5d1b1a32f92c465f18b16acf43d827e2b2d3d44eff6e149d0f07f38b7544",
    19383221,
)
# Window scores from DirectML must match CPU within this (measured: ~3e-7).
PARITY_TOLERANCE = 1e-3


def identity() -> str:
    """Model identity for cache keys: the pinned export plus the label mapping and
    thresholds, so changing either invalidates cached tags."""
    mapping = repr(
        (
            [(lb.kind, lb.name, lb.classes) for lb in LABELS],
            DETECT_THRESHOLD,
            UNCERTAIN_THRESHOLD,
            WINDOW,
            WINDOW_HOP,
            sorted(GENRE_EVIDENCE),
            sorted(EVALUATED),
        )
    ).encode()
    return f"{MODEL_ID}:{PINNED[0][:16]}:{hashlib.sha256(mapping).hexdigest()[:8]}"


def model_file(models_dir: Path) -> Path:
    return models_dir / REGISTRY_ID / FILE


_verified: dict[Path, tuple[int, int]] = {}


def installed(models_dir: Path) -> bool:
    """The export exists and matches the pinned size and SHA-256 (memoised by size/mtime
    for the API; the worker re-hashes the bytes it loads)."""
    from xsample.runtime.registry import sha256_file

    path = model_file(models_dir)
    try:
        st = path.stat()
    except OSError:
        return False
    stamp = (st.st_size, st.st_mtime_ns)
    if _verified.get(path) == stamp:
        return True
    if st.st_size != PINNED[1] or sha256_file(path) != PINNED[0]:
        return False
    _verified[path] = stamp
    return True


def load_verified(models_dir: Path) -> bytes:
    data = model_file(models_dir).read_bytes()
    if len(data) != PINNED[1] or hashlib.sha256(data).hexdigest() != PINNED[0]:
        raise ValueError(f"{FILE} does not match its pinned SHA-256")
    return data


class TaggingCancelledError(Exception):
    pass


@dataclass
class TaggerReport:
    backend: str
    notes: list[str]
    inference_s: float


class Tagger:
    """ONNX session(s) for the tagger. Runs in a worker process only.

    DirectML is used, and reported, only if the strict session can be created, the profiled
    first window ran every node on DirectML and its output matches CPU on the same window.
    Otherwise the whole piece runs on CPU and the reason is noted.
    """

    def __init__(self, models_dir: Path, backend: str) -> None:
        from xsample.runtime.adapters import BackendUnsupportedError, get_adapter

        self._model = load_verified(models_dir)
        self.notes: list[str] = []
        self._cpu: Any = None
        self._gpu: Any = None
        if backend == "directml":
            try:
                self._gpu = get_adapter("directml").create_session(
                    self._model, strict=True, profile=True
                )
            except BackendUnsupportedError as exc:
                self.notes.append(f"DirectML could not run the tagger; it ran on CPU ({exc}).")
        self._checked = False

    @property
    def backend(self) -> str:
        return "directml" if self._gpu is not None else "cpu"

    def _cpu_session(self) -> Any:
        if self._cpu is None:
            from xsample.runtime.adapters import get_adapter

            self._cpu = get_adapter("cpu").create_session(self._model, strict=True)
        return self._cpu

    def _drop_gpu(self, reason: str) -> None:
        self.notes.append(f"{reason}; the tagger ran on CPU.")
        self._gpu.close()
        self._gpu = None

    def _verify(self, inputs: dict[str, Any]) -> None:
        g = self._gpu.run(inputs)
        report = self._gpu.end_profiling()
        fallback = report.fallback_nodes("DmlExecutionProvider") if report else ["no profile"]
        c = self._cpu_session().run(inputs)
        diff = float(np.max(np.abs(np.asarray(g[0]) - np.asarray(c[0]))))
        self._checked = True
        if fallback:
            self._drop_gpu(f"DirectML placed nodes on another provider ({', '.join(fallback)})")
        elif not np.isfinite(diff) or diff > PARITY_TOLERANCE:
            self._drop_gpu(f"DirectML differed from CPU by {diff:.3g} (limit {PARITY_TOLERANCE})")
        else:
            self._cpu.close()  # recreated only if DirectML later fails
            self._cpu = None

    def run(self, inputs: dict[str, Any]) -> list[Any]:
        if self._gpu is not None and not self._checked:
            self._verify(inputs)
        session = self._gpu if self._gpu is not None else self._cpu_session()
        result: list[Any] = session.run(inputs)
        return result

    def tag(
        self,
        logmel: F32,
        *,
        progress: Callable[[float], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[list[int], F32, TaggerReport]:
        t0 = time.perf_counter()
        try:
            starts, scores = predict_windows(logmel, self, progress=progress, cancelled=cancelled)
        except InterruptedError as exc:
            raise TaggingCancelledError from exc
        except Exception as exc:
            if self._gpu is None:
                raise
            # A DirectML runtime failure restarts the piece on CPU, so no result mixes backends.
            self._drop_gpu(f"DirectML failed during inference ({type(exc).__name__}: {exc})")
            try:
                starts, scores = predict_windows(
                    logmel, self, progress=progress, cancelled=cancelled
                )
            except InterruptedError as exc2:
                raise TaggingCancelledError from exc2
        report = TaggerReport(self.backend, list(self.notes), time.perf_counter() - t0)
        return starts, scores, report

    def close(self) -> None:
        for s in (self._gpu, self._cpu):
            if s is not None:
                s.close()
        self._gpu = self._cpu = None
