"""Neural beat and downbeat tracking with the Beat This! "small" model (MIT).

Pre- and postprocessing are NumPy re-implementations of beat_this 1.1.0 (commit b95c8ab):
torchaudio log-mel, chunked prediction (split_predict_aggregate, keep_first) and the
"minimal" peak picking. Inference goes through an ONNX session from runtime/adapters.py;
this module never imports onnxruntime, so importing it is safe in the API process.

Two ONNX exports of the same weights are used. DirectML cannot run the dynamic-shape
export (a Reshape fails), so long pieces use a fixed (1, CHUNK, N_MELS) export; every chunk
of a piece longer than CHUNK - 2 * BORDER frames is exactly CHUNK long. A shorter piece is
one chunk of its own length (upstream behaviour) and needs the dynamic export, which runs
on CPU: zero-padding it to CHUNK changes the predictions materially.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

SAMPLE_RATE = 22050
N_FFT, HOP = 1024, 441  # 50 frames per second
FPS = SAMPLE_RATE / HOP
N_MELS = 128
F_MIN, F_MAX = 30.0, 11000.0
LOG_MULTIPLIER = 1000.0
CHUNK, BORDER = 1500, 6
NO_PREDICTION = -1000.0
_BATCH_FRAMES = 2048  # STFT frames per block: bounds the complex spectrum to ~8 MB


def _hz_to_mel_slaney(f: F64) -> F64:
    f_sp, min_log_hz, min_log_mel = 200.0 / 3, 1000.0, 15.0
    logstep = np.log(6.4) / 27.0
    mel: F64 = f / f_sp
    high = f >= min_log_hz
    mel[high] = min_log_mel + np.log(f[high] / min_log_hz) / logstep
    return mel


def _mel_to_hz_slaney(m: F64) -> F64:
    f_sp, min_log_hz, min_log_mel = 200.0 / 3, 1000.0, 15.0
    logstep = np.log(6.4) / 27.0
    f: F64 = f_sp * m
    high = m >= min_log_mel
    f[high] = min_log_hz * np.exp(logstep * (m[high] - min_log_mel))
    return f


def mel_filterbank() -> F64:
    """torchaudio.functional.melscale_fbanks(513, 30, 11000, 128, 22050, None, "slaney")."""
    all_freqs = np.linspace(0.0, SAMPLE_RATE // 2, N_FFT // 2 + 1)
    m = np.linspace(
        _hz_to_mel_slaney(np.array([F_MIN]))[0], _hz_to_mel_slaney(np.array([F_MAX]))[0], N_MELS + 2
    )
    f_pts = _mel_to_hz_slaney(m)
    f_diff = np.diff(f_pts)
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb: F64 = np.maximum(0.0, np.minimum(down, up))
    return fb  # (513, 128)


class LogMelStream:
    """Streaming equivalent of beat_this LogMelSpect (center=True, reflect padding).

    Frame k is centred on sample k*HOP; a piece of N samples yields 1 + N // HOP frames.
    Only the window overlap is buffered, so memory does not grow with duration.
    """

    def __init__(self) -> None:
        self._buf = np.zeros(0, np.float64)
        self._started = False
        self._total = 0
        self._frames = 0
        self._window = np.hanning(N_FFT + 1)[:-1]  # periodic Hann, as torch.hann_window
        # normalized="frame_length" is torch.stft(normalized=True): 1/sqrt(n_fft).
        self._norm = float(np.sqrt(N_FFT))
        self._fb = mel_filterbank()

    def _emit(self) -> F32:
        count = 0 if len(self._buf) < N_FFT else 1 + (len(self._buf) - N_FFT) // HOP
        out = [np.zeros((0, N_MELS), np.float32)]
        for first in range(0, count, _BATCH_FRAMES):
            n = min(_BATCH_FRAMES, count - first)
            idx = np.arange(N_FFT)[None, :] + HOP * (first + np.arange(n))[:, None]
            mag = np.abs(np.fft.rfft(self._buf[idx] * self._window, axis=1)) / self._norm
            out.append(np.log1p(LOG_MULTIPLIER * (mag @ self._fb)).astype(np.float32))
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
            # Too short for reflect padding (< 23 ms): nothing to track.
            return np.zeros((0, N_MELS), np.float32)
        self._buf = np.concatenate([self._buf, self._buf[-2 : -N_FFT // 2 - 2 : -1]])
        tail = self._emit()
        wanted = 1 + self._total // HOP
        return tail[: max(0, wanted - (self._frames - len(tail)))]


class OnnxRunner(Protocol):
    def run(self, inputs: dict[str, Any]) -> list[Any]: ...


def chunk_starts(n_frames: int) -> list[int]:
    """beat_this split_piece(chunk_size=CHUNK, border_size=BORDER, avoid_short_end=True)."""
    starts = list(range(-BORDER, n_frames - BORDER, CHUNK - 2 * BORDER))
    if n_frames > CHUNK - 2 * BORDER:
        starts[-1] = n_frames - (CHUNK - BORDER)
    return starts


def is_short(n_frames: int) -> bool:
    return n_frames <= CHUNK - 2 * BORDER


def predict_logits(
    spect: F32,
    session: OnnxRunner,
    *,
    short_session: OnnxRunner | None = None,
    progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[F32, F32]:
    """Beat and downbeat logits per frame (beat includes downbeats, as SumHead)."""
    n = len(spect)
    if n == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    if is_short(n):
        if short_session is None:
            raise ValueError("pieces of <= CHUNK - 2 * BORDER frames need the dynamic model")
        padded = np.pad(spect, ((BORDER, BORDER), (0, 0)))[None]
        b, d = short_session.run({"spect": padded})
        if progress is not None:
            progress(1.0)
        return (
            np.asarray(b, np.float32)[0, BORDER:-BORDER],
            np.asarray(d, np.float32)[0, BORDER:-BORDER],
        )
    beat = np.full(n, NO_PREDICTION, np.float32)
    down = np.full(n, NO_PREDICTION, np.float32)
    starts = chunk_starts(n)
    # keep_first: run in reverse so earlier chunks overwrite the overlap of later ones.
    for i, start in enumerate(reversed(starts)):
        if cancelled is not None and cancelled():
            raise InterruptedError("cancelled")
        chunk = np.zeros((CHUNK, N_MELS), np.float32)
        lo, hi = max(start, 0), min(start + CHUNK, n)
        chunk[lo - start : hi - start] = spect[lo:hi]
        b, d = session.run({"spect": chunk[None]})
        b, d = np.asarray(b)[0], np.asarray(d)[0]
        # The model is not trained on the BORDER frames at either edge of a chunk.
        beat[start + BORDER : start + CHUNK - BORDER] = b[BORDER:-BORDER]
        down[start + BORDER : start + CHUNK - BORDER] = d[BORDER:-BORDER]
        if progress is not None:
            progress((i + 1) / len(starts))
    return beat, down


def _peaks(logits: F32) -> F64:
    padded = np.pad(logits.astype(np.float64), 3, constant_values=-np.inf)
    window = np.lib.stride_tricks.sliding_window_view(padded, 7).max(axis=1)
    frames = np.flatnonzero((logits == window) & (logits > 0))
    # deduplicate_peaks(width=1): runs of adjacent frames become their running mean.
    out: list[float] = []
    p, c = 0.0, 0
    for f in frames:
        if c and f - p <= 1:
            c += 1
            p += (f - p) / c
        else:
            if c:
                out.append(p)
            p, c = float(f), 1
    if c:
        out.append(p)
    return np.asarray(out, np.float64)


def postprocess(beat: F32, down: F32) -> tuple[F64, F64]:
    """beat_this Postprocessor(type="minimal"): beat and downbeat times in seconds."""
    beats = _peaks(beat) / FPS
    downs = _peaks(down) / FPS
    if len(beats) and len(downs):
        # Each downbeat moves to the nearest beat.
        idx = np.abs(downs[:, None] - beats[None, :]).argmin(axis=1)
        downs = np.unique(beats[idx])
    return beats, downs


# --- Installed model ------------------------------------------------------------------------

MODEL_ID = "beat-this-small0"
DISPLAY_NAME = "Beat This! small (small0)"
LICENSE = (
    "MIT (code and weights, CPJKU/beat_this). Some training audio is copyrighted or under "
    "limited Creative Commons licenses; assess whether that matters for your use."
)
SOURCE_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/small0.ckpt"
CHECKPOINT_SHA256 = "6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b"
CHECKPOINT_BYTES = 8451101
# ONNX exports of that checkpoint made by tools/export_beat_this.py; the pins are the trust
# root (registry.json on disk is user-writable and only mirrors them).
FULL_FILE = "beat_this_small0_t1500.onnx"  # fixed (1, CHUNK, N_MELS): CPU or DirectML
SHORT_FILE = "beat_this_small0.onnx"  # dynamic time axis: CPU only, short pieces
# Registry ids (one per file: the store keeps each model in models/<id>/<file>).
REGISTRY_IDS = {FULL_FILE: f"{MODEL_ID}-t1500", SHORT_FILE: f"{MODEL_ID}-dynamic"}
PINNED: dict[str, tuple[str, int]] = {  # filename -> (sha256, size); export is deterministic
    FULL_FILE: ("5f4f4182efefb66a578e8d1a6a013bce6957f044a446a80b935bd1d2f7765ca6", 12068437),
    SHORT_FILE: ("bc45629cca16eb41f30c234d7e962d68e6ada810e3822e10039dc56f2f980987", 10390369),
}
# Logits from DirectML must match the CPU run of the same chunk within this, or the job
# falls back to CPU (measured max difference on real audio: ~1e-4).
PARITY_TOLERANCE = 1e-2


def identity() -> str:
    """Model identity for cache keys: changes whenever either pinned export changes."""
    return ":".join([MODEL_ID] + [PINNED[f][0][:16] for f in (FULL_FILE, SHORT_FILE)])


def model_files(models_dir: Path) -> dict[str, Path]:
    return {name: models_dir / REGISTRY_IDS[name] / name for name in (FULL_FILE, SHORT_FILE)}


_verified: dict[Path, tuple[int, int]] = {}


def installed(models_dir: Path) -> bool:
    """Both exports exist and match the pinned size and SHA-256.

    Memoised by size and mtime for the API's cache-key and backend decisions only; the
    worker re-hashes the exact bytes it loads (load_verified).
    """
    from xsample.runtime.registry import sha256_file

    for name, path in model_files(models_dir).items():
        sha, size = PINNED[name]
        try:
            st = path.stat()
        except OSError:
            return False
        stamp = (st.st_size, st.st_mtime_ns)
        if _verified.get(path) == stamp:
            continue
        if st.st_size != size or sha256_file(path) != sha:
            return False
        _verified[path] = stamp
    return True


class BeatTrackingCancelledError(Exception):
    pass


def load_verified(models_dir: Path) -> dict[str, bytes]:
    """Model bytes, each checked against its pin after reading (no check-then-use gap)."""
    out = {}
    for name, path in model_files(models_dir).items():
        data = path.read_bytes()
        sha, size = PINNED[name]
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"{name} does not match its pinned SHA-256")
        out[name] = data
    return out


@dataclass
class TrackerReport:
    backend: str  # where the model ran: "cpu" | "directml"
    notes: list[str]
    inference_s: float


class BeatTracker:
    """ONNX sessions for both exports. Runs in a worker process only (imports onnxruntime).

    DirectML is used, and reported, only if the strict session (CPU fallback disabled) can
    be created, the profiled first chunk ran every node on DirectML and its logits match the
    CPU run of the same chunk. Otherwise the whole piece runs on CPU and the reason is noted.
    """

    def __init__(self, models_dir: Path, backend: str) -> None:
        from xsample.runtime.adapters import BackendUnsupportedError, get_adapter

        files = load_verified(models_dir)
        self.notes: list[str] = []
        # CPU sessions are created on first use (each holds a few hundred MiB of arena):
        # the dynamic one only for short pieces, the fixed one for CPU runs and the parity
        # check, after which it is released when DirectML is used.
        cpu = get_adapter("cpu")
        self._cpu_short = _LazySession(lambda: cpu.create_session(files[SHORT_FILE], strict=True))
        self._cpu_full = _LazySession(lambda: cpu.create_session(files[FULL_FILE], strict=True))
        self._gpu: Any = None
        if backend == "directml":
            try:
                self._gpu = get_adapter("directml").create_session(
                    files[FULL_FILE], strict=True, profile=True
                )
            except BackendUnsupportedError as exc:
                self.notes.append(f"DirectML could not run the beat model; it ran on CPU ({exc}).")
        self._checked = False

    @property
    def backend(self) -> str:
        return "directml" if self._gpu is not None else "cpu"

    def _drop_gpu(self, reason: str) -> None:
        self.notes.append(f"{reason}; the beat model ran on CPU.")
        self._gpu.close()
        self._gpu = None

    def _verify(self, chunk: F32) -> None:
        g = self._gpu.run({"spect": chunk})
        report = self._gpu.end_profiling()
        fallback = report.fallback_nodes("DmlExecutionProvider") if report else ["no profile"]
        c = self._cpu_full.run({"spect": chunk})
        diff = max(
            float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(g, c, strict=True)
        )
        self._checked = True
        if fallback:
            self._drop_gpu(f"DirectML placed nodes on another provider ({', '.join(fallback)})")
        elif not np.isfinite(diff) or diff > PARITY_TOLERANCE:
            self._drop_gpu(f"DirectML differed from CPU by {diff:.3g} (limit {PARITY_TOLERANCE})")
        else:
            self._cpu_full.close()  # recreated only if DirectML later fails

    def run(self, inputs: dict[str, Any]) -> list[Any]:
        if self._gpu is not None and not self._checked:
            self._verify(inputs["spect"])
        session = self._gpu if self._gpu is not None else self._cpu_full
        out: list[Any] = session.run(inputs)
        return out

    def track(
        self,
        spect: F32,
        *,
        progress: Callable[[float], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[F64, F64, TrackerReport]:
        t0 = time.perf_counter()
        short = is_short(len(spect))
        piece_notes = []
        if short and self._gpu is not None:
            piece_notes.append(
                "Shorter than one 30 s model chunk: the dynamic-length model ran on CPU."
            )
        try:
            logits = predict_logits(
                spect, self, short_session=self._cpu_short, progress=progress, cancelled=cancelled
            )
        except InterruptedError as exc:
            raise BeatTrackingCancelledError from exc
        except Exception as exc:
            if self._gpu is None:
                raise
            # A DirectML runtime failure restarts the piece on CPU, so no result mixes backends.
            self._drop_gpu(f"DirectML failed during inference ({type(exc).__name__}: {exc})")
            try:
                logits = predict_logits(
                    spect,
                    self,
                    short_session=self._cpu_short,
                    progress=progress,
                    cancelled=cancelled,
                )
            except InterruptedError as exc2:
                raise BeatTrackingCancelledError from exc2
        backend = "cpu" if short else self.backend
        beats, downs = postprocess(*logits)
        report = TrackerReport(
            backend=backend,
            notes=self.notes + piece_notes,
            inference_s=time.perf_counter() - t0,
        )
        return beats, downs, report

    def close(self) -> None:
        if self._gpu is not None:
            self._gpu.close()
            self._gpu = None
        self._cpu_full.close()
        self._cpu_short.close()


class _LazySession:
    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._session: Any = None

    def run(self, inputs: dict[str, Any]) -> list[Any]:
        if self._session is None:
            self._session = self._factory()
        out: list[Any] = self._session.run(inputs)
        return out

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
