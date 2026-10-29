"""Beat This! integration: preprocessing parity, chunking, postprocessing, model loading,
backend selection and fallback, cancellation, short clips and cache invalidation.

Golden values come from torchaudio 2.11 / beat_this 1.1.0 (commit b95c8ab), computed once in
an isolated environment; the engine does not depend on torch. Most tests use a tiny ONNX
model generated here with the same input/output signature. Tests marked "real model" use
the installed exports from %LOCALAPPDATA%\\X-Sample\\models and are skipped without them.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper

from tests.analysis_fixtures import Spec, f_measure, render
from tests.conftest import wait_for_job
from xsample.analysis import beat_model as bm
from xsample.analysis.run import AnalysisCancelledError, analyze_file, cache_key
from xsample.config import default_data_dir
from xsample.runtime.adapters import DirectMLAdapter
from xsample.runtime.registry import sha256_file

REAL_MODELS = default_data_dir() / "models"
real_model = pytest.mark.skipif(
    not bm.installed(REAL_MODELS), reason="Beat This! exports are not installed"
)

# --- golden values (torchaudio LogMelSpect, beat_this split_piece and Postprocessor) ---------

GOLDEN_SHAPE = (101, 128)
GOLDEN_FRAME_SUMS = {0: 582.4906, 1: 251.7177, 50: 568.8569, 100: 629.7746}
GOLDEN_BIN_SUMS = {0: 69.3699, 10: 126.4194, 64: 66.0447, 100: 78.8718, 127: 93.6332}
GOLDEN_POINTS = [(0, 0, 6.96735), (37, 20, 2.90041), (75, 90, 4.82388), (100, 127, 5.86432)]
GOLDEN_STARTS = {
    100: [-6],
    1488: [-6],
    1489: [-6, -5],
    1500: [-6, 6],
    3000: [-6, 1482, 1506],
    4466: [-6, 1482, 2970, 2972],
    15000: [-6, 1482, 2970, 4458, 5946, 7434, 8922, 10410, 11898, 13386, 13506],
}
GOLDEN_BEATS = [
    0.0, 0.18, 0.42, 0.52, 0.74, 0.92, 1.01, 1.04, 1.34, 1.44, 1.52, 1.6, 1.76, 1.94, 2.18,
    2.34, 2.48, 2.56, 2.68, 2.84, 3.3, 3.48, 3.62, 3.78, 4.0, 4.18, 4.48, 4.84, 4.94, 5.04,
    5.16, 5.32, 5.46, 5.54, 5.62, 5.72, 5.88, 6.2, 6.58, 6.7, 6.86, 7.08, 7.22, 7.32, 7.54,
    7.72, 7.94,
]  # fmt: skip
GOLDEN_DOWNS = [
    0.18, 1.34, 1.76, 2.18, 2.68, 3.3, 3.62, 4.48, 5.46, 5.62, 5.88, 6.2, 7.54,
]  # fmt: skip


def golden_signal() -> np.ndarray:
    sr = 22050
    t = np.arange(sr * 2 + 123) / sr
    x = (
        0.3 * np.sin(2 * np.pi * (110 * t + 400 * t**2)) + 0.2 * np.sin(2 * np.pi * 3000 * t)
    ).astype(np.float32)
    x[:: sr // 4] += 0.8
    return x


@pytest.mark.parametrize("chunk", [512, 4096, 65536])
def test_log_mel_matches_torchaudio_and_streams(chunk: int) -> None:
    x = golden_signal()
    stream = bm.LogMelStream()
    spect = np.concatenate(
        [stream.push(x[i : i + chunk]) for i in range(0, len(x), chunk)] + [stream.finish()]
    )
    assert spect.shape == GOLDEN_SHAPE == (1 + len(x) // bm.HOP, bm.N_MELS)
    for i, v in GOLDEN_FRAME_SUMS.items():
        assert spect[i].sum() == pytest.approx(v, abs=2e-2)
    for j, v in GOLDEN_BIN_SUMS.items():
        assert spect[:, j].sum() == pytest.approx(v, abs=2e-2)
    for i, j, v in GOLDEN_POINTS:
        assert spect[i, j] == pytest.approx(v, abs=1e-3)


def test_log_mel_too_short_for_reflect_padding_is_empty() -> None:
    stream = bm.LogMelStream()
    assert len(stream.push(np.zeros(400, np.float32))) == 0
    assert stream.finish().shape == (0, bm.N_MELS)


@pytest.mark.parametrize(("n", "starts"), list(GOLDEN_STARTS.items()))
def test_chunk_starts_match_upstream(n: int, starts: list[int]) -> None:
    assert bm.chunk_starts(n) == starts


def test_postprocess_matches_upstream_minimal() -> None:
    rng = np.random.default_rng(3)
    b = rng.normal(-3, 3, 400).astype(np.float32)
    d = rng.normal(-5, 3, 400).astype(np.float32)
    b[[10, 11, 50, 51, 52, 200]] = 6.0
    beats, downs = bm.postprocess(b, d)
    np.testing.assert_allclose(beats, GOLDEN_BEATS, atol=1e-6)
    np.testing.assert_allclose(downs, GOLDEN_DOWNS, atol=1e-6)


# --- chunked prediction with a stand-in session --------------------------------------------


def frame_fn(spect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beat = spect.mean(axis=-1) - 2.0
    return beat, beat - 1.0


class FrameSession:
    """Per-frame function, so chunking + aggregation must reproduce it on the whole piece."""

    def __init__(self, fixed: int | None) -> None:
        self.fixed = fixed
        self.shapes: list[tuple[int, ...]] = []

    def run(self, inputs: dict[str, Any]) -> list[Any]:
        x = inputs["spect"]
        self.shapes.append(x.shape)
        if self.fixed is not None:
            assert x.shape == (1, self.fixed, bm.N_MELS)
        return list(frame_fn(x))


@pytest.mark.parametrize("n", [1, 700, 1488, 1489, 3000, 4466])
def test_predict_logits_aggregates_chunks(n: int) -> None:
    spect = np.random.default_rng(n).random((n, bm.N_MELS), dtype=np.float32) * 4
    full, short = FrameSession(bm.CHUNK), FrameSession(None)
    beat, down = bm.predict_logits(spect, full, short_session=short)
    want_b, want_d = frame_fn(spect)
    np.testing.assert_allclose(beat, want_b, atol=1e-6)
    np.testing.assert_allclose(down, want_d, atol=1e-6)
    if bm.is_short(n):
        # Short pieces: one exact-length chunk on the dynamic model, never zero-padded to CHUNK.
        assert full.shapes == [] and short.shapes == [(1, n + 2 * bm.BORDER, bm.N_MELS)]
    else:
        assert short.shapes == [] and len(full.shapes) == len(bm.chunk_starts(n))


def test_predict_logits_cancels_between_chunks() -> None:
    spect = np.zeros((6000, bm.N_MELS), np.float32)
    session = FrameSession(bm.CHUNK)
    calls = {"n": 0}

    def cancelled() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(InterruptedError):
        bm.predict_logits(spect, session, cancelled=cancelled)
    assert len(session.shapes) == 2


def test_short_piece_requires_the_dynamic_model() -> None:
    with pytest.raises(ValueError, match="dynamic"):
        bm.predict_logits(np.zeros((100, bm.N_MELS), np.float32), FrameSession(bm.CHUNK))


# --- model files: install, verification and loading ----------------------------------------


def tiny_model(fixed: bool) -> bytes:
    """spect (1, T, 128) -> beat = mean(spect) - 2, downbeat = beat - 1 (per frame)."""
    t: int | str = bm.CHUNK if fixed else "time"
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["spect"], ["m"], axes=[2], keepdims=0),
            helper.make_node("Sub", ["m", "two"], ["beat"]),
            helper.make_node("Sub", ["beat", "one"], ["downbeat"]),
        ],
        "tiny_beat",
        [helper.make_tensor_value_info("spect", TensorProto.FLOAT, [1, t, bm.N_MELS])],
        [
            helper.make_tensor_value_info("beat", TensorProto.FLOAT, [1, t]),
            helper.make_tensor_value_info("downbeat", TensorProto.FLOAT, [1, t]),
        ],
        [
            helper.make_tensor("two", TensorProto.FLOAT, [], [2.0]),
            helper.make_tensor("one", TensorProto.FLOAT, [], [1.0]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model.SerializeToString()


@pytest.fixture
def tiny_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    models = tmp_path / "models"
    pins = {}
    for name, fixed in ((bm.FULL_FILE, True), (bm.SHORT_FILE, False)):
        path = models / bm.REGISTRY_IDS[name] / name
        path.parent.mkdir(parents=True)
        path.write_bytes(tiny_model(fixed))
        pins[name] = (sha256_file(path), path.stat().st_size)
    monkeypatch.setattr(bm, "PINNED", pins)
    bm._verified.clear()
    return models


def test_installed_requires_both_files_with_pinned_hashes(tiny_models: Path) -> None:
    assert bm.installed(tiny_models)
    full = bm.model_files(tiny_models)[bm.FULL_FILE]
    data = bytearray(full.read_bytes())
    data[-1] ^= 1  # same size, different content
    full.write_bytes(bytes(data))
    # The worker never trusts the memo: it hashes the bytes it loads.
    with pytest.raises(ValueError, match="pinned SHA-256"):
        bm.BeatTracker(tiny_models, "cpu")
    bm._verified.clear()
    assert not bm.installed(tiny_models)
    full.unlink()
    assert not bm.installed(tiny_models)
    assert not bm.installed(tiny_models.parent / "missing")


def test_tracker_cpu_and_short_clip(tiny_models: Path) -> None:
    tracker = bm.BeatTracker(tiny_models, "cpu")
    try:
        for n in (300, 4000):
            spect = np.random.default_rng(n).random((n, bm.N_MELS), dtype=np.float32) * 4
            beats, downs, report = tracker.track(spect)
            assert report.backend == "cpu" and report.notes == []
            want_b, want_d = bm.postprocess(*frame_fn(spect))
            np.testing.assert_allclose(beats, want_b)
            np.testing.assert_allclose(downs, want_d)
    finally:
        tracker.close()


def test_tracker_cancellation(tiny_models: Path) -> None:
    tracker = bm.BeatTracker(tiny_models, "cpu")
    with pytest.raises(bm.BeatTrackingCancelledError):
        tracker.track(np.zeros((5000, bm.N_MELS), np.float32), cancelled=lambda: True)
    tracker.close()


def test_directml_unavailable_falls_back_to_cpu_and_says_so(
    tiny_models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xsample.runtime import adapters

    def refuse(self: Any, model: bytes, **kw: Any) -> Any:
        raise adapters.BackendUnsupportedError("directml: no device")

    monkeypatch.setattr(adapters.DirectMLAdapter, "create_session", refuse)
    tracker = bm.BeatTracker(tiny_models, "directml")
    _, _, report = tracker.track(np.ones((3000, bm.N_MELS), np.float32))
    tracker.close()
    assert report.backend == "cpu"
    assert any("DirectML could not run" in n for n in report.notes)


def test_directml_runtime_failure_restarts_piece_on_cpu(tiny_models: Path) -> None:
    tracker = bm.BeatTracker(tiny_models, "cpu")

    class Broken:
        def run(self, inputs: dict[str, Any]) -> list[Any]:
            raise RuntimeError("Reshape failed")

        def end_profiling(self) -> None:
            return None

        def close(self) -> None:
            pass

    tracker._gpu, tracker._checked = Broken(), True
    beats, _, report = tracker.track(np.ones((3000, bm.N_MELS), np.float32) * 3)
    tracker.close()
    assert report.backend == "cpu" and any("failed during inference" in n for n in report.notes)
    assert len(beats) > 0


def test_analyze_file_uses_model_and_keys_cache_by_model(tiny_models: Path) -> None:
    y, _ = render(Spec(bpm=120, key_root=0))

    def decode(_p: Path, **kw: Any) -> Iterator[np.ndarray]:
        for i in range(0, len(y), kw["chunk_frames"]):
            yield y[i : i + kw["chunk_frames"], None]

    common: dict[str, Any] = {
        "source_id": "src_0123456789ab",
        "media_sha256": "0" * 64,
        "sample_rate": 22050,
        "frames": len(y),
        "progress": lambda f, m: None,
        "cancelled": lambda: False,
        "decode": decode,
    }
    with_model = analyze_file(Path("x.wav"), beat_model_dir=tiny_models, **common)
    dsp = analyze_file(Path("x.wav"), **common)
    assert with_model.beat_tracker.method == "model" and with_model.backend == "cpu"
    assert with_model.beat_tracker.model_sha256 == [
        bm.PINNED[bm.FULL_FILE][0],
        bm.PINNED[bm.SHORT_FILE][0],
    ]
    assert dsp.beat_tracker.method == "dsp" and dsp.timings.beat_model_s == 0.0
    assert with_model.cache_key == cache_key("0" * 64, bm.identity())
    assert dsp.cache_key == cache_key("0" * 64) != with_model.cache_key
    assert any("Beat This!" in n for n in with_model.rhythm.notes)

    # A broken install falls back to DSP inside the job and says why.
    broken = analyze_file(Path("x.wav"), beat_model_dir=tiny_models / "nope", **common)
    assert broken.beat_tracker.method == "dsp"
    assert "could not be loaded" in broken.beat_tracker.notes[0]
    assert broken.rhythm.beats == dsp.rhythm.beats

    calls = {"n": 0}

    def cancel_in_model() -> bool:
        calls["n"] += 1
        return calls["n"] > len(y) // 65536 + 3

    with pytest.raises(AnalysisCancelledError):
        analyze_file(
            Path("x.wav"),
            beat_model_dir=tiny_models,
            **{**common, "cancelled": cancel_in_model},
        )


@pytest.mark.gpu
def test_tiny_model_on_directml_is_strict_and_reported(tiny_models: Path) -> None:
    if not DirectMLAdapter.is_available():
        pytest.skip("DirectML is not available")
    tracker = bm.BeatTracker(tiny_models, "directml")
    spect = np.random.default_rng(1).random((4000, bm.N_MELS), dtype=np.float32) * 4
    beats, _, report = tracker.track(spect)
    short_beats, _, short_report = tracker.track(spect[:500])
    tracker.close()
    assert report.backend == "directml" and report.notes == []
    np.testing.assert_allclose(beats, bm.postprocess(*frame_fn(spect))[0])
    assert short_report.backend == "cpu" and any("dynamic-length" in n for n in short_report.notes)
    assert len(short_beats) == len(bm.postprocess(*frame_fn(spect[:500]))[0])


# --- real model (skipped unless installed) --------------------------------------------------


def click_track_spect(bpm: float, seconds: float) -> tuple[np.ndarray, list[float]]:
    sr = 22050
    y = np.zeros(int(sr * seconds), np.float32)
    beats = list(np.arange(0.5, seconds - 0.2, 60.0 / bpm))
    t = np.arange(int(0.03 * sr)) / sr
    click = (np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 150)).astype(np.float32)
    for i, b in enumerate(beats):
        s = int(b * sr)
        y[s : s + len(click)] += click * (1.0 if i % 4 == 0 else 0.5)
    stream = bm.LogMelStream()
    return np.concatenate([stream.push(y), stream.finish()]), beats


@real_model
def test_real_model_tracks_a_click_track_on_cpu() -> None:
    for seconds in (20.0, 75.0):  # short (dynamic model) and chunked (fixed model)
        spect, truth = click_track_spect(120.0, seconds)
        tracker = bm.BeatTracker(REAL_MODELS, "cpu")
        beats, _, report = tracker.track(spect)
        tracker.close()
        assert report.backend == "cpu"
        assert f_measure(list(beats), truth) > 0.9


@real_model
@pytest.mark.gpu
def test_real_model_directml_matches_cpu() -> None:
    if not DirectMLAdapter.is_available():
        pytest.skip("DirectML is not available")
    spect, _ = click_track_spect(97.0, 120.0)
    cpu = bm.BeatTracker(REAL_MODELS, "cpu")
    gpu = bm.BeatTracker(REAL_MODELS, "directml")
    want = bm.predict_logits(spect, cpu, short_session=cpu._cpu_short)
    got = bm.predict_logits(spect, gpu, short_session=gpu._cpu_short)
    assert gpu.backend == "directml" and gpu.notes == []  # strict, profiled, parity-checked
    for a, b in zip(got, want, strict=True):
        assert float(np.max(np.abs(a - b))) < 1e-3
    np.testing.assert_allclose(bm.postprocess(*got)[0], bm.postprocess(*want)[0])
    cpu.close()
    gpu.close()


@real_model
def test_installing_the_model_makes_dsp_results_stale_then_rerun_uses_it(
    client: TestClient, auth: dict[str, str], tmp_path: Path, settings: Any
) -> None:
    from tests.test_analysis import _project_with_source, _wait_ready

    source_id = _project_with_source(client, auth, tmp_path)
    url = f"/api/project/sources/{source_id}/analysis"
    client.post(url, headers=auth, json={})
    dsp = _wait_ready(client, auth, url)
    assert dsp["status"] == "ready" and dsp["result"]["beat_tracker"]["method"] == "dsp"

    for name, path in bm.model_files(REAL_MODELS).items():
        target = bm.model_files(settings.models_dir)[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    stale = client.get(url, headers=auth).json()
    assert stale["status"] == "stale" and stale["cache_key"] != dsp["cache_key"]

    job = client.post(url, headers=auth, json={}).json()["job"]
    # Detection is off in tests, so the engine requests CPU; DirectML is not claimed.
    assert job["params"]["beat_backend"] == "cpu" and job["uses_gpu"] is False
    assert wait_for_job(client, job["id"], auth, timeout=120)["state"] == "succeeded"
    view = client.get(url, headers=auth).json()
    assert view["status"] == "ready"
    tracker = view["result"]["beat_tracker"]
    assert tracker["method"] == "model" and tracker["backend"] == "cpu"
    assert view["result"]["backend"] == "cpu" and "MIT" in tracker["license"]
    assert abs(view["result"]["rhythm"]["bpm"] - 76) < 3
