"""PANNs MobileNetV1 tagger: front-end parity, windowing, scoring, model loading, backend
selection and fallback, cancellation, sidecar/cache integration and tag corrections.

Golden values come from torchlibrosa 0.1.0 and the complete torch model (audioset_tagging_cnn
commit d2f4b8c, MobileNetV1_mAP=0.389.pth), computed once in an isolated environment; the
engine does not depend on torch. Most tests use a tiny ONNX model with the same signature.
Tests marked "real model" use the installed export and are skipped without it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from tests.analysis_fixtures import Spec, render
from xsample.analysis import tagger as tg
from xsample.analysis.effective import effective
from xsample.analysis.run import AnalysisCancelledError, analyze_file, cache_key
from xsample.analysis.schema import TagsOut, Unavailable
from xsample.config import default_data_dir
from xsample.project.model import AnalysisCorrections, TagCorrection
from xsample.runtime.adapters import DirectMLAdapter
from xsample.runtime.registry import sha256_file

REAL_MODELS = default_data_dir() / "models"
real_model = pytest.mark.skipif(
    not tg.installed(REAL_MODELS), reason="PANNs tagger export is not installed"
)
SR = tg.SAMPLE_RATE


def tone(seconds: float, hz: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


# --- front end -------------------------------------------------------------------------------


def test_mel_filterbank_matches_the_checkpoint() -> None:
    fb = tg.mel_filterbank()
    assert fb.shape == (513, 64)
    # (bin, band, weight, band sum) from the checkpoint's logmel_extractor.melW.
    for band, peak_bin, peak, total in (
        (0, 3, 0.01501181349158287, 0.031203873455524445),
        (20, 39, 0.012666620314121246, 0.03168105706572533),
        (63, 424, 0.0013412120752036572, 0.03199377655982971),
    ):
        assert int(fb[:, band].argmax()) == peak_bin
        assert fb[peak_bin, band] == pytest.approx(peak, rel=1e-6)
        assert fb[:, band].sum() == pytest.approx(total, rel=1e-6)


@pytest.mark.parametrize("chunk", [7777, 65536, 10**6])
def test_log_mel_matches_torchlibrosa_and_streams(chunk: int) -> None:
    x = tone(10.0)
    stream = tg.LogMelStream()
    lm = np.concatenate([stream.push(x[i : i + chunk]) for i in range(0, len(x), chunk)])
    lm = np.concatenate([lm, stream.finish()])
    assert lm.shape == (1 + len(x) // tg.HOP, tg.N_MELS) == (1001, 64)
    # (frame, band, torchlibrosa dB); edge frames see the reflect padding.
    for frame, band, want in (
        (0, 8, 10.8167),
        (500, 8, -19.9786),
        (500, 9, -37.9093),
        (1000, 8, 10.7727),
        (500, 40, -100.0),
    ):
        assert lm[frame, band] == pytest.approx(want, abs=2e-3)


def test_log_mel_of_a_tiny_input_is_empty() -> None:
    stream = tg.LogMelStream()
    assert len(stream.push(np.zeros(100, np.float32))) == 0
    assert len(stream.finish()) == 0


# --- windowing and scoring -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "starts"),
    [
        (100, [0]),
        (1001, [0]),
        (1002, [0, 1]),
        (2001, [0, 500, 1000]),
        (2300, [0, 500, 1000, 1299]),
    ],
)
def test_window_starts(n: int, starts: list[int]) -> None:
    assert tg.window_starts(n) == starts


def label_index(name: str) -> int:
    return next(i for i, lb in enumerate(tg.LABELS) if lb.name == name)


def test_summarise_thresholds_activity_and_genre_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    starts = [0, 500, 1000, 1500]
    scores = np.zeros((4, len(tg.LABELS)), np.float32)
    piano, violin, rnb = label_index("Piano"), label_index("Violin"), label_index("R&B")
    scores[:, piano] = [0.9, 0.1, 0.3, 0.6]
    scores[:, violin] = [0.25, 0.0, 0.0, 0.0]
    scores[:, rnb] = [0.8, 0.8, 0.0, 0.0]
    monkeypatch.setattr(tg, "GENRE_EVIDENCE", frozenset())
    out = {s.label.name: s for s in tg.summarise(starts, scores, 2501)}
    assert out["Piano"].status == "detected" and out["Piano"].score == pytest.approx(0.9)
    # Windows 0 (0-10 s) and 2+3 (10-25.01 s) overlap and merge; window 1 is below 0.2.
    assert [(a, z) for a, z, _ in out["Piano"].activity] == [(0.0, 25.01)]
    assert out["Violin"].status == "uncertain" and out["Violin"].activity == [(0.0, 10.01, 0.25)]
    assert out["Saxophone"].status == "not_detected" and out["Saxophone"].activity == []
    # A genre without measured evidence is at most uncertain, whatever its score.
    assert out["R&B"].status == "uncertain"
    monkeypatch.setattr(tg, "GENRE_EVIDENCE", frozenset({"R&B"}))
    out = {s.label.name: s for s in tg.summarise(starts, scores, 2501)}
    assert out["R&B"].status == "detected"


def test_every_label_maps_to_valid_audioset_classes() -> None:
    names = [(lb.kind, lb.name) for lb in tg.LABELS]
    assert len(names) == len(set(names))
    for lb in tg.LABELS:
        assert lb.classes and len(lb.classes) == len(lb.class_names)
        assert all(0 <= c < tg.N_CLASSES for c in lb.classes)


class Echo:
    """Fake session: class c's probability = mean of mel band c % 64, as a sigmoid."""

    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def run(self, inputs: dict[str, Any]) -> list[Any]:
        x = inputs["logmel"]
        self.shapes.append(x.shape)
        m = x.mean(axis=1)[0]
        probs = 1 / (1 + np.exp(-m[np.arange(tg.N_CLASSES) % tg.N_MELS]))
        return [probs[None].astype(np.float32)]


def test_predict_windows_shapes_and_cancellation() -> None:
    logmel = np.zeros((2300, tg.N_MELS), np.float32)
    session = Echo()
    starts, scores = tg.predict_windows(logmel, session)
    assert starts == [0, 500, 1000, 1299] and scores.shape == (4, len(tg.LABELS))
    assert set(session.shapes) == {(1, tg.WINDOW, tg.N_MELS)}
    np.testing.assert_allclose(scores, 0.5)
    _, short = tg.predict_windows(logmel[:300], session)
    assert session.shapes[-1] == (1, 300, tg.N_MELS) and short.shape == (1, len(tg.LABELS))
    with pytest.raises(InterruptedError):
        tg.predict_windows(logmel, session, cancelled=lambda: True)


# --- model file, sessions and backends -------------------------------------------------------


def tiny_model() -> bytes:
    """logmel (1, T, 64) -> probs (1, 527) = sigmoid(mean over time and bands / 100 + bias)."""
    bias = (np.arange(tg.N_CLASSES) % 7 - 3).astype(np.float32)  # every label range spread
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["logmel"], ["m"], axes=[1, 2], keepdims=0),
            helper.make_node("Div", ["m", "ten"], ["s"]),
            helper.make_node("Unsqueeze", ["s", "axis"], ["s1"]),
            helper.make_node("Add", ["s1", "bias"], ["z"]),
            helper.make_node("Sigmoid", ["z"], ["probs"]),
        ],
        "tiny_tagger",
        [helper.make_tensor_value_info("logmel", TensorProto.FLOAT, [1, "frames", tg.N_MELS])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, tg.N_CLASSES])],
        [
            helper.make_tensor("ten", TensorProto.FLOAT, [], [100.0]),
            helper.make_tensor("axis", TensorProto.INT64, [1], [1]),
            helper.make_tensor("bias", TensorProto.FLOAT, [1, tg.N_CLASSES], bias.tolist()),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model.SerializeToString()


@pytest.fixture
def tiny_tagger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    models = tmp_path / "models"
    path = tg.model_file(models)
    path.parent.mkdir(parents=True)
    path.write_bytes(tiny_model())
    monkeypatch.setattr(tg, "PINNED", (sha256_file(path), path.stat().st_size))
    tg._verified.clear()
    return models


def test_installed_and_load_verify_the_pin(tiny_tagger: Path) -> None:
    assert tg.installed(tiny_tagger)
    path = tg.model_file(tiny_tagger)
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="pinned SHA-256"):
        tg.Tagger(tiny_tagger, "cpu")
    tg._verified.clear()
    assert not tg.installed(tiny_tagger)
    path.unlink()
    assert not tg.installed(tiny_tagger)


def test_tagger_cpu_and_cancellation(tiny_tagger: Path) -> None:
    model = tg.Tagger(tiny_tagger, "cpu")
    starts, scores, report = model.tag(np.zeros((1600, tg.N_MELS), np.float32))
    assert starts == [0, 500, 599] and report.backend == "cpu" and report.notes == []
    assert scores.shape == (3, len(tg.LABELS))
    with pytest.raises(tg.TaggingCancelledError):
        model.tag(np.zeros((1600, tg.N_MELS), np.float32), cancelled=lambda: True)
    model.close()


def test_directml_unavailable_falls_back_to_cpu_and_says_so(
    tiny_tagger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xsample.runtime import adapters

    def refuse(self: Any, model: bytes, **kw: Any) -> Any:
        raise adapters.BackendUnsupportedError("directml: no device")

    monkeypatch.setattr(adapters.DirectMLAdapter, "create_session", refuse)
    model = tg.Tagger(tiny_tagger, "directml")
    _, _, report = model.tag(np.zeros((1200, tg.N_MELS), np.float32))
    model.close()
    assert report.backend == "cpu" and any("DirectML could not run" in n for n in report.notes)


class Broken:
    def run(self, inputs: dict[str, Any]) -> list[Any]:
        raise RuntimeError("device removed")

    def end_profiling(self) -> None:
        return None

    def close(self) -> None:
        pass


def test_directml_runtime_failure_restarts_piece_on_cpu(tiny_tagger: Path) -> None:
    model = tg.Tagger(tiny_tagger, "cpu")
    model._gpu, model._checked = Broken(), True
    _, scores, report = model.tag(np.zeros((1200, tg.N_MELS), np.float32))
    model.close()
    assert report.backend == "cpu" and any("failed during inference" in n for n in report.notes)
    assert scores.shape[0] == 2


class Wrong:
    """A 'GPU' session that runs every node on DirectML but returns different numbers."""

    def run(self, inputs: dict[str, Any]) -> list[Any]:
        return [np.zeros((1, tg.N_CLASSES), np.float32)]

    def end_profiling(self) -> Any:
        class Report:
            def fallback_nodes(self, _: str) -> list[str]:
                return []

        return Report()

    def close(self) -> None:
        pass


def test_directml_parity_failure_uses_cpu(tiny_tagger: Path) -> None:
    model = tg.Tagger(tiny_tagger, "cpu")
    model._gpu = Wrong()
    _, scores, report = model.tag(np.zeros((1200, tg.N_MELS), np.float32))
    model.close()
    assert report.backend == "cpu" and any("differed from CPU" in n for n in report.notes)
    assert float(scores.max()) > 0.0  # CPU numbers, not the zeros from the bad session


@pytest.mark.gpu
def test_tiny_model_on_directml_is_strict_and_reported(tiny_tagger: Path) -> None:
    if not DirectMLAdapter.is_available():
        pytest.skip("DirectML is not available")
    logmel = np.random.default_rng(2).normal(-40, 20, (2300, tg.N_MELS)).astype(np.float32)
    gpu, cpu = tg.Tagger(tiny_tagger, "directml"), tg.Tagger(tiny_tagger, "cpu")
    _, got, report = gpu.tag(logmel)
    _, want, _ = cpu.tag(logmel)
    gpu.close()
    cpu.close()
    assert report.backend == "directml" and report.notes == []
    np.testing.assert_allclose(got, want, atol=1e-5)


# --- analysis integration --------------------------------------------------------------------


def _decoder(y22: np.ndarray, y32: np.ndarray) -> Any:
    def decode(_p: Path, **kw: Any) -> Iterator[np.ndarray]:
        y = y32 if kw["sample_rate"] == SR else y22
        for i in range(0, len(y), kw["chunk_frames"]):
            yield y[i : i + kw["chunk_frames"], None]

    return decode


def test_analyze_file_tags_and_keys_cache_by_tagger(tiny_tagger: Path) -> None:
    y22, _ = render(Spec(bpm=120, key_root=0))
    seconds = len(y22) / 22050
    common: dict[str, Any] = {
        "source_id": "src_0123456789ab",
        "media_sha256": "0" * 64,
        "sample_rate": 22050,
        "frames": len(y22),
        "progress": lambda f, m: None,
        "cancelled": lambda: False,
        "decode": _decoder(y22, tone(seconds)),
    }
    tagged = analyze_file(Path("x.wav"), tagger_dir=tiny_tagger, **common)
    plain = analyze_file(Path("x.wav"), **common)
    assert isinstance(tagged.instruments, TagsOut) and isinstance(tagged.genres, TagsOut)
    assert isinstance(tagged.vocals, TagsOut)
    assert {t.kind for t in tagged.vocals.tags} == {"vocal"}
    assert len(tagged.instruments.tags) == sum(lb.kind == "instrument" for lb in tg.LABELS)
    assert tagged.tagger is not None and tagged.tagger.backend == "cpu"
    assert tagged.tagger.model_sha256 == tg.PINNED[0] and "CC BY 4.0" in tagged.tagger.license
    assert tagged.timings.tagger_s > 0
    for tag in tagged.instruments.tags:
        for seg in tag.activity:
            assert 0 <= seg.start < seg.end <= len(y22)
    assert isinstance(plain.instruments, Unavailable) and plain.tagger is None
    assert tagged.cache_key == cache_key("0" * 64, None, tg.identity())
    assert plain.cache_key == cache_key("0" * 64) != tagged.cache_key

    # A broken install leaves tags unavailable and says why; the rest of the analysis runs.
    broken = analyze_file(Path("x.wav"), tagger_dir=tiny_tagger / "nope", **common)
    assert isinstance(broken.genres, Unavailable) and "could not be loaded" in broken.genres.reason
    assert broken.tagger is not None and broken.rhythm.beats == plain.rhythm.beats

    # Too short to tag (under 1 s): unavailable, not guessed.
    short = analyze_file(
        Path("x.wav"),
        tagger_dir=tiny_tagger,
        **{**common, "frames": 11025, "decode": _decoder(y22[:11025], tone(0.5))},
    )
    assert isinstance(short.instruments, Unavailable) and "Too short" in short.instruments.reason

    calls = {"n": 0}

    def cancel_while_tagging() -> bool:
        calls["n"] += 1
        return calls["n"] > len(y22) // 65536 + 8

    with pytest.raises(AnalysisCancelledError):
        analyze_file(
            Path("x.wav"), tagger_dir=tiny_tagger, **{**common, "cancelled": cancel_while_tagging}
        )


def test_effective_tags_merge_detected_uncertain_and_user_corrections(tiny_tagger: Path) -> None:
    y22, _ = render(Spec(bpm=100, key_root=2))
    result = analyze_file(
        Path("x.wav"),
        source_id="src_0123456789ab",
        media_sha256="1" * 64,
        sample_rate=22050,
        frames=len(y22),
        progress=lambda f, m: None,
        cancelled=lambda: False,
        decode=_decoder(y22, tone(len(y22) / 22050)),
        tagger_dir=tiny_tagger,
    )
    assert isinstance(result.instruments, TagsOut)
    shown = [t for t in result.instruments.tags if t.status != "not_detected"]
    assert shown, "the tiny model's bias spreads scores across the thresholds"
    eff = effective(result, AnalysisCorrections())
    detected = [t for t in eff.tags if t.origin == "detected"]
    assert {t.label for t in detected} >= {t.label for t in shown}
    assert all(
        t.uncertain == (t.confidence is not None and t.confidence < 0.5)
        for t in detected
        if t.kind != "genre"
    )
    assert all(t.uncertain for t in detected if t.kind == "genre")  # no genre evidence yet
    first = shown[0]
    corr = AnalysisCorrections(
        tags=[
            TagCorrection(kind="instrument", label=first.label.upper(), present=False),
            TagCorrection(kind="vocal", label="Singing"),
            TagCorrection(kind="genre", label="Neo soul"),
        ]
    )
    eff = effective(result, corr)
    labels = [(t.kind, t.label.lower(), t.origin) for t in eff.tags]
    assert ("instrument", first.label.lower(), "detected") not in labels
    assert all(lb != first.label.lower() for _, lb, _ in labels)
    assert labels[:2] == [("vocal", "singing", "user"), ("genre", "neo soul", "user")]
    assert "tags" in eff.corrected


# --- real model (skipped unless installed) ---------------------------------------------------


@real_model
def test_real_model_matches_torch_on_a_tone() -> None:
    stream = tg.LogMelStream()
    x = tone(10.0)
    logmel = np.concatenate([stream.push(x), stream.finish()])[None]
    model = tg.Tagger(REAL_MODELS, "cpu")
    probs = np.asarray(model.run({"logmel": logmel})[0])[0]
    model.close()
    # Top five AudioSet classes of the complete torch model for the same signal.
    for index, want in ((501, 0.63131), (393, 0.3308), (394, 0.1179), (503, 0.02494)):
        assert probs[index] == pytest.approx(want, abs=2e-3)


@real_model
@pytest.mark.gpu
def test_real_model_directml_matches_cpu() -> None:
    if not DirectMLAdapter.is_available():
        pytest.skip("DirectML is not available")
    rng = np.random.default_rng(3)
    x = (0.2 * rng.standard_normal(SR * 40)).astype(np.float32) + tone(40.0, 220.0)
    stream = tg.LogMelStream()
    logmel = np.concatenate([stream.push(x), stream.finish()])
    cpu, gpu = tg.Tagger(REAL_MODELS, "cpu"), tg.Tagger(REAL_MODELS, "directml")
    _, want, _ = cpu.tag(logmel)
    _, got, report = gpu.tag(logmel)
    cpu.close()
    gpu.close()
    assert report.backend == "directml" and report.notes == []  # strict, profiled, parity
    assert float(np.max(np.abs(got - want))) < 1e-3
