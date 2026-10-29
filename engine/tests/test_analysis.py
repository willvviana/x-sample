"""v0.2.0 musical analysis: DSP accuracy on synthetic fixtures with exact ground truth,
the sidecar/effective-view logic, manifest corrections, and the API through real workers.

Thresholds are acceptance floors measured on synthetic material only; real-audio accuracy is
measured separately by the evaluation script (not part of the unit suite, datasets are local).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.analysis_fixtures import (
    Spec,
    chord_overlap,
    f_measure,
    melody_note_accuracy,
    mirex_key_score,
    render,
)
from tests.audio_fixtures import write_wav
from tests.conftest import wait_for_job
from xsample.analysis.dsp import ANALYSIS_RATE, FeatureExtractor
from xsample.analysis.effective import apply_beat_grid, apply_chords, effective
from xsample.analysis.harmony import analyze_harmony
from xsample.analysis.rhythm import analyze_rhythm
from xsample.analysis.run import AnalysisCancelledError, analyze_file, cache_key
from xsample.analysis.schema import ANALYZER_VERSION, ChordOut
from xsample.project import store
from xsample.project.model import (
    AnalysisCorrections,
    BeatGridCorrection,
    ChordCorrection,
    KeyCorrection,
    Project,
    TagCorrection,
)

CASES = {
    "pop": Spec(bpm=120, key_root=0),
    "minor": Spec(bpm=95, key_root=9, minor=True),
    "rnb_sax": Spec(
        bpm=76, key_root=5, minor=True, groove="rnb", sevenths=True, lead="sax", bars=12
    ),
    "rnb_violin": Spec(bpm=96, key_root=8, groove="rnb", sevenths=True, lead="violin"),
    "piano_no_drums": Spec(bpm=90, key_root=1, minor=True, drums=False),
}


def _analyse(spec: Spec) -> tuple[Any, Any, Any]:
    y, truth = render(spec)
    fx = FeatureExtractor()
    for i in range(0, len(y), 65536):
        fx.push(y[i : i + 65536])
    features = fx.finish()
    rhythm = analyze_rhythm(features)
    harmony = analyze_harmony(features, rhythm.beats, rhythm.downbeats, rhythm.onsets)
    return truth, rhythm, harmony


@pytest.fixture(scope="module")
def analysed() -> dict[str, tuple[Any, Any, Any]]:
    return {name: _analyse(spec) for name, spec in CASES.items()}


@pytest.mark.parametrize("name", list(CASES))
def test_tempo_beats_and_downbeats(analysed: dict[str, Any], name: str) -> None:
    truth, rhythm, _ = analysed[name]
    true_bpm = (truth.bpm_start + truth.bpm_end) / 2
    assert abs(rhythm.bpm / true_bpm - 1) < 0.04, (rhythm.bpm, true_bpm)
    assert f_measure(rhythm.beats, truth.beats) >= 0.9
    assert f_measure(rhythm.downbeats, truth.downbeats) >= 0.85
    assert rhythm.meter == truth.meter


@pytest.mark.parametrize("name", list(CASES))
def test_key_and_chords(analysed: dict[str, Any], name: str) -> None:
    truth, _, harmony = analysed[name]
    top = harmony.key_candidates[0]
    assert mirex_key_score((top.tonic, top.mode), truth.key) == 1.0, top.label
    chords = [(c.start, c.end, c.root, c.quality) for c in harmony.chords]
    assert chord_overlap(chords, truth.chords, True) >= 0.7


@pytest.mark.parametrize("name", ["rnb_sax", "rnb_violin"])
def test_lead_melody_notes_and_vibrato(analysed: dict[str, Any], name: str) -> None:
    truth, _, harmony = analysed[name]
    notes = [(n.start, n.end, n.midi) for n in harmony.melody_notes]
    assert melody_note_accuracy(notes, truth.melody) >= 0.9
    assert sum(1 for n in harmony.melody_notes if n.vibrato_hz) >= len(truth.melody) // 2


def test_rnb_groove_is_syncopated(analysed: dict[str, Any]) -> None:
    _, rhythm, _ = analysed["rnb_sax"]
    _, straight, _ = analysed["piano_no_drums"]
    assert rhythm.syncopation is not None and straight.syncopation is not None
    assert rhythm.syncopation > straight.syncopation + 0.2


def test_silence_reports_no_rhythm_instead_of_inventing_one() -> None:
    fx = FeatureExtractor()
    fx.push(np.zeros(ANALYSIS_RATE * 10, np.float32))
    features = fx.finish()
    rhythm = analyze_rhythm(features)
    assert rhythm.bpm == 0 or rhythm.beats == [] or rhythm.confidence < 0.3
    harmony = analyze_harmony(features, rhythm.beats, rhythm.downbeats, rhythm.onsets)
    assert harmony.melody_notes == []
    assert all(c.root is None for c in harmony.chords)


# --- orchestrator, sidecar, effective view ---------------------------------------------------


def _chunks(y: np.ndarray) -> Any:
    def decode(_path: Path, **kw: Any) -> Iterator[np.ndarray]:
        step = kw["chunk_frames"]
        for i in range(0, len(y), step):
            if kw["cancelled"]():
                raise AnalysisCancelledError
            yield y[i : i + step, None]

    return decode


def test_orchestrator_converts_to_project_samples() -> None:
    y, truth = render(CASES["pop"])
    project_rate = 44100
    frames = len(y) * 2  # the working copy is at the project rate; analysis runs at 22050
    progress: list[float] = []
    result = analyze_file(
        Path("unused.wav"),
        source_id="src_0123456789ab",
        media_sha256="0" * 64,
        sample_rate=project_rate,
        frames=frames,
        progress=lambda f, _m: progress.append(f),
        cancelled=lambda: False,
        decode=_chunks(y),
    )
    assert progress == sorted(progress) and progress[-1] >= 0.9
    assert result.backend == "cpu" and result.analyzer_version == ANALYZER_VERSION
    assert result.cache_key == cache_key("0" * 64)
    beats = result.rhythm.beats
    assert all(isinstance(b, int) and 0 <= b < frames for b in beats)
    true_beats = [round(t * project_rate) for t in truth.beats]
    assert min(abs(beats[3] - t) for t in true_beats) < 0.07 * project_rate
    assert result.genres.status == "unavailable" and result.instruments.status == "unavailable"
    b = result.boundaries
    assert b.start.position == 0 and b.end.position == frames
    assert b.start.beat_offset is not None and abs(b.start.beat_offset) < 0.1
    assert b.start.chord_is_tonic is True
    json.loads(result.model_dump_json())  # serialisable


def test_orchestrator_cancels_between_chunks() -> None:
    y, _ = render(CASES["pop"])
    calls = {"n": 0}

    def cancelled() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    with pytest.raises(AnalysisCancelledError):
        analyze_file(
            Path("unused.wav"),
            source_id="src_0123456789ab",
            media_sha256="0" * 64,
            sample_rate=22050,
            frames=len(y),
            progress=lambda f, m: None,
            cancelled=cancelled,
            decode=_chunks(y),
        )


def test_cache_key_changes_with_audio() -> None:
    assert cache_key("a" * 64) != cache_key("b" * 64)
    assert cache_key("a" * 64) == cache_key("a" * 64)


def test_beat_grid_corrections() -> None:
    sr, frames = 1000, 10_000
    beats = list(range(500, 10_000, 500))  # 120 BPM at sr=1000
    downs = beats[::4]
    same = apply_beat_grid(beats, downs, 4, BeatGridCorrection(), sr, frames)
    assert same == (beats, downs, 4)
    double, _, _ = apply_beat_grid(beats, downs, 4, BeatGridCorrection(scale="double"), sr, frames)
    assert len(double) == 2 * len(beats) - 1 and double[1] == 750
    half, _, _ = apply_beat_grid(
        beats, downs, 4, BeatGridCorrection(scale="half", half_phase=1), sr, frames
    )
    assert half == beats[1::2]
    const, _, _ = apply_beat_grid(beats, downs, 4, BeatGridCorrection(bpm=60.0), sr, frames)
    assert np.all(np.diff(const) == 1000) and const[0] == 500
    shifted, s_down, meter = apply_beat_grid(
        beats, downs, 4, BeatGridCorrection(offset=100, meter=3, downbeat_shift=1), sr, frames
    )
    assert shifted[0] == 600 and meter == 3 and s_down[:2] == [shifted[1], shifted[4]]


def test_chord_corrections_split_detected_segments() -> None:
    def chord(a: int, z: int, root: int) -> ChordOut:
        return ChordOut(
            start=a, end=z, root=root, quality="maj", bass=root, label="x", confidence=0.5
        )

    detected = [chord(0, 100, 0), chord(100, 200, 7)]
    out = apply_chords(detected, [ChordCorrection(start=50, end=150, root=9, quality="min")])
    assert [(c.start, c.end, c.root, c.label) for c in out] == [
        (0, 50, 0, "x"),
        (50, 150, 9, "Am"),
        (150, 200, 7, "x"),
    ]


def test_effective_view_applies_key_tags_and_marks_corrections() -> None:
    y, _ = render(CASES["minor"])
    result = analyze_file(
        Path("unused.wav"),
        source_id="src_0123456789ab",
        media_sha256="1" * 64,
        sample_rate=22050,
        frames=len(y),
        progress=lambda f, m: None,
        cancelled=lambda: False,
        decode=_chunks(y),
    )
    plain = effective(result, AnalysisCorrections())
    assert plain.corrected == [] and plain.key is not None and plain.key.label == "A minor"
    corr = AnalysisCorrections(
        key=KeyCorrection(tonic=0, mode="major"),
        tags=[
            TagCorrection(kind="genre", label="R&B"),
            TagCorrection(kind="instrument", label="Piano"),
        ],
        beat_grid=BeatGridCorrection(scale="double"),
    )
    eff = effective(result, corr)
    assert eff.key is not None and eff.key.label == "C major"
    assert [t.label for t in eff.tags] == ["R&B", "Piano"] and eff.tags[0].origin == "user"
    assert set(eff.corrected) == {"key", "tags", "beat_grid"}
    assert abs(eff.bpm - 2 * plain.bpm) < 2


# --- manifest -------------------------------------------------------------------------------


def test_chord_correction_validation() -> None:
    with pytest.raises(ValueError):
        ChordCorrection(start=10, end=5, root=0, quality="maj")
    with pytest.raises(ValueError):
        ChordCorrection(start=0, end=5, root=0, quality=None)
    ChordCorrection(start=0, end=5)  # explicit "no chord"


def test_format_1_manifest_loads_and_is_upgraded(tmp_path: Path) -> None:
    root = tmp_path / "p"
    store.create_project(root, "Old", 44100)
    manifest = root / "project.json"
    raw = json.loads(manifest.read_text("utf-8"))
    raw["format_version"] = 1
    raw.pop("analysis", None)
    manifest.write_text(json.dumps(raw), "utf-8")
    loaded = store.load_project(root).project
    assert loaded.analysis == {}
    store.save_project(root, loaded)
    assert json.loads(manifest.read_text("utf-8"))["format_version"] == 2


def test_corrections_for_unknown_source_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "p"
    store.create_project(root, "P", 44100)
    raw = json.loads((root / "project.json").read_text("utf-8"))
    raw["analysis"] = {"src_0123456789ab": {}}
    with pytest.raises(ValueError):
        Project.model_validate(raw)


# --- API through real worker processes ---------------------------------------------------------


@pytest.mark.parametrize(
    "path,method",
    [
        ("/api/project/sources/src_0123456789ab/analysis", "GET"),
        ("/api/project/sources/src_0123456789ab/analysis", "POST"),
        ("/api/project/sources/src_0123456789ab/analysis/corrections", "PUT"),
        ("/api/project/sources/src_0123456789ab/analysis/corrections", "DELETE"),
    ],
)
def test_analysis_routes_require_token(client: TestClient, path: str, method: str) -> None:
    assert client.request(method, path).status_code == 401


def _project_with_source(client: TestClient, auth: dict[str, str], tmp_path: Path) -> str:
    y, _ = render(CASES["rnb_sax"])
    wav = write_wav(tmp_path / "rnb sax é.wav", y[:, None], 22050)
    r = client.post("/api/projects", headers=auth, json={"name": "Analysis", "sample_rate": 44100})
    assert r.status_code == 200, r.text
    r = client.post(
        "/api/project/import", headers=auth, params={"filename": wav.name}, content=wav.read_bytes()
    )
    assert r.status_code == 202, r.text
    assert wait_for_job(client, r.json()["id"], auth)["state"] == "succeeded"
    source_id: str = client.get("/api/project", headers=auth).json()["sources"][0]["id"]
    return source_id


def _wait_ready(client: TestClient, auth: dict[str, str], url: str) -> dict[str, Any]:
    view: dict[str, Any] = client.get(url, headers=auth).json()
    if view["job"]:
        wait_for_job(client, view["job"]["id"], auth, timeout=120)
    view = client.get(url, headers=auth).json()
    return view


def test_analysis_job_cache_corrections_and_rerun(
    client: TestClient, auth: dict[str, str], tmp_path: Path
) -> None:
    source_id = _project_with_source(client, auth, tmp_path)
    url = f"/api/project/sources/{source_id}/analysis"
    assert client.get(url, headers=auth).json()["status"] == "none"
    r = client.post(url, headers=auth, json={})
    assert r.status_code == 200 and r.json()["status"] in ("queued", "running")
    view = _wait_ready(client, auth, url)
    assert view["status"] == "ready", view
    result = view["result"]
    assert result["backend"] == "cpu" and result["sample_rate"] == 44100
    assert abs(result["rhythm"]["bpm"] - 76) < 3
    assert view["effective"]["key"]["label"] == "F minor"
    assert result["genres"]["status"] == "unavailable"
    job = view["job"]
    assert job["uses_gpu"] is False and job["result"]["backend"] == "cpu"

    # Cached: a second POST without force does not start another job.
    again = client.post(url, headers=auth, json={}).json()
    assert again["status"] == "ready" and again["job"]["id"] == job["id"]

    corrections = {
        "beat_grid": {"scale": "half"},
        "key": {"tonic": 8, "mode": "major"},
        "chords": [{"start": 0, "end": 44100, "root": 5, "quality": "min7", "bass": 5}],
        "tags": [{"kind": "genre", "label": "R&B"}, {"kind": "instrument", "label": "Saxophone"}],
    }
    r = client.put(f"{url}/corrections", headers=auth, json=corrections)
    assert r.status_code == 200, r.text
    eff = r.json()["effective"]
    assert eff["key"]["label"] == "Ab major" and eff["chords"][0]["label"] == "Fmin7"
    assert set(eff["corrected"]) == {"beat_grid", "key", "chords", "tags"}

    # Re-running keeps the corrections (they live in the manifest, not the cache).
    client.post(url, headers=auth, json={"force": True})
    view = _wait_ready(client, auth, url)
    assert view["status"] == "ready" and view["job"]["id"] != job["id"]
    assert view["corrections"]["key"] == {"tonic": 8, "mode": "major"}
    sidecars = list(
        (
            Path(client.get("/api/project", headers=auth).json()["path"])
            / "analysis"
            / "results"
            / source_id
        ).glob("*.json")
    )
    assert [p.stem for p in sidecars] == [view["cache_key"]]

    # Invalid corrections are rejected with a clear error and change nothing.
    bad = client.put(
        f"{url}/corrections",
        headers=auth,
        json={"chords": [{"start": 0, "end": 10**9, "root": 0, "quality": "maj"}]},
    )
    assert bad.status_code in (409, 422)
    dup = client.put(
        f"{url}/corrections",
        headers=auth,
        json={"tags": [{"kind": "genre", "label": "Soul"}, {"kind": "genre", "label": "soul"}]},
    )
    assert dup.status_code in (409, 422)
    assert client.get(url, headers=auth).json()["corrections"]["key"]["tonic"] == 8

    # Reset, then reopen: the manifest is the source of truth for corrections.
    r = client.delete(f"{url}/corrections", headers=auth)
    assert r.json()["corrections"]["key"] is None and r.json()["effective"]["corrected"] == []
    client.put(f"{url}/corrections", headers=auth, json={"key": {"tonic": 5, "mode": "minor"}})
    path = client.get("/api/project", headers=auth).json()["path"]
    assert client.post("/api/project/close", headers=auth).status_code == 204
    client.post("/api/projects/open", headers=auth, json={"path": path})
    reopened = client.get(url, headers=auth).json()
    assert reopened["status"] == "ready" and reopened["corrections"]["key"]["tonic"] == 5


def test_analysis_job_can_be_cancelled(
    client: TestClient, auth: dict[str, str], tmp_path: Path
) -> None:
    source_id = _project_with_source(client, auth, tmp_path)
    url = f"/api/project/sources/{source_id}/analysis"
    job_id = client.post(url, headers=auth, json={}).json()["job"]["id"]
    client.post(f"/api/jobs/{job_id}/cancel", headers=auth)
    final = wait_for_job(client, job_id, auth, timeout=60)
    view = client.get(url, headers=auth).json()
    if final["state"] == "cancelled":
        assert view["status"] == "cancelled" and view["result"] is None
    else:  # finished before the cancel arrived; still a consistent result
        assert final["state"] == "succeeded" and view["status"] == "ready"


def test_unknown_source_is_404(client: TestClient, auth: dict[str, str], tmp_path: Path) -> None:
    _project_with_source(client, auth, tmp_path)
    r = client.get("/api/project/sources/src_ffffffffffff/analysis", headers=auth)
    assert r.status_code == 404
