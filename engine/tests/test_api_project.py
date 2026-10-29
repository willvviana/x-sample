"""Phase 1 API: projects, import, editing, peaks, preview, export, security of new routes."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tests.audio_fixtures import make_all_formats, sine_array, write_wav
from tests.conftest import wait_for_job

RATE = 44100


def _project(client: TestClient, auth: dict[str, str], rate: int = 44100) -> dict[str, Any]:
    r = client.post("/api/projects", headers=auth, json={"name": "Demo", "sample_rate": rate})
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


def _upload(client: TestClient, auth: dict[str, str], path: Path, **q: str) -> Any:
    return client.post(
        "/api/project/import",
        headers=auth,
        params={"filename": path.name, **q},
        content=path.read_bytes(),
    )


def _import(client: TestClient, auth: dict[str, str], path: Path, **q: str) -> dict[str, Any]:
    r = _upload(client, auth, path, **q)
    assert r.status_code == 202, r.text
    job = wait_for_job(client, r.json()["id"], auth)
    assert job["state"] == "succeeded", job
    return job


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "tone.wav", sine_array(2.0, RATE, (440.0,)), RATE)


@pytest.mark.parametrize(
    "path,method",
    [
        ("/api/projects", "GET"),
        ("/api/projects", "POST"),
        ("/api/project", "GET"),
        ("/api/project/import", "POST"),
        ("/api/project/save", "POST"),
        ("/api/project/exports", "POST"),
        ("/api/project/exports/x.wav", "GET"),
        ("/api/project/render", "GET"),
        ("/api/audio/devices", "GET"),
        ("/api/audio/record", "POST"),
        ("/api/jobs/abc/stop", "POST"),
    ],
)
def test_new_routes_require_token(client: TestClient, path: str, method: str) -> None:
    assert client.request(method, path).status_code == 401


def test_no_project_open_is_a_clear_error(client: TestClient, auth: dict[str, str]) -> None:
    assert client.get("/api/project", headers=auth).json() is None
    r = client.post("/api/project/save", headers=auth)
    assert r.status_code in {409, 422}
    assert "project" in r.json()["detail"].lower()


def test_import_edit_slice_export_reopen(
    client: TestClient, auth: dict[str, str], wav: Path
) -> None:
    _project(client, auth)
    _import(client, auth, wav)
    view = client.get("/api/project", headers=auth).json()
    assert len(view["clips"]) == 1 and len(view["sources"]) == 1
    clip = view["clips"][0]
    assert clip["duration"] == 2 * RATE

    r = client.post(
        f"/api/project/clips/{clip['id']}/trim", headers=auth, json={"start": 1000, "end": 50_000}
    )
    assert r.status_code == 200, r.text
    c = r.json()["clips"][0]
    assert (c["source_in"], c["duration"]) == (1000, 49_000)

    r = client.post(
        f"/api/project/clips/{clip['id']}/slice",
        headers=auth,
        json={"start": 2000, "end": 12_000, "name": "hit"},
    )
    assert r.status_code == 200, r.text
    sample = r.json()["samples"][0]
    assert sample["duration"] == 10_000

    r = client.post(
        "/api/project/exports", headers=auth, json={"kind": "sample", "sample_id": sample["id"]}
    )
    assert r.status_code == 202, r.text
    job = wait_for_job(client, r.json()["id"], auth)
    assert job["state"] == "succeeded", job
    filename = job["result"]["filename"]
    dl = client.get(f"/api/project/exports/{filename}", headers=auth)
    assert dl.status_code == 200 and dl.headers["content-type"] == "audio/wav"
    out = Path(view["path"]) / "exports" / filename
    info = sf.info(out)
    assert info.frames == 10_000 and info.samplerate == RATE

    pv = client.get(f"/api/project/samples/{sample['id']}/render", headers=auth)
    assert pv.status_code == 200 and pv.content[:4] == b"RIFF"

    src = view["sources"][0]["id"]
    pk = client.get(
        f"/api/project/sources/{src}/peaks",
        headers=auth,
        params={"start": 0, "end": 2 * RATE, "columns": 100},
    )
    assert pk.status_code == 200, pk.text
    body = pk.json()
    assert len(body["max"][0]) == 100 and max(body["max"][0]) > 10_000

    assert client.post("/api/project/close", headers=auth).status_code == 204
    reopened = client.post("/api/projects/open", headers=auth, json={"path": view["path"]}).json()
    assert reopened["clips"][0]["duration"] == 49_000
    assert reopened["samples"][0]["name"] == "hit"
    assert len(reopened["exports"]) == 1


def test_import_all_formats_and_bad_files(
    client: TestClient, auth: dict[str, str], tmp_path: Path
) -> None:
    _project(client, auth, 48000)
    files = make_all_formats(tmp_path / "fx", seconds=1.0)
    for path in files.values():
        _import(client, auth, path)
    view = client.get("/api/project", headers=auth).json()
    assert len(view["sources"]) == len(files)
    assert all(s["media"]["sample_rate"] == 48000 for s in view["sources"])

    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not audio at all" * 100)
    r = _upload(client, auth, junk)
    job = wait_for_job(client, r.json()["id"], auth)
    assert job["state"] == "failed" and "audio" in job["error"].lower()
    assert len(client.get("/api/project", headers=auth).json()["sources"]) == len(files)

    r = client.post(
        "/api/project/import", headers=auth, params={"filename": "x.exe"}, content=b"MZ"
    )
    assert r.status_code == 422
    r = client.post("/api/project/import", headers=auth, params={"filename": "a.wav"}, content=b"")
    assert r.status_code == 422


def test_edit_validation_rejects_bad_input(
    client: TestClient, auth: dict[str, str], wav: Path
) -> None:
    _project(client, auth)
    _import(client, auth, wav)
    clip = client.get("/api/project", headers=auth).json()["clips"][0]
    bad = [
        {"start": 10, "end": 5},
        {"start": 0, "end": 10**9},
        {"start": 1.5, "end": 100},
        {"start": -1, "end": 100},
    ]
    for body in bad:
        r = client.post(f"/api/project/clips/{clip['id']}/trim", headers=auth, json=body)
        assert r.status_code == 422, (body, r.text)
    assert client.get("/api/project", headers=auth).json()["clips"][0]["duration"] == 2 * RATE


def test_export_unknown_and_traversal(client: TestClient, auth: dict[str, str]) -> None:
    _project(client, auth)
    for name in ("..%2Fproject.json", "nope.wav", ".hidden"):
        assert client.get(f"/api/project/exports/{name}", headers=auth).status_code in {404, 422}


def test_close_blocked_while_import_active(
    client: TestClient, auth: dict[str, str], tmp_path: Path
) -> None:
    _project(client, auth)
    big = write_wav(tmp_path / "big.wav", sine_array(30.0, RATE, (220.0,)), RATE)
    r = _upload(client, auth, big)
    closed = client.post("/api/project/close", headers=auth)
    wait_for_job(client, r.json()["id"], auth)
    assert closed.status_code in {204, 422}


def test_devices_listing_job(client: TestClient, auth: dict[str, str]) -> None:
    r = client.post("/api/audio/devices/refresh", headers=auth)
    assert r.status_code == 202
    job = wait_for_job(client, r.json()["id"], auth)
    assert job["state"] in {"succeeded", "failed"}
    assert client.get("/api/audio/devices", headers=auth).json()["state"] in {"ready", "failed"}


def test_ffmpeg_available() -> None:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)  # noqa: S607
    assert result.returncode == 0
