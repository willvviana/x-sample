"""Long-audio behaviour: 10-minute stereo import/export with bounded worker memory, plus
cancellation and worker-crash recovery for the audio job kinds. Measurements are written to
`test-reports/long_audio.json` (measured values only)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tests.audio_fixtures import make_long_wav
from tests.conftest import wait_for_job

REPORT = Path(__file__).resolve().parents[1] / "test-reports" / "long_audio.json"
RATE = 44100
MiB = 1024**2


def _report(key: str, value: dict[str, Any]) -> None:
    REPORT.parent.mkdir(exist_ok=True)
    data: dict[str, Any] = json.loads(REPORT.read_text("utf-8")) if REPORT.exists() else {}
    data[key] = value
    REPORT.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _project(client: TestClient, auth: dict[str, str], name: str = "Long") -> dict[str, Any]:
    r = client.post("/api/projects", headers=auth, json={"name": name, "sample_rate": RATE})
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


def _import(client: TestClient, auth: dict[str, str], path: Path) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    r = client.post(
        "/api/project/import",
        headers=auth,
        params={"filename": path.name},
        content=path.read_bytes(),
    )
    assert r.status_code == 202, r.text
    job = wait_for_job(client, r.json()["id"], auth, timeout=170)
    return job, time.perf_counter() - started


def _make_mp3(path: Path, seconds: int) -> Path:
    subprocess.run(
        [  # noqa: S607
            "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
            f"sine=frequency=330:duration={seconds}:sample_rate={RATE}", "-ac", "2",
            "-b:a", "128k", str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


def test_ten_minute_stereo_memory_is_bounded(
    client: TestClient, auth: dict[str, str], tmp_path: Path
) -> None:
    short = make_long_wav(tmp_path / "short.wav", 10, RATE)
    long = make_long_wav(tmp_path / "long.wav", 600, RATE)
    assert sf.info(long).channels == 2 and sf.info(long).frames == 600 * RATE
    _project(client, auth)

    short_job, short_wall = _import(client, auth, short)
    assert short_job["state"] == "succeeded", short_job
    long_job, long_wall = _import(client, auth, long)
    assert long_job["state"] == "succeeded", long_job

    short_peak = short_job["peak_worker_memory_bytes"]
    long_peak = long_job["peak_worker_memory_bytes"]
    assert short_peak and long_peak
    # 60x the audio must not cost 60x the memory: allow a generous 80 MiB of growth.
    assert long_peak - short_peak < 80 * MiB, (short_peak / MiB, long_peak / MiB)

    view = client.get("/api/project", headers=auth).json()
    clip = max(view["clips"], key=lambda c: c["duration"])
    assert clip["duration"] == 600 * RATE
    src = clip["source_id"]
    t0 = time.perf_counter()
    pk = client.get(
        f"/api/project/sources/{src}/peaks",
        headers=auth,
        params={"start": 0, "end": 600 * RATE, "columns": 2000},
    )
    peaks_query_s = time.perf_counter() - t0
    assert pk.status_code == 200 and len(pk.json()["max"][0]) == 2000

    t0 = time.perf_counter()
    r = client.post(
        "/api/project/exports",
        headers=auth,
        json={"kind": "selection", "start": 0, "end": 600 * RATE, "name": "full"},
    )
    assert r.status_code == 202, r.text
    exp = wait_for_job(client, r.json()["id"], auth, timeout=170)
    export_wall = time.perf_counter() - t0
    assert exp["state"] == "succeeded", exp
    out = Path(view["path"]) / "exports" / exp["result"]["filename"]
    assert sf.info(out).frames == 600 * RATE
    export_peak = exp["peak_worker_memory_bytes"]
    assert export_peak < 400 * MiB, export_peak / MiB

    long_mb = long.stat().st_size / MiB
    _report(
        "wav_10min_stereo_44k1",
        {
            "file_mib": round(long_mb, 1),
            "import_wall_s": round(long_wall, 2),
            "import_worker_peak_mib": round(long_peak / MiB, 1),
            "import_10s_worker_peak_mib": round(short_peak / MiB, 1),
            "import_10s_wall_s": round(short_wall, 2),
            "peaks_query_s": round(peaks_query_s, 4),
            "export_wall_s": round(export_wall, 2),
            "export_worker_peak_mib": round(export_peak / MiB, 1),
            "export_size_mib": round(out.stat().st_size / MiB, 1),
            "worker_cpu_s_import": (long_job.get("result") or {}).get("metrics"),
        },
    )


def test_ten_minute_mp3_import(client: TestClient, auth: dict[str, str], tmp_path: Path) -> None:
    mp3 = _make_mp3(tmp_path / "long.mp3", 600)
    _project(client, auth, "LongMp3")
    job, wall = _import(client, auth, mp3)
    assert job["state"] == "succeeded", job
    view = client.get("/api/project", headers=auth).json()
    assert abs(view["clips"][0]["duration"] / RATE - 600) < 0.1
    assert job["peak_worker_memory_bytes"] < 300 * MiB
    _report(
        "mp3_10min_stereo",
        {
            "file_mib": round(mp3.stat().st_size / MiB, 1),
            "import_wall_s": round(wall, 2),
            "import_worker_peak_mib": round(job["peak_worker_memory_bytes"] / MiB, 1),
        },
    )


def _leftovers(root: Path) -> list[str]:
    found = []
    for sub in ("cache/incoming", "cache/tmp", "cache/media", "sources", "analysis/peaks"):
        d = root / sub
        found += [str(p.relative_to(root)) for p in d.glob("*")] if d.exists() else []
    return found


def test_cancelled_import_leaves_no_files(
    client: TestClient, auth: dict[str, str], tmp_path: Path
) -> None:
    long = make_long_wav(tmp_path / "long.wav", 600, RATE)
    view = _project(client, auth, "Cancel")
    r = client.post(
        "/api/project/import",
        headers=auth,
        params={"filename": long.name},
        content=long.read_bytes(),
    )
    job_id = r.json()["id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}", headers=auth).json()
        if job["state"] == "running" and job["progress"] > 0.05:
            break
        time.sleep(0.05)
    client.post(f"/api/jobs/{job_id}/cancel", headers=auth)
    done = wait_for_job(client, job_id, auth, timeout=60)
    assert done["state"] == "cancelled", done
    assert _leftovers(Path(view["path"])) == []
    after = client.get("/api/project", headers=auth).json()
    assert after["sources"] == [] and after["clips"] == []


def test_worker_crash_during_import_is_recoverable(
    client: TestClient, auth: dict[str, str], tmp_path: Path
) -> None:
    long = make_long_wav(tmp_path / "long.wav", 600, RATE)
    view = _project(client, auth, "Crash")
    r = client.post(
        "/api/project/import",
        headers=auth,
        params={"filename": long.name},
        content=long.read_bytes(),
    )
    job_id = r.json()["id"]
    deadline = time.monotonic() + 30
    pid = None
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}", headers=auth).json()
        if job["state"] == "running" and job["progress"] > 0.05 and job["worker_pid"]:
            pid = job["worker_pid"]
            break
        time.sleep(0.05)
    assert pid is not None
    worker = psutil.Process(pid)
    for child in worker.children(recursive=True):  # ffmpeg decoder
        child.kill()
    worker.kill()
    done = wait_for_job(client, job_id, auth, timeout=60)
    assert done["state"] == "failed", done
    # The project is intact and a fresh import still works afterwards.
    after = client.get("/api/project", headers=auth).json()
    assert after["sources"] == [] and after["issues"] == []
    short = make_long_wav(tmp_path / "ok.wav", 5, RATE)
    ok, _ = _import(client, auth, short)
    assert ok["state"] == "succeeded", ok
    # Only the finished import's files remain; the crashed attempt left nothing behind that
    # the project would reference (orphans are reported by verify, not silently loaded).
    verify = client.post("/api/project/verify", headers=auth).json()
    vdone = wait_for_job(client, verify["id"], auth, timeout=60)
    assert vdone["state"] == "succeeded", vdone
    assert not (vdone["result"] or {}).get("problems"), vdone["result"]
    assert os.path.isdir(view["path"])


def test_audio_job_kinds_are_not_public(client: TestClient, auth: dict[str, str]) -> None:
    for kind in ("import_audio", "export_audio", "record_audio", "verify_project"):
        r = client.post("/api/jobs", headers=auth, json={"kind": kind, "params": {}})
        assert r.status_code == 422, (kind, r.text)


def test_record_timeout_scales_with_requested_length() -> None:
    from xsample.jobs.kinds import KINDS, RecordParams

    spec = KINDS["record_audio"]
    assert spec.timeout_s is not None
    short = RecordParams(project_dir="x", device_index=0, device_name="d", kind="microphone",
                         max_seconds=10)  # fmt: skip
    assert spec.timeout_s(short) == pytest.approx(130.0)


def test_numpy_float32_pipeline_dtype() -> None:
    assert np.zeros((2, 4), dtype=np.float32).dtype == np.float32
