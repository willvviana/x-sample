"""Real-hardware recording verification (opt-in: `uv run pytest -m hardware`).

These tests open the default microphone and a WASAPI loopback device on the machine they run
on. They never fabricate captured audio: the microphone test records whatever the room
produces; the loopback test plays a short, quiet 440 Hz tone on the default output (so that
there is system audio to capture) and checks that the loopback recording contains it.
Recorded audio is deleted when the test ends; only measurements are kept, in
`test-reports/hardware_recording.json`.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tests.conftest import wait_for_job

pytestmark = pytest.mark.hardware

REPORT = Path(__file__).resolve().parents[1] / "test-reports" / "hardware_recording.json"
TONE_HZ = 440.0
MIC_SECONDS = float(os.environ.get("XSAMPLE_HW_SECONDS", "3"))  # raise for a stability run
TONE_AMPLITUDE = 0.02  # quiet: about -34 dBFS


def _record_report(key: str, value: dict[str, Any]) -> None:
    REPORT.parent.mkdir(exist_ok=True)
    data: dict[str, Any] = {}
    if REPORT.exists():
        data = json.loads(REPORT.read_text(encoding="utf-8"))
    data[key] = value
    REPORT.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _devices(client: TestClient, auth: dict[str, str]) -> dict[str, Any]:
    job = client.post("/api/audio/devices/refresh", headers=auth).json()
    assert wait_for_job(client, job["id"], auth)["state"] == "succeeded"
    devices: dict[str, Any] = client.get("/api/audio/devices", headers=auth).json()
    return devices


def _play_tone(output_name: str, seconds: float, started: threading.Event) -> None:
    """Plays a quiet tone on the render device that the loopback device mirrors."""
    import pyaudiowpatch as pyaudio

    p = pyaudio.PyAudio()
    try:
        base = output_name.removesuffix(" [Loopback]")
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        info = next(
            p.get_device_info_by_index(i)
            for i in range(p.get_device_count())
            if p.get_device_info_by_index(i)["name"] == base
            and p.get_device_info_by_index(i)["hostApi"] == wasapi["index"]
            and p.get_device_info_by_index(i)["maxOutputChannels"] > 0
            and not p.get_device_info_by_index(i).get("isLoopbackDevice")
        )
        rate = int(info["defaultSampleRate"])
        channels = int(info["maxOutputChannels"])
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            output=True,
            output_device_index=int(info["index"]),
        )
        t = np.arange(int(seconds * rate)) / rate
        mono = (TONE_AMPLITUDE * np.sin(2 * math.pi * TONE_HZ * t)).astype(np.float32)
        block = 2048
        started.set()
        for i in range(0, len(mono), block):
            chunk = mono[i : i + block]
            stream.write(np.repeat(chunk[:, None], channels, axis=1).tobytes())
        stream.stop_stream()
        stream.close()
    finally:
        p.terminate()


def _run_recording(
    client: TestClient,
    auth: dict[str, str],
    device: dict[str, Any],
    seconds: float,
    during: Any = None,
) -> dict[str, Any]:
    r = client.post(
        "/api/audio/record",
        headers=auth,
        json={
            "device_index": device["index"],
            "device_name": device["name"],
            "kind": device["kind"],
            "max_seconds": 60,
        },
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["id"]
    # Wait for the stream to deliver real frames.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}", headers=auth).json()
        assert job["state"] in {"queued", "running"}, job
        if (job.get("data") or {}).get("captured_s", 0) > 0.2:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("no audio frames arrived within 15 s")
    thread = None
    if during is not None:
        thread = threading.Thread(target=during, daemon=True)
        thread.start()
    time.sleep(seconds)
    telemetry = client.get(f"/api/jobs/{job_id}", headers=auth).json().get("data") or {}
    assert client.post(f"/api/jobs/{job_id}/stop", headers=auth).status_code == 200
    done = wait_for_job(client, job_id, auth, timeout=30)
    if thread is not None:
        thread.join(timeout=30)
    assert done["state"] == "succeeded", done
    result: dict[str, Any] = done["result"]
    result["_telemetry"] = telemetry
    return result


def _verify_and_import(
    client: TestClient, auth: dict[str, str], result: dict[str, Any], label: str
) -> dict[str, Any]:
    info = sf.info(result["path"]) if Path(result["path"]).exists() else None
    # The engine chains an import job when recording finishes; wait for the source to appear.
    deadline = time.monotonic() + 60
    view: dict[str, Any] = {}
    while time.monotonic() < deadline:
        view = client.get("/api/project", headers=auth).json()
        if any(s["origin"] == "recording" for s in view["sources"]):
            break
        time.sleep(0.2)
    else:
        raise AssertionError("recording was not imported into the project")
    source = next(s for s in view["sources"] if s["origin"] == "recording")
    original = Path(view["path"]) / source["stored_path"]
    original_info = sf.info(original)
    assert original_info.format == "WAV"
    assert original_info.frames == result["frames"] and original_info.frames > 0
    assert original_info.samplerate == result["sample_rate"]
    data, _ = sf.read(original, dtype="float32", always_2d=True)
    assert np.isfinite(data).all()
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    duration = original_info.frames / original_info.samplerate
    assert abs(duration - result["captured_s"]) < 0.01
    del info
    return {
        "label": label,
        "device": source["device"],
        "frames": int(original_info.frames),
        "sample_rate": int(original_info.samplerate),
        "channels": int(original_info.channels),
        "duration_s": round(duration, 3),
        "wall_s": round(result["elapsed_s"], 3),
        "peak_dbfs": round(20 * math.log10(max(peak, 1e-12)), 1),
        "rms_dbfs": round(20 * math.log10(max(rms, 1e-12)), 1),
        "dropped_blocks": result["dropped_blocks"],
        "clip_samples": result["clip_samples"],
        "_data": data,
        "_peak": peak,
        "_view": view,
    }


def _reopen_and_play(
    client: TestClient, auth: dict[str, str], view: dict[str, Any]
) -> dict[str, Any]:
    assert client.post("/api/project/close", headers=auth).status_code == 204
    reopened = client.post("/api/projects/open", headers=auth, json={"path": view["path"]}).json()
    assert any(s["origin"] == "recording" for s in reopened["sources"])
    length = reopened["length_samples"]
    assert length > 0
    render = client.get(
        "/api/project/render", headers=auth, params={"start": 0, "end": min(length, 10 * 48000)}
    )
    assert render.status_code == 200
    import io

    audio, rate = sf.read(io.BytesIO(render.content), dtype="float32", always_2d=True)
    return {
        "playback_frames": int(audio.shape[0]),
        "playback_rate": int(rate),
        "playback_peak": float(np.max(np.abs(audio))),
    }


@pytest.fixture
def project(client: TestClient, auth: dict[str, str]) -> Any:
    r = client.post("/api/projects", headers=auth, json={"name": "HW", "sample_rate": 48000})
    assert r.status_code == 200
    root = Path(r.json()["path"])
    yield r.json()
    client.post("/api/project/close", headers=auth)
    shutil.rmtree(root, ignore_errors=True)  # no recorded audio is left behind


def test_microphone_recording(client: TestClient, auth: dict[str, str], project: Any) -> None:
    devices = _devices(client, auth)
    mics = devices["microphones"]
    if not mics:
        _record_report("microphone", {"status": "unverified", "reason": "no microphone found"})
        pytest.skip("no microphone available")
    mic = next((m for m in mics if m["is_default"]), mics[0])
    try:
        result = _run_recording(client, auth, mic, seconds=MIC_SECONDS)
    except AssertionError as exc:
        _record_report("microphone", {"status": "failed", "device": mic["name"], "error": str(exc)})
        raise
    m = _verify_and_import(client, auth, result, "microphone")
    play = _reopen_and_play(client, auth, m["_view"])
    # Non-silence: any real input device has a noise floor well above digital zero.
    non_silent = m["_peak"] > 1e-4
    report = {k: v for k, v in m.items() if not k.startswith("_")}
    report.update(play)
    report["non_silent"] = non_silent
    report["status"] = "verified" if non_silent else "recorded-but-silent"
    _record_report("microphone", report)
    assert non_silent, f"microphone recording is silent (peak {m['_peak']:.2e}); is it muted?"
    assert play["playback_peak"] > 1e-4
    assert m["dropped_blocks"] == 0


def test_loopback_recording(client: TestClient, auth: dict[str, str], project: Any) -> None:
    devices = _devices(client, auth)
    loops = devices["loopbacks"]
    if not loops:
        _record_report("loopback", {"status": "unverified", "reason": "no loopback device"})
        pytest.skip("no loopback device available")
    loop = next((d for d in loops if d["is_default"]), loops[0])
    started = threading.Event()
    tone_seconds = 3.0
    extra = max(0.0, MIC_SECONDS - tone_seconds)
    result = _run_recording(
        client,
        auth,
        loop,
        seconds=tone_seconds + 0.8 + extra,
        during=lambda: _play_tone(loop["name"], tone_seconds, started),
    )
    m = _verify_and_import(client, auth, result, "loopback")
    play = _reopen_and_play(client, auth, m["_view"])
    data = m["_data"]
    rate = m["sample_rate"]
    mono = data.mean(axis=1)
    window = mono[: rate * 3]
    spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
    freqs = np.fft.rfftfreq(len(window), 1 / rate)
    peak_freq = float(freqs[int(np.argmax(spectrum))])
    # The loopback must have kept a continuous timeline (silence included).
    drift = abs(m["duration_s"] - m["wall_s"])
    report = {k: v for k, v in m.items() if not k.startswith("_")}
    report.update(play)
    report["spectral_peak_hz"] = round(peak_freq, 1)
    report["captured_vs_wall_s"] = round(drift, 3)
    report["tone_played"] = {"hz": TONE_HZ, "amplitude": TONE_AMPLITUDE, "seconds": tone_seconds}
    ok = abs(peak_freq - TONE_HZ) < 5 and m["_peak"] > 1e-3
    report["status"] = "verified" if ok else "failed"
    _record_report("loopback", report)
    assert started.is_set()
    assert ok, f"loopback spectral peak {peak_freq:.1f} Hz, level {m['_peak']:.2e}"
    assert play["playback_peak"] > 1e-3
