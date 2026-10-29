"""Task implementations that run inside worker processes.

Each task receives a context (progress reporting, cancellation) and its validated
params, and returns a JSON-serialisable result.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any, Protocol


class TaskCancelledError(Exception):
    pass


class TaskContext(Protocol):
    def progress(
        self, fraction: float, message: str | None = None, data: dict[str, Any] | None = None
    ) -> None: ...

    def cancelled(self) -> bool: ...

    def stop_requested(self) -> bool: ...


def detect_backends(ctx: TaskContext, params: dict[str, Any]) -> dict[str, Any]:
    from xsample.runtime import vulkan, win_gpu
    from xsample.runtime.adapters import DirectMLAdapter, runtime_version

    ctx.progress(0.1, "enumerating adapters")
    try:
        adapters = [
            {
                "index": a.index,
                "name": a.name,
                "vendor_id": a.vendor_id,
                "dedicated_video_memory": a.dedicated_video_memory,
                "software": a.software,
            }
            for a in win_gpu.dxgi_adapters()
        ]
        adapter_error = None
    except OSError as exc:
        adapters, adapter_error = [], str(exc)
    hardware = [a for a in adapters if not a["software"]]

    ctx.progress(0.4, "checking DirectML")
    directml: dict[str, Any]
    if not DirectMLAdapter.is_available():
        directml = {"state": "unavailable", "detail": "DirectML execution provider not installed"}
    elif not hardware:
        directml = {"state": "unavailable", "detail": "no hardware DirectX 12 adapter found"}
    else:
        directml = {
            "state": "available",
            "detail": "DirectML execution provider present; models must pass a probe",
            "devices": [a["name"] for a in hardware],
        }

    ctx.progress(0.7, "checking Vulkan")
    vk = vulkan.detect()
    vk_devices = [d.name for d in vk.devices if d.device_type != "cpu"]
    vulkan_state = {
        "state": "available" if vk_devices else "unavailable",
        "detail": (
            "Vulkan driver present; no Vulkan model runner installed, so no model support "
            "is claimed"
            if vk_devices
            else (vk.error or "no Vulkan devices")
        ),
        "devices": vk_devices,
        "instance_version": vk.instance_version,
    }

    return {
        "runtime_version": runtime_version(),
        "dxgi_adapters": adapters,
        "dxgi_error": adapter_error,
        "backends": {
            "cpu": {"state": "available", "detail": "ONNX Runtime CPU execution provider"},
            "directml": directml,
            "vulkan": vulkan_state,
        },
    }


def probe(ctx: TaskContext, params: dict[str, Any]) -> dict[str, Any]:
    from xsample.runtime.probe import run_probe

    ctx.progress(0.1, f"probing {params['backend']}")
    result = run_probe(
        params["backend"],
        device_id=int(params.get("device_id", 0)),
        cache_path=params.get("cache_path"),
        force=bool(params.get("force", False)),
    )
    ctx.progress(1.0)
    return result


def smoke_inference(ctx: TaskContext, params: dict[str, Any]) -> dict[str, Any]:
    from xsample.runtime.inference_check import CheckCancelledError, run_check

    chunk = int(params["chunk_samples"])
    sample_rate = int(params.get("sample_rate", 44100))
    chunks = max(1, round(float(params["seconds"]) * sample_rate / chunk))
    last = [0.0]

    def on_progress(done: int, total: int) -> None:
        now = time.monotonic()
        if done == total or now - last[0] > 0.1:
            last[0] = now
            ctx.progress(done / total, f"chunk {done}/{total}")

    try:
        return run_check(
            params["backend"],
            chunk_samples=chunk,
            chunks=chunks,
            strict=bool(params.get("strict", True)),
            device_id=int(params.get("device_id", 0)),
            verify_every=int(params.get("verify_every", 8)),
            sample_rate=sample_rate,
            progress=on_progress,
            cancelled=ctx.cancelled,
        )
    except CheckCancelledError as exc:
        raise TaskCancelledError() from exc


# --- audio / project tasks (Phase 1) ---------------------------------------------------


def import_audio(ctx: TaskContext, params: dict[str, Any]) -> dict[str, Any]:
    from xsample.audio import media

    return media.import_audio(
        params["project_dir"],
        params["incoming_path"],
        params["original_name"],
        params["sample_rate"],
        params["channels"],
        params["origin"],
        params.get("device"),
        ctx.progress,
        ctx.cancelled,
        rebuild_stored_path=params.get("rebuild_stored_path"),
    )


def export_audio(ctx: TaskContext, params: dict[str, Any]) -> dict[str, Any]:
    from pathlib import Path

    from xsample.audio import render
    from xsample.project import layout

    return render.export_audio(
        render.RenderPlan.from_dict(params["plan"]),
        params["start"],
        params["end"],
        Path(params["project_dir"]) / layout.EXPORTS,
        layout.safe_stem(params["name"]),
        params["format"],
        params["subtype"],
        ctx.progress,
        ctx.cancelled,
        params.get("describe", {}),
    )


def record_audio(ctx: TaskContext, params: dict[str, Any]) -> dict[str, Any]:
    from xsample.audio import capture

    return capture.record(ctx, params)


def list_audio_devices(ctx: TaskContext, params: dict[str, Any]) -> dict[str, Any]:
    from xsample.audio import capture

    ctx.progress(0.2, "Listing audio devices")
    return capture.list_devices()


def verify_project(ctx: TaskContext, params: dict[str, Any]) -> dict[str, Any]:
    from xsample.project import verify

    return verify.verify_sources(
        params["project_dir"], params["sources"], ctx.progress, ctx.cancelled
    )


def analyze_audio(ctx: TaskContext, params: dict[str, Any]) -> dict[str, Any]:
    from pathlib import Path

    from xsample.analysis import beat_model, run, tagger
    from xsample.project import layout

    root = Path(params["project_dir"])
    media = layout.resolve_inside(root, params["media_path"])
    model_dir = params.get("beat_model_dir")
    tagger_dir = params.get("tagger_dir")
    key = run.cache_key(
        params["media_sha256"],
        beat_model.identity() if model_dir is not None else None,
        tagger.identity() if tagger_dir is not None else None,
    )
    out = layout.resolve_inside(root, f"{layout.ANALYSIS_RESULTS}/{params['source_id']}/{key}.json")
    ctx.progress(0.0, "Reading audio")
    try:
        result = run.analyze_file(
            media,
            source_id=params["source_id"],
            media_sha256=params["media_sha256"],
            sample_rate=int(params["sample_rate"]),
            frames=int(params["frames"]),
            progress=lambda f, m: ctx.progress(f, m),
            cancelled=ctx.cancelled,
            beat_model_dir=Path(model_dir) if model_dir is not None else None,
            beat_backend=params.get("beat_backend", "cpu"),
            tagger_dir=Path(tagger_dir) if tagger_dir is not None else None,
            tagger_backend=params.get("tagger_backend", "cpu"),
        )
    except run.AnalysisCancelledError as exc:
        raise TaskCancelledError() from exc
    if ctx.cancelled():
        raise TaskCancelledError()
    run.write_sidecar(out, result)
    # Sidecars for older audio or analyzer versions are invalid now; corrections are elsewhere.
    for old in out.parent.glob("*.json"):
        if old != out:
            old.unlink(missing_ok=True)
    ctx.progress(1.0, "Done")
    return {
        "source_id": params["source_id"],
        "cache_key": key,
        "backend": result.backend,
        "beat_tracker": result.beat_tracker.model_dump(),
        "timings": result.timings.model_dump(),
    }


# --- test-only tasks (enabled with --enable-test-tasks) ----------------------------


def test_steps(ctx: TaskContext, params: dict[str, Any]) -> dict[str, Any]:
    steps = int(params.get("steps", 5))
    delay = float(params.get("delay_s", 0.05))
    for i in range(steps):
        if ctx.cancelled():
            raise TaskCancelledError()
        time.sleep(delay)
        ctx.progress((i + 1) / steps, f"step {i + 1}/{steps}")
    return {"steps": steps, "pid": os.getpid()}


def test_crash(ctx: TaskContext, params: dict[str, Any]) -> dict[str, Any]:
    ctx.progress(0.5, "about to crash")
    os._exit(int(params.get("exit_code", 3)))


def test_hang(ctx: TaskContext, params: dict[str, Any]) -> dict[str, Any]:
    ctx.progress(0.1, "hanging and ignoring cancellation")
    while True:
        time.sleep(0.1)


def test_alloc(ctx: TaskContext, params: dict[str, Any]) -> dict[str, Any]:
    size = int(params["megabytes"]) * 1024 * 1024
    ctx.progress(0.1, "allocating")
    block = bytearray(size)
    block[::4096] = b"\x01" * len(block[::4096])
    return {"allocated_bytes": len(block)}


def test_network(ctx: TaskContext, params: dict[str, Any]) -> dict[str, Any]:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        sock.connect(("192.0.2.1", 80))  # TEST-NET-1, never routable
    return {"connected": True}


TaskFn = Callable[[TaskContext, dict[str, Any]], dict[str, Any]]

TASKS: dict[str, TaskFn] = {
    "detect_backends": detect_backends,
    "probe": probe,
    "smoke_inference": smoke_inference,
    "import_audio": import_audio,
    "export_audio": export_audio,
    "record_audio": record_audio,
    "list_audio_devices": list_audio_devices,
    "verify_project": verify_project,
    "analyze_audio": analyze_audio,
    "test.steps": test_steps,
    "test.crash": test_crash,
    "test.hang": test_hang,
    "test.alloc": test_alloc,
    "test.network": test_network,
}
