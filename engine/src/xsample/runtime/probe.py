"""Backend compatibility probes (worker side; imports onnxruntime)."""

from __future__ import annotations

import time
from typing import Any

from xsample.runtime import smoke_model, vulkan, win_gpu
from xsample.runtime.adapters import BackendId, BackendUnsupportedError, runtime_version
from xsample.runtime.inference_check import run_check
from xsample.runtime.probe_cache import ProbeCache, probe_key

PROBE_CHUNK_SAMPLES = 16384
PROBE_CHUNKS = 4


def _environment(backend: BackendId, device_id: int) -> dict[str, Any]:
    if backend is BackendId.CPU:
        return {"device": win_gpu.cpu_name() or "cpu", "driver_version": None}
    if backend is BackendId.DIRECTML:
        hardware = [a for a in win_gpu.dxgi_adapters() if not a.software]
        name = hardware[device_id].name if device_id < len(hardware) else f"adapter {device_id}"
        drivers = {g.name: g.driver_version for g in win_gpu.registry_gpus()}
        return {"device": name, "driver_version": drivers.get(name)}
    status = vulkan.detect()
    device = status.devices[0] if status.devices else None
    return {
        "device": device.name if device else None,
        "driver_version": str(device.driver_version_raw) if device else None,
    }


def run_probe(
    backend: BackendId | str,
    *,
    device_id: int = 0,
    cache_path: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    backend = BackendId(backend)
    env = _environment(backend, device_id)
    model_sha = smoke_model.build(PROBE_CHUNK_SAMPLES).sha256
    key = probe_key(
        model_sha=model_sha,
        backend=str(backend),
        device=env["device"],
        driver_version=env["driver_version"],
        runtime_version=runtime_version(),
    )
    if cache_path and not force:
        cached = ProbeCache(cache_path).get(key)
        if cached is not None:
            return {"cached": True, "record": cached}

    record: dict[str, Any] = {
        "key": key,
        "model_id": smoke_model.MODEL_ID,
        "model_sha256": model_sha,
        "backend": str(backend),
        "device": env["device"],
        "driver_version": env["driver_version"],
        "runtime_version": runtime_version(),
        "probed_at": time.time(),
    }
    try:
        check = run_check(
            backend,
            chunk_samples=PROBE_CHUNK_SAMPLES,
            chunks=PROBE_CHUNKS,
            strict=True,
            device_id=device_id,
            compare_cpu=backend is not BackendId.CPU,
        )
    except BackendUnsupportedError as exc:
        record.update(status="unsupported", detail=str(exc))
        return {"cached": False, "record": record}

    record.update(
        status="pass" if check["passed"] else "fail",
        detail=None if check["passed"] else "numerical or placement check failed",
        max_abs_error=check["max_abs_error_vs_numpy"],
        max_abs_error_vs_cpu=check["max_abs_error_vs_cpu"],
        tolerance=check["tolerance"],
        first_run_ms=check["first_run_ms"],
        chunk_ms_mean=check["chunk_ms_mean"],
        active_providers=check["active_providers"],
        nodes_by_provider=(check["profile"] or {}).get("nodes_by_provider"),
        fallback_nodes=check["fallback_nodes"],
        adapter_used=check["adapter_used"],
        gpu_dedicated_peak_bytes=check["gpu_dedicated_peak_bytes"],
    )
    return {"cached": False, "record": record}
