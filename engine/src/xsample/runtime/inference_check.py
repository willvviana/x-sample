"""Run the smoke model on a backend and measure correctness, placement and cost.

Used by backend probes, the `smoke_inference` job and the DirectML GPU tests.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import psutil

from xsample.runtime import smoke_model, win_gpu
from xsample.runtime.adapters import (
    PROVIDER_FOR_BACKEND,
    BackendId,
    Session,
    get_adapter,
    runtime_version,
)

FP32_TOLERANCE = 1e-4


class CheckCancelledError(Exception):
    pass


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def _gpu_memory_by_adapter() -> dict[str, Any]:
    """Dedicated GPU memory held by this process, attributed to DXGI adapter names."""
    try:
        adapters = {a.luid: a for a in win_gpu.dxgi_adapters()}
        usage = win_gpu.process_gpu_memory(os.getpid())
    except OSError as exc:
        return {"error": str(exc), "adapters": []}
    rows: list[dict[str, Any]] = []
    for entry in usage:
        adapter = adapters.get(entry.luid)
        rows.append(
            {
                "adapter": adapter.name if adapter else f"luid {entry.luid}",
                "software": adapter.software if adapter else None,
                "dedicated_bytes": entry.dedicated_bytes,
            }
        )
    rows.sort(key=lambda r: int(r["dedicated_bytes"]), reverse=True)
    return {"adapters": rows}


def run_check(
    backend: BackendId | str,
    *,
    chunk_samples: int = 65536,
    chunks: int = 8,
    strict: bool = True,
    device_id: int = 0,
    verify_every: int = 1,
    sample_rate: int = 44100,
    compare_cpu: bool = False,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    backend = BackendId(backend)
    expected_provider = PROVIDER_FOR_BACKEND.get(backend)
    model = smoke_model.build(chunk_samples)
    adapter = get_adapter(backend, device_id)

    t0 = time.perf_counter()
    session = adapter.create_session(model.model_bytes, strict=strict, profile=True)
    create_ms = (time.perf_counter() - t0) * 1000

    cpu_session: Session | None = None
    if compare_cpu and backend is not BackendId.CPU:
        cpu_session = get_adapter(BackendId.CPU).create_session(model.model_bytes)

    gpu_samples: list[dict[str, Any]] = []
    times_ms: list[float] = []
    max_err_ref = 0.0
    max_err_cpu: float | None = 0.0 if cpu_session else None
    try:
        warm = smoke_model.test_signal(0, chunk_samples, sample_rate)
        t0 = time.perf_counter()
        session.run({"x": warm})
        first_run_ms = (time.perf_counter() - t0) * 1000

        wall0 = time.perf_counter()
        for i in range(chunks):
            if cancelled and cancelled():
                raise CheckCancelledError()
            x = smoke_model.test_signal(i, chunk_samples, sample_rate)
            t0 = time.perf_counter()
            y = session.run({"x": x})[0]
            times_ms.append((time.perf_counter() - t0) * 1000)
            if i % verify_every == 0:
                ref = model.reference(x)
                max_err_ref = max(max_err_ref, float(np.max(np.abs(y - ref))))
                if cpu_session is not None and max_err_cpu is not None:
                    y_cpu = cpu_session.run({"x": x})[0]
                    max_err_cpu = max(max_err_cpu, float(np.max(np.abs(y - y_cpu))))
            if backend is BackendId.DIRECTML and i in (0, chunks - 1):
                gpu_samples.append(_gpu_memory_by_adapter())
            if progress:
                progress(i + 1, chunks)
        wall_s = time.perf_counter() - wall0
        active_providers = session.active_providers
        profile = session.end_profiling()
    finally:
        session.close()
        if cpu_session is not None:
            cpu_session.close()

    audio_s = chunks * chunk_samples / sample_rate
    processing_s = sum(times_ms) / 1000
    fallback = profile.fallback_nodes(expected_provider) if profile and expected_provider else []
    placement_ok = bool(profile and expected_provider in profile.nodes_by_provider)

    adapter_used = None
    gpu_dedicated_peak = None
    for sample in gpu_samples:
        for row in sample.get("adapters", []):
            if row["dedicated_bytes"] > (gpu_dedicated_peak or 0):
                gpu_dedicated_peak = int(row["dedicated_bytes"])
                adapter_used = row["adapter"]

    numerics_ok = max_err_ref <= FP32_TOLERANCE and (
        max_err_cpu is None or max_err_cpu <= FP32_TOLERANCE
    )
    passed = (
        numerics_ok
        and placement_ok
        and bool(active_providers)
        and active_providers[0] == expected_provider
        and (not strict or not fallback)
    )
    mem = psutil.Process().memory_info()
    return {
        "backend": str(backend),
        "model_id": smoke_model.MODEL_ID,
        "model_sha256": model.sha256,
        "runtime_version": runtime_version(),
        "strict": strict,
        "passed": passed,
        "expected_provider": expected_provider,
        "active_providers": active_providers,
        "profile": profile.to_dict() if profile else None,
        "fallback_nodes": fallback,
        "adapter_used": adapter_used,
        "gpu_memory_samples": gpu_samples,
        "gpu_dedicated_peak_bytes": gpu_dedicated_peak,
        "chunk_samples": chunk_samples,
        "chunks": chunks,
        "sample_rate": sample_rate,
        "audio_seconds": audio_s,
        "session_create_ms": create_ms,
        "first_run_ms": first_run_ms,
        "chunk_ms_mean": float(np.mean(times_ms)) if times_ms else 0.0,
        "chunk_ms_p95": _percentile(times_ms, 95),
        "chunk_ms_max": max(times_ms, default=0.0),
        "wall_seconds": wall_s,
        "realtime_factor": audio_s / processing_s if processing_s > 0 else None,
        "max_abs_error_vs_numpy": max_err_ref,
        "max_abs_error_vs_cpu": max_err_cpu,
        "tolerance": FP32_TOLERANCE,
        "peak_rss_bytes": int(getattr(mem, "peak_wset", mem.rss)),
    }
