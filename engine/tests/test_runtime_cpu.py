"""CPU reference path, smoke model, profiling parser and hardware detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xsample.runtime import smoke_model, vulkan, win_gpu
from xsample.runtime.adapters import BackendUnsupportedError, get_adapter
from xsample.runtime.inference_check import run_check
from xsample.runtime.probe_cache import ProbeCache, probe_key


def test_smoke_model_is_deterministic() -> None:
    assert smoke_model.build(4096).sha256 == smoke_model.build(4096).sha256
    assert smoke_model.build(4096).sha256 != smoke_model.build(8192).sha256


def test_cpu_session_matches_numpy_reference() -> None:
    model = smoke_model.build(4096)
    session = get_adapter("cpu").create_session(model.model_bytes)
    x = smoke_model.test_signal(3, 4096)
    y = session.run({"x": x})[0]
    assert y.shape == x.shape and y.dtype == np.float32
    np.testing.assert_allclose(y, model.reference(x), atol=1e-5)


def test_cpu_check_reports_placement_and_metrics() -> None:
    result = run_check("cpu", chunk_samples=8192, chunks=4)
    assert result["passed"]
    assert result["active_providers"][0] == "CPUExecutionProvider"
    assert result["profile"]["kernel_events"] > 0
    assert set(result["profile"]["nodes_by_provider"]) == {"CPUExecutionProvider"}
    assert result["fallback_nodes"] == []
    assert result["chunk_ms_mean"] > 0 and result["realtime_factor"] > 0


def test_vulkan_adapter_claims_nothing() -> None:
    with pytest.raises(BackendUnsupportedError, match="no Vulkan model runner"):
        get_adapter("vulkan").create_session(smoke_model.build(4096).model_bytes)


def test_probe_key_changes_with_environment() -> None:
    base = {
        "model_sha": "a" * 64,
        "backend": "directml",
        "device": "GPU",
        "driver_version": "1",
        "runtime_version": "1.24.4",
    }
    key = probe_key(**base)
    for field, value in [("driver_version", "2"), ("runtime_version", "1.25"), ("device", "Other")]:
        assert probe_key(**{**base, field: value}) != key


def test_probe_cache_round_trip(tmp_path: Path) -> None:
    cache = ProbeCache(tmp_path / "cache.json")
    assert cache.get("k") is None
    cache.put({"key": "k", "backend": "cpu", "probed_at": 1.0})
    assert cache.get("k") == {"key": "k", "backend": "cpu", "probed_at": 1.0}
    assert len(cache.all()) == 1


def test_hardware_detection_is_well_formed() -> None:
    for gpu in win_gpu.registry_gpus():
        assert gpu.name
    for adapter in win_gpu.dxgi_adapters():
        assert adapter.name and isinstance(adapter.luid, tuple)
    status = vulkan.detect()
    assert isinstance(status.loader_present, bool)
