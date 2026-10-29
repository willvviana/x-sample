"""DirectML verification on the reference GPU (run with `uv run pytest -m gpu`).

These tests run real inference on a locally generated ONNX model. They check that:
- the session is strict (CPU-EP fallback disabled) and every profiled kernel ran on the
  DirectML EP;
- the process actually allocated GPU memory on the expected adapter (PDH counters,
  matched to DXGI adapter names), not a software/WARP device;
- outputs match the NumPy and CPU-EP references;
- unsupported operators are reported explicitly (strict) or as fallbacks (non-strict).

Metrics are written to engine/test-reports/directml_smoke.json.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import pytest

from xsample.runtime import smoke_model, win_gpu
from xsample.runtime.adapters import BackendUnsupportedError, DirectMLAdapter, get_adapter
from xsample.runtime.inference_check import run_check

pytestmark = pytest.mark.gpu

EXPECTED_GPU = os.environ.get("XSAMPLE_EXPECT_GPU", "RX 5700 XT")
REPORT = Path(__file__).resolve().parents[1] / "test-reports" / "directml_smoke.json"
MiB = 1024**2


def _record(section: str, data: dict[str, Any]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        report = {}
    report["environment"] = {
        "python": platform.python_version(),
        "gpus": [g.__dict__ for g in win_gpu.registry_gpus()],
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    report[section] = data
    REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


@pytest.fixture(scope="module", autouse=True)
def _require_directml() -> None:
    if not DirectMLAdapter.is_available():
        pytest.skip("DirectML execution provider not installed")
    hardware = [a for a in win_gpu.dxgi_adapters() if not a.software]
    if not hardware:
        pytest.skip("no hardware DirectX 12 adapter")


def test_expected_gpu_is_the_first_hardware_adapter() -> None:
    hardware = [a for a in win_gpu.dxgi_adapters() if not a.software]
    assert EXPECTED_GPU in hardware[0].name, [a.name for a in hardware]
    assert hardware[0].dedicated_video_memory > 4 * 1024**3


def test_strict_directml_inference_on_reference_gpu() -> None:
    result = run_check(
        "directml",
        chunk_samples=65536,
        chunks=64,
        strict=True,
        verify_every=1,
        compare_cpu=True,
    )
    _record("strict_inference", result)

    # Real execution on DirectML, no fallback.
    assert result["active_providers"][0] == "DmlExecutionProvider"
    assert result["profile"]["kernel_events"] >= 64
    assert set(result["profile"]["nodes_by_provider"]) == {"DmlExecutionProvider"}
    assert result["fallback_nodes"] == []
    # Memory was allocated on the expected physical adapter.
    assert result["adapter_used"] is not None and EXPECTED_GPU in result["adapter_used"]
    assert result["gpu_dedicated_peak_bytes"] > 0
    # Numerics.
    assert result["max_abs_error_vs_numpy"] <= result["tolerance"]
    assert result["max_abs_error_vs_cpu"] <= result["tolerance"]
    assert result["passed"]


def test_ten_minutes_of_audio_on_directml_via_worker(manager_factory: Any) -> None:
    manager = manager_factory(job_timeout_s=600)
    job = manager.submit(
        "smoke_inference",
        {"backend": "directml", "seconds": 600, "chunk_samples": 65536, "verify_every": 16},
    )
    final = manager.wait(job.id, timeout=600)
    assert final["state"] == "succeeded", final["error"]
    result = final["result"]
    _record(
        "worker_10min",
        {
            **{k: v for k, v in result.items() if k != "gpu_memory_samples"},
            "peak_worker_memory_bytes": final["peak_worker_memory_bytes"],
            "job_wall_seconds": final["finished_at"] - final["started_at"],
        },
    )
    assert result["passed"]
    assert EXPECTED_GPU in (result["adapter_used"] or "")
    assert result["fallback_nodes"] == []
    assert final["peak_worker_memory_bytes"] < 600 * MiB


def test_unsupported_operator_is_reported_in_strict_mode() -> None:
    with pytest.raises(
        BackendUnsupportedError, match="fallback to CPU EP has been explicitly disabled"
    ):
        get_adapter("directml").create_session(smoke_model.build_cpu_only_op_model(), strict=True)


def test_fallback_is_reported_in_non_strict_mode() -> None:
    import numpy as np

    session = get_adapter("directml").create_session(
        smoke_model.build_cpu_only_op_model(), strict=False, profile=True
    )
    session.run({"x": np.eye(3, dtype=np.float32)[None].repeat(4, axis=0)})
    report = session.end_profiling()
    session.close()
    assert report is not None
    fallback = report.fallback_nodes("DmlExecutionProvider")
    _record(
        "fallback_report", {"nodes_by_provider": report.nodes_by_provider, "fallback": fallback}
    )
    assert fallback == ["Det"]
    assert "DmlExecutionProvider" in report.nodes_by_provider


def test_directml_probe_passes(tmp_path: Path) -> None:
    from xsample.runtime.probe import run_probe

    result = run_probe("directml", cache_path=str(tmp_path / "probes.json"))
    record = result["record"]
    _record("probe", record)
    assert record["status"] == "pass", record
    assert EXPECTED_GPU in (record["device"] or "")
    assert record["fallback_nodes"] == []
