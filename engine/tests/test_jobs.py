"""ADR-0005/0007: worker isolation, progress, cancellation, crashes, limits, budgets."""

from __future__ import annotations

import time
from typing import Any

import psutil
import pytest

from tests.conftest import wait_for_state
from xsample.config import GiB
from xsample.jobs.manager import JobRejectedError

MiB = 1024**2


def test_job_succeeds_with_ordered_progress(manager_factory: Any) -> None:
    manager = manager_factory()
    job = manager.submit("test.steps", {"steps": 5, "delay_s": 0.1})
    seen: list[float] = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        snapshot = manager.get(job.id)
        if not seen or snapshot["progress"] != seen[-1]:
            seen.append(snapshot["progress"])
        if snapshot["state"] == "succeeded":
            break
        time.sleep(0.01)
    final = manager.get(job.id)
    assert final["state"] == "succeeded"
    assert final["exit_code"] == 0
    assert final["progress"] == 1.0
    assert seen == sorted(seen), "progress must never go backwards"
    assert len(seen) >= 3
    assert final["result"]["pid"] != psutil.Process().pid, "job must run in a separate process"
    assert final["peak_worker_memory_bytes"] and final["peak_worker_memory_bytes"] > 0


def test_cooperative_cancellation(manager_factory: Any) -> None:
    manager = manager_factory()
    job = manager.submit("test.steps", {"steps": 200, "delay_s": 0.05})
    wait_for_state(manager, job.id, {"running"})
    time.sleep(0.3)
    manager.cancel(job.id)
    final = manager.wait(job.id, timeout=30)
    assert final["state"] == "cancelled"
    assert final["forced_termination"] is False
    assert final["exit_code"] == 0
    assert final["progress"] < 1.0


def test_unresponsive_worker_is_force_terminated(manager_factory: Any) -> None:
    manager = manager_factory(cancel_grace_s=0.5)
    job = manager.submit("test.hang")
    running = wait_for_state(manager, job.id, {"running"})
    pid = running["worker_pid"]
    time.sleep(0.5)
    manager.cancel(job.id)
    final = manager.wait(job.id, timeout=30)
    assert final["state"] == "cancelled"
    assert final["forced_termination"] is True
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_cancel_queued_job_never_starts(manager_factory: Any) -> None:
    manager = manager_factory(max_concurrent_jobs=1)
    blocker = manager.submit("test.steps", {"steps": 40, "delay_s": 0.05})
    queued = manager.submit("test.steps", {"steps": 1})
    assert manager.get(queued.id)["state"] == "queued"
    manager.cancel(queued.id)
    final = manager.wait(queued.id, timeout=30)
    assert final["state"] == "cancelled"
    assert final["worker_pid"] is None
    manager.cancel(blocker.id)


def test_worker_crash_is_reported_and_engine_recovers(manager_factory: Any) -> None:
    manager = manager_factory()
    crashed = manager.wait(manager.submit("test.crash", {"exit_code": 7}).id, timeout=30)
    assert crashed["state"] == "failed"
    assert crashed["exit_code"] == 7
    assert "exited unexpectedly with code 7" in crashed["error"]

    after = manager.wait(manager.submit("test.steps", {"steps": 2}).id, timeout=30)
    assert after["state"] == "succeeded"


def test_task_exception_is_reported(manager_factory: Any) -> None:
    manager = manager_factory()
    final = manager.wait(manager.submit("test.network").id, timeout=30)
    assert final["state"] == "failed"
    assert "NetworkBlockedError" in final["error"], "workers must inherit the network block"


def test_worker_memory_limit_is_enforced(manager_factory: Any) -> None:
    manager = manager_factory(worker_memory_limit_bytes=256 * MiB)
    final = manager.wait(manager.submit("test.alloc", {"megabytes": 512}).id, timeout=60)
    assert final["state"] == "failed"
    assert "Memory" in (final["error"] or "") or final["exit_code"] != 0
    assert final["peak_worker_memory_bytes"] <= 256 * MiB


def test_timeout_terminates_worker(manager_factory: Any) -> None:
    manager = manager_factory(job_timeout_s=2.0)
    final = manager.wait(manager.submit("test.hang").id, timeout=30)
    assert final["state"] == "failed"
    assert "timed out" in final["error"]
    assert final["forced_termination"] is True


def test_vram_budget_rejects_oversized_job(manager_factory: Any) -> None:
    manager = manager_factory(vram_budget_bytes=1 * MiB)
    with pytest.raises(JobRejectedError, match="VRAM"):
        manager.submit("smoke_inference", {"backend": "directml"})
    # CPU jobs do not use VRAM and are unaffected.
    assert manager.submit("smoke_inference", {"backend": "cpu", "seconds": 1})


def test_ram_estimate_over_worker_limit_is_rejected(manager_factory: Any) -> None:
    manager = manager_factory(worker_memory_limit_bytes=100 * MiB)
    with pytest.raises(JobRejectedError, match="RAM"):
        manager.submit("smoke_inference", {"backend": "cpu"})


@pytest.mark.parametrize(
    "kind,params",
    [
        ("nope", {}),
        ("smoke_inference", {"backend": "cuda"}),
        ("smoke_inference", {"backend": "cpu", "seconds": 601}),
        ("smoke_inference", {"backend": "cpu", "chunk_samples": 10}),
        ("test.steps", {"steps": 2, "unexpected": 1}),
    ],
)
def test_invalid_jobs_are_rejected(manager_factory: Any, kind: str, params: dict[str, Any]) -> None:
    with pytest.raises(JobRejectedError):
        manager_factory().submit(kind, params)


def test_test_tasks_are_disabled_by_default(manager_factory: Any) -> None:
    with pytest.raises(JobRejectedError, match="unknown job kind"):
        manager_factory(enable_test_tasks=False).submit("test.crash")


def test_gpu_jobs_run_one_at_a_time(manager_factory: Any) -> None:
    manager = manager_factory(max_concurrent_jobs=4)
    # Force the GPU lock path without needing a GPU: hold it, then check the job waits.
    manager._gpu.acquire()
    try:
        job = manager.submit("smoke_inference", {"backend": "directml", "seconds": 1})
        time.sleep(0.5)
        assert manager.get(job.id)["state"] == "queued"
        manager.cancel(job.id)
    finally:
        manager._gpu.release()
    assert manager.wait(job.id, timeout=30)["state"] == "cancelled"


def test_ten_minutes_of_audio_on_cpu_with_bounded_memory(manager_factory: Any) -> None:
    manager = manager_factory(worker_memory_limit_bytes=1 * GiB)
    peaks = {}
    for seconds in (10, 600):
        job = manager.submit(
            "smoke_inference",
            {"backend": "cpu", "seconds": seconds, "chunk_samples": 65536, "verify_every": 16},
        )
        final = manager.wait(job.id, timeout=300)
        assert final["state"] == "succeeded", final["error"]
        result = final["result"]
        assert result["passed"]
        assert result["audio_seconds"] >= seconds * 0.99
        assert result["max_abs_error_vs_numpy"] <= result["tolerance"]
        peaks[seconds] = final["peak_worker_memory_bytes"]
    # Chunked processing: committed memory must not scale with duration. A full
    # 10-minute stereo float32 buffer alone would add ~212 MB.
    assert peaks[600] - peaks[10] < 64 * MiB, peaks
    assert peaks[600] < 400 * MiB, peaks
