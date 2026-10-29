from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from xsample import netguard
from xsample.api.app import create_app
from xsample.config import EngineSettings
from xsample.jobs.manager import JobManager

TOKEN = "test-token-0123456789abcdef0123456789"
PORT = 8765
BASE_URL = f"http://127.0.0.1:{PORT}"

# Tests never touch the network; workers inherit the block through the environment.
netguard.install()
os.environ["XSAMPLE_BLOCK_NETWORK"] = "1"


@pytest.fixture
def settings(tmp_path: Path) -> EngineSettings:
    return EngineSettings(
        token=TOKEN,
        port=PORT,
        data_dir=tmp_path / "data",
        projects_dir=tmp_path / "projects",
        enable_test_tasks=True,
        detect_on_startup=False,
        cancel_grace_s=1.0,
        job_timeout_s=120.0,
    )


@pytest.fixture
def manager_factory(settings: EngineSettings) -> Iterator[Any]:
    managers: list[JobManager] = []

    def make(**overrides: Any) -> JobManager:
        manager = JobManager(replace(settings, **overrides))
        managers.append(manager)
        return manager

    yield make
    for manager in managers:
        manager.shutdown()


@pytest.fixture
def client(settings: EngineSettings) -> Iterator[TestClient]:
    with TestClient(create_app(settings), base_url=BASE_URL) as test_client:
        yield test_client


@pytest.fixture
def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def wait_for_job(
    client: TestClient, job_id: str, headers: dict[str, str], timeout: float = 60.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job: dict[str, Any] = client.get(f"/api/jobs/{job_id}", headers=headers).json()
        if job["state"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout} s")


def wait_for_state(
    manager: JobManager, job_id: str, states: set[str], timeout: float = 30.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job["state"] in states:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {states}: {manager.get(job_id)['state']}")
