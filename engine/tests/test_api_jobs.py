"""Job lifecycle, detection and probes through the HTTP API."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from tests.conftest import wait_for_job


def test_submit_poll_and_list(client: TestClient, auth: dict[str, str]) -> None:
    response = client.post(
        "/api/jobs", headers=auth, json={"kind": "test.steps", "params": {"steps": 3}}
    )
    assert response.status_code == 202
    job = wait_for_job(client, response.json()["id"], auth)
    assert job["state"] == "succeeded"
    ids = [j["id"] for j in client.get("/api/jobs", headers=auth).json()]
    assert job["id"] in ids


def test_cancel_through_api(client: TestClient, auth: dict[str, str]) -> None:
    job_id = client.post(
        "/api/jobs",
        headers=auth,
        json={"kind": "test.steps", "params": {"steps": 200, "delay_s": 0.05}},
    ).json()["id"]
    time.sleep(1.0)
    assert client.post(f"/api/jobs/{job_id}/cancel", headers=auth).status_code == 200
    assert wait_for_job(client, job_id, auth)["state"] == "cancelled"


def test_crash_through_api(client: TestClient, auth: dict[str, str]) -> None:
    job_id = client.post(
        "/api/jobs", headers=auth, json={"kind": "test.crash", "params": {}}
    ).json()["id"]
    job = wait_for_job(client, job_id, auth)
    assert job["state"] == "failed"
    assert job["exit_code"] == 3
    assert client.get("/api/health").status_code == 200


def test_unknown_job_is_404(client: TestClient, auth: dict[str, str]) -> None:
    assert client.get("/api/jobs/doesnotexist", headers=auth).status_code == 404
    assert client.post("/api/jobs/doesnotexist/cancel", headers=auth).status_code == 404


def test_detection_reports_backends_without_claiming_support(
    client: TestClient, auth: dict[str, str]
) -> None:
    before = client.get("/api/system", headers=auth).json()
    assert before["detection"]["state"] == "pending"
    job_id = client.post("/api/backends/detect", headers=auth).json()["id"]
    assert wait_for_job(client, job_id, auth)["state"] == "succeeded"

    system = client.get("/api/system", headers=auth).json()
    assert system["detection"]["state"] == "ready"
    backends = {b["id"]: b for b in system["backends"]}
    assert set(backends) == {"cpu", "directml", "vulkan"}
    assert backends["cpu"]["state"] == "available"
    assert backends["directml"]["state"] in {"available", "unavailable"}
    assert backends["vulkan"]["state"] in {"available", "unavailable"}
    # Nothing has been probed yet, so no backend may claim model support.
    assert not any(b["claims_model_support"] for b in backends.values())
    assert system["memory"]["total_bytes"] > 0
    assert system["budgets"]["vram_budget_bytes"] > 0


def test_cpu_probe_passes_and_is_cached(client: TestClient, auth: dict[str, str]) -> None:
    first = wait_for_job(
        client, client.post("/api/probes", headers=auth, json={"backend": "cpu"}).json()["id"], auth
    )
    assert first["state"] == "succeeded"
    assert first["result"]["cached"] is False
    assert first["result"]["record"]["status"] == "pass"

    second = wait_for_job(
        client, client.post("/api/probes", headers=auth, json={"backend": "cpu"}).json()["id"], auth
    )
    assert second["result"]["cached"] is True
    assert second["result"]["record"]["key"] == first["result"]["record"]["key"]

    forced = wait_for_job(
        client,
        client.post("/api/probes", headers=auth, json={"backend": "cpu", "force": True}).json()[
            "id"
        ],
        auth,
    )
    assert forced["result"]["cached"] is False

    records = client.get("/api/probes", headers=auth).json()
    assert len(records) == 1 and records[0]["backend"] == "cpu"
    system = client.get("/api/system", headers=auth).json()
    assert {b["id"]: b for b in system["backends"]}["cpu"]["claims_model_support"] is True


def test_vulkan_probe_is_unsupported_not_claimed(client: TestClient, auth: dict[str, str]) -> None:
    job = wait_for_job(
        client,
        client.post("/api/probes", headers=auth, json={"backend": "vulkan"}).json()["id"],
        auth,
    )
    assert job["state"] == "succeeded"
    record = job["result"]["record"]
    assert record["status"] == "unsupported"
    assert "no Vulkan model runner" in record["detail"]
    system = client.get("/api/system", headers=auth).json()
    assert {b["id"]: b for b in system["backends"]}["vulkan"]["claims_model_support"] is False


def test_models_endpoint_lists_empty_registry(client: TestClient, auth: dict[str, str]) -> None:
    assert client.get("/api/models", headers=auth).json() == []
