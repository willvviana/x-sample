"""API response and request models (also the source of the UI's generated types)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

BackendState = Literal["available", "unavailable", "unknown"]
JobStateName = Literal["queued", "running", "cancelling", "succeeded", "failed", "cancelled"]


class Health(BaseModel):
    status: Literal["ok"]
    version: str


class CpuInfo(BaseModel):
    name: str | None
    cores: int | None
    threads: int | None


class MemoryInfo(BaseModel):
    total_bytes: int
    available_bytes: int


class GpuInfo(BaseModel):
    name: str
    vram_bytes: int | None
    driver_version: str | None


class BackendStatus(BaseModel):
    id: Literal["cpu", "directml", "vulkan"]
    state: BackendState
    detail: str | None
    devices: list[str]
    claims_model_support: bool = Field(
        description="True only when at least one model has passed a probe on this backend"
    )


class DetectionInfo(BaseModel):
    state: Literal["pending", "running", "ready", "failed"]
    job_id: str | None = None
    updated_at: float | None = None
    error: str | None = None
    runtime_version: str | None = None


class Budgets(BaseModel):
    vram_budget_bytes: int
    worker_memory_limit_bytes: int
    max_concurrent_jobs: int


class SystemInfo(BaseModel):
    engine_version: str
    python_version: str
    os: str
    cpu: CpuInfo
    memory: MemoryInfo
    gpus: list[GpuInfo]
    backends: list[BackendStatus]
    detection: DetectionInfo
    budgets: Budgets
    test_tasks_enabled: bool


class ProbeRequest(BaseModel):
    backend: Literal["cpu", "directml", "vulkan"]
    force: bool = False


class ProbeRecord(BaseModel):
    key: str
    model_id: str
    model_sha256: str
    backend: str
    status: Literal["pass", "fail", "unsupported"]
    detail: str | None = None
    device: str | None = None
    driver_version: str | None = None
    runtime_version: str
    probed_at: float
    max_abs_error: float | None = None
    max_abs_error_vs_cpu: float | None = None
    tolerance: float | None = None
    first_run_ms: float | None = None
    chunk_ms_mean: float | None = None
    active_providers: list[str] | None = None
    nodes_by_provider: dict[str, list[str]] | None = None
    fallback_nodes: list[str] | None = None
    adapter_used: str | None = None
    gpu_dedicated_peak_bytes: int | None = None


class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    license: str
    size_bytes: int
    candidate_backends: list[str]
    status: Literal["missing", "ok", "size_mismatch", "hash_mismatch"]


class JobRequest(BaseModel):
    kind: str = Field(max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)


class JobOut(BaseModel):
    id: str
    kind: str
    params: dict[str, Any]
    state: JobStateName
    progress: float
    message: str | None
    uses_gpu: bool
    created_at: float
    started_at: float | None
    finished_at: float | None
    result: dict[str, Any] | None
    error: str | None
    exit_code: int | None
    worker_pid: int | None
    forced_termination: bool
    peak_worker_memory_bytes: int | None
    stderr_tail: list[str]
    data: dict[str, Any] | None


# --- Phase 1: projects, audio, recording --------------------------------------------------

# Sample positions are integers; JSON floats such as 1.0 are rejected on purpose.
Sample = Annotated[int, Field(strict=True, ge=0)]


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectCreate(_Req):
    name: str = Field(min_length=1, max_length=120)
    sample_rate: int = 48000


class ProjectOpen(_Req):
    path: str = Field(min_length=1, max_length=1024)


class ProjectPatch(_Req):
    name: str = Field(min_length=1, max_length=120)


class ProjectEntry(BaseModel):
    name: str
    path: str
    last_opened: str | None


class ProjectList(BaseModel):
    projects_dir: str
    projects: list[ProjectEntry]


class TrackCreate(_Req):
    name: str | None = Field(default=None, max_length=120)


class TrackPatch(_Req):
    name: str | None = Field(default=None, max_length=120)
    muted: bool | None = None
    solo: bool | None = None
    gain: float | None = Field(default=None, ge=0.0, le=4.0)


class TrimRequest(_Req):
    start: Sample
    end: Sample


class MoveRequest(_Req):
    timeline_start: Sample


class SliceRequest(_Req):
    start: Sample
    end: Sample
    name: str | None = Field(default=None, max_length=120)


class SamplePatch(_Req):
    name: str = Field(min_length=1, max_length=120)


class PlaceRequest(_Req):
    track_id: str | None = None
    at: Sample = 0


class ExportRequest(_Req):
    kind: Literal["sample", "selection"]
    sample_id: str | None = None
    start: Sample | None = None
    end: Sample | None = None
    name: str | None = Field(default=None, max_length=120)
    format: Literal["wav", "flac"] = "wav"
    subtype: Literal["PCM_16", "PCM_24", "FLOAT"] = "PCM_24"


class PeaksOut(BaseModel):
    start: int
    end: int
    columns: int
    channels: int
    scale: int = Field(description="Values are int16-quantised: sample = value / scale")
    min: list[list[int]] = Field(description="[channel][column]")
    max: list[list[int]] = Field(description="[channel][column]")


class AudioDevice(BaseModel):
    index: int
    name: str
    kind: Literal["microphone", "loopback"]
    channels: int
    sample_rate: int
    is_default: bool


class AudioDevices(BaseModel):
    state: Literal["pending", "running", "ready", "failed"]
    updated_at: float | None
    error: str | None
    microphones: list[AudioDevice]
    loopbacks: list[AudioDevice]


class RecordRequest(_Req):
    device_index: int = Field(ge=0)
    device_name: str = Field(min_length=1, max_length=300)
    kind: Literal["microphone", "loopback"]
    max_seconds: int = Field(default=3600, ge=1, le=4 * 3600)
