"""Allow-listed job kinds, their parameter schemas and resource estimates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MiB = 1024**2


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectParams(_Params):
    pass


class ProbeParams(_Params):
    backend: Literal["cpu", "directml", "vulkan"]
    device_id: int = Field(0, ge=0, le=7)
    force: bool = False


class SmokeParams(_Params):
    backend: Literal["cpu", "directml"]
    seconds: float = Field(10.0, gt=0, le=600)
    chunk_samples: int = Field(65536, ge=1024, le=262144)
    strict: bool = True
    device_id: int = Field(0, ge=0, le=7)
    verify_every: int = Field(8, ge=1, le=1000)


class StepsParams(_Params):
    steps: int = Field(5, ge=1, le=1000)
    delay_s: float = Field(0.05, ge=0, le=5)


class CrashParams(_Params):
    exit_code: int = Field(3, ge=1, le=255)


class EmptyParams(_Params):
    pass


class AllocParams(_Params):
    megabytes: int = Field(ge=1, le=65536)


class ImportParams(_Params):
    project_dir: str
    incoming_path: str = ""
    original_name: str = Field(max_length=255)
    sample_rate: int = Field(ge=8000, le=192000)
    channels: Literal["keep", "mono", "stereo"] = "keep"
    origin: Literal["import", "recording"] = "import"
    device: str | None = None
    rebuild_source_id: str | None = None
    rebuild_stored_path: str | None = None


class ExportParams(_Params):
    project_dir: str
    kind: Literal["sample", "selection"]
    plan: dict[str, Any]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    name: str = Field(min_length=1, max_length=120)
    format: Literal["wav", "flac"]
    subtype: Literal["PCM_16", "PCM_24", "FLOAT"]
    describe: dict[str, Any] = Field(default_factory=dict)


class RecordParams(_Params):
    project_dir: str
    device_index: int = Field(ge=0)
    device_name: str = Field(min_length=1, max_length=300)
    kind: Literal["microphone", "loopback"]
    max_seconds: int = Field(ge=1, le=4 * 3600)


class VerifyParams(_Params):
    project_dir: str
    sources: list[dict[str, Any]]


class AnalyzeParams(_Params):
    project_dir: str
    source_id: str = Field(pattern=r"^src_[0-9a-f]{12}$")
    media_path: str = Field(min_length=1, max_length=300)
    media_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate: int = Field(ge=8000, le=192000)
    frames: int = Field(gt=0)
    # Set by the engine when the Beat This! exports are installed and verified.
    beat_model_dir: str | None = Field(None, min_length=1, max_length=400)
    beat_backend: Literal["cpu", "directml"] = "cpu"
    # Set by the engine when the PANNs tagger export is installed and verified.
    tagger_dir: str | None = Field(None, min_length=1, max_length=400)
    tagger_backend: Literal["cpu", "directml"] = "cpu"


# Tagger (PANNs MobileNetV1, 10 s windows), measured on the reference machine: ~48 MiB
# dedicated VRAM, process RSS 147 MiB (CPU) / 194 MiB (DirectML) for 10 min. With the beat
# model the job peak did not rise (the models run in turn); the RAM margin is kept anyway.
TAGGER_RAM = 300 * MiB
TAGGER_VRAM = 128 * MiB


def _beat_gpu(p: AnalyzeParams) -> bool:
    return p.beat_model_dir is not None and p.beat_backend == "directml"


def _tagger_gpu(p: AnalyzeParams) -> bool:
    return p.tagger_dir is not None and p.tagger_backend == "directml"


def _analyze_uses_gpu(p: BaseModel) -> bool:
    assert isinstance(p, AnalyzeParams)
    return _beat_gpu(p) or _tagger_gpu(p)


def _analyze_vram(p: BaseModel) -> int:
    assert isinstance(p, AnalyzeParams)
    # Measured peaks on the RX 5700 XT: beat model ~570 MiB dedicated (three 1500-frame
    # attention maps); tagger ~48 MiB. The models run one after the other.
    vram = 0
    if _beat_gpu(p):
        vram = 768 * MiB
    if _tagger_gpu(p):
        vram = max(vram, TAGGER_VRAM)
    return vram


def _analyze_ram(p: BaseModel) -> int:
    assert isinstance(p, AnalyzeParams)
    # Measured worker peak commit for 10 min: DSP 215 MiB, + model on CPU 1.06 GiB (the
    # 1500-frame attention arena), + DirectML 1.64 GiB (CPU reference session for the
    # parity check alongside the DirectML session). Flat with duration.
    base = 600 * MiB if p.beat_model_dir is None else (1900 if _beat_gpu(p) else 1300) * MiB
    return base + (TAGGER_RAM if p.tagger_dir is not None else 0)


def _zero(_: BaseModel) -> int:
    return 0


def _smoke_vram(p: BaseModel) -> int:
    assert isinstance(p, SmokeParams)
    # input + hidden activations + output, float32, plus runtime overhead
    return p.chunk_samples * 4 * (2 + 16 + 2 + 2) + 64 * MiB if p.backend == "directml" else 0


def _smoke_ram(p: BaseModel) -> int:
    assert isinstance(p, SmokeParams)
    return 300 * MiB + p.chunk_samples * 4 * 64


def _alloc_ram(p: BaseModel) -> int:
    # Deliberately not estimated: the test exercises the hard worker memory limit.
    return 0


def _uses_gpu_backend(p: BaseModel) -> bool:
    return getattr(p, "backend", "cpu") != "cpu"


@dataclass(frozen=True)
class JobKind:
    params_model: type[_Params]
    uses_gpu: Callable[[BaseModel], bool]
    est_vram_bytes: Callable[[BaseModel], int] = _zero
    est_ram_bytes: Callable[[BaseModel], int] = _zero
    public: bool = True  # submittable through POST /api/jobs
    test_only: bool = False
    timeout_s: Callable[[BaseModel], float] | None = None


KINDS: dict[str, JobKind] = {
    "detect_backends": JobKind(DetectParams, lambda _: False, public=False),
    "probe": JobKind(ProbeParams, _uses_gpu_backend, public=False),
    "smoke_inference": JobKind(SmokeParams, _uses_gpu_backend, _smoke_vram, _smoke_ram),
    "import_audio": JobKind(
        ImportParams,
        lambda _: False,
        est_ram_bytes=lambda _: 400 * MiB,
        public=False,
        timeout_s=lambda _: 4 * 3600.0,
    ),
    "export_audio": JobKind(
        ExportParams,
        lambda _: False,
        est_ram_bytes=lambda _: 300 * MiB,
        public=False,
        timeout_s=lambda _: 4 * 3600.0,
    ),
    "record_audio": JobKind(
        RecordParams,
        lambda _: False,
        est_ram_bytes=lambda _: 200 * MiB,
        public=False,
        timeout_s=lambda p: float(getattr(p, "max_seconds", 0)) + 120.0,
    ),
    "list_audio_devices": JobKind(DetectParams, lambda _: False, public=False),
    "verify_project": JobKind(
        VerifyParams,
        lambda _: False,
        est_ram_bytes=lambda _: 200 * MiB,
        public=False,
        timeout_s=lambda _: 4 * 3600.0,
    ),
    "analyze_audio": JobKind(
        AnalyzeParams,
        _analyze_uses_gpu,  # only the beat model may use DirectML; DSP is CPU-only
        est_vram_bytes=_analyze_vram,
        est_ram_bytes=_analyze_ram,
        public=False,
        timeout_s=lambda _: 4 * 3600.0,
    ),
    "test.steps": JobKind(StepsParams, lambda _: False, test_only=True),
    "test.crash": JobKind(CrashParams, lambda _: False, test_only=True),
    "test.hang": JobKind(EmptyParams, lambda _: False, test_only=True),
    "test.alloc": JobKind(AllocParams, lambda _: False, est_ram_bytes=_alloc_ram, test_only=True),
    "test.network": JobKind(EmptyParams, lambda _: False, test_only=True),
}
