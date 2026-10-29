"""Backend adapters (ADR-0003).

Each adapter creates inference sessions for one backend. Availability of a backend
says nothing about whether a given model runs on it; that is what probes establish.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import onnxruntime as ort


class BackendId(StrEnum):
    CPU = "cpu"
    DIRECTML = "directml"
    VULKAN = "vulkan"


class BackendUnsupportedError(RuntimeError):
    """The backend cannot run this model (no runner, missing kernels, or disabled)."""


PROVIDER_FOR_BACKEND = {
    BackendId.CPU: "CPUExecutionProvider",
    BackendId.DIRECTML: "DmlExecutionProvider",
}


@dataclass
class ProfileReport:
    """Which execution provider actually ran each graph node, from ORT profiling."""

    kernel_events: int = 0
    nodes_by_provider: dict[str, list[str]] = field(default_factory=dict)

    def fallback_nodes(self, expected_provider: str) -> list[str]:
        # MemcpyToHost/FromHost are transfer nodes owned by the GPU EP, not fallbacks.
        return sorted(
            node
            for provider, nodes in self.nodes_by_provider.items()
            if provider != expected_provider
            for node in nodes
        )

    def to_dict(self) -> dict[str, Any]:
        return {"kernel_events": self.kernel_events, "nodes_by_provider": self.nodes_by_provider}


def parse_profile(path: str | Path) -> ProfileReport:
    events = json.loads(Path(path).read_text(encoding="utf-8"))
    report = ProfileReport()
    seen: dict[str, set[str]] = {}
    for event in events:
        if event.get("cat") != "Node" or not str(event.get("name", "")).endswith("_kernel_time"):
            continue
        args = event.get("args", {})
        provider = str(args.get("provider", "unknown"))
        seen.setdefault(provider, set()).add(str(args.get("op_name", event["name"])))
        report.kernel_events += 1
    report.nodes_by_provider = {p: sorted(n) for p, n in seen.items()}
    return report


@dataclass
class Session:
    backend: BackendId
    inner: ort.InferenceSession
    strict: bool
    profile_dir: Path | None

    @property
    def active_providers(self) -> list[str]:
        return list(self.inner.get_providers())

    def run(self, inputs: dict[str, Any]) -> list[Any]:
        return list(self.inner.run(None, inputs))

    def end_profiling(self) -> ProfileReport | None:
        if self.profile_dir is None:
            return None
        return parse_profile(self.inner.end_profiling())

    def close(self) -> None:
        # Drop the native session so device memory is released before the worker exits.
        del self.inner


class OnnxAdapter:
    backend: BackendId

    def providers(self, strict: bool) -> list[Any]:
        raise NotImplementedError

    def configure(self, options: ort.SessionOptions, strict: bool) -> None:
        pass

    def create_session(
        self, model: bytes, *, strict: bool = True, profile: bool = False
    ) -> Session:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.configure(options, strict)
        profile_dir: Path | None = None
        if profile:
            profile_dir = Path(tempfile.mkdtemp(prefix="xsample-profile-"))
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_dir / "ort")
        try:
            inner = ort.InferenceSession(model, options, providers=self.providers(strict))
        except Exception as exc:
            raise BackendUnsupportedError(f"{self.backend}: {exc}") from exc
        return Session(self.backend, inner, strict, profile_dir)


class CpuAdapter(OnnxAdapter):
    backend = BackendId.CPU

    def providers(self, strict: bool) -> list[Any]:
        return ["CPUExecutionProvider"]


class DirectMLAdapter(OnnxAdapter):
    backend = BackendId.DIRECTML

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = device_id

    def providers(self, strict: bool) -> list[Any]:
        dml = ("DmlExecutionProvider", {"device_id": self.device_id})
        return [dml] if strict else [dml, "CPUExecutionProvider"]

    def configure(self, options: ort.SessionOptions, strict: bool) -> None:
        # Required by the DirectML EP.
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if strict:
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    @staticmethod
    def is_available() -> bool:
        return "DmlExecutionProvider" in ort.get_available_providers()


class VulkanAdapter(OnnxAdapter):
    """Vulkan runs through model-specific native runners; none are registered yet."""

    backend = BackendId.VULKAN

    def create_session(
        self, model: bytes, *, strict: bool = True, profile: bool = False
    ) -> Session:
        raise BackendUnsupportedError(
            "vulkan: no Vulkan model runner is registered; Vulkan support is detected, "
            "not claimed for any model"
        )


def get_adapter(backend: BackendId | str, device_id: int = 0) -> OnnxAdapter:
    backend = BackendId(backend)
    if backend is BackendId.CPU:
        return CpuAdapter()
    if backend is BackendId.DIRECTML:
        return DirectMLAdapter(device_id)
    return VulkanAdapter()


def runtime_version() -> str:
    return str(ort.__version__)
