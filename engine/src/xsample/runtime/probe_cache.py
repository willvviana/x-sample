"""Persistent cache of backend probe results (engine side; no onnxruntime import).

A result is valid only for the exact model hash, backend, device, driver and runtime
version it was measured with; any change produces a new key and a fresh probe.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any


def probe_key(
    *,
    model_sha: str,
    backend: str,
    device: str | None,
    driver_version: str | None,
    runtime_version: str,
) -> str:
    material = json.dumps(
        [model_sha, backend, device, driver_version, runtime_version], separators=(",", ":")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ProbeCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self._load().get(key)

    def all(self) -> list[dict[str, Any]]:
        records = list(self._load().values())
        records.sort(key=lambda r: float(r.get("probed_at", 0)), reverse=True)
        return records

    def put(self, record: dict[str, Any]) -> None:
        with self._lock:
            data = self._load()
            data[str(record["key"])] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
