"""Model registry with SHA-256 verification.

Phase 0 never downloads anything. Models can only be installed from a local file the
user supplies, and only if its size and SHA-256 match the registry entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

_CHUNK = 1024 * 1024


class ModelSource(StrEnum):
    URL = "url"
    LOCAL = "local"


class ModelEntry(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,63}$")
    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=50)
    source: ModelSource
    source_url: str | None = None
    license: str = Field(min_length=1, max_length=200)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    filename: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    candidate_backends: list[Literal["cpu", "directml", "vulkan"]] = Field(min_length=1)
    est_vram_bytes: int = Field(default=0, ge=0)
    est_ram_bytes: int = Field(default=0, ge=0)
    approved_for_download: bool = False

    @field_validator("filename")
    @classmethod
    def _no_dot_segments(cls, value: str) -> str:
        if ".." in value:
            raise ValueError("filename must not contain '..'")
        return value

    @model_validator(mode="after")
    def _url_rules(self) -> ModelEntry:
        if self.source is ModelSource.URL and not (
            self.source_url and self.source_url.startswith("https://")
        ):
            raise ValueError("url models require an https:// source_url")
        return self


class VerifyStatus(StrEnum):
    MISSING = "missing"
    OK = "ok"
    SIZE_MISMATCH = "size_mismatch"
    HASH_MISMATCH = "hash_mismatch"


class DownloadNotPermittedError(PermissionError):
    pass


class IntegrityError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


class ModelRegistry:
    def __init__(self, models_dir: Path, entries: list[ModelEntry] | None = None) -> None:
        self.models_dir = models_dir
        self._entries: dict[str, ModelEntry] = {}
        for entry in entries or []:
            if entry.id in self._entries:
                raise ValueError(f"duplicate model id {entry.id}")
            self._entries[entry.id] = entry

    @classmethod
    def load(cls, models_dir: Path) -> ModelRegistry:
        path = models_dir / "registry.json"
        if not path.exists():
            return cls(models_dir, [])
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(models_dir, [ModelEntry.model_validate(item) for item in raw.get("models", [])])

    def entries(self) -> list[ModelEntry]:
        return list(self._entries.values())

    def get(self, model_id: str) -> ModelEntry:
        return self._entries[model_id]

    def path_for(self, entry: ModelEntry) -> Path:
        root = self.models_dir.resolve()
        path = (root / entry.id / entry.filename).resolve()
        if root not in path.parents:
            raise IntegrityError("model path escapes the models directory")
        return path

    def verify(self, entry: ModelEntry) -> VerifyStatus:
        path = self.path_for(entry)
        if not path.is_file():
            return VerifyStatus.MISSING
        if path.stat().st_size != entry.size_bytes:
            return VerifyStatus.SIZE_MISMATCH
        if sha256_file(path) != entry.sha256:
            return VerifyStatus.HASH_MISMATCH
        return VerifyStatus.OK

    def install_from_file(self, entry: ModelEntry, source: Path) -> Path:
        """Copy a user-supplied file into the model store after verifying it."""
        if source.stat().st_size != entry.size_bytes:
            raise IntegrityError(f"{entry.id}: size mismatch")
        target = self.path_for(entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".partial")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            shutil.copyfile(source, tmp)
            if sha256_file(tmp) != entry.sha256:
                raise IntegrityError(f"{entry.id}: SHA-256 mismatch")
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        return target

    def download(self, entry: ModelEntry) -> Path:
        raise DownloadNotPermittedError(
            f"{entry.id}: model downloads are disabled in this build; "
            "install from a verified local file instead"
        )
