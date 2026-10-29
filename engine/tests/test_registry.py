from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from xsample.runtime.registry import (
    DownloadNotPermittedError,
    IntegrityError,
    ModelEntry,
    ModelRegistry,
    VerifyStatus,
)

PAYLOAD = b"not really a model, just bytes for hashing" * 100


def _entry(**overrides: Any) -> ModelEntry:
    data: dict[str, Any] = {
        "id": "test.model",
        "name": "Test model",
        "version": "1",
        "source": "local",
        "license": "MIT",
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
        "size_bytes": len(PAYLOAD),
        "filename": "model.onnx",
        "candidate_backends": ["cpu"],
    }
    data.update(overrides)
    return ModelEntry.model_validate(data)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sha256": "abc"},
        {"sha256": "G" * 64},
        {"filename": "../evil.onnx"},
        {"filename": "sub/dir.onnx"},
        {"filename": "a..b"},
        {"id": "Bad ID"},
        {"source": "url", "source_url": "http://insecure.example/m.onnx"},
        {"source": "url"},
        {"candidate_backends": []},
        {"candidate_backends": ["cuda"]},
        {"size_bytes": 0},
    ],
)
def test_invalid_entries_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _entry(**overrides)


def test_verify_and_install(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models", [_entry()])
    entry = registry.get("test.model")
    assert registry.verify(entry) is VerifyStatus.MISSING

    source = tmp_path / "download.onnx"
    source.write_bytes(PAYLOAD)
    installed = registry.install_from_file(entry, source)
    assert installed.read_bytes() == PAYLOAD
    assert registry.verify(entry) is VerifyStatus.OK

    installed.write_bytes(PAYLOAD[:-1] + b"X")
    assert registry.verify(entry) is VerifyStatus.HASH_MISMATCH
    installed.write_bytes(PAYLOAD + b"extra")
    assert registry.verify(entry) is VerifyStatus.SIZE_MISMATCH


def test_install_rejects_tampered_file_and_leaves_nothing(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models", [_entry()])
    entry = registry.get("test.model")
    source = tmp_path / "tampered.onnx"
    source.write_bytes(PAYLOAD[:-1] + b"X")
    with pytest.raises(IntegrityError, match="SHA-256"):
        registry.install_from_file(entry, source)
    target_dir = registry.path_for(entry).parent
    assert list(target_dir.iterdir()) == []


def test_downloads_are_not_permitted(tmp_path: Path) -> None:
    entry = _entry(
        source="url", source_url="https://example.invalid/m.onnx", approved_for_download=True
    )
    with pytest.raises(DownloadNotPermittedError):
        ModelRegistry(tmp_path, [entry]).download(entry)


def test_load_from_disk_and_duplicates(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    raw = _entry().model_dump(mode="json")
    (models / "registry.json").write_text(json.dumps({"models": [raw]}), encoding="utf-8")
    assert [e.id for e in ModelRegistry.load(models).entries()] == ["test.model"]
    with pytest.raises(ValueError, match="duplicate"):
        ModelRegistry(models, [_entry(), _entry()])
