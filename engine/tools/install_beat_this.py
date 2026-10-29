r"""Install the Beat This! small0 ONNX exports into the local model store.

    uv run python tools/install_beat_this.py <dir with the two .onnx files>

Files are verified against the size and SHA-256 pinned in xsample.analysis.beat_model and
copied atomically to %LOCALAPPDATA%\X-Sample\models\beat-this-small0-*\; registry.json gets
matching entries (with the license) so the Models list shows them. Nothing is downloaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from xsample.analysis import beat_model
from xsample.config import default_data_dir
from xsample.runtime.registry import ModelEntry, ModelRegistry, ModelSource

MiB = 1024**2


def entries() -> list[ModelEntry]:
    out = []
    for filename, label, backends, vram in (
        (beat_model.FULL_FILE, "30 s chunks", ["cpu", "directml"], 1024 * MiB),
        (beat_model.SHORT_FILE, "pieces under 30 s", ["cpu"], 0),
    ):
        sha, size = beat_model.PINNED[filename]
        out.append(
            ModelEntry(
                id=beat_model.REGISTRY_IDS[filename],
                name=f"{beat_model.DISPLAY_NAME}, {label}",
                version="1.1.0+b95c8ab",
                source=ModelSource.LOCAL,
                source_url=beat_model.SOURCE_URL,
                license=beat_model.LICENSE[:200],
                sha256=sha,
                size_bytes=size,
                filename=filename,
                candidate_backends=backends,  # type: ignore[arg-type]
                est_vram_bytes=vram,
                est_ram_bytes=300 * MiB,
            )
        )
    return out


def main() -> None:
    source = Path(sys.argv[1])
    models_dir = default_data_dir() / "models"
    new = entries()
    for entry in new:
        target = ModelRegistry(models_dir, [entry]).install_from_file(
            entry, source / entry.filename
        )
        print("installed", target)
    path = models_dir / "registry.json"
    current = json.loads(path.read_text("utf-8")) if path.exists() else {"models": []}
    kept = [
        m
        for m in current.get("models", [])
        if (m.get("id"), m.get("filename")) not in {(e.id, e.filename) for e in new}
    ]
    current["models"] = kept + [e.model_dump(mode="json") for e in new]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2), "utf-8")
    tmp.replace(path)
    print("installed:", beat_model.installed(models_dir))


if __name__ == "__main__":
    main()
