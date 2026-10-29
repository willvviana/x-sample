r"""Install the PANNs MobileNetV1 ONNX export (tagger) into the local model store.

    uv run python tools/install_panns.py <dir with panns_mobilenetv1.onnx>

The file is verified against the size and SHA-256 pinned in xsample.analysis.tagger and
copied atomically to %LOCALAPPDATA%\X-Sample\models\panns-mobilenetv1-audioset\;
registry.json gets a matching entry (with the license). Nothing is downloaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from xsample.analysis import tagger
from xsample.config import default_data_dir
from xsample.runtime.registry import ModelEntry, ModelRegistry, ModelSource

MiB = 1024**2


def entry() -> ModelEntry:
    sha, size = tagger.PINNED
    return ModelEntry(
        id=tagger.REGISTRY_ID,
        name=tagger.DISPLAY_NAME,
        version="mAP0.389+d2f4b8c",
        source=ModelSource.LOCAL,
        source_url=tagger.SOURCE_URL,
        license=tagger.LICENSE[:200],
        sha256=sha,
        size_bytes=size,
        filename=tagger.FILE,
        candidate_backends=["cpu", "directml"],
        est_vram_bytes=128 * MiB,
        est_ram_bytes=200 * MiB,
    )


def main() -> None:
    source = Path(sys.argv[1])
    models_dir = default_data_dir() / "models"
    new = entry()
    target = ModelRegistry(models_dir, [new]).install_from_file(new, source / new.filename)
    print("installed", target)
    path = models_dir / "registry.json"
    current = json.loads(path.read_text("utf-8")) if path.exists() else {"models": []}
    kept = [m for m in current.get("models", []) if m.get("id") != new.id]
    current["models"] = [*kept, new.model_dump(mode="json")]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2), "utf-8")
    tmp.replace(path)
    print("installed:", tagger.installed(models_dir))


if __name__ == "__main__":
    main()
