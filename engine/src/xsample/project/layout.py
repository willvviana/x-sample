"""Project folder layout (ADR-0006). Paths stored in the manifest are relative, with '/'."""

from __future__ import annotations

from pathlib import Path

MANIFEST = "project.json"
BACKUP = "project.json.bak"
LOCK = ".xsample.lock"

SOURCES = "sources"  # immutable originals, content-addressed
PEAKS = "analysis/peaks"  # waveform peak pyramids
MEDIA = "cache/media"  # decoded working copies at the project sample rate
TMP = "cache/tmp"
INCOMING = "cache/incoming"  # uploads awaiting import
RECORDING = "cache/recording"
EXPORTS = "exports"

DIRECTORIES = (SOURCES, PEAKS, MEDIA, TMP, INCOMING, RECORDING, EXPORTS)
SOURCE_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".aif", ".aiff"}
)


def ensure_layout(root: Path) -> None:
    for name in DIRECTORIES:
        (root / name).mkdir(parents=True, exist_ok=True)


def resolve_inside(root: Path, relative: str) -> Path:
    """Resolve a manifest-relative path, refusing anything that escapes the project."""
    base = root.resolve()
    path = (base / relative).resolve()
    if base != path and base not in path.parents:
        raise ValueError(f"path escapes the project folder: {relative}")
    return path


def safe_stem(name: str, fallback: str = "export") -> str:
    """A Windows-safe file name stem (no separators, reserved names or trailing dots)."""
    import re

    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().strip(".")
    if not cleaned or cleaned.split(".")[0].lower() in {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }:
        cleaned = fallback
    return cleaned[:80]
