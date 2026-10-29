"""Project integrity check (runs in a worker): originals, decoded media, peaks, strays."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import soundfile as sf

from xsample.audio.errors import Cancelled
from xsample.audio.media import sha256_file
from xsample.project import layout


def verify_sources(
    project_dir: str,
    sources: list[dict[str, Any]],
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
) -> dict[str, Any]:
    root = Path(project_dir)
    report: list[dict[str, Any]] = []
    known: set[str] = set()
    for n, src in enumerate(sources):
        if cancelled():
            raise Cancelled()
        progress(n / max(1, len(sources)), f"Checking {src['original_name']}")
        issues: list[str] = []
        stored = root / src["stored_path"]
        known.add(stored.name)
        if not stored.is_file():
            issues.append("original file is missing")
        else:
            if stored.stat().st_size != src["size_bytes"]:
                issues.append("original file size changed")
            elif sha256_file(stored, cancelled) != src["sha256"]:
                issues.append("original file content no longer matches its checksum")
        media = src.get("media")
        if media is None:
            issues.append("no decoded audio (rebuild needed)")
        else:
            media_file, peaks = root / media["path"], root / media["peaks_path"]
            if not media_file.is_file():
                issues.append("decoded audio is missing (rebuild needed)")
            else:
                try:
                    if sf.info(media_file).frames != media["frames"]:
                        issues.append("decoded audio length differs from the manifest")
                except (RuntimeError, sf.LibsndfileError):
                    issues.append("decoded audio is unreadable (rebuild needed)")
            if not peaks.is_file():
                issues.append("waveform data is missing (rebuild needed)")
        report.append(
            {"id": src["id"], "name": src["original_name"], "ok": not issues, "issues": issues}
        )

    orphans = [
        f"{layout.SOURCES}/{p.name}"
        for p in sorted((root / layout.SOURCES).glob("*"))
        if p.is_file() and p.name not in known
    ]
    recordings = [
        f"{layout.RECORDING}/{p.name}" for p in sorted((root / layout.RECORDING).glob("*.wav"))
    ]
    progress(1.0, "Done")
    return {
        "ok": all(r["ok"] for r in report),
        "sources": report,
        "unreferenced_originals": orphans,
        "unimported_recordings": recordings,
    }
