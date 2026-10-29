"""Reading and writing project folders: atomic saves, backup recovery, locking."""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import psutil
from pydantic import ValidationError

from xsample import __version__
from xsample.project import layout
from xsample.project.model import FORMAT_VERSION, Project

_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


class ProjectError(Exception):
    """A project problem the user can understand. `status` maps to an HTTP status code."""

    status = 422

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ProjectNotFoundError(ProjectError):
    status = 404


class ProjectExistsError(ProjectError):
    status = 409


class ProjectLockedError(ProjectError):
    status = 409


class InvalidProjectError(ProjectError):
    status = 422


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def sanitize_name(name: str) -> str:
    """A safe single Windows folder name derived from a project title."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().rstrip(". ")
    if not cleaned:
        raise ProjectError("Project name is empty")
    if cleaned.split(".")[0].lower() in _RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned[:80]


# --- save / load --------------------------------------------------------------------


def save_project(root: Path, project: Project) -> None:
    """Validate, then write project.json atomically, keeping the previous one as .bak."""
    Project.model_validate(project.model_dump())  # never persist an inconsistent manifest
    project.revision += 1
    project.modified_at = now_iso()
    data = project.model_dump_json(indent=2)
    manifest = root / layout.MANIFEST
    tmp = root / (layout.MANIFEST + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if manifest.exists():
        backup_tmp = root / (layout.BACKUP + ".tmp")
        shutil.copyfile(manifest, backup_tmp)
        os.replace(backup_tmp, root / layout.BACKUP)
    os.replace(tmp, manifest)


def _parse(path: Path) -> Project:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidProjectError(f"{path.name} is not valid JSON ({exc})") from exc
    version = raw.get("format_version") if isinstance(raw, dict) else None
    if not isinstance(version, int):
        raise InvalidProjectError(f"{path.name} is not an X-Sample project manifest")
    if version > FORMAT_VERSION:
        raise InvalidProjectError(
            f"This project was created by a newer X-Sample (format {version}); "
            f"this version understands format {FORMAT_VERSION}"
        )
    try:
        return Project.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        where = ".".join(str(p) for p in first["loc"]) or "manifest"
        raise InvalidProjectError(f"{path.name} is invalid at {where}: {first['msg']}") from exc


@dataclass
class Loaded:
    project: Project
    recovered_from_backup: bool


def load_project(root: Path) -> Loaded:
    if not root.is_dir():
        raise ProjectNotFoundError(f"Folder not found: {root}")
    manifest = root / layout.MANIFEST
    backup = root / layout.BACKUP
    if not manifest.exists() and not backup.exists():
        raise InvalidProjectError("This folder does not contain an X-Sample project")
    errors: list[str] = []
    for candidate, recovered in ((manifest, False), (backup, True)):
        if not candidate.exists():
            continue
        try:
            return Loaded(_parse(candidate), recovered)
        except InvalidProjectError as exc:
            errors.append(exc.message)
    raise InvalidProjectError(
        "The project could not be read and no usable backup exists: " + " | ".join(errors)
    )


def create_project(root: Path, name: str, sample_rate: int) -> Project:
    if sample_rate not in (22050, 32000, 44100, 48000, 88200, 96000):
        raise ProjectError(f"Unsupported project sample rate: {sample_rate}")
    if root.exists() and any(root.iterdir()):
        raise ProjectExistsError(f"The folder already exists and is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    layout.ensure_layout(root)
    stamp = now_iso()
    project = Project(
        id=uuid.uuid4().hex,
        name=name.strip()[:120] or root.name,
        created_at=stamp,
        modified_at=stamp,
        engine_version=__version__,
        sample_rate=sample_rate,
    )
    save_project(root, project)
    return project


def clean_stale_temp_files(root: Path) -> None:
    """Remove leftovers of crashed imports/saves. Recordings are kept: they may be recoverable."""
    for sub in (layout.TMP, layout.INCOMING):
        folder = root / sub
        if folder.is_dir():
            for item in folder.iterdir():
                with contextlib.suppress(OSError):
                    item.unlink() if item.is_file() else shutil.rmtree(item)
    for leftover in (layout.MANIFEST + ".tmp", layout.BACKUP + ".tmp"):
        (root / leftover).unlink(missing_ok=True)
    exports = root / layout.EXPORTS
    if exports.is_dir():
        for part in exports.glob(".*.part"):
            part.unlink(missing_ok=True)


# --- single-writer lock --------------------------------------------------------------


class ProjectLock:
    """`.xsample.lock` holds (pid, process start time); locks of dead processes are taken over."""

    def __init__(self, root: Path) -> None:
        self.path = root / layout.LOCK
        self._token = secrets.token_hex(8)
        self._held = False

    @staticmethod
    def _owner_alive(info: dict[str, object]) -> bool:
        pid = info.get("pid")
        created = info.get("created")
        if not isinstance(pid, int) or not psutil.pid_exists(pid):
            return False
        try:
            actual = psutil.Process(pid).create_time()
        except psutil.Error:
            return False
        return isinstance(created, (int, float)) and abs(actual - float(created)) < 2.0

    def acquire(self) -> None:
        me = psutil.Process()
        payload = json.dumps({"pid": me.pid, "created": me.create_time(), "token": self._token})
        for _ in range(3):
            try:
                with self.path.open("x", encoding="utf-8") as handle:
                    handle.write(payload)
                self._held = True
                return
            except FileExistsError:
                try:
                    info = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    info = {}
                if isinstance(info, dict) and self._owner_alive(info):
                    raise ProjectLockedError(
                        f"The project is already open in another X-Sample process (pid {info['pid']})"  # noqa: E501
                    ) from None
                with contextlib.suppress(OSError):
                    self.path.unlink()  # stale: its process is gone
        raise ProjectLockedError("Could not lock the project folder")

    def release(self) -> None:
        if self._held:
            with contextlib.suppress(OSError):
                self.path.unlink()
            self._held = False
