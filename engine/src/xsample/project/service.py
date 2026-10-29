"""The single open project of a running engine: lifecycle, edits, job-result application."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from xsample.audio.render import RenderPlan
from xsample.config import EngineSettings
from xsample.project import layout, ops, store
from xsample.project.model import ExportRecord, Media, ProcessingRecord, Project, Source
from xsample.project.store import ProjectError, ProjectLock, ProjectNotFoundError, now_iso
from xsample.project.view import ProjectView, build_view

log = logging.getLogger("xsample.project")
T = TypeVar("T")
MAX_RECENT = 20


class NoProjectOpenError(ProjectError):
    status = 409

    def __init__(self) -> None:
        super().__init__("No project is open")


class ProjectService:
    def __init__(self, settings: EngineSettings) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        self._root: Path | None = None
        self._project: Project | None = None
        self._recovered = False
        self._file_lock: ProjectLock | None = None

    # --- lifecycle ------------------------------------------------------------------------

    def create(self, name: str, sample_rate: int) -> ProjectView:
        folder = store.sanitize_name(name)
        root = self._settings.projects_dir / folder
        with self._lock:
            self._ensure_closed()
            store.create_project(root, name, sample_rate)
            return self._open_locked(root)

    def open(self, path: str) -> ProjectView:
        root = Path(path).expanduser()
        with self._lock:
            if self._root is not None and root.resolve() == self._root.resolve():
                return self.view_required()
            self._ensure_closed()
            return self._open_locked(root)

    def _ensure_closed(self) -> None:
        if self._root is not None:
            raise ProjectError("Close the current project first")

    def _open_locked(self, root: Path) -> ProjectView:
        loaded = store.load_project(root)  # raises a clear error for invalid folders
        file_lock = ProjectLock(root)
        file_lock.acquire()
        try:
            layout.ensure_layout(root)
            store.clean_stale_temp_files(root)
            if loaded.recovered_from_backup:
                store.save_project(root, loaded.project)  # rewrite a good manifest
        except BaseException:
            file_lock.release()
            raise
        self._root, self._project = root.resolve(), loaded.project
        self._recovered, self._file_lock = loaded.recovered_from_backup, file_lock
        self._remember(self._root, loaded.project.name)
        return self.view_required()

    def close(self) -> None:
        with self._lock:
            if self._root is None:
                return
            if self._file_lock is not None:
                self._file_lock.release()
            self._root = self._project = self._file_lock = None
            self._recovered = False

    @property
    def is_open(self) -> bool:
        return self._root is not None

    def root(self) -> Path:
        if self._root is None:
            raise NoProjectOpenError
        return self._root

    def _current(self) -> tuple[Path, Project]:
        if self._root is None or self._project is None:
            raise NoProjectOpenError
        return self._root, self._project

    def view(self) -> ProjectView | None:
        with self._lock:
            if self._root is None or self._project is None:
                return None
            return build_view(self._root, self._project, self._recovered)

    def view_required(self) -> ProjectView:
        view = self.view()
        if view is None:
            raise NoProjectOpenError
        return view

    # --- edits ------------------------------------------------------------------------------

    def edit(self, change: Callable[[Project], T]) -> T:
        """Apply `change` to a copy, validate and persist it, then swap it in (all or nothing)."""
        with self._lock:
            root, current = self._current()
            candidate = current.model_copy(deep=True)
            result = change(candidate)
            store.save_project(root, candidate)
            self._project = candidate
            self._recovered = False
            return result

    def save(self) -> ProjectView:
        self.edit(lambda p: None)
        return self.view_required()

    def rename(self, name: str) -> ProjectView:
        if not name.strip():
            raise ProjectError("Project name is empty")

        def change(p: Project) -> None:
            p.name = name.strip()[:120]
            ops.log(p, "project.rename", name=p.name)

        self.edit(change)
        return self.view_required()

    def read(self, fn: Callable[[Path, Project], T]) -> T:
        with self._lock:
            root, project = self._current()
            return fn(root, project)

    def mix_plan(self) -> RenderPlan:
        return self.read(lambda root, p: ops.mix_plan(p, root))

    def sample_plan(self, sample_id: str) -> RenderPlan:
        return self.read(lambda root, p: ops.sample_plan(p, root, sample_id))

    def new_incoming_path(self, suffix: str) -> Path:
        root = self.root()
        (root / layout.INCOMING).mkdir(parents=True, exist_ok=True)
        return root / layout.INCOMING / f"{uuid.uuid4().hex}{suffix}"

    def source_for_rebuild(self, source_id: str) -> Source:
        def get(_: Path, p: Project) -> Source:
            try:
                return p.source(source_id)
            except KeyError as exc:
                raise ProjectError("Source not found") from exc

        return self.read(get)

    # --- applying finished jobs -----------------------------------------------------------------

    def _same_project(self, project_dir: str) -> bool:
        return self._root is not None and Path(project_dir).resolve() == self._root

    def apply_import(self, result: dict[str, Any], params: dict[str, Any]) -> None:
        with self._lock:
            if not self._same_project(params["project_dir"]):
                log.warning("import finished for a project that is no longer open")
                return
            media = Media(**result["media"])
            if params.get("rebuild_source_id"):
                source_id = params["rebuild_source_id"]

                def rebuild(p: Project) -> None:
                    p.source(source_id).media = media
                    ops.log(p, "source.rebuild", source=source_id)

                self.edit(rebuild)
                return
            source = Source(**result["source"], imported_at=now_iso(), media=media)
            self.edit(lambda p: ops.add_imported_source(p, source))

    def apply_export(self, result: dict[str, Any], params: dict[str, Any]) -> None:
        with self._lock:
            if not self._same_project(params["project_dir"]):
                return
            record = ExportRecord(
                id=store.new_id("exp"),
                filename=result["filename"],
                kind=params["kind"],
                format=result["format"],
                subtype=result["subtype"],
                frames=result["frames"],
                sample_rate=result["sample_rate"],
                channels=result["channels"],
                size_bytes=result["size_bytes"],
                created_at=now_iso(),
                processing=ProcessingRecord(**result["processing"]),
            )

            def add(p: Project) -> None:
                p.exports.append(record)
                ops.log(p, "export", file=record.filename, kind=record.kind)

            self.edit(add)

    def export_path(self, filename: str) -> Path:
        root = self.root()
        if Path(filename).name != filename or filename.startswith("."):
            raise ProjectNotFoundError("Export not found")
        path = root / layout.EXPORTS / filename
        known = self.read(lambda _r, p: {e.filename for e in p.exports})
        if filename not in known or not path.is_file():
            raise ProjectNotFoundError("Export not found")
        return path

    # --- project discovery ----------------------------------------------------------------------

    def _recent_file(self) -> Path:
        return self._settings.data_dir / "recent_projects.json"

    def _remember(self, root: Path, name: str) -> None:
        try:
            entries = [e for e in self._read_recent() if e["path"] != str(root)]
            entries.insert(0, {"path": str(root), "name": name, "opened_at": now_iso()})
            self._recent_file().parent.mkdir(parents=True, exist_ok=True)
            self._recent_file().write_text(
                json.dumps(entries[:MAX_RECENT], indent=2), encoding="utf-8"
            )
        except OSError:
            log.warning("could not update the recent projects list", exc_info=True)

    def _read_recent(self) -> list[dict[str, str]]:
        try:
            data = json.loads(self._recent_file().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [e for e in data if isinstance(e, dict) and isinstance(e.get("path"), str)]

    def list_projects(self) -> dict[str, Any]:
        found: dict[str, dict[str, Any]] = {}
        for entry in self._read_recent():
            path = Path(entry["path"])
            if (path / layout.MANIFEST).is_file() or (path / layout.BACKUP).is_file():
                found[str(path)] = {
                    "name": entry.get("name") or path.name,
                    "path": str(path),
                    "last_opened": entry.get("opened_at"),
                }
        folder = self._settings.projects_dir
        if folder.is_dir():
            for child in sorted(folder.iterdir()):
                if (
                    child.is_dir()
                    and (child / layout.MANIFEST).is_file()
                    and str(child) not in found
                ):
                    found[str(child)] = {
                        "name": child.name,
                        "path": str(child),
                        "last_opened": None,
                    }
        return {"projects_dir": str(folder), "projects": list(found.values())}
