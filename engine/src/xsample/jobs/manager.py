"""Job queue: scheduling, budgets, worker supervision, cancellation and crash handling."""

from __future__ import annotations

import builtins
import collections
import contextlib
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import IO, Any

import psutil
from pydantic import ValidationError

from xsample.config import EngineSettings
from xsample.jobs.kinds import KINDS, JobKind
from xsample.workers.cancel_event import CancelEvent
from xsample.workers.winjob import WorkerJobObject

log = logging.getLogger("xsample.jobs")

_CREATE_NO_WINDOW = 0x08000000
_POLL_S = 0.05


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


class JobRejectedError(ValueError):
    """The job kind is unknown, its params are invalid, or it exceeds a budget."""


class JobNotFoundError(KeyError):
    pass


@dataclass
class Job:
    id: str
    kind: str
    params: dict[str, Any]
    uses_gpu: bool
    est_vram_bytes: int
    est_ram_bytes: int
    state: JobState = JobState.QUEUED
    progress: float = 0.0
    message: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    exit_code: int | None = None
    worker_pid: int | None = None
    forced_termination: bool = False
    peak_worker_memory_bytes: int | None = None
    stderr_tail: list[str] = field(default_factory=list)
    data: dict[str, Any] | None = None  # latest worker telemetry (e.g. live recording meters)
    timeout_s: float | None = None
    stop_requested: bool = False
    cancel_requested: threading.Event = field(default_factory=threading.Event, repr=False)
    extra_params: dict[str, Any] = field(default_factory=dict, repr=False)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "params": dict(self.params),
            "state": str(self.state),
            "progress": self.progress,
            "message": self.message,
            "uses_gpu": self.uses_gpu,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "exit_code": self.exit_code,
            "worker_pid": self.worker_pid,
            "forced_termination": self.forced_termination,
            "peak_worker_memory_bytes": self.peak_worker_memory_bytes,
            "stderr_tail": list(self.stderr_tail),
            "data": dict(self.data) if self.data is not None else None,
        }


class JobManager:
    def __init__(
        self,
        settings: EngineSettings,
        on_finished: Callable[[Job], None] | None = None,
    ) -> None:
        self.settings = settings
        self._on_finished = on_finished
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(settings.max_concurrent_jobs)
        self._gpu = threading.Lock()
        self._job_objects: dict[str, WorkerJobObject] = {}
        self._cancel_events: dict[str, CancelEvent] = {}
        self._stop_events: dict[str, CancelEvent] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._closed = False

    # --- public API -------------------------------------------------------------

    def submit(
        self,
        kind: str,
        params: dict[str, Any] | None = None,
        *,
        public: bool = False,
        extra_params: dict[str, Any] | None = None,
    ) -> Job:
        spec = KINDS.get(kind)
        if spec is None or (spec.test_only and not self.settings.enable_test_tasks):
            raise JobRejectedError(f"unknown job kind {kind!r}")
        if public and not spec.public:
            raise JobRejectedError(f"job kind {kind!r} cannot be submitted directly")
        try:
            validated = spec.params_model.model_validate(params or {})
        except ValidationError as exc:
            raise JobRejectedError(f"invalid params: {exc.errors(include_url=False)}") from exc

        self._check_budget(kind, spec, validated)
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            params=validated.model_dump(),
            uses_gpu=spec.uses_gpu(validated),
            est_vram_bytes=spec.est_vram_bytes(validated),
            est_ram_bytes=spec.est_ram_bytes(validated),
            extra_params=dict(extra_params or {}),
            timeout_s=spec.timeout_s(validated) if spec.timeout_s else None,
        )
        with self._lock:
            if self._closed:
                raise JobRejectedError("engine is shutting down")
            self._jobs[job.id] = job
            thread = threading.Thread(target=self._run, args=(job,), daemon=True)
            self._threads[job.id] = thread
        thread.start()
        return job

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            return job.snapshot()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.snapshot() for j in jobs]

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.state not in TERMINAL:
                job.cancel_requested.set()
                if job.state is JobState.RUNNING:
                    job.state = JobState.CANCELLING
            event = self._cancel_events.get(job_id)
            if event is not None:
                event.set()
        return self.get(job_id)

    def stop(self, job_id: str) -> dict[str, Any]:
        """Ask a running job to finish normally and keep its output (recording: save)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.state not in TERMINAL:
                job.stop_requested = True
                event = self._stop_events.get(job_id)
                if event is not None:
                    event.set()
        return self.get(job_id)

    def active(self, kinds: set[str] | None = None) -> builtins.list[dict[str, Any]]:
        with self._lock:
            return [
                j.snapshot()
                for j in self._jobs.values()
                if j.state not in TERMINAL and (kinds is None or j.kind in kinds)
            ]

    def wait(self, job_id: str, timeout: float | None = None) -> dict[str, Any]:
        thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout)
        return self.get(job_id)

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            pending = list(self._jobs.values())
            objects = list(self._job_objects.values())
        for job in pending:
            job.cancel_requested.set()
        for job_object in objects:
            job_object.terminate(1)
        for thread in list(self._threads.values()):
            thread.join(timeout=10)

    # --- internals --------------------------------------------------------------

    def _check_budget(self, kind: str, spec: JobKind, params: Any) -> None:
        vram = spec.est_vram_bytes(params)
        if vram > self.settings.vram_budget_bytes:
            raise JobRejectedError(
                f"{kind}: estimated VRAM {vram} B exceeds budget "
                f"{self.settings.vram_budget_bytes} B"
            )
        ram = spec.est_ram_bytes(params)
        if ram > self.settings.worker_memory_limit_bytes:
            raise JobRejectedError(
                f"{kind}: estimated RAM {ram} B exceeds worker limit "
                f"{self.settings.worker_memory_limit_bytes} B"
            )
        available = int(psutil.virtual_memory().available)
        if ram > available:
            raise JobRejectedError(f"{kind}: estimated RAM {ram} B exceeds available {available} B")

    def _acquire(self, lock: threading.Semaphore | threading.Lock, job: Job) -> bool:
        while not lock.acquire(timeout=_POLL_S):
            if job.cancel_requested.is_set() or self._closed:
                return False
        return True

    def _update(self, job: Job, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(job, key, value)

    def _run(self, job: Job) -> None:
        try:
            if not self._acquire(self._slots, job):
                self._finish(job, JobState.CANCELLED)
                return
            try:
                if job.uses_gpu and not self._acquire(self._gpu, job):
                    self._finish(job, JobState.CANCELLED)
                    return
                try:
                    self._execute(job)
                finally:
                    if job.uses_gpu:
                        self._gpu.release()
            finally:
                self._slots.release()
        except Exception as exc:  # supervisor bug: never leave a job hanging
            log.exception("job %s: supervisor error", job.id)
            self._finish(job, JobState.FAILED, error=f"supervisor error: {exc}")

    @staticmethod
    def _send(stdin: IO[str], message: dict[str, Any]) -> None:
        try:
            stdin.write(json.dumps(message) + "\n")
            stdin.flush()
        except (OSError, ValueError):
            pass  # worker already gone

    def _execute(self, job: Job) -> None:
        if job.cancel_requested.is_set():
            self._finish(job, JobState.CANCELLED)
            return
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("OPENBLAS_NUM_THREADS", str(self.settings.worker_blas_threads))
        if self.settings.block_network_in_workers:
            env["XSAMPLE_BLOCK_NETWORK"] = "1"

        job_object = WorkerJobObject(self.settings.worker_memory_limit_bytes)
        cancel_event = CancelEvent.create()
        stop_event = CancelEvent.create()
        proc = subprocess.Popen(
            [sys.executable, "-m", "xsample.workers.main"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=_CREATE_NO_WINDOW,
        )
        assert proc.stdin and proc.stdout and proc.stderr
        try:
            job_object.assign(int(proc._handle))  # type: ignore[attr-defined]
        except OSError:
            proc.kill()
            job_object.close()
            cancel_event.close()
            stop_event.close()
            raise
        with self._lock:
            self._job_objects[job.id] = job_object
            self._cancel_events[job.id] = cancel_event
            self._stop_events[job.id] = stop_event
            if job.stop_requested:
                stop_event.set()
            job.state = JobState.CANCELLING if job.cancel_requested.is_set() else JobState.RUNNING
            job.started_at = time.time()
            job.worker_pid = proc.pid
            if job.cancel_requested.is_set():
                cancel_event.set()

        spec = {
            "kind": job.kind,
            "params": {**job.params, **job.extra_params},
            "cancel_event": cancel_event.name,
            "stop_event": stop_event.name,
        }
        self._send(proc.stdin, spec)
        # Close stdin so the worker never has a pending pipe read (see cancel_event.py).
        with contextlib.suppress(OSError):
            proc.stdin.close()

        lines: queue.Queue[str | None] = queue.Queue()
        stderr_tail: collections.deque[str] = collections.deque(maxlen=40)
        stdout, stderr = proc.stdout, proc.stderr

        def pump_stdout() -> None:
            for line in stdout:
                lines.put(line)
            lines.put(None)

        def pump_stderr() -> None:
            for line in stderr:
                stderr_tail.append(line.rstrip())

        readers = [
            threading.Thread(target=pump_stdout, daemon=True),
            threading.Thread(target=pump_stderr, daemon=True),
        ]
        for reader in readers:
            reader.start()

        outcome: dict[str, Any] | None = None
        cancel_sent_at: float | None = None
        timed_out = False
        limit = job.timeout_s if job.timeout_s is not None else self.settings.job_timeout_s
        deadline = time.monotonic() + limit
        while True:
            try:
                line = lines.get(timeout=_POLL_S)
            except queue.Empty:
                line = ""
            if line is None:
                break
            if line:
                outcome = self._handle_message(job, line, outcome)
            now = time.monotonic()
            if job.cancel_requested.is_set() and cancel_sent_at is None:
                cancel_sent_at = now
            if (
                cancel_sent_at is not None
                and now - cancel_sent_at > self.settings.cancel_grace_s
                and proc.poll() is None
            ):
                job.forced_termination = True
                job_object.terminate(1)
            if now > deadline and proc.poll() is None:
                timed_out = True
                job.forced_termination = True
                job_object.terminate(1)

        try:
            exit_code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            job_object.terminate(1)
            exit_code = proc.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=2)
        with self._lock:
            self._cancel_events.pop(job.id, None)
            self._stop_events.pop(job.id, None)
            self._job_objects.pop(job.id, None)
        peak = job_object.peak_process_memory()
        job_object.close()
        cancel_event.close()
        stop_event.close()

        common: dict[str, Any] = {"exit_code": exit_code, "peak_worker_memory_bytes": peak}
        kind = outcome.get("type") if outcome else None
        if kind == "result":
            self._finish(
                job, JobState.SUCCEEDED, result=outcome["result"] if outcome else None, **common
            )
        elif kind == "cancelled" or (job.cancel_requested.is_set() and not timed_out):
            self._finish(job, JobState.CANCELLED, **common)
        elif timed_out:
            self._finish(
                job,
                JobState.FAILED,
                error=f"timed out after {limit:.0f} s",
                stderr_tail=list(stderr_tail),
                **common,
            )
        elif kind == "error" and outcome:
            self._finish(
                job,
                JobState.FAILED,
                error=str(outcome.get("error")),
                stderr_tail=list(stderr_tail),
                **common,
            )
        else:
            self._finish(
                job,
                JobState.FAILED,
                error=f"worker exited unexpectedly with code {exit_code}",
                stderr_tail=list(stderr_tail),
                **common,
            )

    def _handle_message(
        self, job: Job, line: str, outcome: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return outcome
        if not isinstance(message, dict):
            return outcome
        if message.get("type") == "progress":
            self._update(
                job,
                progress=float(message.get("progress", job.progress)),
                message=message.get("message"),
                data=message.get("data") or job.data,
            )
            return outcome
        if message.get("type") in {"result", "error", "cancelled"}:
            return message
        return outcome

    def _finish(self, job: Job, state: JobState, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(job, key, value)
            job.state = state
            job.finished_at = time.time()
            if state is JobState.SUCCEEDED:
                job.progress = 1.0
        if self._on_finished is not None:
            try:
                self._on_finished(job)
            except Exception:  # a listener must not break the queue
                log.exception("job %s: on_finished listener failed", job.id)
