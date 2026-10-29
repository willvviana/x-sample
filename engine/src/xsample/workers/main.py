"""Worker process entry point: `python -m xsample.workers.main`.

Protocol (ADR-0007): stdin carries one line, the job spec
`{"kind", "params", "cancel_event", "stop_event"}`, and is then closed. Cancel and stop
requests are signalled through the named Windows events in those fields. Messages go to a
dedicated copy of the original stdout as JSON lines; anything libraries print is redirected to
stderr.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from typing import IO, Any

from xsample.audio.errors import AudioError, Cancelled
from xsample.workers.cancel_event import CancelEvent


class _Context:
    def __init__(
        self, out: IO[str], cancel: CancelEvent | None, stop: CancelEvent | None = None
    ) -> None:
        self._out = out
        self._lock = threading.Lock()
        self._cancel = cancel
        self._stop = stop

    def emit(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, default=str)
        with self._lock:
            self._out.write(line + "\n")
            self._out.flush()

    def progress(
        self, fraction: float, message: str | None = None, data: dict[str, Any] | None = None
    ) -> None:
        self.emit(
            {
                "type": "progress",
                "progress": max(0.0, min(1.0, fraction)),
                "message": message,
                "data": data,
            }
        )

    def cancelled(self) -> bool:
        return self._cancel is not None and self._cancel.is_set()

    def stop_requested(self) -> bool:
        """'Finish and keep what you have' (e.g. end a recording); distinct from cancel."""
        return self._stop is not None and self._stop.is_set()


def _protocol_stream() -> IO[str]:
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return os.fdopen(protocol_fd, "w", encoding="utf-8", buffering=1)


def main() -> int:
    out = _protocol_stream()
    if os.environ.get("XSAMPLE_BLOCK_NETWORK") == "1":
        from xsample import netguard

        netguard.install()

    try:
        spec = json.loads(sys.stdin.readline())
        kind = str(spec["kind"])
        params = dict(spec.get("params") or {})
        cancel_name = spec.get("cancel_event")
        stop_name = spec.get("stop_event")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        _Context(out, None).emit({"type": "error", "error": f"invalid job spec: {exc}"})
        return 2

    ctx = _Context(
        out,
        CancelEvent.open(cancel_name) if cancel_name else None,
        CancelEvent.open(stop_name) if stop_name else None,
    )

    from xsample.workers.tasks import TASKS, TaskCancelledError

    task = TASKS.get(kind)
    if task is None:
        ctx.emit({"type": "error", "error": f"unknown task kind {kind!r}"})
        return 2
    try:
        result = task(ctx, params)
    except (TaskCancelledError, Cancelled):
        ctx.emit({"type": "cancelled"})
        return 0
    except AudioError as exc:  # expected, user-facing: no traceback noise
        ctx.emit({"type": "error", "error": str(exc)})
        return 1
    except MemoryError:
        ctx.emit({"type": "error", "error": "MemoryError: worker memory limit exceeded"})
        return 1
    except Exception as exc:
        ctx.emit(
            {
                "type": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=20),
            }
        )
        return 1
    ctx.emit({"type": "result", "result": result})
    return 0


if __name__ == "__main__":
    sys.exit(main())
