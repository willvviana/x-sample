"""Cross-process cancellation flag backed by a named Windows event.

A thread blocked reading a pipe on Windows can stall other operations on that handle
(observed as workers hanging during imports), so cancellation does not use stdin.
"""

from __future__ import annotations

import ctypes
import secrets
from ctypes import wintypes

_SYNCHRONIZE = 0x00100000
_EVENT_MODIFY_STATE = 0x0002
_WAIT_OBJECT_0 = 0

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateEventW.restype = wintypes.HANDLE
_kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
_kernel32.OpenEventW.restype = wintypes.HANDLE
_kernel32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
_kernel32.SetEvent.restype = wintypes.BOOL
_kernel32.SetEvent.argtypes = [wintypes.HANDLE]
_kernel32.WaitForSingleObject.restype = wintypes.DWORD
_kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_kernel32.CloseHandle.restype = wintypes.BOOL
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


class CancelEvent:
    def __init__(self, handle: int, name: str) -> None:
        self._handle: int | None = handle
        self.name = name

    @classmethod
    def create(cls) -> CancelEvent:
        name = f"Local\\xsample-cancel-{secrets.token_hex(16)}"
        handle = _kernel32.CreateEventW(None, True, False, name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        return cls(handle, name)

    @classmethod
    def open(cls, name: str) -> CancelEvent:
        handle = _kernel32.OpenEventW(_SYNCHRONIZE | _EVENT_MODIFY_STATE, False, name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        return cls(handle, name)

    def set(self) -> None:
        if self._handle:
            _kernel32.SetEvent(self._handle)

    def is_set(self) -> bool:
        if not self._handle:
            return False
        return bool(_kernel32.WaitForSingleObject(self._handle, 0) == _WAIT_OBJECT_0)

    def close(self) -> None:
        if self._handle:
            _kernel32.CloseHandle(self._handle)
            self._handle = None
