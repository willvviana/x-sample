"""Block outbound network connections from this process (tests and offline workers).

Only loopback destinations are allowed. This guards Python-level sockets; native
libraries that open their own sockets are not covered.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

_installed = False
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex


class NetworkBlockedError(OSError):
    pass


def _is_loopback(address: Any) -> bool:
    if not isinstance(address, tuple) or not address:
        # AF_UNIX paths and similar are local by definition.
        return True
    host = str(address[0])
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def _guarded_connect(self: socket.socket, address: Any) -> None:
    if not _is_loopback(address):
        raise NetworkBlockedError(f"network access blocked by X-Sample: {address!r}")
    _original_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
    if not _is_loopback(address):
        raise NetworkBlockedError(f"network access blocked by X-Sample: {address!r}")
    return _original_connect_ex(self, address)


def install() -> None:
    global _installed
    if _installed:
        return
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign,assignment]
    socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign,assignment]
    _installed = True


def is_installed() -> bool:
    return _installed
