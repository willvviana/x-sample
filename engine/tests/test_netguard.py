from __future__ import annotations

import socket

import pytest

from xsample import netguard


def test_guard_is_active_in_tests() -> None:
    assert netguard.is_installed()


def test_external_connection_is_blocked() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(netguard.NetworkBlockedError),
    ):
        sock.connect(("192.0.2.1", 80))


def test_loopback_connection_is_allowed() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=2):
            pass
