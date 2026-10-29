from __future__ import annotations


class AudioError(Exception):
    """A user-facing audio problem (bad file, missing tool, device failure)."""


class Cancelled(Exception):
    """The operation was cancelled by the user."""
