"""Fail closed when a supervised Modal snapshot emits a fatal platform log."""

from __future__ import annotations

import re
from pathlib import Path


class SnapshotLogGuard:
    def __init__(self, path: Path, *, maximum_bytes: int = 4_000_000):
        self.path = path
        self.maximum_bytes = maximum_bytes
        self.offset = 0
        self.carry = ""
        self.failure: str | None = None

    def check(self) -> str | None:
        if self.failure is not None:
            return self.failure
        if self.path.is_symlink() or not self.path.is_file():
            self.failure = "platform_log_unavailable"
        elif self.path.stat().st_size > self.maximum_bytes:
            self.failure = "platform_log_exceeded_bound"
        elif self.path.stat().st_size < self.offset:
            self.failure = "platform_log_truncated"
        else:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                chunk = stream.read(self.maximum_bytes + 1)
                self.offset = stream.tell()
            if self.offset > self.maximum_bytes:
                self.failure = "platform_log_exceeded_bound"
                return self.failure
            text = self.carry + chunk.decode("utf-8", errors="replace")
            if re.search(
                r"(?:failed|failure)[^\n]*(?:snapshot|checkpoint)"
                r"|(?:snapshot|checkpoint)[^\n]*(?:failed|failure)"
                r"|snapshot capture allowance exhausted",
                text,
                re.IGNORECASE,
            ):
                self.failure = "snapshot_platform_failure"
            elif re.search(r"runner failed with exception", text, re.IGNORECASE):
                self.failure = "snapshot_worker_failure"
            elif re.search(r"(?:retry|retri)[^\n]*snapshot", text, re.IGNORECASE):
                self.failure = "snapshot_platform_retry"
            self.carry = text[-4096:]
        return self.failure
