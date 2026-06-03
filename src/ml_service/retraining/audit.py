from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class RetrainingAuditLedger:
    """Append-only JSONL audit of retraining requests, used for the audit trail and debounce.

    Each record carries at least ``trigger`` and ``requested_at`` (ISO-8601). The default path
    lives under ``data/`` (gitignored runtime artifact); tests pass a tmp path.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def _read_all(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        with self._path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def recent(self, trigger: str, within: timedelta, now: datetime) -> list[dict[str, Any]]:
        """Emitted requests for ``trigger`` whose ``requested_at`` is within ``within`` of ``now``.

        Only emitted requests gate debounce — suppressed records never block future requests.
        """
        cutoff = now - within
        out: list[dict[str, Any]] = []
        for rec in self._read_all():
            if rec.get("trigger") != trigger or not rec.get("emitted", False):
                continue
            ts = datetime.fromisoformat(rec["requested_at"])
            if ts >= cutoff:
                out.append(rec)
        return out


class RunLedger:
    """Append-only JSONL log of orchestrated retraining runs (observability for FRAUD-116)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def all(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        with self._path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
