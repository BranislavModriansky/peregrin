from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class LogEntry:
    level: str          # "debug" | "info" | "warn" | "error"
    timestamp: str
    message: str


class LogQueue:
    """Module-level singleton log buffer.

    Any module can push messages; the Shiny UI layer drains them.
    """

    _instance: LogQueue | None = None

    def __new__(cls) -> LogQueue:
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._entries = deque(maxlen=2000)
            cls._instance = obj
        return cls._instance

    def push(self, level: str, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._entries.append(LogEntry(level=level, timestamp=ts, message=message))

    def drain(self) -> list[LogEntry]:
        """Return all buffered entries and clear the buffer."""
        items = list(self._entries)
        self._entries.clear()
        return items

    def peek(self) -> list[LogEntry]:
        """Return all buffered entries *without* clearing."""
        return list(self._entries)

    @property
    def count(self) -> int:
        return len(self._entries)


# ── logging.Handler that writes into the singleton queue ──────────────

class _QueueHandler(logging.Handler):
    """Bridges Python's logging module → LogQueue singleton."""

    LEVEL_MAP = {
        logging.DEBUG:    "debug",
        logging.INFO:     "info",
        logging.WARNING:  "warn",
        logging.ERROR:    "error",
        logging.CRITICAL: "error",
        logging.NOTSET:   "brkpnt",
    }

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            level = self.LEVEL_MAP.get(record.levelno, "info")
            LogQueue().push(level, msg)
        except Exception:
            self.handleError(record)


def get_logger(name: str = "peregrin") -> logging.Logger:
    """Return a logger pre-configured with the queue handler.

    Safe to call repeatedly — the handler is attached only once.
    """
    logger = logging.getLogger(name)
    if not any(isinstance(h, _QueueHandler) for h in logger.handlers):
        handler = _QueueHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger