"""Standard-library logging setup with an optional per-run log file."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(
    name: str = "amazon_recsys", log_file: str | Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Configure and return the package logger. Safe to call repeatedly."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT)

    if not any(getattr(h, "_recsys_console", False) for h in logger.handlers):
        console = logging.StreamHandler(sys.stderr)
        console._recsys_console = True  # type: ignore[attr-defined]
        console.setFormatter(formatter)
        logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        already = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path
            for h in logger.handlers
        )
        if not already:
            handler = logging.FileHandler(path)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger
