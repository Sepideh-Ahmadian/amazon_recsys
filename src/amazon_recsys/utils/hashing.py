"""Stable hashes for configs, files and prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def hash_text(text: str) -> str:
    """SHA-256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_obj(obj: Any) -> str:
    """Hash any JSON-serialisable object; independent of dict key order."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hash_text(payload)


def hash_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
