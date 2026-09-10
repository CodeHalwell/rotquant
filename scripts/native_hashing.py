"""Bounded-memory artifact hashing, including Python 3.10 compatibility."""
from __future__ import annotations

import hashlib
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()
