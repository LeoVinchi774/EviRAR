"""Writable user cache paths, kept outside packaged card resources."""
from __future__ import annotations

import os
from pathlib import Path


def cache_file(name: str) -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    directory = root / "evirar"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name
