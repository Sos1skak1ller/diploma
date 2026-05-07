"""Lazy thumbnail generation backed by Pillow with on-disk cache."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps


def _cache_path(cache_dir: Path, source: Path, size: int) -> Path:
    key = hashlib.sha1(f"{source.resolve()}::{int(source.stat().st_mtime)}::{size}".encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.jpg"


def generate_thumbnail(source: Path, cache_dir: Path, size: int = 256) -> Path:
    """Return a path to a JPEG thumbnail for *source*, generating it if missing.

    Raises FileNotFoundError if the source does not exist.
    """
    if not source.is_file():
        raise FileNotFoundError(str(source))

    size = max(32, min(int(size), 1024))
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _cache_path(cache_dir, source, size)
    if target.is_file():
        return target

    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.thumbnail((size, size))
        save_kwargs = {"format": "JPEG", "quality": 85, "optimize": True}
        if im.mode == "L":
            im.save(target, **save_kwargs)
        else:
            im.convert("RGB").save(target, **save_kwargs)

    return target
