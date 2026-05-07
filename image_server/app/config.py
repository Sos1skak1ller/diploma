"""Settings for the demo image server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_DEFAULT_STORAGE = Path(__file__).resolve().parent.parent / "storage"


@dataclass(frozen=True)
class Settings:
    storage_dir: Path
    api_key: str | None
    host: str
    port: int
    thumbnail_dir: Path
    allowed_extensions: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        storage_env = os.environ.get("STORAGE_DIR")
        storage_dir = Path(storage_env).expanduser() if storage_env else _DEFAULT_STORAGE
        storage_dir = storage_dir.resolve()

        thumb_env = os.environ.get("THUMBNAIL_DIR")
        if thumb_env:
            thumbnail_dir = Path(thumb_env).expanduser().resolve()
        else:
            thumbnail_dir = (storage_dir.parent / ".thumbnails").resolve()

        api_key = os.environ.get("API_KEY") or None
        host = os.environ.get("HOST", "0.0.0.0")
        try:
            port = int(os.environ.get("PORT", "8000"))
        except ValueError:
            port = 8000

        exts_env = os.environ.get("ALLOWED_EXTENSIONS")
        if exts_env:
            allowed = tuple(
                e.strip().lower() if e.startswith(".") else f".{e.strip().lower()}"
                for e in exts_env.split(",")
                if e.strip()
            )
        else:
            allowed = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        return cls(
            storage_dir=storage_dir,
            api_key=api_key,
            host=host,
            port=port,
            thumbnail_dir=thumbnail_dir,
            allowed_extensions=allowed,
        )


def get_settings() -> Settings:
    return Settings.from_env()
