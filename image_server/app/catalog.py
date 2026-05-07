"""Storage catalog: scans the storage dir, indexes images by stable id."""

from __future__ import annotations

import hashlib
import mimetypes
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    id: str
    name: str
    relative_path: str
    absolute_path: Path
    size: int
    modified: str
    content_type: str

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "relative_path": self.relative_path,
            "size": self.size,
            "modified": self.modified,
            "content_type": self.content_type,
        }


class ImageCatalog:
    """Thread-safe catalog of images on disk.

    The catalog assigns a stable id (sha1 of the relative path) to each file
    so that the GUI can address files by id rather than by potentially
    sensitive absolute paths.
    """

    def __init__(self, storage_dir: Path, allowed_extensions: tuple[str, ...]):
        self._storage_dir = storage_dir
        self._allowed = tuple(e.lower() for e in allowed_extensions)
        self._lock = threading.RLock()
        self._records: dict[str, ImageRecord] = {}

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    @staticmethod
    def _stable_id(relative_path: str) -> str:
        digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()
        return digest[:16]

    def refresh(self) -> int:
        """Re-scan the storage directory. Returns the number of files indexed."""
        records: dict[str, ImageRecord] = {}
        if not self._storage_dir.exists():
            with self._lock:
                self._records = records
            return 0

        for path in sorted(self._storage_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self._allowed:
                continue
            try:
                rel = path.relative_to(self._storage_dir).as_posix()
            except ValueError:
                continue
            stat = path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            ctype, _ = mimetypes.guess_type(path.name)
            record = ImageRecord(
                id=self._stable_id(rel),
                name=path.name,
                relative_path=rel,
                absolute_path=path.resolve(),
                size=stat.st_size,
                modified=modified,
                content_type=ctype or "application/octet-stream",
            )
            records[record.id] = record

        with self._lock:
            self._records = records
        return len(records)

    def list(self) -> list[ImageRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.relative_path)

    def get(self, image_id: str) -> ImageRecord | None:
        with self._lock:
            return self._records.get(image_id)
