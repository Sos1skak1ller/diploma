"""HTTP client for the demo SAR image storage server.

Used by the GUI in "Сервер" mode: list available images on the server and
download a chosen image to the local cache for further processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin

import requests


class RemoteClientError(RuntimeError):
    """Raised when the remote server returns an error or is unreachable."""


@dataclass(frozen=True)
class RemoteImage:
    id: str
    name: str
    relative_path: str
    size: int
    modified: str
    content_type: str

    @classmethod
    def from_dict(cls, data: dict) -> "RemoteImage":
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            relative_path=str(data.get("relative_path", data.get("name", ""))),
            size=int(data.get("size", 0) or 0),
            modified=str(data.get("modified", "")),
            content_type=str(data.get("content_type", "application/octet-stream")),
        )


class RemoteImageClient:
    """Thin synchronous client for the demo image server."""

    DEFAULT_TIMEOUT = 15.0
    DOWNLOAD_TIMEOUT = 60.0

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float | None = None):
        if not base_url:
            raise ValueError("base_url is required")
        self._base_url = base_url.rstrip("/") + "/"
        self._api_key = api_key or None
        self._timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self._session = requests.Session()

    @property
    def base_url(self) -> str:
        return self._base_url

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers

    def _url(self, path: str) -> str:
        return urljoin(self._base_url, path.lstrip("/"))

    def health(self) -> dict:
        try:
            resp = self._session.get(self._url("api/v1/health"), headers=self._headers(), timeout=self._timeout)
        except requests.RequestException as exc:
            raise RemoteClientError(f"Сервер недоступен: {exc}") from exc
        if resp.status_code != 200:
            raise RemoteClientError(f"Health check вернул статус {resp.status_code}")
        try:
            return resp.json()
        except ValueError as exc:
            raise RemoteClientError(f"Некорректный JSON в ответе health: {exc}") from exc

    def list_images(self) -> list[RemoteImage]:
        try:
            resp = self._session.get(self._url("api/v1/images"), headers=self._headers(), timeout=self._timeout)
        except requests.RequestException as exc:
            raise RemoteClientError(f"Не удалось получить список снимков: {exc}") from exc
        if resp.status_code == 401:
            raise RemoteClientError("Сервер требует API-ключ (или ключ неверен).")
        if resp.status_code != 200:
            raise RemoteClientError(f"Сервер ответил статусом {resp.status_code}: {resp.text[:200]}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise RemoteClientError(f"Некорректный JSON: {exc}") from exc
        items = payload.get("items", payload if isinstance(payload, list) else [])
        return [RemoteImage.from_dict(it) for it in items]

    def download(
        self,
        image_id: str,
        dest: Path,
        progress: callable | None = None,
        chunk_size: int = 64 * 1024,
    ) -> Path:
        """Download image *image_id* to *dest* (file path). Returns the resulting path.

        If *progress* is provided, it is called as ``progress(bytes_downloaded, total_bytes)``
        where ``total_bytes`` may be 0 if the server does not report ``Content-Length``.
        """
        if not image_id:
            raise ValueError("image_id is required")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        url = self._url(f"api/v1/images/{image_id}/raw")
        try:
            with self._session.get(url, headers=self._headers(), stream=True, timeout=self.DOWNLOAD_TIMEOUT) as resp:
                if resp.status_code == 404:
                    raise RemoteClientError(f"Снимок '{image_id}' не найден на сервере")
                if resp.status_code == 401:
                    raise RemoteClientError("Сервер требует API-ключ (или ключ неверен).")
                if resp.status_code != 200:
                    raise RemoteClientError(f"Сервер ответил статусом {resp.status_code}")

                total = int(resp.headers.get("Content-Length", "0") or 0)
                downloaded = 0
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if progress is not None:
                            try:
                                progress(downloaded, total)
                            except Exception:
                                pass
                tmp.replace(dest)
        except requests.RequestException as exc:
            raise RemoteClientError(f"Ошибка скачивания: {exc}") from exc
        return dest

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "RemoteImageClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
