"""FastAPI application for the demo SAR image storage server."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .catalog import ImageCatalog, ImageRecord
from .config import Settings, get_settings
from .thumbnails import generate_thumbnail

logger = logging.getLogger("image_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="SAR Demo Image Server",
        version=__version__,
        description=(
            "Демонстрационный сервер‑хранилище SAR снимков. "
            "Используется системой анализа спутниковых изображений в режиме «Сервер»."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    catalog = ImageCatalog(settings.storage_dir, settings.allowed_extensions)
    indexed = catalog.refresh()
    logger.info("Storage dir: %s", settings.storage_dir)
    logger.info("Indexed %d image(s) at startup", indexed)

    app.state.settings = settings
    app.state.catalog = catalog

    def _check_api_key(x_api_key: str | None) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    def _require_record(image_id: str) -> ImageRecord:
        record = catalog.get(image_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found")
        if not record.absolute_path.is_file():
            raise HTTPException(status_code=410, detail="Image file is no longer available")
        return record

    @app.get("/")
    def root():
        return {
            "name": "SAR Demo Image Server",
            "version": __version__,
            "endpoints": {
                "health": "/api/v1/health",
                "list": "/api/v1/images",
                "raw": "/api/v1/images/{id}/raw",
                "thumbnail": "/api/v1/images/{id}/thumbnail",
                "refresh": "/api/v1/refresh",
            },
        }

    @app.get("/api/v1/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "indexed": len(catalog.list()),
            "storage_dir": str(settings.storage_dir),
        }

    @app.get("/api/v1/images")
    def list_images(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_api_key(x_api_key)
        items = [r.to_public_dict() for r in catalog.list()]
        return JSONResponse({"items": items, "count": len(items)})

    @app.post("/api/v1/refresh")
    def refresh_index(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_api_key(x_api_key)
        n = catalog.refresh()
        return {"indexed": n}

    @app.get("/api/v1/images/{image_id}")
    def get_image_meta(image_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_api_key(x_api_key)
        record = _require_record(image_id)
        return record.to_public_dict()

    @app.get("/api/v1/images/{image_id}/raw")
    def get_image_raw(image_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_api_key(x_api_key)
        record = _require_record(image_id)
        return FileResponse(
            path=str(record.absolute_path),
            media_type=record.content_type,
            filename=record.name,
        )

    @app.get("/api/v1/images/{image_id}/thumbnail")
    def get_image_thumbnail(
        image_id: str,
        size: int = Query(256, ge=32, le=1024),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        _check_api_key(x_api_key)
        record = _require_record(image_id)
        try:
            thumb = generate_thumbnail(record.absolute_path, settings.thumbnail_dir, size=size)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Thumbnail generation failed for %s", image_id)
            raise HTTPException(status_code=500, detail=f"Thumbnail error: {exc}") from exc
        return FileResponse(path=str(thumb), media_type="image/jpeg")

    return app


app = create_app()
