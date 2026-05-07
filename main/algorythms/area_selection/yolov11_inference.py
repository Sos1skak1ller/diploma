"""YOLOv11 inference helpers for ROI crops.

This module replaces the old SAR-HUB ResNet classification path in the GUI.
It runs a YOLOv11 ``.pt`` model on already cropped ROI frames and returns the
same lightweight "predictions / best" shape that the interface expects.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
import torch


_MODEL = None
_LOADED_WEIGHTS_PATH: Optional[str] = None


def get_default_yolov11_weights_path(project_root: Path) -> Path:
    """Default project-local YOLOv11m weights path."""
    return project_root / "Веса" / "YOLOv11m" / "best.pt"


def select_torch_device() -> str:
    """Prefer Apple Silicon MPS, fall back to CPU."""
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _to_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _get_model(weights_path: Path):
    """Lazy-load and cache Ultralytics YOLO model."""
    global _MODEL, _LOADED_WEIGHTS_PATH

    path_str = str(weights_path)
    if _MODEL is not None and _LOADED_WEIGHTS_PATH == path_str:
        return _MODEL

    # Keep Ultralytics runtime settings outside the repo root.
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        tempfile.gettempdir(),
    )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Не установлен ultralytics. Выполните: pip install -r requirements.txt"
        ) from exc

    _MODEL = YOLO(path_str)
    _LOADED_WEIGHTS_PATH = path_str
    return _MODEL


def extract_yolo_predictions(result: Any, max_predictions: int = 3) -> Dict[str, Any]:
    """Convert one Ultralytics result to GUI-friendly predictions."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return {"predictions": [], "best": None}

    xyxy = _to_list(getattr(boxes, "xyxy", []))
    confs = _to_list(getattr(boxes, "conf", []))
    classes = _to_list(getattr(boxes, "cls", []))
    names = getattr(result, "names", {}) or {}

    predictions: List[Dict[str, Any]] = []
    for idx, (bbox, conf, cls_idx) in enumerate(zip(xyxy, confs, classes), start=1):
        class_index = int(cls_idx)
        label = names.get(class_index, f"class_{class_index}") if isinstance(names, dict) else f"class_{class_index}"
        bbox_tuple = tuple(int(round(float(v))) for v in bbox[:4])
        predictions.append(
            {
                "rank": idx,
                "index": class_index,
                "prob": round(float(conf), 4),
                "label": label,
                "bbox": bbox_tuple,
            }
        )

    predictions.sort(key=lambda p: p["prob"], reverse=True)
    predictions = predictions[:max_predictions]
    for rank, pred in enumerate(predictions, start=1):
        pred["rank"] = rank

    return {
        "predictions": predictions,
        "best": predictions[0] if predictions else None,
    }


def detect_pil_image(
    img: Image.Image,
    weights_path: Path,
    conf: float = 0.25,
    imgsz: int = 1024,
    max_predictions: int = 3,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run YOLOv11 on one ROI crop and return top detections."""
    if img.mode != "RGB":
        img = img.convert("RGB")

    model = _get_model(weights_path)
    selected_device = device or select_torch_device()
    results = model.predict(
        source=img,
        conf=conf,
        imgsz=imgsz,
        device=selected_device,
        verbose=False,
    )
    first = results[0] if results else None
    extracted = extract_yolo_predictions(first, max_predictions=max_predictions) if first else {
        "predictions": [],
        "best": None,
    }
    extracted["weights"] = str(weights_path)
    extracted["device"] = selected_device
    return extracted


__all__ = [
    "detect_pil_image",
    "extract_yolo_predictions",
    "get_default_yolov11_weights_path",
    "select_torch_device",
]
