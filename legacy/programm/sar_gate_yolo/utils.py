import os
import time
import math
import json
import contextlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import cv2
import yaml


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_gray_or_rgb(img: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if channels == 3:
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported channels={channels}")


def to_log_amplitude(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    return np.log1p(np.maximum(img, 0.0))


@contextlib.contextmanager
def timeit(name: str):
    t0 = time.time()
    yield
    dt = (time.time() - t0) * 1000.0
    print(f"[{name}] {dt:.2f} ms")


def set_num_threads(n_threads: Optional[int] = None):
    if n_threads is None:
        return
    try:
        import torch

        torch.set_num_threads(n_threads)
    except Exception:
        pass
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")


def letterbox(img: np.ndarray, new_shape: int = 640, color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    shape = img.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw //= 2
    dh //= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh, dh
    left, right = dw, dw
    if img.ndim == 2:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color[0])
    else:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def overlay_mask(img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.4) -> np.ndarray:
    if img.ndim == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
    m = (mask > 0).astype(np.uint8)
    overlay = vis.copy()
    overlay[m > 0] = (overlay[m > 0] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return overlay


def save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def set_seed(seed: int = 42):
    import random

    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


