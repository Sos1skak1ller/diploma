from typing import Dict, Tuple

import numpy as np
import cv2


def compute_tile_features(img: np.ndarray, tile: int = 128, stride: int = 64) -> Dict[str, np.ndarray]:
    """Fast tile features using local mean/variance via box filters.

    - energy ~ local mean of squared values
    - entropy proxy ~ normalized local variance
    """
    img32 = img.astype(np.float32)
    ksize = (tile, tile)
    mean = cv2.boxFilter(img32, ddepth=-1, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(img32 * img32, ddepth=-1, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(0.0, mean_sq - mean * mean)
    energy = cv2.normalize(mean_sq, None, 0, 1, cv2.NORM_MINMAX)
    entropy_proxy = cv2.normalize(var, None, 0, 1, cv2.NORM_MINMAX)
    mask = ((energy > 0.5) | (entropy_proxy > 0.5)).astype(np.uint8) * 255
    return {"energy": energy, "entropy": entropy_proxy, "mask": mask}


