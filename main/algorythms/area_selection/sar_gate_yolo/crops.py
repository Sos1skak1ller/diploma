from typing import Dict, List, Tuple

import numpy as np
import cv2


def build_crops(
    img: np.ndarray, rois: List[Tuple[int, int, int, int]], target: int = 640
) -> Tuple[np.ndarray, List[Dict]]:
    crops = []
    metas = []
    for (x1, y1, x2, y2) in rois:
        crop = img[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if crop.ndim == 2:
            crop_c = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        else:
            crop_c = crop
        # letterbox to target
        scale = min(target / h, target / w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(
            crop_c, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        canvas = np.full((target, target, 3), 114, dtype=resized.dtype)
        dx = (target - new_w) // 2
        dy = (target - new_h) // 2
        canvas[dy : dy + new_h, dx : dx + new_w] = resized
        crops.append(canvas)
        metas.append(
            {
                "roi": (x1, y1, x2, y2),
                "scale": scale,
                "pad": (dx, dy),
                "target": target,
            }
        )
    if len(crops) == 0:
        return np.zeros((0, target, target, 3), dtype=np.uint8), metas
    return np.stack(crops, axis=0), metas


def remap_boxes(local_boxes: List[np.ndarray], metas: List[Dict]) -> List[np.ndarray]:
    global_boxes = []
    for boxes, meta in zip(local_boxes, metas):
        x1, y1, x2, y2 = meta["roi"]
        scale = meta["scale"]
        dx, dy = meta["pad"]
        target = meta["target"]
        # boxes: [cx, cy, w, h, score, cls] in pixels on letterboxed crop
        if boxes.size == 0:
            global_boxes.append(boxes)
            continue
        b = boxes.copy()
        # remove pad, invert scale
        b[:, 0] = (b[:, 0] - dx) / scale + x1
        b[:, 1] = (b[:, 1] - dy) / scale + y1
        b[:, 2] = b[:, 2] / scale
        b[:, 3] = b[:, 3] / scale
        global_boxes.append(b)
    return global_boxes



