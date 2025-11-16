from typing import Tuple

import numpy as np
import cv2


def cfar_ca(img: np.ndarray, guard: int, train: int, k: float) -> np.ndarray:
    """Vectorized CA-CFAR via box filters.

    - Compute mean over (guard+train) window and over guard window.
    - Estimate training mean as (mean_win*area_win - mean_guard*area_guard) / (area_win - area_guard).
    - Threshold = k * mean_train.
    """
    img32 = img.astype(np.float32)
    g = int(guard)
    t = int(train)
    win = g + t
    ksize_win = (2 * win + 1, 2 * win + 1)
    ksize_g = (2 * g + 1, 2 * g + 1)

    mean_win = cv2.boxFilter(
        img32,
        ddepth=-1,
        ksize=ksize_win,
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    if g > 0:
        mean_guard = cv2.boxFilter(
            img32,
            ddepth=-1,
            ksize=ksize_g,
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        area_win = float(ksize_win[0] * ksize_win[1])
        area_guard = float(ksize_g[0] * ksize_g[1])
        sum_train = mean_win * area_win - mean_guard * area_guard
        area_train = max(1.0, area_win - area_guard)
        mean_train = sum_train / area_train
    else:
        mean_train = mean_win

    thr = mean_train * float(k)
    mask = (img32 > thr).astype(np.uint8) * 255
    return mask


def cfar_os(img: np.ndarray, guard: int, train: int, q: float, k: float) -> np.ndarray:
    # Approximate OS-CFAR by using a quantile filter on the train window
    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    win = guard + train
    ksize = 2 * win + 1
    # Compute rolling quantile via OpenCV medianBlur for q=0.5 or fallback to percentile using patches
    if abs(q - 0.5) < 1e-6:
        # OpenCV medianBlur requires 8-bit input; normalize consistently
        img_u8 = cv2.normalize(
            img, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        bg_u8 = cv2.medianBlur(img_u8, ksize)
        # Threshold in the same domain to maintain ratio consistency
        thr = (bg_u8.astype(np.float32)) * float(k)
        mask[img_u8.astype(np.float32) > thr] = 255
        return mask
    else:
        pad = win
        padded = cv2.copyMakeBorder(
            img,
            pad,
            pad,
            pad,
            pad,
            borderType=cv2.BORDER_REFLECT,
        )
        bg = np.zeros_like(img, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                ys = y
                xs = x
                patch = padded[
                    ys : ys + 2 * win + 1, xs : xs + 2 * win + 1
                ].astype(np.float32)
                # mask out guard cells
                gy1 = win - guard
                gy2 = win + guard + 1
                gx1 = win - guard
                gx2 = win + guard + 1
                patch[gy1:gy2, gx1:gx2] = np.nan
                vals = patch.reshape(-1)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    bg[y, x] = 0.0
                else:
                    bg[y, x] = np.percentile(vals, q * 100.0)
    thr = bg * k
    mask[img.astype(np.float32) > thr] = 255
    return mask



