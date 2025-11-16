from typing import Dict, List, Tuple
import os

import numpy as np
import cv2

from .tiles import compute_tile_features
from .cfar import cfar_ca, cfar_os


class MiniNetONNX:
    def __init__(self, onnx_path: str, num_threads: int = None):
        try:
            import onnxruntime as ort  # lazy import to avoid hard dependency for gate-only
        except Exception as e:
            raise RuntimeError(f"onnxruntime not available: {e}")
        sess_opts = ort.SessionOptions()
        if num_threads is not None:
            sess_opts.intra_op_num_threads = num_threads
            sess_opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.session.get_inputs()[0]
        self.in_name = meta.name
        self.out_name = self.session.get_outputs()[0].name

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # img: (H, W) or (H, W, 3) in [0,255]
        if img.ndim == 2:
            x = img[None, None, :, :].astype(np.float32) / 255.0
        else:
            x = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        y = self.session.run([self.out_name], {self.in_name: x})[0]
        return y[0, 0]


def unify_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    mask = ((mask_a > 0) | (mask_b > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    return mask


def connected_components_from_heatmap(
    heat: np.ndarray, thr: float, dilate_iter: int = 1
) -> List[Tuple[int, int, int, int, float]]:
    prob = 1.0 / (1.0 + np.exp(-heat))
    binm = (prob >= thr).astype(np.uint8)
    binm = cv2.dilate(
        binm,
        np.ones((3, 3), np.uint8),
        iterations=max(1, int(dilate_iter)),
    )
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binm, connectivity=8
    )
    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area <= 0:
            continue
        m = prob[y : y + h, x : x + w]
        score = float(np.max(m))
        boxes.append((x, y, x + w, y + h, score))
    return boxes


def run_gate(img: np.ndarray, cfg: Dict) -> List[Tuple[int, int, int, int, float]]:
    # Step A: tile + CFAR
    if cfg.get("input", {}).get("log_transform", True):
        img_proc = np.log1p(img.astype(np.float32))
    else:
        img_proc = img.astype(np.float32)
    ksize = int(cfg.get("input", {}).get("median_ksize", 0))
    if ksize and ksize >= 3:
        img_proc = cv2.medianBlur(img_proc.astype(np.float32), ksize)
    tiles = compute_tile_features(img_proc.astype(np.float32))

    cfar_cfg = cfg.get("gate", {}).get("cfar", {})
    if cfar_cfg.get("enabled", True):
        guard = int(cfar_cfg.get("guard", 2))
        train = int(cfar_cfg.get("train", 8))
        k_val = float(cfar_cfg.get("k", 4.0))
        q_val = float(cfar_cfg.get("q", 0.75))
        scales = cfar_cfg.get("scales")
        masks = []
        if isinstance(scales, list) and len(scales) > 0:
            for s in scales:
                s = float(s)
                small = (
                    img_proc
                    if s == 1.0
                    else cv2.resize(
                        img_proc,
                        (
                            int(img_proc.shape[1] * s),
                            int(img_proc.shape[0] * s),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                )
                if cfar_cfg.get("type", "OS").upper() == "OS":
                    m_small = cfar_os(
                        small, guard=guard, train=train, q=q_val, k=k_val
                    )
                else:
                    m_small = cfar_ca(
                        small, guard=guard, train=train, k=k_val
                    )
                m_full = (
                    m_small
                    if s == 1.0
                    else cv2.resize(
                        m_small,
                        (img_proc.shape[1], img_proc.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                )
                masks.append(m_full)
            mask_cfar = np.zeros_like(img_proc, dtype=np.uint8)
            for m in masks:
                mask_cfar = cv2.bitwise_or(mask_cfar, m)
        else:
            scale = float(cfar_cfg.get("scale", 1.0))
            small = (
                img_proc
                if scale == 1.0
                else cv2.resize(
                    img_proc,
                    (
                        int(img_proc.shape[1] * scale),
                        int(img_proc.shape[0] * scale),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            )
            if cfar_cfg.get("type", "OS").upper() == "OS":
                mask_small = cfar_os(
                    small, guard=guard, train=train, q=q_val, k=k_val
                )
            else:
                mask_small = cfar_ca(
                    small, guard=guard, train=train, k=k_val
                )
            mask_cfar = (
                mask_small
                if scale == 1.0
                else cv2.resize(
                    mask_small,
                    (img_proc.shape[1], img_proc.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
    else:
        mask_cfar = np.zeros_like(tiles["mask"])
    mask_a = unify_mask(tiles["mask"], mask_cfar)

    # Step B: mini net over whole frame (fallback to Step A components if model not present)
    mini_cfg = cfg.get("gate", {}).get("mininet", {})
    mini_path = mini_cfg.get("onnx_path", "models/mini_net.onnx")
    comps: List[Tuple[int, int, int, int, float]]
    use_mininet = False
    if mini_path and os.path.exists(mini_path):
        try:
            _ = MiniNetONNX(mini_path)
            use_mininet = True
        except Exception:
            use_mininet = False
    roi_cfg = cfg.get("gate", {}).get("roi", {})
    dilate_iter = int(roi_cfg.get("dilate", 2))
    if use_mininet:
        mininet = MiniNetONNX(mini_path)
        heat = mininet(img)
        thr = float(mini_cfg.get("threshold", 0.4))
        comps = connected_components_from_heatmap(
            heat, thr=thr, dilate_iter=dilate_iter
        )
    else:
        # Fallback: use Step A mask to form components
        binm = (mask_a > 0).astype(np.uint8)
        binm = cv2.dilate(
            binm,
            np.ones((3, 3), np.uint8),
            iterations=max(1, dilate_iter),
        )
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            binm, connectivity=8
        )
        comps = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area <= 0:
                continue
            comps.append((x, y, x + w, y + h, 0.5))

    # Filter by area and aspect ratio
    roi_cfg = cfg.get("gate", {}).get("roi", {})
    min_area = int(roi_cfg.get("min_area", 12))
    max_area = int(roi_cfg.get("max_area", 0))  # 0 = no limit
    min_aspect = float(roi_cfg.get("min_aspect", 0.1))
    max_aspect = float(roi_cfg.get("max_aspect", 10.0))
    filtered: List[Tuple[int, int, int, int, float]] = []
    for x1, y1, x2, y2, s in comps:
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        ar = (w + 1e-6) / (h + 1e-6)
        if ar < min_aspect or ar > max_aspect:
            continue
        filtered.append((x1, y1, x2, y2, s))

    # Simple NMS on comps using IoU
    def iou_xyxy(
        a: Tuple[int, int, int, int, float],
        b: Tuple[int, int, int, int, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
        bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    nms_iou = float(roi_cfg.get("nms_iou", 0.6))
    filtered.sort(key=lambda x: x[4], reverse=True)
    comps_nms: List[Tuple[int, int, int, int, float]] = []
    for cand in filtered:
        if all(iou_xyxy(cand, kept) < nms_iou for kept in comps_nms):
            comps_nms.append(cand)

    # Top-k
    # Жёсткий колпак topk для скорости, если в конфиге указали слишком большой
    topk = min(200, int(roi_cfg.get("topk", 100)))
    comps = comps_nms[:topk]

    # Padding and max_side
    pad = int(roi_cfg.get("pad", 16))
    max_side = int(roi_cfg.get("max_side", 512))
    h, w = img.shape[:2]
    rois = []
    for x1, y1, x2, y2, s in comps:
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        # split if too large
        bw = x2 - x1
        bh = y2 - y1
        if max(bw, bh) > max_side:
            # tile into chunks
            step = max_side
            for ty in range(y1, y2, step):
                for tx in range(x1, x2, step):
                    rois.append(
                        (tx, ty, min(tx + step, w), min(ty + step, h))
                    )
        else:
            rois.append((x1, y1, x2, y2))
    return rois


__all__ = ["run_gate"]



