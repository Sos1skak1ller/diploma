from typing import List

import numpy as np


def diou_nms(boxes: np.ndarray, iou_thr: float = 0.6) -> np.ndarray:
    # boxes: [cx, cy, w, h, score, cls]
    if boxes.size == 0:
        return boxes
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    scores = boxes[:, 4]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_o = (x2[order[1:]] - x1[order[1:]]) * (
            y2[order[1:]] - y1[order[1:]]
        )
        union = area_i + area_o - inter + 1e-6
        iou = inter / union
        # DIoU term
        cx_i = boxes[i, 0]
        cy_i = boxes[i, 1]
        cx_o = boxes[order[1:], 0]
        cy_o = boxes[order[1:], 1]
        cw = np.maximum(x2[i], x2[order[1:]]) - np.minimum(
            x1[i], x1[order[1:]]
        )
        ch = np.maximum(y2[i], y2[order[1:]]) - np.minimum(
            y1[i], y1[order[1:]]
        )
        c2 = cw * cw + ch * ch + 1e-6
        d2 = (cx_i - cx_o) ** 2 + (cy_i - cy_o) ** 2
        diou = iou - d2 / c2
        inds = np.where(diou <= iou_thr)[0]
        order = order[inds + 1]
    return boxes[keep]


def soft_nms(
    boxes: np.ndarray,
    iou_thr: float = 0.6,
    sigma: float = 0.5,
    score_thresh: float = 0.001,
) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    dets = boxes.copy()
    N = dets.shape[0]
    for i in range(N):
        maxpos = i + np.argmax(dets[i:, 4])
        dets[[i, maxpos]] = dets[[maxpos, i]]
        x1 = dets[i, 0] - dets[i, 2] / 2
        y1 = dets[i, 1] - dets[i, 3] / 2
        x2 = dets[i, 0] + dets[i, 2] / 2
        y2 = dets[i, 1] + dets[i, 3] / 2
        area_i = (x2 - x1) * (y2 - y1)
        for j in range(i + 1, N):
            xx1 = max(x1, dets[j, 0] - dets[j, 2] / 2)
            yy1 = max(y1, dets[j, 1] - dets[j, 3] / 2)
            xx2 = min(x2, dets[j, 0] + dets[j, 2] / 2)
            yy2 = min(y2, dets[j, 1] + dets[j, 3] / 2)
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            area_j = dets[j, 2] * dets[j, 3]
            union = area_i + area_j - inter + 1e-6
            iou = inter / union
            dets[j, 4] *= np.exp(-(iou * iou) / sigma)
    keep = dets[:, 4] > score_thresh
    return dets[keep]



