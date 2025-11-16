import os
from typing import List, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def load_image(path: str, in_ch: int = 1) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if in_ch == 1:
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if in_ch == 3 and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def load_yolo_labels(label_path: str, img_w: int, img_h: int) -> List[Tuple[float, float, float, float, int]]:
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((cx * img_w, cy * img_h, w * img_w, h * img_h, cls))
    return boxes


def draw_heatmap_from_boxes(h: int, w: int, boxes: List[Tuple[float, float, float, float, int]], sigma: float = 3.0) -> np.ndarray:
    heat = np.zeros((h, w), dtype=np.float32)
    for cx, cy, bw, bh, _ in boxes:
        x = int(round(cx))
        y = int(round(cy))
        cv2.circle(heat, (x, y), max(1, int(0.5 * (bw + bh) / 4)), 1.0, -1)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)
    heat = np.clip(heat, 0.0, 1.0)
    return heat


class HeatmapDataset(Dataset):
    def __init__(self, root: str, in_ch: int = 1, size: int = 512):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.lbl_dir = os.path.join(root, "labels")
        self.in_ch = in_ch
        self.size = size
        self.paths = [
            os.path.join(self.img_dir, f)
            for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = load_image(p, in_ch=self.in_ch)
        h, w = img.shape[:2]
        label_path = os.path.join(
            self.lbl_dir, os.path.splitext(os.path.basename(p))[0] + ".txt"
        )
        boxes = load_yolo_labels(label_path, w, h)
        heat = draw_heatmap_from_boxes(h, w, boxes, sigma=3.0)

        if self.in_ch == 1:
            img_resized = cv2.resize(
                img, (self.size, self.size), interpolation=cv2.INTER_LINEAR
            )
            img_t = torch.from_numpy(img_resized).float().unsqueeze(0) / 255.0
        else:
            img_resized = cv2.resize(
                img, (self.size, self.size), interpolation=cv2.INTER_LINEAR
            )
            img_t = (
                torch.from_numpy(img_resized)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

        heat_resized = cv2.resize(
            heat, (img_t.shape[-1], img_t.shape[-2])
        )
        heat_t = torch.from_numpy(heat_resized).float().unsqueeze(0)
        return img_t, heat_t



