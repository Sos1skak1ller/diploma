import os
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .mini_net import MiniObjNet, export_onnx
from .utils import read_yaml, set_seed
from .dataset import HeatmapDataset


class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = targets * p + (1 - targets) * (1 - p)
        loss = ((1 - p_t) ** self.gamma) * ce
        return loss.mean()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu().item())
    return total / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, required=True, help="root with images/ and labels/"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="models/mini_net.onnx")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = read_yaml(args.config)
    in_ch = int(cfg.get("input", {}).get("channels", 1))
    stride2 = bool(cfg.get("gate", {}).get("mininet", {}).get("stride2", False))

    dataset = HeatmapDataset(args.data, in_ch=in_ch)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=0
    )

    device = torch.device("cpu")
    model = MiniObjNet(in_ch=in_ch, base=24, blocks=3, stride2=stride2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = FocalBCELoss(gamma=2.0)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"epoch {epoch+1}/{args.epochs} loss={loss:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    export_onnx(model, args.out, in_ch=in_ch, h=512, w=512)
    print(f"Saved ONNX to {args.out}")


if __name__ == "__main__":
    main()


