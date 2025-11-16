from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MiniObjNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 24, blocks: int = 3, stride2: bool = False):
        super().__init__()
        ch = base
        layers = [ConvBNReLU(in_ch, ch, 3, 2 if stride2 else 1, 1)]
        for _ in range(blocks - 1):
            layers.append(ConvBNReLU(ch, ch, 3, 1, 1))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        logits = self.head(x)
        return logits


def export_onnx(model: nn.Module, out_path: str, in_ch: int = 1, h: int = 512, w: int = 512):
    model.eval()
    dummy = torch.randn(1, in_ch, h, w)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={"input": {2: "h", 3: "w"}, "logits": {2: "h_out", 3: "w_out"}},
    )



