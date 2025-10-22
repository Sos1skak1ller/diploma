import torch
from ..mini_net import MiniObjNet


def test_forward_output_shape():
    model = MiniObjNet(in_ch=1, base=24, blocks=3, stride2=False)
    x = torch.randn(2, 1, 128, 160)
    y = model(x)
    assert y.shape[:2] == (2, 1)
    assert y.shape[-2:] == (128, 160)


