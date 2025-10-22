import numpy as np
from ..cfar import cfar_ca


def test_cfar_ca_detects_points():
    img = np.zeros((64, 64), dtype=np.float32)
    pts = [(10, 10), (20, 40), (50, 30)]
    for (x, y) in pts:
        img[y, x] = 10.0
    mask = cfar_ca(img, guard=1, train=4, k=2.0)
    for (x, y) in pts:
        assert mask[y, x] > 0


