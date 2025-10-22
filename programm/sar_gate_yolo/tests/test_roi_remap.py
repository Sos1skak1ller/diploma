import numpy as np
from ..crops import build_crops, remap_boxes


def test_roundtrip_remap():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    rois = [(100, 50, 300, 250)]
    crops, metas = build_crops(img, rois, target=200)
    # Fake YOLO box on crop center
    cx = cy = 100
    w = h = 50
    boxes = [np.array([[cx, cy, w, h, 0.9, 0]])]
    glob = remap_boxes(boxes, metas)[0]
    assert glob.shape[1] == 6
    # Center should map near original ROI center
    gx, gy = glob[0, 0], glob[0, 1]
    assert 100 <= gx <= 300
    assert 50 <= gy <= 250


