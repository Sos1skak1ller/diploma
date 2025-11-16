import numpy as np
from ..crops import build_crops, remap_boxes
from ..nms import diou_nms


def test_end2end_dummy():
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    rois = [(0, 0, 256, 256), (256, 256, 512, 512)]
    crops, metas = build_crops(img, rois, target=256)
    # Mock YOLO outputs: one per crop
    b1 = np.array([[128, 128, 40, 40, 0.9, 0]])
    b2 = np.array([[128, 128, 40, 40, 0.8, 0]])
    glob = remap_boxes([b1, b2], metas)
    allb = np.concatenate(glob, axis=0)
    out = diou_nms(allb, iou_thr=0.6)
    assert out.shape[1] == 6
    assert out.shape[0] >= 1


