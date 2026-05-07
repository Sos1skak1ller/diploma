#!/usr/bin/env python3
"""Unit tests for YOLOv11 ROI inference helpers."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class _FakeArray:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def tolist(self):
        return self.value


class Yolov11InferenceTests(unittest.TestCase):
    def test_default_weights_path_points_to_yolov11m_best_pt(self):
        from main.algorythms.area_selection.yolov11_inference import (
            get_default_yolov11_weights_path,
        )

        root = Path("/project")

        self.assertEqual(
            get_default_yolov11_weights_path(root),
            root / "Веса" / "YOLOv11m" / "best.pt",
        )

    def test_select_device_prefers_mps_when_available(self):
        from main.algorythms.area_selection.yolov11_inference import select_torch_device

        with patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(select_torch_device(), "mps")

    def test_select_device_falls_back_to_cpu(self):
        from main.algorythms.area_selection.yolov11_inference import select_torch_device

        with patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(select_torch_device(), "cpu")

    def test_extract_predictions_sorts_by_confidence_and_marks_best(self):
        from main.algorythms.area_selection.yolov11_inference import (
            extract_yolo_predictions,
        )

        boxes = SimpleNamespace(
            xyxy=_FakeArray([[10, 20, 30, 40], [1, 2, 3, 4]]),
            conf=_FakeArray([0.55, 0.92]),
            cls=_FakeArray([1, 0]),
        )
        result = SimpleNamespace(names={0: "ship", 1: "land"}, boxes=boxes)

        output = extract_yolo_predictions(result, max_predictions=2)

        self.assertEqual(output["best"]["label"], "ship")
        self.assertEqual(output["best"]["prob"], 0.92)
        self.assertEqual(output["best"]["bbox"], (1, 2, 3, 4))
        self.assertEqual([p["label"] for p in output["predictions"]], ["ship", "land"])

    def test_extract_predictions_handles_empty_boxes(self):
        from main.algorythms.area_selection.yolov11_inference import (
            extract_yolo_predictions,
        )

        boxes = SimpleNamespace(
            xyxy=_FakeArray([]),
            conf=_FakeArray([]),
            cls=_FakeArray([]),
        )
        result = SimpleNamespace(names={}, boxes=boxes)

        output = extract_yolo_predictions(result)

        self.assertIsNone(output["best"])
        self.assertEqual(output["predictions"], [])


if __name__ == "__main__":
    unittest.main()
