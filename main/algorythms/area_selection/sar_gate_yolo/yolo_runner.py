from typing import Dict, List, Tuple

import numpy as np
import cv2

try:  # onnxruntime опционален: нужен только для YOLO
    import onnxruntime as ort  # type: ignore[import]
except Exception:
    ort = None


class YoloONNX:
    def __init__(self, onnx_path: str, num_threads: int = None):
        if ort is None:
            raise RuntimeError(
                "onnxruntime не установлен. Установите пакет onnxruntime, "
                "чтобы использовать YOLO-детектор."
            )
        sess_opts = ort.SessionOptions()
        if num_threads is not None:
            sess_opts.intra_op_num_threads = num_threads
            sess_opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.in_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name

    def __call__(self, images: np.ndarray) -> List[np.ndarray]:
        # images: (N, H, W, 3) uint8
        x = images.astype(np.float32) / 255.0
        x = x.transpose(0, 3, 1, 2)
        y = self.session.run([self.out_name], {self.in_name: x})[0]
        # Expecting output per image as [num, 6] [cx,cy,w,h,score,cls]
        if isinstance(y, list):
            return [yy for yy in y]
        if y.ndim == 3:
            return [y[i] for i in range(y.shape[0])]
        return [y]



