import os
import argparse
import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static


class ImageFolderDataReader(CalibrationDataReader):
    def __init__(self, folder: str, input_name: str, size: int = 512, channels: int = 3):
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        self.input_name = input_name
        self.size = size
        self.channels = channels
        self.iter = iter(self.files)

    def get_next(self):
        try:
            import cv2

            path = next(self.iter)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            img = cv2.resize(img, (self.size, self.size))
            if self.channels == 1 and img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.channels == 3 and img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            x = img.astype(np.float32) / 255.0
            if self.channels == 1:
                x = x[None, None, :, :]
            else:
                x = x.transpose(2, 0, 1)[None]
            return {self.input_name: x}
        except StopIteration:
            return None


def quantize_mininet(onnx_in: str, calib_dir: str, onnx_out: str):
    model = onnx.load(onnx_in)
    input_name = model.graph.input[0].name
    dr = ImageFolderDataReader(calib_dir, input_name, size=512, channels=1)
    quantize_static(onnx_in, onnx_out, dr, weight_type=QuantType.QInt8)


def quantize_yolo(onnx_in: str, calib_dir: str, onnx_out: str, size: int = 640):
    model = onnx.load(onnx_in)
    input_name = model.graph.input[0].name
    dr = ImageFolderDataReader(calib_dir, input_name, size=size, channels=3)
    quantize_static(onnx_in, onnx_out, dr, weight_type=QuantType.QInt8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yolo", action="store_true", help="Quantize YOLO instead of MiniNet")
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.yolo:
        quantize_yolo(args.onnx, args.calib, args.out, size=args.size)
    else:
        quantize_mininet(args.onnx, args.calib, args.out)
    print(f"Saved INT8 model to {args.out}")


if __name__ == "__main__":
    main()


