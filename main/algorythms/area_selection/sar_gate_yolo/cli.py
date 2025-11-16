import os
import argparse
import glob
from typing import List

import cv2
import numpy as np

from .utils import read_yaml, ensure_gray_or_rgb, save_json
from .inference_gate import run_gate
from .crops import build_crops, remap_boxes
from .nms import diou_nms, soft_nms


def run_pipeline(
    images_path: str, config_path: str, save_vis: str = None, save_json_dir: str = None
):
    cfg = read_yaml(config_path)
    # Lazy import to avoid onnxruntime dependency for gate-only workflows
    from .yolo_runner import YoloONNX

    yolo = YoloONNX(cfg["yolo"]["onnx_path"])  # CPU-only
    size = int(cfg["yolo"]["size"]) if "size" in cfg["yolo"] else 640

    if os.path.isdir(images_path):
        paths = sorted(
            [
                p
                for p in glob.glob(os.path.join(images_path, "**", "*"), recursive=True)
                if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
        )
    else:
        paths = [images_path]

    results = {}
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_rgb = img

        rois = run_gate(
            img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cfg
        )
        crops, metas = build_crops(img_rgb, rois, target=size)
        if crops.shape[0] == 0:
            results[p] = []
            continue
        batch_boxes = yolo(crops)
        global_boxes = remap_boxes(batch_boxes, metas)
        all_boxes = (
            np.concatenate(global_boxes, axis=0)
            if len(global_boxes) > 0
            else np.zeros((0, 6))
        )
        if all_boxes.size > 0:
            if bool(cfg["yolo"].get("use_soft_nms", False)):
                final = soft_nms(
                    all_boxes, iou_thr=float(cfg["yolo"].get("iou_nms", 0.6))
                )
            else:
                final = diou_nms(
                    all_boxes, iou_thr=float(cfg["yolo"].get("iou_nms", 0.6))
                )
        else:
            final = all_boxes
        results[p] = final.tolist()
        if save_vis:
            os.makedirs(save_vis, exist_ok=True)
            vis = img_rgb.copy()
            for cx, cy, w, h, s, c in final:
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            out_p = os.path.join(save_vis, os.path.basename(p))
            cv2.imwrite(out_p, vis)

    if save_json_dir:
        os.makedirs(save_json_dir, exist_ok=True)
        save_json(os.path.join(save_json_dir, "results.json"), results)


def cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    ap_train = sub.add_parser("train-mininet")
    ap_train.add_argument("--data", required=True)
    ap_train.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    )
    ap_train.add_argument("--epochs", type=int, default=30)

    ap_quant = sub.add_parser("quantize-mininet")
    ap_quant.add_argument("--onnx", required=True)
    ap_quant.add_argument("--calib", required=True)
    ap_quant.add_argument("--out", required=True)
    ap_quant.add_argument("--yolo", action="store_true")
    ap_quant.add_argument("--size", type=int, default=640)

    ap_ov = sub.add_parser("convert-openvino")
    ap_ov.add_argument("--onnx", required=True)
    ap_ov.add_argument("--out", required=True)

    ap_run = sub.add_parser("run")
    ap_run.add_argument("--images", required=True)
    ap_run.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    )
    ap_run.add_argument("--save-vis")
    ap_run.add_argument("--save-json")
    ap_run.add_argument(
        "--gate-only", action="store_true", help="Run gate only and save ROI visualization"
    )

    args = ap.parse_args()

    if args.cmd == "train-mininet":
        from .train_mininet import main as train_main

        import sys

        sys.argv = [
            "train_mininet.py",
            "--data",
            args.data,
            "--config",
            args.config,
            "--epochs",
            str(args.epochs),
        ]
        return train_main()
    if args.cmd == "quantize-mininet":
        from .quantization import main as quant_main
        import sys

        sys_argv = [
            "quantization.py",
            "--onnx",
            args.onnx,
            "--calib",
            args.calib,
            "--out",
            args.out,
        ]
        if args.yolo:
            sys_argv += ["--yolo", "--size", str(args.size)]
        sys.argv = sys_argv
        return quant_main()
    if args.cmd == "convert-openvino":
        try:
            from openvino.tools.mo import convert_model
        except Exception:
            raise RuntimeError(
                "OpenVINO is not installed. Please pip install openvino."
            )
        os.makedirs(args.out, exist_ok=True)
        ir = convert_model(args.onnx)
        from openvino.runtime import serialize

        serialize(ir, os.path.join(args.out, "model.xml"))
        print(f"Saved OpenVINO IR to {args.out}")
        return
    if args.cmd == "run":
        if args.gate_only:
            cfg = read_yaml(args.config)
            import glob

            paths = (
                sorted(
                    [
                        p
                        for p in glob.glob(
                            os.path.join(args.images, "**", "*"), recursive=True
                        )
                        if p.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".bmp")
                        )
                    ]
                )
                if os.path.isdir(args.images)
                else [args.images]
            )
            os.makedirs(args.save_vis or "out/vis_gate", exist_ok=True)
            for p in paths:
                import cv2

                img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rois = run_gate(gray, cfg)
                vis = (
                    img.copy()
                    if img.ndim == 3
                    else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                )
                for x1, y1, x2, y2 in [
                    (r[0], r[1], r[2], r[3]) for r in rois
                ]:
                    cv2.rectangle(
                        vis, (x1, y1), (x2, y2), (0, 255, 255), 2
                    )
                out_p = os.path.join(
                    args.save_vis or "out/vis_gate", os.path.basename(p)
                )
                cv2.imwrite(out_p, vis)
            return
        return run_pipeline(
            args.images, args.config, args.save_vis, args.save_json
        )

    ap.print_help()


if __name__ == "__main__":
    cli()



