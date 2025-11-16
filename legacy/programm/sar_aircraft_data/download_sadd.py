import os
import sys
import textwrap


def main():
    target_root = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(target_root, "images")
    labels_dir = os.path.join(target_root, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    msg = f"""
    SADD dataset preparation helper
    --------------------------------
    This script creates the expected folders:
      - {images_dir}
      - {labels_dir}

    Please download the SAR Aircraft Detection Dataset (SADD) from the official source and
    extract images to 'images' and YOLO txt labels to 'labels'.

    Source repo: https://github.com/hust-rslab/SAR-aircraft-data
    BaiduYun: https://pan.baidu.com/s/11SBfMkGszu3Lr_Pe1lIEkg (Code: d2uw)

    After extraction, you can train the mini-gate network:
      python -m sar_gate_yolo.train_mininet --data {os.path.join(target_root, '..')} --config sar_gate_yolo/config.yaml
    """
    print(textwrap.dedent(msg))


if __name__ == "__main__":
    sys.exit(main())


