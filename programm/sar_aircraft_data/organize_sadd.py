import os
import shutil
import argparse


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMG_EXTS


def main():
    ap = argparse.ArgumentParser(description="Organize extracted SADD into images/ and labels/")
    ap.add_argument("--src", required=True, help="Path to extracted SADD root (will be scanned recursively)")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(root, "images")
    labels_dir = os.path.join(root, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    moved_images = 0
    moved_labels = 0
    for dirpath, _, filenames in os.walk(args.src):
        for fn in filenames:
            src_path = os.path.join(dirpath, fn)
            name, ext = os.path.splitext(fn)
            if is_image(src_path):
                dst = os.path.join(images_dir, fn)
                if not os.path.exists(dst):
                    shutil.copy2(src_path, dst)
                    moved_images += 1
            elif ext.lower() == ".txt":
                dst = os.path.join(labels_dir, fn)
                if not os.path.exists(dst):
                    shutil.copy2(src_path, dst)
                    moved_labels += 1

    print(f"Copied images: {moved_images}, labels: {moved_labels}")
    # Sanity check: matching basenames
    imgs = {os.path.splitext(f)[0] for f in os.listdir(images_dir)}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(labels_dir)}
    missing_labels = sorted(list(imgs - lbls))
    if missing_labels:
        print(f"Warning: {len(missing_labels)} images have no matching labels (first 10): {missing_labels[:10]}")
    else:
        print("All images have matching labels.")


if __name__ == "__main__":
    main()


