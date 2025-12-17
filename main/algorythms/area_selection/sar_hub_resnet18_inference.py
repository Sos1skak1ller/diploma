import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from torch import nn
from PIL import Image
from torchvision import models, transforms


_DEVICE = torch.device("cpu")
_MODEL: Optional[torch.nn.Module] = None
_PREPROCESS: Optional[transforms.Compose] = None
_LABELS: Optional[List[str]] = None
_LOADED_WEIGHTS_PATH: Optional[str] = None
_LOADED_LABELS_PATH: Optional[str] = None


def build_preprocess(image_size: int = 128) -> transforms.Compose:
    """
    Preprocessing pipeline for SAR-HUB ResNet-18 TSX.
    
    Для TSX‑чекпоинта используем вход 128×128, поэтому
    приводим изображение ровно к этому размеру.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # При необходимости можно заменить нормализацию на SAR-специфичную
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_labels(labels_path: Optional[Path]) -> Optional[List[str]]:
    if labels_path is None:
        return None
    with labels_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_sarhub_resnet18(weights_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Load SAR-HUB ResNet-18 (SAR-pretrained) weights into a torchvision ResNet-18.
    
    Для TSX‑чекпоинта голова имеет 32 класса (fc.weight: [32, 512]).
    Соответственно, переопределяем последний полносвязный слой под 32 класса.
    """
    model = models.resnet18()
    # Заменяем классификатор под 32 класса, как в чекпоинте TSX
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 32)

    checkpoint = torch.load(str(weights_path), map_location=device)

    # Many research repos save weights under "state_dict" and/or with "module." prefix.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        cleaned_state_dict[k] = v

    # strict=False на случай, если в чекпоинте есть дополнительные ключи
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.to(device)
    model.eval()

    return model


def _get_preprocess() -> transforms.Compose:
    global _PREPROCESS
    if _PREPROCESS is None:
        _PREPROCESS = build_preprocess()
    return _PREPROCESS


def _get_model(weights_path: Path) -> torch.nn.Module:
    """
    Lazy loader/cacher for SAR-HUB ResNet-18 model.
    Reuses already загруженную модель, если путь к весам не изменился.
    """
    global _MODEL, _LOADED_WEIGHTS_PATH
    path_str = str(weights_path)
    if _MODEL is None or _LOADED_WEIGHTS_PATH != path_str:
        _MODEL = load_sarhub_resnet18(weights_path, _DEVICE)
        _LOADED_WEIGHTS_PATH = path_str
    return _MODEL


def _get_labels(labels_path: Optional[Path]) -> Optional[List[str]]:
    global _LABELS, _LOADED_LABELS_PATH
    if labels_path is None:
        _LABELS = None
        _LOADED_LABELS_PATH = None
        return None

    path_str = str(labels_path)
    if _LABELS is None or _LOADED_LABELS_PATH != path_str:
        _LABELS = load_labels(labels_path)
        _LOADED_LABELS_PATH = path_str
    return _LABELS


def classify_pil_image(
    img: Image.Image,
    weights_path: Path,
    labels_path: Optional[Path] = None,
    topk: int = 5,
) -> Dict[str, Any]:
    """
    Классификация уже загруженного PIL-изображения (например, обрезанного ROI).
    Используется в GUI для работы с зонами интересов.
    """
    model = _get_model(weights_path)
    preprocess = _get_preprocess()
    labels = _get_labels(labels_path)

    if img.mode != "RGB":
        img = img.convert("L").convert("RGB")

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        logits = model(input_batch)
        probs = torch.softmax(logits, dim=1)[0]

    topk = min(topk, probs.numel())
    top_probs, top_indices = torch.topk(probs, topk)

    predictions = []
    for rank, (p, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1):
        if labels is not None and idx < len(labels):
            label = labels[idx]
        else:
            label = f"class_{idx}"
        predictions.append(
            {
                "rank": rank,
                "index": int(idx),
                "prob": float(p),
                "label": label,
            }
        )

    return {
        "weights": str(weights_path),
        "device": str(_DEVICE),
        "predictions": predictions,
    }


def classify_image(
    image_path: Path,
    weights_path: Path,
    labels_path: Optional[Path] = None,
    topk: int = 5,
) -> Dict[str, Any]:
    """
    High-level helper for GUI/CLI: классифицирует изображение по пути и
    возвращает структуру данных с топ-предсказаниями, БЕЗ print.
    """
    img = Image.open(image_path)
    result = classify_pil_image(
        img=img,
        weights_path=weights_path,
        labels_path=labels_path,
        topk=topk,
    )

    return {
        "image": str(image_path),
        "weights": result["weights"],
        "device": result["device"],
        "predictions": result["predictions"],
    }


def run_inference(
    weights_path: Path,
    image_path: Path,
    labels_path: Optional[Path] = None,
    topk: int = 5,
) -> None:
    """
    CLI-обёртка вокруг classify_image: оставлена для совместимости при запуске из консоли.
    """
    result = classify_image(
        image_path=image_path,
        weights_path=weights_path,
        labels_path=labels_path,
        topk=topk,
    )

    print(f"Image: {result['image']}")
    print(f"Weights: {result['weights']}")
    print(f"Device: {result['device']}")
    print("Top predictions:")
    for p in result["predictions"]:
        print(f"{p['rank']:2d}. {p['label']:20s} prob={p['prob']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with SAR-HUB ResNet-18 (SAR-pretrained) on CPU."
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to SAR-HUB ResNet-18 weights file (e.g. resnet18_I_nwpu_cate45.pth).",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input SAR image (any format supported by Pillow).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional path to text file with class labels (one label per line).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="How many top predictions to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    image_path = Path(args.image)
    labels_path = Path(args.labels) if args.labels is not None else None

    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if labels_path is not None and not labels_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    run_inference(
        weights_path=weights_path,
        image_path=image_path,
        labels_path=labels_path,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()


