"""classifier.py — EfficientNet-B0 plant species classifier.

``PlantClassifier.predict()`` is the primary public interface consumed by
the Streamlit UI and any downstream code.

Typical usage::

    clf = PlantClassifier(class_names, weights_path="model/weights/best_model.pth")
    result = clf.predict(pil_image)
    # {"species": "Pothos (Epipremnum aureum)", "confidence": 0.9421, "top3": [...]}
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "weights" / "best_model.pth"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Construct an EfficientNet-B0 with a custom classification head.

    Args:
        num_classes:      Number of output classes.
        freeze_backbone:  If True, freeze all layers except the classifier head.
                          Set False when loading weights for inference or fine-tuning.

    Returns:
        Configured ``nn.Module`` with requires_grad set appropriately.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # EfficientNet-B0 classifier: Sequential(Dropout(0.2), Linear(1280, 1000))
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    # Classifier head is always trainable (re-enable after freeze_backbone loop)
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def load_model(
    num_classes: int,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    device: Optional[str] = None,
) -> nn.Module:
    """
    Load a trained model state dict from disk and set it to eval mode.

    Args:
        num_classes:   Must match the value used during training.
        weights_path:  Path to the saved ``state_dict`` (``.pth`` file).
        device:        ``'cuda'``, ``'cpu'``, or ``None`` (auto-detect).

    Returns:
        Model in eval mode placed on *device*.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(num_classes, freeze_backbone=False)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


class PlantClassifier:
    """
    High-level inference wrapper around the trained EfficientNet-B0.

    This is the API boundary for the Streamlit UI — create one instance at
    startup and call ``predict()`` for every image.

    Args:
        class_names:   Ordered list of species names matching the training
                       label mapping (returned by ``get_dataloaders()``).
        weights_path:  Path to the saved model weights.
        device:        ``'cuda'``, ``'cpu'``, or ``None`` (auto-detect).
    """

    def __init__(
        self,
        class_names: list[str],
        weights_path: Path = DEFAULT_WEIGHTS_PATH,
        device: Optional[str] = None,
    ) -> None:
        self.class_names = class_names
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(len(class_names), weights_path, self.device)

    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a single PIL image.

        If a W&B run is active (``wandb.run is not None``), the top-1 species
        and confidence are logged as ``prediction/species`` and
        ``prediction/confidence`` so evaluation sessions appear on the W&B
        dashboard automatically.

        Args:
            image: Any-mode PIL image (converted to RGB internally).

        Returns:
            ``{
                "species":    str,          # top-1 predicted class name
                "confidence": float,        # top-1 softmax probability [0, 1]
                "top3":       list[dict],   # up to 3 {"species", "confidence"} dicts
            }``
        """
        tensor = (
            _INFERENCE_TRANSFORMS(image.convert("RGB"))
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        k = min(3, len(self.class_names))
        top_probs, top_idxs = torch.topk(probs, k=k)

        top3 = [
            {
                "species":    self.class_names[idx.item()],
                "confidence": round(prob.item(), 4),
            }
            for prob, idx in zip(top_probs, top_idxs)
        ]

        result = {
            "species":    top3[0]["species"],
            "confidence": top3[0]["confidence"],
            "top3":       top3,
        }

        # Optional W&B prediction logging — no-ops when wandb is absent or no
        # run is active, so the Streamlit UI and offline use are unaffected.
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "prediction/species":    result["species"],
                    "prediction/confidence": result["confidence"],
                })
        except ImportError:
            pass

        return result
