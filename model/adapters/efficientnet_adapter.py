"""efficientnet_adapter.py — PlantClassifier adapter for the model registry.

Loads the trained EfficientNet-B0 from model/weights/ and exposes the
standard predict() interface expected by the registry.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from model.classifier import PlantClassifier

WEIGHTS_PATH     = Path("model/weights/best_model.pth")
CLASS_NAMES_PATH = Path("model/weights/class_names.json")


class EfficientNetAdapter:
    """Wraps PlantClassifier for use via the model registry."""

    def __init__(self) -> None:
        class_names = json.loads(CLASS_NAMES_PATH.read_text())
        self._classifier = PlantClassifier(class_names, weights_path=WEIGHTS_PATH)

    def predict(self, image: Image.Image) -> dict:
        """Returns {"species": str, "confidence": float, "top3": list[dict]}."""
        return self._classifier.predict(image)

    @property
    def model(self):
        """Underlying nn.Module — used by the /explain Grad-CAM endpoint."""
        return self._classifier.model
