"""vit_adapter.py — HuggingFace ViT adapter for the model registry.

Uses google/vit-base-patch16-224 via the transformers image-classification
pipeline. No custom weights — pretrained ImageNet labels only.

Activate by setting the environment variable:
    ACTIVE_MODEL=vit
"""
from __future__ import annotations

from PIL import Image


class ViTAdapter:
    """Wraps a HuggingFace ViT pipeline for use via the model registry."""

    def __init__(self) -> None:
        from transformers import pipeline
        self._pipeline = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",
        )

    def predict(self, image: Image.Image) -> dict:
        """Returns {"species": str, "confidence": float, "top3": list[dict]}."""
        results = self._pipeline(image.convert("RGB"), top_k=3)
        top3 = [
            {"species": r["label"], "confidence": round(r["score"], 4)}
            for r in results
        ]
        return {
            "species":    top3[0]["species"],
            "confidence": top3[0]["confidence"],
            "top3":       top3,
        }

    @property
    def model(self):
        """Underlying nn.Module — exposed for Grad-CAM compatibility.

        Note: generate_heatmap() targets model.features[-1] which is
        EfficientNet-specific. /explain will raise a runtime error when
        this adapter is active.
        """
        return self._pipeline.model
