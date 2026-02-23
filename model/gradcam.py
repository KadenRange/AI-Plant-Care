"""gradcam.py — Grad-CAM visualization for the plant species classifier.

Produces a heatmap overlay on the input image highlighting the spatial regions
the model used when making its prediction.

Public interface::

    overlay = generate_heatmap(pil_image, model)
    overlay.show()                    # PIL Image, 224×224

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for a CNN.

    Registers forward and backward hooks on *target_layer* to capture
    activations and gradients. Call ``generate()`` for inference, then
    ``remove_hooks()`` when done (or use ``generate_heatmap()`` which handles
    cleanup automatically).

    Args:
        model:        Trained ``nn.Module``. EfficientNet-B0 expected.
        target_layer: Conv layer to visualize. Defaults to the last block in
                      ``model.features`` (EfficientNet-B0 ``features[-1]``).
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model  = model
        self.device = next(model.parameters()).device

        layer = target_layer if target_layer is not None else model.features[-1]

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        self._fwd_hook = layer.register_forward_hook(self._save_activations)
        self._bwd_hook = layer.register_full_backward_hook(self._save_gradients)

    # ── Hook callbacks ────────────────────────────────────────────────────────

    def _save_activations(self, _module, _input, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output: tuple) -> None:
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Detach forward and backward hooks. Call after inference is done."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    # ── Core Grad-CAM ─────────────────────────────────────────────────────────

    def generate(
        self,
        image: Image.Image,
        class_idx: Optional[int] = None,
        alpha: float = 0.5,
    ) -> Image.Image:
        """
        Compute Grad-CAM for *image* and blend it with the resized original.

        Args:
            image:     Input PIL image (any mode; converted to RGB internally).
            class_idx: Class index to explain. ``None`` uses the predicted class.
            alpha:     Blend weight for the heatmap overlay (0 = original only,
                       1 = heatmap only). Default 0.5.

        Returns:
            224×224 PIL image with the Grad-CAM heatmap blended onto the input.
        """
        self.model.eval()
        tensor = _PREPROCESS(image.convert("RGB")).unsqueeze(0).to(self.device)

        # Forward — retain graph so we can call backward on a scalar
        logits = self.model(tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward pass for the chosen class score
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global-average-pool gradients → channel importance weights
        # activations: (1, C, H, W)   gradients: (1, C, H, W)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze()    # (H, W)
        cam = torch.relu(cam).cpu().numpy()                         # negative → 0

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        # Resize to 224×224 and apply jet colormap
        cam_img  = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        cam_arr  = np.array(cam_img, dtype=np.float32) / 255.0     # [0, 1]

        # Jet colormap (manual, no matplotlib dependency):
        #   0.0 → blue  0.25 → cyan  0.5 → green  0.75 → yellow  1.0 → red
        r = np.clip(1.5 - np.abs(cam_arr * 4.0 - 3.0), 0, 1)
        g = np.clip(1.5 - np.abs(cam_arr * 4.0 - 2.0), 0, 1)
        b = np.clip(1.5 - np.abs(cam_arr * 4.0 - 1.0), 0, 1)
        heatmap_rgb = Image.fromarray(
            (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)
        )

        # Blend with the (resized) original
        base = image.convert("RGB").resize((224, 224), Image.BILINEAR)
        return Image.blend(base, heatmap_rgb, alpha=alpha)


# ── Convenience function (public interface) ───────────────────────────────────

def generate_heatmap(
    image: Image.Image,
    model: nn.Module,
    class_idx: Optional[int] = None,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Run Grad-CAM on *image* and return a heatmap overlay as a PIL image.

    Hooks are automatically registered and removed around inference.

    Args:
        image:     Input PIL image.
        model:     Trained plant classifier (``nn.Module`` in eval mode).
        class_idx: Class to explain. ``None`` → the model's top prediction.
        alpha:     Heatmap blend strength (0–1). Default 0.5.

    Returns:
        224×224 PIL image with the Grad-CAM overlay.
    """
    cam = GradCAM(model)
    try:
        overlay = cam.generate(image, class_idx=class_idx, alpha=alpha)
    finally:
        cam.remove_hooks()
    return overlay
