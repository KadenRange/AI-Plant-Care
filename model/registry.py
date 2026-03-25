"""registry.py — Model registry for the plant classifier API.

Maintains a catalogue of named adapter classes and loads the active one
based on the ACTIVE_MODEL environment variable (default: "efficientnet").

Usage::

    registry = ModelRegistry()
    registry.register("efficientnet", EfficientNetAdapter)
    registry.register("vit", ViTAdapter)
    registry.load_active()          # instantiates only the active adapter

    adapter = registry.active()     # the loaded instance
    registry.list()                 # ["efficientnet", "vit"]
    registry.active_name            # "efficientnet"

To swap models, set ACTIVE_MODEL=vit in the environment and restart.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type]  = {}
        self._instance: Any | None      = None
        self._active_name: str          = os.environ.get("ACTIVE_MODEL", "efficientnet")

    def register(self, name: str, adapter_class: type) -> None:
        """Register an adapter class under *name*."""
        self._classes[name] = adapter_class

    def load_active(self) -> None:
        """Instantiate the active adapter. Called once at API startup."""
        if self._active_name not in self._classes:
            raise ValueError(
                f"Unknown model '{self._active_name}'. "
                f"Available: {list(self._classes)}"
            )
        self._instance = self._classes[self._active_name]()
        logger.info("ModelRegistry: loaded adapter '%s'.", self._active_name)

    def active(self) -> Any:
        """Return the loaded adapter instance."""
        if self._instance is None:
            raise RuntimeError("No adapter loaded. Call load_active() first.")
        return self._instance

    def get(self, name: str) -> type:
        """Return the registered adapter class for *name*."""
        if name not in self._classes:
            raise KeyError(f"No adapter registered for '{name}'.")
        return self._classes[name]

    def list(self) -> list[str]:
        """Return all registered adapter names."""
        return list(self._classes)

    @property
    def active_name(self) -> str:
        return self._active_name
