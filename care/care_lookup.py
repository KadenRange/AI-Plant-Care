"""care_lookup.py — Plant care instruction lookup.

``get_care()`` is the public interface for the Streamlit UI. It accepts the
species name returned by ``PlantClassifier.predict()`` and returns the
corresponding care guide from ``care_data.json``.

Usage::

    care = get_care("Pothos (Epipremnum aureum)")
    print(care["watering"])
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_CARE_DATA_PATH = Path(__file__).parent / "care_data.json"


@lru_cache(maxsize=1)
def _load() -> dict[str, dict]:
    """Load and cache care_data.json. Executed at most once per process."""
    with open(_CARE_DATA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize(s: str) -> str:
    """Lowercase and strip non-alphanumeric chars for comparison."""
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def get_care(species_name: str) -> dict:
    """
    Return care instructions for the given species.

    Performs a case-insensitive exact match first, then falls back to
    substring matching so that partial names (e.g. ``"Snake Plant"``) still
    resolve when the JSON key includes a parenthetical suffix.

    Args:
        species_name: Species name as returned by ``PlantClassifier.predict()``.

    Returns:
        Care instruction dict with keys:
        ``watering``, ``light``, ``humidity``, ``temperature``,
        ``common_issues``, ``toxicity``.

        On lookup failure returns ``{"error": "<message>"}``.
    """
    data = _load()
    norm_query = _normalize(species_name)

    # 1. Exact match (case-insensitive)
    for key, value in data.items():
        if _normalize(key) == norm_query:
            return value

    # 2. Substring match in either direction
    for key, value in data.items():
        norm_key = _normalize(key)
        if norm_query in norm_key or norm_key in norm_query:
            return value

    return {"error": f"No care data found for '{species_name}'."}
