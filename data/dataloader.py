"""dataloader.py — HuggingFace dataset loader for plant species classification.

Filters the 47-class ``kakasher/house-plant-species`` dataset down to
TARGET_CLASSES, remaps labels 0-N, and returns train / val / test DataLoaders.

Usage::

    train_loader, val_loader, test_loader, class_names = get_dataloaders()
"""
from __future__ import annotations

import re
from typing import Callable

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Canonical target classes (index 0-6 in the model output) ─────────────────
TARGET_CLASSES: list[str] = [
    "Aloe Vera",
    "Monstera Deliciosa",
    "Dumb Cane (Dieffenbachia)",
    "Pothos (Epipremnum aureum)",
    "Peace Lily (Spathiphyllum wallisii)",
    "Snake Plant (Sansevieria)",
    "ZZ Plant (Zamioculcas zamiifolia)",
]

# Per-class keyword aliases used for fuzzy matching against dataset label names.
# Keys are canonical names; values are lower-case substrings that indicate a match.
_ALIASES: dict[str, list[str]] = {
    "Aloe Vera": ["aloe vera", "aloe"],
    "Monstera Deliciosa": ["monstera deliciosa", "monstera"],
    "Dumb Cane (Dieffenbachia)": ["dumb cane", "dieffenbachia"],
    "Pothos (Epipremnum aureum)": ["pothos", "epipremnum aureum", "golden pothos"],
    "Peace Lily (Spathiphyllum wallisii)": ["peace lily", "spathiphyllum"],
    "Snake Plant (Sansevieria)": ["snake plant", "sansevieria", "dracaena trifasciata"],
    "ZZ Plant (Zamioculcas zamiifolia)": ["zz plant", "zamioculcas"],
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

_EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


def _normalize(s: str) -> str:
    """Lowercase and strip non-alphanumeric chars for fuzzy comparison."""
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _find_canonical(dataset_label: str) -> str | None:
    """
    Map a raw dataset label string to one of the TARGET_CLASSES canonical names.

    Returns None if no alias matches.
    """
    norm = _normalize(dataset_label)
    for canonical, aliases in _ALIASES.items():
        if any(alias in norm or norm in alias for alias in aliases):
            return canonical
    return None


class _PlantSubset(Dataset):
    """
    Lazy-loading Dataset that wraps a slice of a HuggingFace dataset.

    Stores (hf_index, remapped_label) pairs; images are decoded on demand so
    the full dataset is never loaded into RAM at once.
    """

    def __init__(
        self,
        hf_dataset,
        records: list[tuple[int, int]],
        transform: Callable,
        image_key: str = "image",
    ) -> None:
        self.hf_dataset = hf_dataset
        self.records = records
        self.transform = transform
        self.image_key = image_key

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        hf_idx, label = self.records[idx]
        img = self.hf_dataset[hf_idx][self.image_key]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return self.transform(img.convert("RGB")), label


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Load ``kakasher/house-plant-species``, filter to TARGET_CLASSES, remap
    labels 0-N, and return DataLoaders plus the ordered class name list.

    Split: 70 % train · 15 % val · 15 % test (randomly shuffled, seeded).

    Args:
        batch_size:  Samples per batch.
        num_workers: DataLoader worker count (set 0 on macOS if you hit
                     multiprocessing errors).
        seed:        RNG seed for reproducible splits.

    Returns:
        ``(train_loader, val_loader, test_loader, class_names)`` where
        ``class_names[i]`` is the canonical species name for label *i*.
    """
    raw = load_dataset("kakasher/house-plant-species", split="train")
    label_feature = raw.features.get("label") or raw.features.get("labels")

    # ── Detect image field name (WebDataset stores images as "jpg", not "image") ─
    _IMAGE_KEYS = ("image", "jpg", "jpeg", "png", "webp")
    first = raw[0]
    image_key = next((k for k in _IMAGE_KEYS if k in first), None)
    if image_key is None:
        raise RuntimeError(
            f"No image field found in dataset. Available keys: {list(first.keys())}"
        )

    # ── Build {original_int → canonical_name} ────────────────────────────────
    if hasattr(label_feature, "names"):
        # ClassLabel feature: example["label"] returns an int
        int_to_name: dict[int, str] = {i: n for i, n in enumerate(label_feature.names)}
        label_is_int = True
    else:
        # String-valued labels: enumerate unique values
        unique = sorted({str(ex["__key__"]) for ex in raw})
        int_to_name = {i: v for i, v in enumerate(unique)}
        str_to_int  = {v: k for k, v in int_to_name.items()}
        label_is_int = False

    # ── Remap original ints → 0-based target indices ──────────────────────────
    orig_to_new: dict[int, int] = {}
    class_names: list[str] = []

    for orig_int, dataset_name in int_to_name.items():
        canonical = _find_canonical(dataset_name)
        if canonical is None:
            continue
        if canonical not in class_names:
            class_names.append(canonical)
        orig_to_new[orig_int] = class_names.index(canonical)

    if not class_names:
        sample = list(int_to_name.values())[:10]
        raise RuntimeError(
            "No target classes matched any dataset labels.\n"
            f"Sample dataset labels: {sample}\n"
            "Adjust TARGET_CLASSES or _ALIASES in dataloader.py."
        )

    # ── Single pass: collect valid (hf_index, remapped_label) pairs ──────────
    valid_orig = set(orig_to_new)
    all_records: list[tuple[int, int]] = []

    for hf_idx, example in enumerate(raw):
        raw_label = example["label"] if label_is_int else example["__key__"]
        orig_int: int | None = int(raw_label) if label_is_int else str_to_int.get(str(raw_label))
        if orig_int is not None and orig_int in valid_orig:
            all_records.append((hf_idx, orig_to_new[orig_int]))

    # ── Reproducible 70 / 15 / 15 split ──────────────────────────────────────
    total   = len(all_records)
    n_train = int(0.70 * total)
    n_val   = int(0.15 * total)
    n_test  = total - n_train - n_val

    rng  = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=rng).tolist()
    shuffled = [all_records[i] for i in perm]

    train_rec = shuffled[:n_train]
    val_rec   = shuffled[n_train : n_train + n_val]
    test_rec  = shuffled[n_train + n_val :]

    train_ds = _PlantSubset(raw, train_rec, _TRAIN_TRANSFORMS, image_key)
    val_ds   = _PlantSubset(raw, val_rec,   _EVAL_TRANSFORMS, image_key)
    test_ds  = _PlantSubset(raw, test_rec,  _EVAL_TRANSFORMS, image_key)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        class_names,
    )
