"""dataloader.py — HuggingFace dataset loader for plant species classification.

Loads all classes from ``kakasher/house-plant-species`` and returns
train / val / test DataLoaders.

Usage::

    train_loader, val_loader, test_loader, class_names = get_dataloaders()
"""
from __future__ import annotations

import io
from typing import Callable

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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
        if isinstance(img, Image.Image):
            pass
        elif isinstance(img, bytes):
            img = Image.open(io.BytesIO(img))
        elif img is not None:
            img = Image.fromarray(img)
        else:
            # Corrupt/missing record — return a blank image so the batch continues
            img = Image.new("RGB", (224, 224))
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
        raw_pairs: list[tuple[int, str]] | None = None
    else:
        # String-valued labels: single pass to collect keys and cache index pairs
        # so the dataset is not iterated a second time below.
        raw_pairs = [(i, str(ex["__key__"])) for i, ex in enumerate(raw)]
        unique    = sorted({key for _, key in raw_pairs})
        int_to_name = {i: v for i, v in enumerate(unique)}
        str_to_int  = {v: k for k, v in int_to_name.items()}
        label_is_int = False

    # ── class_names is just the full sorted label list ────────────────────────
    class_names: list[str] = [int_to_name[i] for i in sorted(int_to_name)]

    # ── Collect all (hf_index, label) pairs ──────────────────────────────────
    all_records: list[tuple[int, int]] = []

    if label_is_int:
        for hf_idx, example in enumerate(raw):
            all_records.append((hf_idx, int(example["label"])))
    else:
        assert raw_pairs is not None
        for hf_idx, raw_key in raw_pairs:
            orig_int = str_to_int.get(raw_key)
            if orig_int is not None:
                all_records.append((hf_idx, orig_int))

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
