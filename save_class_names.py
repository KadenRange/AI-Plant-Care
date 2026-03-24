"""save_class_names.py — Generate model/weights/class_names.json from the dataset.

Run this once to produce the class names file the API needs at startup.
Does not train — only loads the dataset metadata.

    python save_class_names.py
"""
import json
from pathlib import Path

from datasets import load_dataset

WEIGHTS_DIR = Path("model/weights")
OUT_PATH    = WEIGHTS_DIR / "class_names.json"

raw   = load_dataset("kakasher/house-plant-species", split="train")
keys  = [str(ex["__key__"]).split("/")[1] for ex in raw]
class_names = sorted(set(keys))

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(class_names, indent=2))

print(f"Saved {len(class_names)} classes to {OUT_PATH}")
print(class_names)
