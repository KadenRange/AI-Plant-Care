"""train.py — Training entry point for the plant species classifier.

Trains only the EfficientNet-B0 classifier head (backbone frozen) for up to
10 epochs using AdamW + CrossEntropyLoss + ReduceLROnPlateau. The checkpoint
with the highest validation accuracy is saved to ``model/weights/best_model.pth``.

Weights & Biases logging is enabled by default (``USE_WANDB = True``).
If wandb is not installed or you want plain stdout-only training, set
``USE_WANDB = False`` — everything else works identically.

Run::

    python train.py
    python train.py --epochs 20 --batch-size 64 --lr 5e-4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.dataloader import get_dataloaders
from model.classifier import build_model

# ── W&B toggle ────────────────────────────────────────────────────────────────
# Set to False to disable all wandb calls without removing any code.
USE_WANDB = True

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    if USE_WANDB:
        print(
            "[train] wandb not installed — logging disabled.\n"
            "        Install with: pip install wandb"
        )

# ── Hyperparameter config (single source of truth; used by wandb and CLI) ─────
# Edit here to change defaults; CLI flags override individual keys at runtime.
HPARAMS: dict = {
    # Optimisation
    "epochs":             10,
    "batch_size":         32,
    "lr":                 1e-3,
    "weight_decay":       1e-4,
    "scheduler":          "ReduceLROnPlateau",
    "scheduler_factor":   0.5,
    "scheduler_patience": 2,
    # Data
    "dataset":            "kakasher/house-plant-species",
    "num_workers":        0,
    "seed":               42,
    "input_size":         224,
    # Model
    "model":              "efficientnet_b0",
    "freeze_backbone":    True,
    "num_classes":        7,         # updated to actual value after dataset load
}

WEIGHTS_DIR = Path("model/weights")


# ── W&B helpers ───────────────────────────────────────────────────────────────

def _wb_active() -> bool:
    """True only when wandb is installed, enabled, and a run is live."""
    return USE_WANDB and _WANDB_AVAILABLE and wandb.run is not None


# ── Per-epoch helpers ─────────────────────────────────────────────────────────

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """One training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Evaluate on val or test split. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ── Main training routine ─────────────────────────────────────────────────────

def train(
    epochs: int      = HPARAMS["epochs"],
    batch_size: int  = HPARAMS["batch_size"],
    lr: float        = HPARAMS["lr"],
    num_workers: int = HPARAMS["num_workers"],
    seed: int        = HPARAMS["seed"],
) -> None:
    """Full training run with stdout logging and optional W&B integration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading dataset …")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(
        f"Samples — train: {len(train_loader.dataset)}"
        f"  val: {len(val_loader.dataset)}"
        f"  test: {len(test_loader.dataset)}"
    )

    # Merge runtime values into config before passing to wandb
    run_cfg = {
        **HPARAMS,
        "epochs":      epochs,
        "batch_size":  batch_size,
        "lr":          lr,
        "num_workers": num_workers,
        "seed":        seed,
        "num_classes": num_classes,
        "classes":     class_names,
        "device":      device,
    }

    # ── W&B init ──────────────────────────────────────────────────────────────
    if USE_WANDB and _WANDB_AVAILABLE:
        wandb.init(
            project="plant-classifier",
            config=run_cfg,
            # Group runs by model architecture for easy sweep comparison
            group=HPARAMS["model"],
        )
        # Keep a clean alias so we can refer to the config via wandb.config
        cfg = wandb.config
    else:
        cfg = run_cfg   # plain dict fallback, same key access pattern

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes, freeze_backbone=True).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = AdamW(trainable, lr=lr, weight_decay=HPARAMS["weight_decay"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=HPARAMS["scheduler_factor"],
        patience=HPARAMS["scheduler_patience"],
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path    = WEIGHTS_DIR / "best_model.pth"
    best_val_acc = 0.0

    header = f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}"
    sep    = "─" * len(header)
    print(f"\n{header}\n{sep}")

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = _train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = _evaluate(model, val_loader, criterion, device)

        scheduler.step(v_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        flag = " ✓" if v_acc > best_val_acc else ""
        print(
            f"{epoch:>6}  {t_loss:>10.4f}  {t_acc:>8.2%}  "
            f"{v_loss:>8.4f}  {v_acc:>6.2%}  {current_lr:>8.2e}{flag}"
        )

        # ── W&B per-epoch metrics ─────────────────────────────────────────────
        if _wb_active():
            wandb.log({
                "epoch":      epoch,
                "train/loss": t_loss,
                "train/acc":  t_acc,
                "val/loss":   v_loss,
                "val/acc":    v_acc,
                "lr":         current_lr,
            }, step=epoch)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_path)

    print(f"\nBest val accuracy: {best_val_acc:.2%}  →  {best_path}")

    # ── Final test evaluation on best checkpoint ───────────────────────────────
    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    test_loss, test_acc = _evaluate(model, test_loader, criterion, device)
    print(f"Test  accuracy:    {test_acc:.2%}  |  Test loss: {test_loss:.4f}")

    # ── W&B: summary metrics + model artifact ─────────────────────────────────
    if _wb_active():
        # Summary metrics appear in the run overview panel
        wandb.run.summary["best_val_acc"] = best_val_acc
        wandb.run.summary["test_acc"]     = test_acc
        wandb.run.summary["test_loss"]    = test_loss

        # Log final test metrics as their own step so they appear on charts
        wandb.log({
            "test/loss": test_loss,
            "test/acc":  test_acc,
        })

        # Save the best checkpoint as a versioned model artifact
        artifact = wandb.Artifact(
            name="plant-classifier-best",
            type="model",
            description="Best EfficientNet-B0 checkpoint (highest val accuracy)",
            metadata={
                "best_val_acc": best_val_acc,
                "test_acc":     test_acc,
                "classes":      class_names,
                "epochs":       epochs,
            },
        )
        artifact.add_file(str(best_path))
        wandb.log_artifact(artifact)

        wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the plant species classifier")
    parser.add_argument("--epochs",      type=int,   default=HPARAMS["epochs"],      help=f"Training epochs (default {HPARAMS['epochs']})")
    parser.add_argument("--batch-size",  type=int,   default=HPARAMS["batch_size"],  help=f"Batch size (default {HPARAMS['batch_size']})")
    parser.add_argument("--lr",          type=float, default=HPARAMS["lr"],          help=f"AdamW learning rate (default {HPARAMS['lr']})")
    parser.add_argument("--num-workers", type=int,   default=HPARAMS["num_workers"], help="DataLoader workers (use 0 on macOS if issues)")
    parser.add_argument("--seed",        type=int,   default=HPARAMS["seed"],        help=f"Random seed (default {HPARAMS['seed']})")
    parser.add_argument("--no-wandb",    action="store_true",                        help="Disable W&B logging for this run")
    args = parser.parse_args()

    if args.no_wandb:
        USE_WANDB = False

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
    )
