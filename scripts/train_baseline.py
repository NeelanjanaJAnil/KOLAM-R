"""Train multi-task CNN baseline on KOLAM-R canonical dataset.

Usage:
    python scripts/train_baseline.py [--epochs 40] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kolam_r.dataset.loader import KolamDataset
from kolam_r.models.cnn_baseline import KolamCNNBaseline
from kolam_r.training.losses import MultiTaskKolamLoss
from kolam_r.training.trainer import KolamTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN Baseline for KOLAM-R")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    data_dir = project_root / "data"
    splits_dir = data_dir / "splits"
    checkpoints_dir = project_root / "checkpoints"
    results_dir = project_root / "results"

    for d in [checkpoints_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.json"
    val_path = splits_dir / "val.json"

    if not train_path.exists() or not val_path.exists():
        print("Error: Dataset splits not found. Run scripts/build_dataset.py first.")
        sys.exit(1)

    print("=" * 75)
    print("KOLAM-R Stage 3 — Multi-Task CNN Baseline Training")
    print("=" * 75)

    train_dataset = KolamDataset(train_path)
    val_dataset = KolamDataset(val_path)

    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    model = KolamCNNBaseline()
    criterion = MultiTaskKolamLoss(
        weight_rule=1.0,
        weight_sym=1.0,
        weight_motif=0.5,
        weight_depth=1.0,
        weight_grid=1.0,
        weight_angle=2.0,
    )

    trainer = KolamTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        criterion=criterion,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_checkpoint = trainer.train(
        num_epochs=args.epochs,
        checkpoint_dir=checkpoints_dir,
        verbose=True,
    )

    curves_path = trainer.plot_training_curves(results_dir / "training_curves.png")
    print(f"Training curves plotted to: {curves_path}")
    print(f"Best model weights saved to: {best_checkpoint}")


if __name__ == "__main__":
    main()
