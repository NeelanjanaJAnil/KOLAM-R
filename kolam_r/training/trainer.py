"""Training loop and validation engine for KOLAM-R CNN Baseline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from kolam_r.dataset.loader import KolamDataset
from kolam_r.models.cnn_baseline import KolamCNNBaseline
from kolam_r.training.losses import MultiTaskKolamLoss, GRID_SIZE_TO_IDX


class KolamTrainer:
    """Manages model training, validation checkpointing, and metric history."""

    def __init__(
        self,
        model: KolamCNNBaseline,
        train_dataset: KolamDataset,
        val_dataset: KolamDataset,
        criterion: MultiTaskKolamLoss | None = None,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | None = None,
    ) -> None:
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.criterion = (criterion or MultiTaskKolamLoss()).to(self.device)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=40, eta_min=1e-5
        )

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_rule_acc": [],
            "val_rule_acc": [],
            "train_sym_acc": [],
            "val_sym_acc": [],
            "train_structural_match": [],
            "val_structural_match": [],
        }

    def _collate_batch(self, batch: tuple[Any, dict[str, Any]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Convert numpy batch to device tensors."""
        imgs, targets_raw = batch
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs).float()
        imgs = imgs.to(self.device)

        targets: dict[str, torch.Tensor] = {}
        for k, v in targets_raw.items():
            if isinstance(v, (list, tuple)):
                v = torch.tensor(v)
            elif isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            targets[k] = v.to(self.device)

        return imgs, targets

    def _compute_accuracies(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Compute batch-level classification accuracies and exact structural match."""
        pred_rule = outputs["logits_rule"].argmax(dim=-1)
        pred_sym = outputs["logits_symmetry"].argmax(dim=-1)
        pred_motif = outputs["logits_motif"].argmax(dim=-1)
        pred_depth = outputs["logits_depth"].argmax(dim=-1) + 1

        pred_grid_idx = outputs["logits_grid_size"].argmax(dim=-1)
        idx_to_grid = torch.tensor([3, 5, 7, 9], device=self.device)
        pred_grid = idx_to_grid[pred_grid_idx]

        pred_angle = outputs["pred_angle_norm"] * 180.0

        rule_correct = (pred_rule == targets["rule_id"]).float()
        sym_correct = (pred_sym == targets["symmetry"]).float()
        motif_correct = (pred_motif == targets["motif"]).float()
        depth_correct = (pred_depth == targets["recursion_depth"]).float()
        grid_correct = (pred_grid == targets["grid_size"]).float()
        angle_correct = (torch.abs(pred_angle - targets["angle"].float()) <= 5.0).float()

        # Structural Exact Match: all 5 structural params correct simultaneously
        structural_match = (
            rule_correct * sym_correct * depth_correct * grid_correct * angle_correct
        ).mean().item()

        return {
            "rule_acc": rule_correct.mean().item(),
            "sym_acc": sym_correct.mean().item(),
            "motif_acc": motif_correct.mean().item(),
            "depth_acc": depth_correct.mean().item(),
            "grid_acc": grid_correct.mean().item(),
            "angle_acc": angle_correct.mean().item(),
            "structural_match": structural_match,
        }

    def train_epoch(self) -> tuple[float, dict[str, float]]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        metrics_sum: dict[str, float] = {}

        for batch in self.train_loader:
            imgs, targets = self._collate_batch(batch)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss_dict = self.criterion(outputs, targets)
            loss = loss_dict["total_loss"]
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            accs = self._compute_accuracies(outputs, targets)
            for k, v in accs.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v * imgs.size(0)

        n_samples = len(self.train_loader.dataset)
        avg_loss = total_loss / n_samples
        avg_metrics = {k: v / n_samples for k, v in metrics_sum.items()}
        return avg_loss, avg_metrics

    def validate_epoch(self) -> tuple[float, dict[str, float]]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        metrics_sum: dict[str, float] = {}

        with torch.no_grad():
            for batch in self.val_loader:
                imgs, targets = self._collate_batch(batch)
                outputs = self.model(imgs)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict["total_loss"]

                total_loss += loss.item() * imgs.size(0)
                accs = self._compute_accuracies(outputs, targets)
                for k, v in accs.items():
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + v * imgs.size(0)

        n_samples = len(self.val_loader.dataset)
        avg_loss = total_loss / n_samples
        avg_metrics = {k: v / n_samples for k, v in metrics_sum.items()}
        return avg_loss, avg_metrics

    def train(
        self,
        num_epochs: int = 40,
        checkpoint_dir: Path | str | None = None,
        verbose: bool = True,
    ) -> Path:
        """Run full multi-epoch training and save the best model checkpoint."""
        chk_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        chk_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = chk_dir / "best_val_model.pt"

        best_val_loss = float("inf")
        best_epoch = 0

        if verbose:
            print(f"Training on device: {self.device} | {num_epochs} epochs | Batch size: {self.train_loader.batch_size}")
            print("-" * 75)

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()
            self.scheduler.step()
            dt = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_rule_acc"].append(train_metrics["rule_acc"])
            self.history["val_rule_acc"].append(val_metrics["rule_acc"])
            self.history["train_sym_acc"].append(train_metrics["sym_acc"])
            self.history["val_sym_acc"].append(val_metrics["sym_acc"])
            self.history["train_structural_match"].append(train_metrics["structural_match"])
            self.history["val_structural_match"].append(val_metrics["structural_match"])

            # Save best checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                    },
                    best_model_path,
                )

            if verbose and (epoch % 5 == 0 or epoch == num_epochs or is_best):
                marker = "*" if is_best else " "
                print(
                    f"Epoch {epoch:2d}/{num_epochs:2d} [{dt:.1f}s]{marker} | "
                    f"Train Loss: {train_loss:.4f} (Rule: {train_metrics['rule_acc']*100:.1f}%, Sym: {train_metrics['sym_acc']*100:.1f}%, StructMatch: {train_metrics['structural_match']*100:.1f}%) | "
                    f"Val Loss: {val_loss:.4f} (Rule: {val_metrics['rule_acc']*100:.1f}%, Sym: {val_metrics['sym_acc']*100:.1f}%, StructMatch: {val_metrics['structural_match']*100:.1f}%)"
                )

        if verbose:
            print("-" * 75)
            print(f"Training complete. Best model checkpoint saved to: {best_model_path} (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")

        return best_model_path

    def plot_training_curves(self, output_path: Path | str) -> Path:
        """Plot loss and accuracy training curves."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # Loss Curve
        ax = axes[0]
        ax.plot(epochs, self.history["train_loss"], label="Train Loss", color="#2563eb", lw=2)
        ax.plot(epochs, self.history["val_loss"], label="Val Loss", color="#dc2626", lw=2, linestyle="--")
        ax.set_title("Total Multi-Task Loss", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rule & Symmetry Accuracy
        ax = axes[1]
        ax.plot(epochs, [v * 100 for v in self.history["val_rule_acc"]], label="Val Rule Acc", color="#10b981", lw=2)
        ax.plot(epochs, [v * 100 for v in self.history["val_sym_acc"]], label="Val Symmetry Acc", color="#8b5cf6", lw=2)
        ax.set_title("Validation Head Accuracies", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Structural Exact Match
        ax = axes[2]
        ax.plot(epochs, [v * 100 for v in self.history["train_structural_match"]], label="Train Struct Match", color="#3b82f6", lw=1.5)
        ax.plot(epochs, [v * 100 for v in self.history["val_structural_match"]], label="Val Struct Match", color="#f59e0b", lw=2)
        ax.set_title("Exact Structural Match (All 5 Params)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Exact Match (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path
