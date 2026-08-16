"""Benchmark evaluator across clean, held-out depth, and corrupted test splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from kolam_r.dataset.corruption import CorruptionType
from kolam_r.dataset.loader import KolamDataset
from kolam_r.evaluation.metrics import (
    compute_benchmark_metrics,
    plot_confusion_matrices,
)
from kolam_r.models.cnn_baseline import KolamCNNBaseline


class BaselineEvaluator:
    """Evaluates a trained KolamCNNBaseline across multiple benchmark splits."""

    def __init__(
        self,
        model: KolamCNNBaseline,
        device: str | None = None,
    ) -> None:
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.model.eval()

    def predict_dataset(
        self,
        dataset: KolamDataset,
        batch_size: int = 32,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Run batched inference on a dataset and return prediction and target arrays."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds_rule, preds_sym, preds_motif, preds_depth, preds_grid, preds_angle = [], [], [], [], [], []
        t_rule, t_sym, t_motif, t_depth, t_grid, t_angle = [], [], [], [], [], []

        idx_to_grid = torch.tensor([3, 5, 7, 9], device=self.device)

        with torch.no_grad():
            for imgs, targets in loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)

                p_rule = outputs["logits_rule"].argmax(dim=-1).cpu().numpy()
                p_sym = outputs["logits_symmetry"].argmax(dim=-1).cpu().numpy()
                p_motif = outputs["logits_motif"].argmax(dim=-1).cpu().numpy()
                p_depth = (outputs["logits_depth"].argmax(dim=-1) + 1).cpu().numpy()
                p_grid = idx_to_grid[outputs["logits_grid_size"].argmax(dim=-1)].cpu().numpy()
                p_angle = (outputs["pred_angle_norm"] * 180.0).cpu().numpy()

                preds_rule.append(p_rule)
                preds_sym.append(p_sym)
                preds_motif.append(p_motif)
                preds_depth.append(p_depth)
                preds_grid.append(p_grid)
                preds_angle.append(p_angle)

                t_rule.append(np.array(targets["rule_id"]))
                t_sym.append(np.array(targets["symmetry"]))
                t_motif.append(np.array(targets["motif"]))
                t_depth.append(np.array(targets["recursion_depth"]))
                t_grid.append(np.array(targets["grid_size"]))
                t_angle.append(np.array(targets["angle"]))

        predictions = {
            "rule_id": np.concatenate(preds_rule),
            "symmetry": np.concatenate(preds_sym),
            "motif": np.concatenate(preds_motif),
            "recursion_depth": np.concatenate(preds_depth),
            "grid_size": np.concatenate(preds_grid),
            "angle": np.concatenate(preds_angle),
        }

        targets_dict = {
            "rule_id": np.concatenate(t_rule),
            "symmetry": np.concatenate(t_sym),
            "motif": np.concatenate(t_motif),
            "recursion_depth": np.concatenate(t_depth),
            "grid_size": np.concatenate(t_grid),
            "angle": np.concatenate(t_angle),
        }

        return predictions, targets_dict

    def evaluate_split(self, split_file: Path | str) -> dict[str, Any]:
        """Evaluate a single split file."""
        dataset = KolamDataset(split_file)
        preds, targets = self.predict_dataset(dataset)
        return compute_benchmark_metrics(preds, targets)

    def evaluate_all(
        self,
        splits_dir: Path | str,
        results_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Run full evaluation suite across all benchmark splits and corruptions."""
        splits_dir = Path(splits_dir)
        res_dir = Path(results_dir) if results_dir else Path("results")
        res_dir.mkdir(parents=True, exist_ok=True)

        full_results: dict[str, Any] = {
            "model_type": "KolamCNNBaseline",
            "benchmarks": {},
            "corruptions": {},
        }

        # 1. Standard Splits
        for split_name in ["val", "test_iid", "test_heldout_depth"]:
            split_path = splits_dir / f"{split_name}.json"
            if split_path.exists():
                dataset = KolamDataset(split_path)
                preds, targets = self.predict_dataset(dataset)
                metrics = compute_benchmark_metrics(preds, targets)
                full_results["benchmarks"][split_name] = metrics

                # Save confusion matrix for test_iid
                if split_name == "test_iid":
                    plot_confusion_matrices(
                        preds, targets, res_dir / "confusion_matrices_test_iid.png"
                    )
                elif split_name == "test_heldout_depth":
                    plot_confusion_matrices(
                        preds, targets, res_dir / "confusion_matrices_heldout_depth.png"
                    )

        # 2. Corrupted Test Suite (evaluated on test_iid with each corruption)
        test_iid_path = splits_dir / "test_iid.json"
        if test_iid_path.exists():
            for ctype in CorruptionType:
                dataset_corrupt = KolamDataset(
                    test_iid_path, corruption=ctype, corruption_severity=1.0, seed=42
                )
                preds, targets = self.predict_dataset(dataset_corrupt)
                metrics = compute_benchmark_metrics(preds, targets)
                full_results["corruptions"][ctype.value] = metrics

        # Save results JSON
        json_path = res_dir / "baseline_evaluation.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)

        return full_results
