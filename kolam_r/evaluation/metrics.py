"""Diagnostic evaluation metrics and confusion matrix visualizer for KOLAM-R."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kolam_r.schema import VALID_MOTIFS, VALID_RULES, VALID_SYMMETRIES


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix using pure NumPy."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def compute_benchmark_metrics(
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compute comprehensive diagnostic metrics across all parameter heads.

    Explicitly separates Structural Parameters from Rendering-Level Motif.
    """
    pred_rule = predictions["rule_id"]
    pred_sym = predictions["symmetry"]
    pred_motif = predictions["motif"]
    pred_depth = predictions["recursion_depth"]
    pred_grid = predictions["grid_size"]
    pred_angle = predictions["angle"]

    t_rule = targets["rule_id"]
    t_sym = targets["symmetry"]
    t_motif = targets["motif"]
    t_depth = targets["recursion_depth"]
    t_grid = targets["grid_size"]
    t_angle = targets["angle"]

    n_samples = len(t_rule)

    # 1. Per-parameter classification accuracies
    rule_correct = (pred_rule == t_rule)
    sym_correct = (pred_sym == t_sym)
    motif_correct = (pred_motif == t_motif)
    depth_correct = (pred_depth == t_depth)
    grid_correct = (pred_grid == t_grid)

    rule_acc = float(np.mean(rule_correct))
    sym_acc = float(np.mean(sym_correct))
    motif_acc = float(np.mean(motif_correct))
    depth_acc = float(np.mean(depth_correct))
    grid_acc = float(np.mean(grid_correct))

    # 2. Angle regression metrics
    angle_diff = np.abs(pred_angle - t_angle)
    angle_mae = float(np.mean(angle_diff))
    angle_rmse = float(np.sqrt(np.mean(angle_diff**2)))
    angle_correct = (angle_diff <= 5.0)
    angle_acc_5deg = float(np.mean(angle_correct))

    # 3. Structural Exact Match (All 5 structural parameters correct simultaneously)
    # Excludes motif to prevent stroke rendering from inflating structural recovery metrics
    structural_exact_match = float(
        np.mean(rule_correct & sym_correct & depth_correct & grid_correct & angle_correct)
    )

    # 4. Full Exact Match (Structural + Motif)
    full_exact_match = float(
        np.mean(
            rule_correct
            & sym_correct
            & depth_correct
            & grid_correct
            & angle_correct
            & motif_correct
        )
    )

    # 5. R02 vs R05 Specific Confusion Rates
    # R02 is index 1, R05 is index 4 in VALID_RULES ("R01", "R02", "R03", "R04", "R05", "R06")
    r02_idx = 1
    r05_idx = 4
    r02_mask = (t_rule == r02_idx)
    r05_mask = (t_rule == r05_idx)

    r02_confused_as_r05 = (
        float(np.mean(pred_rule[r02_mask] == r05_idx)) if np.any(r02_mask) else 0.0
    )
    r05_confused_as_r02 = (
        float(np.mean(pred_rule[r05_mask] == r02_idx)) if np.any(r05_mask) else 0.0
    )

    # 6. R03 vs R04 Specific Confusion Rates (Sikku Tile vs Mango Leaf)
    r03_idx = 2
    r04_idx = 3
    r03_mask = (t_rule == r03_idx)
    r04_mask = (t_rule == r04_idx)

    r03_confused_as_r04 = (
        float(np.mean(pred_rule[r03_mask] == r04_idx)) if np.any(r03_mask) else 0.0
    )
    r04_confused_as_r03 = (
        float(np.mean(pred_rule[r04_mask] == r03_idx)) if np.any(r04_mask) else 0.0
    )

    return {
        "num_samples": n_samples,
        "structural_exact_match_accuracy": structural_exact_match,
        "full_exact_match_accuracy": full_exact_match,
        "structural_parameters": {
            "rule_id_accuracy": rule_acc,
            "symmetry_accuracy": sym_acc,
            "recursion_depth_accuracy": depth_acc,
            "grid_size_accuracy": grid_acc,
            "angle_mae_degrees": angle_mae,
            "angle_rmse_degrees": angle_rmse,
            "angle_accuracy_within_5deg": angle_acc_5deg,
        },
        "rendering_parameters": {
            "motif_accuracy": motif_acc,
        },
        "pair_confusion_diagnostics": {
            "r02_confused_as_r05_rate": r02_confused_as_r05,
            "r05_confused_as_r02_rate": r05_confused_as_r02,
            "r03_confused_as_r04_rate": r03_confused_as_r04,
            "r04_confused_as_r03_rate": r04_confused_as_r03,
        },
    }


def plot_confusion_matrices(
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    output_path: Path | str,
) -> Path:
    """Generate and plot normalized confusion matrices for categorical heads."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("KOLAM-R CNN Baseline — Normalized Confusion Matrices", fontsize=14, fontweight="bold")

    configs = [
        ("Production Rule", "rule_id", VALID_RULES, axes[0, 0]),
        ("Symmetry Class", "symmetry", VALID_SYMMETRIES, axes[0, 1]),
        ("Recursion Depth", "recursion_depth", ["d=1", "d=2", "d=3", "d=4"], axes[0, 2]),
        ("Grid Size", "grid_size", ["3x3", "5x5", "7x7", "9x9"], axes[1, 0]),
        ("Motif (Rendering)", "motif", VALID_MOTIFS, axes[1, 1]),
    ]

    for title, key, labels, ax in configs:
        t_vec = targets[key]
        p_vec = predictions[key]

        # Map grid_size if values are [3, 5, 7, 9] -> [0, 1, 2, 3]
        if key == "grid_size" and np.max(t_vec) > 3:
            grid_map = {3: 0, 5: 1, 7: 2, 9: 3}
            t_vec = np.array([grid_map.get(v, 0) for v in t_vec])
            p_vec = np.array([grid_map.get(v, 0) for v in p_vec])

        # Map depth if values are [1, 2, 3, 4] -> [0, 1, 2, 3]
        if key == "recursion_depth" and np.min(t_vec) == 1:
            t_vec = t_vec - 1
            p_vec = p_vec - 1

        cm = _confusion_matrix(t_vec, p_vec, num_classes=len(labels))
        cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight="bold")

        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels)

        # Annotate cell values
        thresh = cm_norm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = f"{cm_norm[i, j]:.2f}"
                ax.text(
                    j,
                    i,
                    val,
                    ha="center",
                    va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=8,
                )

        ax.set_ylabel("Ground Truth")
        ax.set_xlabel("Predicted")

    # Leave the 6th subplot for angle regression scatter
    ax_angle = axes[1, 2]
    t_ang = targets["angle"]
    p_ang = predictions["angle"]
    ax_angle.scatter(t_ang, p_ang, alpha=0.6, color="#0284c7", edgecolors="none")
    ax_angle.plot([0, 180], [0, 180], "r--", lw=1.5, label="Ideal")
    ax_angle.set_title("Angle Regression (Degrees)", fontsize=11, fontweight="bold")
    ax_angle.set_xlabel("Ground Truth Angle (°)")
    ax_angle.set_ylabel("Predicted Angle (°)")
    ax_angle.set_xlim(0, 180)
    ax_angle.set_ylim(0, 180)
    ax_angle.legend()
    ax_angle.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
