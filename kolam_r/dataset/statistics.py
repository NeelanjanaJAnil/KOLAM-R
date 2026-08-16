"""Dataset statistics engine and distribution analysis for KOLAM-R.

Computes comprehensive dataset metrics, parameter distributions, split summaries,
and quantitative R02 vs R05 pixel-level overlap metrics (SSIM / MSE / NCC).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from kolam_r.schema import VALID_MOTIFS, VALID_RULES, VALID_SYMMETRIES


def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two grayscale uint8 images."""
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = np.mean(im1)
    mu2 = np.mean(im2)
    var1 = np.var(im1)
    var2 = np.var(im2)
    covar = np.mean((im1 - mu1) * (im2 - mu2))

    num = (2 * mu1 * mu2 + c1) * (2 * covar + c2)
    den = (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
    return float(num / den)


def _compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Squared Error between two uint8 images."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def _compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Normalized Cross-Correlation between two uint8 images."""
    im1 = img1.astype(np.float64) - np.mean(img1)
    im2 = img2.astype(np.float64) - np.mean(img2)
    norm = np.linalg.norm(im1) * np.linalg.norm(im2)
    if norm < 1e-9:
        return 1.0 if np.array_equal(img1, img2) else 0.0
    return float(np.sum(im1 * im2) / norm)


class DatasetStatisticsEngine:
    """Computes and exports complete dataset statistics and overlap metrics."""

    def __init__(self, dataset_dir: Path | str) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.splits_dir = self.dataset_dir / "splits"

    def load_all_records(self) -> dict[str, list[dict]]:
        """Load records for all splits."""
        all_splits: dict[str, list[dict]] = {}
        for p in sorted(self.splits_dir.glob("*.json")):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_splits[data["split_name"]] = data["records"]
        return all_splits

    def compute_r02_r05_overlap_metrics(self) -> dict[str, Any]:
        """Compute pixel-level similarity between R02 (Snake) and R05 (Hilbert).

        Compares R02 and R05 across all identical (symmetry, grid_size, depth, motif)
        tuples and computes mean SSIM, MSE, and NCC.
        """
        images_dir = self.dataset_dir / "images"
        all_splits = self.load_all_records()
        all_recs = [r for recs in all_splits.values() for r in recs]

        # Index by (symmetry, grid_size, recursion_depth, motif)
        r02_map: dict[tuple, dict] = {}
        r05_map: dict[tuple, dict] = {}

        for r in all_recs:
            key = (r["symmetry"], r["grid_size"], r["recursion_depth"], r["motif"])
            if r["production_rule_id"] == "R02":
                r02_map[key] = r
            elif r["production_rule_id"] == "R05":
                r05_map[key] = r

        shared_keys = sorted(set(r02_map.keys()).intersection(set(r05_map.keys())))

        per_depth_ssim: dict[int, list[float]] = defaultdict(list)
        per_depth_mse: dict[int, list[float]] = defaultdict(list)
        per_depth_ncc: dict[int, list[float]] = defaultdict(list)

        for key in shared_keys:
            depth = key[2]
            rec2 = r02_map[key]
            rec5 = r05_map[key]

            img2_path = images_dir / f"{rec2['image_id']}.png"
            img5_path = images_dir / f"{rec5['image_id']}.png"

            if not img2_path.exists() or not img5_path.exists():
                continue

            img2 = np.array(Image.open(img2_path).convert("L"))
            img5 = np.array(Image.open(img5_path).convert("L"))

            ssim = _compute_ssim(img2, img5)
            mse = _compute_mse(img2, img5)
            ncc = _compute_ncc(img2, img5)

            per_depth_ssim[depth].append(ssim)
            per_depth_mse[depth].append(mse)
            per_depth_ncc[depth].append(ncc)

        summary: dict[str, Any] = {
            "num_shared_tuples": len(shared_keys),
            "shared_depths": sorted(per_depth_ssim.keys()),
            "by_depth": {},
        }

        all_ssim = []
        all_mse = []
        all_ncc = []

        for d in sorted(per_depth_ssim.keys()):
            ssim_vals = per_depth_ssim[d]
            mse_vals = per_depth_mse[d]
            ncc_vals = per_depth_ncc[d]

            all_ssim.extend(ssim_vals)
            all_mse.extend(mse_vals)
            all_ncc.extend(ncc_vals)

            summary["by_depth"][f"depth_{d}"] = {
                "count": len(ssim_vals),
                "mean_ssim": float(np.mean(ssim_vals)),
                "mean_mse": float(np.mean(mse_vals)),
                "mean_ncc": float(np.mean(ncc_vals)),
            }

        summary["overall_mean_ssim"] = float(np.mean(all_ssim)) if all_ssim else 0.0
        summary["overall_mean_mse"] = float(np.mean(all_mse)) if all_mse else 0.0
        summary["overall_mean_ncc"] = float(np.mean(all_ncc)) if all_ncc else 0.0
        summary["interpretation"] = (
            "Higher SSIM / NCC indicates severe pixel-level ambiguity between "
            "R02 (Snake Kolam) and R05 (Hilbert Meander), confirming that image-only "
            "classifiers will require topological features (Stage 6) for robust discrimination."
        )

        return summary

    def compute_all_statistics(self) -> dict[str, Any]:
        """Compute full statistical summary across all splits and parameters."""
        all_splits = self.load_all_records()
        total_samples = sum(len(recs) for recs in all_splits.values())

        stats: dict[str, Any] = {
            "total_images": total_samples,
            "splits": {},
            "parameter_distributions": {},
            "cross_tabulations": {},
        }

        all_records = []
        for split_name, recs in all_splits.items():
            unique_structs = len(set(r["structure_id"] for r in recs))
            stats["splits"][split_name] = {
                "count": len(recs),
                "percentage": round(100.0 * len(recs) / max(1, total_samples), 2),
                "num_unique_structures": unique_structs,
            }
            all_records.extend(recs)

        # Marginal distributions
        for param in ["production_rule_id", "symmetry", "motif", "grid_size", "recursion_depth"]:
            counts = Counter(r[param] for r in all_records)
            stats["parameter_distributions"][param] = {
                str(k): v for k, v in sorted(counts.items())
            }

        # Rule x Depth cross-tabulation
        rule_depth = defaultdict(lambda: defaultdict(int))
        rule_sym = defaultdict(lambda: defaultdict(int))
        for r in all_records:
            rule_depth[r["production_rule_id"]][str(r["recursion_depth"])] += 1
            rule_sym[r["production_rule_id"]][r["symmetry"]] += 1

        stats["cross_tabulations"]["rule_x_depth"] = {
            k: dict(v) for k, v in sorted(rule_depth.items())
        }
        stats["cross_tabulations"]["rule_x_symmetry"] = {
            k: dict(v) for k, v in sorted(rule_sym.items())
        }

        # Compute R02 vs R05 overlap metrics
        stats["r02_vs_r05_overlap_metrics"] = self.compute_r02_r05_overlap_metrics()

        return stats

    def export_statistics_json(self, output_path: Path | str | None = None) -> Path:
        """Export computed statistics to JSON."""
        stats = self.compute_all_statistics()
        if output_path is None:
            out_dir = self.dataset_dir / "stats"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / "dataset_statistics.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return output_path

    def plot_distributions(self, output_path: Path | str | None = None) -> Path:
        """Plot a 6-panel chart visualizing all parameter and split distributions."""
        stats = self.compute_all_statistics()
        if output_path is None:
            out_dir = self.dataset_dir / "stats"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / "distributions.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle("KOLAM-R Stage 2 — Dataset Distributions & Split Balance", fontsize=14, fontweight="bold")

        # 1. Splits
        ax = axes[0, 0]
        splits = stats["splits"]
        ax.bar(list(splits.keys()), [v["count"] for v in splits.values()], color="#3b82f6")
        ax.set_title("Sample Counts per Split", fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=20)

        # 2. Production Rules
        ax = axes[0, 1]
        rules = stats["parameter_distributions"]["production_rule_id"]
        ax.bar(list(rules.keys()), list(rules.values()), color="#10b981")
        ax.set_title("Production Rule Distribution", fontsize=11, fontweight="bold")

        # 3. Symmetry
        ax = axes[0, 2]
        syms = stats["parameter_distributions"]["symmetry"]
        ax.bar(list(syms.keys()), list(syms.values()), color="#8b5cf6")
        ax.set_title("Symmetry Class Distribution", fontsize=11, fontweight="bold")

        # 4. Recursion Depth
        ax = axes[1, 0]
        depths = stats["parameter_distributions"]["recursion_depth"]
        ax.bar([f"d={k}" for k in depths.keys()], list(depths.values()), color="#f59e0b")
        ax.set_title("Recursion Depth Distribution", fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")

        # 5. Grid Size
        ax = axes[1, 1]
        grids = stats["parameter_distributions"]["grid_size"]
        ax.bar([f"{k}x{k}" for k in grids.keys()], list(grids.values()), color="#ec4899")
        ax.set_title("Grid Size Distribution", fontsize=11, fontweight="bold")

        # 6. Motifs
        ax = axes[1, 2]
        motifs = stats["parameter_distributions"]["motif"]
        ax.bar(list(motifs.keys()), list(motifs.values()), color="#64748b")
        ax.set_title("Motif Style Distribution (Rendering)", fontsize=11, fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path
