"""Benchmark script for KOLAM-R Stage 5 Reconstruction Pipeline.

Compares Vision-to-Grammar forward synthesis (Stage 4/5) against
Direct Parameter Baseline CNN forward synthesis (Stage 3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from kolam_r.dataset.corruption import CorruptionType, apply_corruption
from kolam_r.dataset.loader import KolamDataset
from kolam_r.reconstruction.metrics import ReconstructionMetrics
from kolam_r.reconstruction.pipeline import ReconstructionPipeline, ReconstructionResult
from kolam_r.reconstruction.visualizer import ReconstructionVisualizer


def evaluate_split(
    pipeline: ReconstructionPipeline,
    dataset: KolamDataset,
    split_name: str,
    corruption_type: str | None = None,
    device: str = "cpu",
) -> tuple[dict[str, float], dict[str, float], list[tuple[np.ndarray, ReconstructionResult, ReconstructionResult, str]]]:
    """Evaluate both Grammar and CNN Baseline paths on a dataset split."""
    grammar_metric_list: list[ReconstructionMetrics] = []
    baseline_metric_list: list[ReconstructionMetrics] = []
    gallery_samples: list[tuple[np.ndarray, ReconstructionResult, ReconstructionResult, str]] = []

    print(f"Evaluating {split_name} ({len(dataset)} samples)...")

    # Sample a few diverse indices for qualitative gallery
    gallery_indices = set(np.linspace(0, len(dataset) - 1, min(6, len(dataset)), dtype=int))

    for idx in range(len(dataset)):
        rec = dataset.records[idx]
        img_raw = dataset.load_image(idx)  # (64, 64) uint8
        motif = rec.get("motif", "M1")
        rule_id = rec.get("production_rule_id", "R01")
        true_depth = int(rec.get("recursion_depth", 1))
        sym = rec.get("symmetry", "C1")

        if corruption_type is not None:
            img_input = apply_corruption(img_raw, corruption_type, severity=2, seed=42 + idx)
        else:
            img_input = img_raw

        # 1. Grammar Program Synthesis Reconstruction
        res_grammar = pipeline.reconstruct_from_grammar(
            image=img_input,
            motif=motif,
            use_discovered_depth=True,
            compute_fidelity=True,
        )
        if res_grammar.metrics:
            grammar_metric_list.append(res_grammar.metrics)

        # 2. Direct Parameter Baseline CNN Reconstruction
        res_baseline = pipeline.reconstruct_from_baseline_cnn(
            image=img_input,
            motif=motif,
            compute_fidelity=True,
        )
        if res_baseline.metrics:
            baseline_metric_list.append(res_baseline.metrics)

        if idx in gallery_indices:
            label = f"{rule_id} d={true_depth} {sym}"
            gallery_samples.append((img_input, res_baseline, res_grammar, label))

    def aggregate_metrics(metrics_list: list[ReconstructionMetrics]) -> dict[str, float]:
        if not metrics_list:
            return {}
        return {
            "mse": float(np.mean([m.mse for m in metrics_list])),
            "psnr": float(np.mean([m.psnr for m in metrics_list])),
            "ssim": float(np.mean([m.ssim for m in metrics_list])),
            "ncc": float(np.mean([m.ncc for m in metrics_list])),
            "iou": float(np.mean([m.iou for m in metrics_list])),
            "dice": float(np.mean([m.dice for m in metrics_list])),
            "chamfer_distance": float(np.mean([m.chamfer_distance for m in metrics_list])),
        }

    return aggregate_metrics(grammar_metric_list), aggregate_metrics(baseline_metric_list), gallery_samples


def main() -> None:
    print("=" * 75)
    print("KOLAM-R Stage 5 — Forward Synthesis Reconstruction Benchmark")
    print("=" * 75)

    device = "cpu"
    pipeline = ReconstructionPipeline(
        grammar_model_path="checkpoints/best_grammar_model.pt",
        baseline_model_path="checkpoints/best_val_model.pt",
        device=device,
    )

    data_root = Path("data")
    splits_root = data_root / "splits"

    splits = {
        "val": KolamDataset(splits_root / "val.json"),
        "test_iid": KolamDataset(splits_root / "test_iid.json"),
        "test_heldout_depth": KolamDataset(splits_root / "test_heldout_depth.json"),
    }

    results: dict = {
        "benchmarks": {},
        "corruptions": {},
    }

    # Evaluate standard splits
    for split_name, ds in splits.items():
        g_metrics, b_metrics, gallery = evaluate_split(pipeline, ds, split_name, device=device)
        results["benchmarks"][split_name] = {
            "grammar_synthesis": g_metrics,
            "baseline_cnn": b_metrics,
        }

        # Export Gallery for IID and Heldout
        if split_name == "test_iid":
            ReconstructionVisualizer.render_gallery(
                gallery,
                "results/reconstruction_gallery_iid.png",
                main_title="KOLAM-R Reconstruction Gallery — Test IID (Clean)",
            )
        elif split_name == "test_heldout_depth":
            ReconstructionVisualizer.render_gallery(
                gallery,
                "results/reconstruction_gallery_heldout.png",
                main_title="KOLAM-R Reconstruction Gallery — Test Held-Out Depth Extrapolation",
            )

    # Evaluate Corruptions on Test_IID
    corruptions = ["gaussian_noise", "rotation", "stroke_dropout", "affine", "blur"]
    test_iid_ds = splits["test_iid"]
    corrupted_gallery_samples = []

    for corr_type in corruptions:
        g_metrics, b_metrics, c_gallery = evaluate_split(
            pipeline, test_iid_ds, f"corr_{corr_type}", corruption_type=corr_type, device=device
        )
        results["corruptions"][corr_type] = {
            "grammar_synthesis": g_metrics,
            "baseline_cnn": b_metrics,
        }
        if c_gallery:
            # Pick one sample per corruption for corrupted gallery
            sample = c_gallery[0]
            corrupted_gallery_samples.append((sample[0], sample[1], sample[2], f"{corr_type.capitalize()} ({sample[3]})"))

    if corrupted_gallery_samples:
        ReconstructionVisualizer.render_gallery(
            corrupted_gallery_samples,
            "results/reconstruction_gallery_corrupted.png",
            main_title="KOLAM-R Reconstruction Gallery — Robustness Under Corruptions",
        )

    # Save detailed JSON
    out_json = Path("results/reconstruction_evaluation.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print("STAGE 5 RECONSTRUCTION BENCHMARK SUMMARY")
    print("=" * 75)

    for split in ["val", "test_iid", "test_heldout_depth"]:
        g = results["benchmarks"][split]["grammar_synthesis"]
        b = results["benchmarks"][split]["baseline_cnn"]
        print(f"\n[{split.upper()}]")
        print(f"  Grammar Synthesis -> PSNR: {g['psnr']:.2f}dB | SSIM: {g['ssim']:.4f} | IoU: {g['iou']:.4f} | Dice: {g['dice']:.4f} | NCC: {g['ncc']:.4f} | Chamfer: {g['chamfer_distance']:.2f}px")
        print(f"  Baseline CNN      -> PSNR: {b['psnr']:.2f}dB | SSIM: {b['ssim']:.4f} | IoU: {b['iou']:.4f} | Dice: {b['dice']:.4f} | NCC: {b['ncc']:.4f} | Chamfer: {b['chamfer_distance']:.2f}px")

    print("\nVisual Galleries Saved:")
    print("  - results/reconstruction_gallery_iid.png")
    print("  - results/reconstruction_gallery_heldout.png")
    print("  - results/reconstruction_gallery_corrupted.png")
    print(f"Detailed JSON saved to: {out_json}")


if __name__ == "__main__":
    main()
