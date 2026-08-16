"""Benchmark script for KOLAM-R Stage 7 — Comparative Experiments.

Evaluates 4 competing paradigms side-by-side:
1. Grammar Program Synthesis (Ours - Vision-to-Grammar + Protocol B Depth Search)
2. Multi-Task CNN Direct Parameter Baseline (Stage 3)
3. Classical Computer Vision Baseline (Symmetry FFT + Multi-Depth Template Matching)
4. End-to-End Neural Autoencoder (Image-to-Image Raster Reconstruction)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from kolam_r.baselines.autoencoder import KolamAutoencoder
from kolam_r.baselines.classical_cv import ClassicalCVBaseline
from kolam_r.dataset.corruption import apply_corruption
from kolam_r.dataset.loader import KolamDataset
from kolam_r.reconstruction.pipeline import ReconstructionPipeline
from kolam_r.topology.validator import validate_reconstruction_topology


def evaluate_comparative_split(
    pipeline: ReconstructionPipeline,
    classical_baseline: ClassicalCVBaseline,
    autoencoder: KolamAutoencoder,
    dataset: KolamDataset,
    split_name: str,
    corruption_type: str | None = None,
) -> tuple[dict, list]:
    """Evaluate all 4 methods on all samples in a benchmark split."""
    methods = ["grammar_synthesis", "baseline_cnn", "classical_cv", "autoencoder"]
    metrics_acc = {m: {"psnr": [], "ssim": [], "iou": [], "dice": [], "ncc": [], "chamfer": [], "delta_b0": [], "delta_b1": [], "exact_topo": []} for m in methods}

    # Meta metrics
    grammar_matches = {"grammar_synthesis": 0, "baseline_cnn": 0, "classical_cv": 0, "autoencoder": "N/A"}
    depth_matches = {"grammar_synthesis": 0, "baseline_cnn": 0, "classical_cv": 0, "autoencoder": "N/A"}

    gallery_samples = []
    gallery_indices = set(np.linspace(0, len(dataset) - 1, min(5, len(dataset)), dtype=int))

    print(f"Evaluating 4-way comparative benchmark on {split_name} ({len(dataset)} samples)...")

    for idx in range(len(dataset)):
        rec = dataset.records[idx]
        img_raw = dataset.load_image(idx)
        motif = rec.get("motif", "M1")
        true_rule_id = rec.get("production_rule_id", "R01")
        true_depth = int(rec.get("recursion_depth", 1))
        true_sym = rec.get("symmetry", "C1")

        if corruption_type is not None:
            img_input = apply_corruption(img_raw, corruption_type, severity=2, seed=42 + idx)
        else:
            img_input = img_raw

        target_mask = (img_raw > 30).astype(np.uint8) * 255

        # 1. Grammar Program Synthesis (Ours)
        res_g = pipeline.reconstruct_from_grammar(img_input, motif=motif, use_discovered_depth=True)
        topo_g = validate_reconstruction_topology(target_mask, res_g.mask)
        m_g = res_g.metrics
        metrics_acc["grammar_synthesis"]["psnr"].append(m_g.psnr)
        metrics_acc["grammar_synthesis"]["ssim"].append(m_g.ssim)
        metrics_acc["grammar_synthesis"]["iou"].append(m_g.iou)
        metrics_acc["grammar_synthesis"]["dice"].append(m_g.dice)
        metrics_acc["grammar_synthesis"]["ncc"].append(m_g.ncc)
        metrics_acc["grammar_synthesis"]["chamfer"].append(m_g.chamfer_distance)
        metrics_acc["grammar_synthesis"]["delta_b0"].append(topo_g.delta_graph_beta_0)
        metrics_acc["grammar_synthesis"]["delta_b1"].append(topo_g.delta_graph_beta_1)
        metrics_acc["grammar_synthesis"]["exact_topo"].append(1.0 if topo_g.exact_graph_topo_match else 0.0)
        if res_g.parameters.get("depth") == true_depth:
            depth_matches["grammar_synthesis"] += 1

        # 2. Baseline CNN (Stage 3)
        res_b = pipeline.reconstruct_from_baseline_cnn(img_input, motif=motif)
        topo_b = validate_reconstruction_topology(target_mask, res_b.mask)
        m_b = res_b.metrics
        metrics_acc["baseline_cnn"]["psnr"].append(m_b.psnr)
        metrics_acc["baseline_cnn"]["ssim"].append(m_b.ssim)
        metrics_acc["baseline_cnn"]["iou"].append(m_b.iou)
        metrics_acc["baseline_cnn"]["dice"].append(m_b.dice)
        metrics_acc["baseline_cnn"]["ncc"].append(m_b.ncc)
        metrics_acc["baseline_cnn"]["chamfer"].append(m_b.chamfer_distance)
        metrics_acc["baseline_cnn"]["delta_b0"].append(topo_b.delta_graph_beta_0)
        metrics_acc["baseline_cnn"]["delta_b1"].append(topo_b.delta_graph_beta_1)
        metrics_acc["baseline_cnn"]["exact_topo"].append(1.0 if topo_b.exact_graph_topo_match else 0.0)
        if res_b.parameters.get("depth") == true_depth:
            depth_matches["baseline_cnn"] += 1

        # 3. Classical CV Baseline
        res_c = classical_baseline.predict_and_reconstruct(img_input, motif=motif)
        topo_c = validate_reconstruction_topology(target_mask, res_c.mask)
        m_c = res_c.metrics
        metrics_acc["classical_cv"]["psnr"].append(m_c.psnr)
        metrics_acc["classical_cv"]["ssim"].append(m_c.ssim)
        metrics_acc["classical_cv"]["iou"].append(m_c.iou)
        metrics_acc["classical_cv"]["dice"].append(m_c.dice)
        metrics_acc["classical_cv"]["ncc"].append(m_c.ncc)
        metrics_acc["classical_cv"]["chamfer"].append(m_c.chamfer_distance)
        metrics_acc["classical_cv"]["delta_b0"].append(topo_c.delta_graph_beta_0)
        metrics_acc["classical_cv"]["delta_b1"].append(topo_c.delta_graph_beta_1)
        metrics_acc["classical_cv"]["exact_topo"].append(1.0 if topo_c.exact_graph_topo_match else 0.0)
        if res_c.parameters.get("rule_id") == true_rule_id:
            grammar_matches["classical_cv"] += 1
        if res_c.parameters.get("depth") == true_depth:
            depth_matches["classical_cv"] += 1

        # 4. Neural Autoencoder Baseline
        res_a = autoencoder.predict_and_reconstruct(img_input, device="cpu")
        topo_a = validate_reconstruction_topology(target_mask, res_a.mask)
        m_a = res_a.metrics
        metrics_acc["autoencoder"]["psnr"].append(m_a.psnr)
        metrics_acc["autoencoder"]["ssim"].append(m_a.ssim)
        metrics_acc["autoencoder"]["iou"].append(m_a.iou)
        metrics_acc["autoencoder"]["dice"].append(m_a.dice)
        metrics_acc["autoencoder"]["ncc"].append(m_a.ncc)
        metrics_acc["autoencoder"]["chamfer"].append(m_a.chamfer_distance)
        metrics_acc["autoencoder"]["delta_b0"].append(topo_a.delta_graph_beta_0)
        metrics_acc["autoencoder"]["delta_b1"].append(topo_a.delta_graph_beta_1)
        metrics_acc["autoencoder"]["exact_topo"].append(1.0 if topo_a.exact_graph_topo_match else 0.0)

        if idx in gallery_indices:
            label = f"{true_rule_id} d={true_depth} {true_sym}"
            gallery_samples.append((img_raw, res_c.image, res_a.image, res_b.image, res_g.image, label))

    total = len(dataset)
    split_summary: dict = {}
    for m in methods:
        split_summary[m] = {
            "psnr": float(np.mean(metrics_acc[m]["psnr"])),
            "ssim": float(np.mean(metrics_acc[m]["ssim"])),
            "iou": float(np.mean(metrics_acc[m]["iou"])),
            "dice": float(np.mean(metrics_acc[m]["dice"])),
            "ncc": float(np.mean(metrics_acc[m]["ncc"])),
            "chamfer_distance": float(np.mean(metrics_acc[m]["chamfer"])),
            "mean_delta_beta_0": float(np.mean(metrics_acc[m]["delta_b0"])),
            "mean_delta_beta_1": float(np.mean(metrics_acc[m]["delta_b1"])),
            "exact_topo_match_rate": float(np.mean(metrics_acc[m]["exact_topo"])),
            "depth_accuracy": float(depth_matches[m] / total) if depth_matches[m] != "N/A" else "N/A",
            "grammar_match_rate": "N/A" if m == "autoencoder" else (float(grammar_matches[m] / total) if m == "classical_cv" else "Computed in Stage 4"),
        }

    return split_summary, gallery_samples


def render_comparative_gallery(gallery_samples: list, output_path: str | Path) -> None:
    """Render 5-column qualitative comparison gallery."""
    num_rows = len(gallery_samples)
    if num_rows == 0:
        return

    fig, axes = plt.subplots(num_rows, 5, figsize=(15, 3.0 * num_rows), dpi=200, constrained_layout=True)
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, (tgt, c_img, a_img, b_img, g_img, label) in enumerate(gallery_samples):
        # Target
        axes[row_idx, 0].imshow(tgt, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 0].set_title(f"Target Ground Truth\n[{label}]", fontsize=8, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Classical CV
        axes[row_idx, 1].imshow(c_img, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 1].set_title("1. Classical CV Baseline\n(Symmetry FFT + Template)", fontsize=8)
        axes[row_idx, 1].axis("off")

        # Autoencoder
        axes[row_idx, 2].imshow(a_img, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 2].set_title("2. Neural Autoencoder\n(Image-to-Image)", fontsize=8)
        axes[row_idx, 2].axis("off")

        # CNN Baseline
        axes[row_idx, 3].imshow(b_img, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 3].set_title("3. CNN Param Baseline\n(Multi-Task Regression)", fontsize=8)
        axes[row_idx, 3].axis("off")

        # Grammar Synthesis (Ours)
        axes[row_idx, 4].imshow(g_img, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 4].set_title("4. Grammar Synthesis (Ours)\n(Vision-to-Grammar + Synthesis)", fontsize=8, color="darkgreen", fontweight="bold")
        axes[row_idx, 4].axis("off")

    fig.suptitle("KOLAM-R Stage 7 — 4-Paradigm Comparative Reconstruction Gallery", fontsize=13, fontweight="bold")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 75)
    print("KOLAM-R Stage 7 — Comparative Experiments Benchmark")
    print("=" * 75)

    pipeline = ReconstructionPipeline(
        grammar_model_path="checkpoints/best_grammar_model.pt",
        baseline_model_path="checkpoints/best_val_model.pt",
        device="cpu",
    )
    classical_baseline = ClassicalCVBaseline()

    autoencoder = KolamAutoencoder(latent_dim=128)
    ae_ckpt = torch.load("checkpoints/best_autoencoder.pt", map_location="cpu", weights_only=True)
    autoencoder.load_state_dict(ae_ckpt["model_state_dict"])
    autoencoder.eval()

    splits_root = Path("data/splits")
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
        summary, gallery = evaluate_comparative_split(pipeline, classical_baseline, autoencoder, ds, split_name)
        results["benchmarks"][split_name] = summary
        if split_name == "test_iid":
            render_comparative_gallery(gallery, "results/comparative_gallery.png")

    # Evaluate Corruptions on Test_IID
    corruptions = ["gaussian_noise", "rotation", "stroke_dropout", "affine", "blur"]
    test_iid_ds = splits["test_iid"]
    for corr in corruptions:
        summary, _ = evaluate_comparative_split(pipeline, classical_baseline, autoencoder, test_iid_ds, f"corr_{corr}", corruption_type=corr)
        results["corruptions"][corr] = summary

    out_json = Path("results/comparative_evaluation.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print("STAGE 7 COMPARATIVE BENCHMARK SUMMARY")
    print("=" * 75)

    for split in ["val", "test_iid", "test_heldout_depth"]:
        print(f"\n[{split.upper()}]")
        for m in ["grammar_synthesis", "baseline_cnn", "classical_cv", "autoencoder"]:
            row = results["benchmarks"][split][m]
            print(f"  {m:<20} -> PSNR: {row['psnr']:5.2f}dB | SSIM: {row['ssim']:.4f} | IoU: {row['iou']:.4f} | NCC: {row['ncc']:.4f} | Chamfer: {row['chamfer_distance']:4.2f}px | TopoMatch: {row['exact_topo_match_rate']*100:5.2f}%")

    print("\nArtifacts Saved:")
    print("  - results/comparative_gallery.png")
    print(f"  - {out_json}")


if __name__ == "__main__":
    main()
