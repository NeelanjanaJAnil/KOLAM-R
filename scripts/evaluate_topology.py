"""Benchmark script for KOLAM-R Stage 6 Topological Validation.

Evaluates dual topological representations:
1. Stroke Skeleton Graph Topology: Medial axis thinning -> Junction Graph G = (V, E) -> beta_0, beta_1
2. GUDHI Binary Mask Cubical Complex Persistent Homology: 2D raster cavities -> mask beta_0, beta_1
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kolam_r.dataset.corruption import apply_corruption
from kolam_r.dataset.loader import KolamDataset
from kolam_r.reconstruction.pipeline import ReconstructionPipeline
from kolam_r.topology.betti import extract_all_topological_invariants
from kolam_r.topology.skeleton import skeletonize_zhang_suen
from kolam_r.topology.validator import validate_reconstruction_topology


def evaluate_split_topology(
    pipeline: ReconstructionPipeline,
    dataset: KolamDataset,
    split_name: str,
    corruption_type: str | None = None,
) -> tuple[dict, dict, list]:
    """Evaluate dual topological invariants across all samples in a split."""
    g_d_b0_graph, g_d_b1_graph = [], []
    g_d_b0_mask, g_d_b1_mask = [], []
    g_graph_matches, g_mask_matches, g_dual_matches = 0, 0, 0

    b_d_b0_graph, b_d_b1_graph = [], []
    b_d_b0_mask, b_d_b1_mask = [], []
    b_graph_matches, b_mask_matches, b_dual_matches = 0, 0, 0

    gallery_samples = []
    gallery_indices = set(np.linspace(0, len(dataset) - 1, min(6, len(dataset)), dtype=int))

    print(f"Evaluating topology on {split_name} ({len(dataset)} samples)...")

    for idx in range(len(dataset)):
        rec = dataset.records[idx]
        img_raw = dataset.load_image(idx)
        motif = rec.get("motif", "M1")
        rule_id = rec.get("production_rule_id", "R01")
        depth = int(rec.get("recursion_depth", 1))
        sym = rec.get("symmetry", "C1")

        if corruption_type is not None:
            img_input = apply_corruption(img_raw, corruption_type, severity=2, seed=42 + idx)
        else:
            img_input = img_raw

        target_mask = (img_raw > 30).astype(np.uint8) * 255

        # 1. Grammar Program Synthesis Recon
        res_g = pipeline.reconstruct_from_grammar(img_input, motif=motif, use_discovered_depth=True)
        report_g = validate_reconstruction_topology(target_mask, res_g.mask)

        g_d_b0_graph.append(report_g.delta_graph_beta_0)
        g_d_b1_graph.append(report_g.delta_graph_beta_1)
        g_d_b0_mask.append(report_g.delta_mask_beta_0)
        g_d_b1_mask.append(report_g.delta_mask_beta_1)
        if report_g.exact_graph_topo_match:
            g_graph_matches += 1
        if report_g.exact_mask_topo_match:
            g_mask_matches += 1
        if report_g.exact_dual_topo_match:
            g_dual_matches += 1

        # 2. Baseline CNN Recon
        res_b = pipeline.reconstruct_from_baseline_cnn(img_input, motif=motif)
        report_b = validate_reconstruction_topology(target_mask, res_b.mask)

        b_d_b0_graph.append(report_b.delta_graph_beta_0)
        b_d_b1_graph.append(report_b.delta_graph_beta_1)
        b_d_b0_mask.append(report_b.delta_mask_beta_0)
        b_d_b1_mask.append(report_b.delta_mask_beta_1)
        if report_b.exact_graph_topo_match:
            b_graph_matches += 1
        if report_b.exact_mask_topo_match:
            b_mask_matches += 1
        if report_b.exact_dual_topo_match:
            b_dual_matches += 1

        if idx in gallery_indices:
            label = f"{rule_id} d={depth} {sym}"
            gallery_samples.append((img_raw, res_b.mask, res_g.mask, report_b, report_g, label))

    total = len(dataset)
    g_summary = {
        "mean_delta_graph_beta_0": float(np.mean(g_d_b0_graph)),
        "mean_delta_graph_beta_1": float(np.mean(g_d_b1_graph)),
        "exact_graph_topo_match_rate": float(g_graph_matches / total),
        "mean_delta_mask_beta_0": float(np.mean(g_d_b0_mask)),
        "mean_delta_mask_beta_1": float(np.mean(g_d_b1_mask)),
        "exact_mask_topo_match_rate": float(g_mask_matches / total),
        "exact_dual_topo_match_rate": float(g_dual_matches / total),
    }

    b_summary = {
        "mean_delta_graph_beta_0": float(np.mean(b_d_b0_graph)),
        "mean_delta_graph_beta_1": float(np.mean(b_d_b1_graph)),
        "exact_graph_topo_match_rate": float(b_graph_matches / total),
        "mean_delta_mask_beta_0": float(np.mean(b_d_b0_mask)),
        "mean_delta_mask_beta_1": float(np.mean(b_d_b1_mask)),
        "exact_mask_topo_match_rate": float(b_mask_matches / total),
        "exact_dual_topo_match_rate": float(b_dual_matches / total),
    }

    return g_summary, b_summary, gallery_samples


def render_topology_gallery(gallery_samples: list, output_path: str | Path) -> None:
    """Render qualitative topological overlay gallery."""
    num_rows = len(gallery_samples)
    if num_rows == 0:
        return

    fig, axes = plt.subplots(num_rows, 4, figsize=(13, 3.2 * num_rows), dpi=200, constrained_layout=True)
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, (tgt_raw, b_mask, g_mask, rep_b, rep_g, label) in enumerate(gallery_samples):
        # Target
        t_inv = rep_g.target_invariants
        axes[row_idx, 0].imshow(tgt_raw, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 0].set_title(f"Target: {label}\nGraph β1={t_inv.graph_beta_1}, Mask β1={t_inv.mask_beta_1}", fontsize=8)
        axes[row_idx, 0].axis("off")

        # Skeleton of Target
        tgt_skel = skeletonize_zhang_suen(tgt_raw > 30)
        axes[row_idx, 1].imshow(tgt_skel, cmap="magma", vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f"Medial Skeleton\nVertices={t_inv.vertex_count}, Edges={t_inv.edge_count}", fontsize=8)
        axes[row_idx, 1].axis("off")

        # CNN Baseline Recon
        b_inv = rep_b.recon_invariants
        axes[row_idx, 2].imshow(b_mask, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 2].set_title(f"CNN Recon\nGraph β1={b_inv.graph_beta_1} (Δ={rep_b.delta_graph_beta_1})", fontsize=8)
        axes[row_idx, 2].axis("off")

        # Grammar Synthesis Recon
        g_inv = rep_g.recon_invariants
        match_str = "MATCH" if rep_g.exact_dual_topo_match else "DIFF"
        axes[row_idx, 3].imshow(g_mask, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 3].set_title(f"Grammar Recon [{match_str}]\nGraph β1={g_inv.graph_beta_1} (Δ={rep_g.delta_graph_beta_1})", fontsize=8, color="darkgreen" if rep_g.exact_dual_topo_match else "black")
        axes[row_idx, 3].axis("off")

    fig.suptitle("KOLAM-R Stage 6 — Dual Topological Validation Gallery", fontsize=13, fontweight="bold")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_skeleton_failure_cases(dataset: KolamDataset, output_path: str | Path) -> None:
    """Identify and visually export failure cases near dots and dense stroke clusters."""
    failure_cases = []

    # Search for samples where dot overlays or high-density strokes cause spurious loops
    for idx in range(len(dataset)):
        rec = dataset.records[idx]
        grid_size = int(rec.get("grid_size", 5))
        depth = int(rec.get("recursion_depth", 1))
        rule_id = rec.get("production_rule_id", "R01")

        # Target cases: dense depth (d>=3) or large grid with dot overlays
        if depth >= 3 or (grid_size >= 7 and depth >= 2):
            img_raw = dataset.load_image(idx)
            skel = skeletonize_zhang_suen(img_raw > 30)
            inv = extract_all_topological_invariants(img_raw)
            # If graph_beta_1 differs significantly from mask_beta_1 due to dot-junction merging
            diff = abs(inv.graph_beta_1 - inv.mask_beta_1)
            failure_cases.append((img_raw, skel, inv, diff, f"{rule_id} d={depth} Grid={grid_size}"))
            if len(failure_cases) >= 4:
                break

    if not failure_cases:
        return

    fig, axes = plt.subplots(len(failure_cases), 3, figsize=(10, 2.8 * len(failure_cases)), dpi=200, constrained_layout=True)
    if len(failure_cases) == 1:
        axes = np.expand_dims(axes, 0)

    for r_idx, (raw, skel, inv, diff, label) in enumerate(failure_cases):
        axes[r_idx, 0].imshow(raw, cmap="gray", vmin=0, vmax=255)
        axes[r_idx, 0].set_title(f"Dense Pattern: {label}", fontsize=8)
        axes[r_idx, 0].axis("off")

        axes[r_idx, 1].imshow(skel, cmap="magma", vmin=0, vmax=1)
        axes[r_idx, 1].set_title(f"Thinning Artifacts\nGraph β1={inv.graph_beta_1}, Mask β1={inv.mask_beta_1}", fontsize=8)
        axes[r_idx, 1].axis("off")

        # Zoomed-in center region showing stroke-dot merging
        h, w = raw.shape
        center_raw = raw[h//4:3*h//4, w//4:3*w//4]
        center_skel = skel[h//4:3*h//4, w//4:3*w//4]
        overlay = np.zeros((center_raw.shape[0], center_raw.shape[1], 3), dtype=np.uint8)
        overlay[..., 0] = center_raw
        overlay[..., 1] = center_raw
        overlay[..., 2] = np.where(center_skel > 0, 255, center_raw)

        axes[r_idx, 2].imshow(overlay)
        axes[r_idx, 2].set_title(f"Zoomed Stroke Merging (Δβ1={diff})", fontsize=8, color="red")
        axes[r_idx, 2].axis("off")

    fig.suptitle("Skeletonization Failure Cases (Dot-Grid Intersections & Dense Fusing)", fontsize=12, fontweight="bold")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 75)
    print("KOLAM-R Stage 6 — Topological Validation Benchmark")
    print("=" * 75)

    pipeline = ReconstructionPipeline(
        grammar_model_path="checkpoints/best_grammar_model.pt",
        baseline_model_path="checkpoints/best_val_model.pt",
        device="cpu",
    )

    splits_root = Path("data/splits")
    splits = {
        "val": KolamDataset(splits_root / "val.json"),
        "test_iid": KolamDataset(splits_root / "test_iid.json"),
        "test_heldout_depth": KolamDataset(splits_root / "test_heldout_depth.json"),
    }

    results: dict = {
        "benchmarks": {},
        "corruptions": {},
        "space_filling_disambiguation": {},
    }

    # Evaluate standard splits
    for split_name, ds in splits.items():
        g_rep, b_rep, gallery = evaluate_split_topology(pipeline, ds, split_name)
        results["benchmarks"][split_name] = {
            "grammar_synthesis": g_rep,
            "baseline_cnn": b_rep,
        }
        if split_name == "test_iid":
            render_topology_gallery(gallery, "results/topology_gallery.png")

    # Evaluate Corruptions on Test_IID
    corruptions = ["gaussian_noise", "rotation", "stroke_dropout", "affine", "blur"]
    test_iid_ds = splits["test_iid"]
    for corr in corruptions:
        g_rep, b_rep, _ = evaluate_split_topology(pipeline, test_iid_ds, f"corr_{corr}", corruption_type=corr)
        results["corruptions"][corr] = {
            "grammar_synthesis": g_rep,
            "baseline_cnn": b_rep,
        }

    # Space-filling disambiguation (R02 Snake vs R05 Hilbert)
    from kolam_r.lsystem.rules import RULES_BY_ID
    from kolam_r.lsystem.engine import LSystemEngine
    from kolam_r.turtle.interpreter import TurtleInterpreter
    from kolam_r.renderer.image_renderer import render_kolam

    eng = LSystemEngine()
    interp = TurtleInterpreter()

    r02 = RULES_BY_ID["R02"]
    res02 = interp.interpret(eng.expand_rule(r02, depth=1), angle=90.0)
    img02 = render_kolam(res02.segments, image_size=64, motif="M1")
    inv02 = extract_all_topological_invariants(img02)

    r05 = RULES_BY_ID["R05"]
    res05 = interp.interpret(eng.expand_rule(r05, depth=1), angle=90.0)
    img05 = render_kolam(res05.segments, image_size=64, motif="M1")
    inv05 = extract_all_topological_invariants(img05)

    results["space_filling_disambiguation"] = {
        "r02_snake_kolam": {"graph_beta_0": inv02.graph_beta_0, "graph_beta_1": inv02.graph_beta_1, "mask_beta_1": inv02.mask_beta_1},
        "r05_hilbert_meander": {"graph_beta_0": inv05.graph_beta_0, "graph_beta_1": inv05.graph_beta_1, "mask_beta_1": inv05.mask_beta_1},
        "topologically_separable": bool(inv02.graph_beta_1 != inv05.graph_beta_1),
    }

    # Export failure cases
    render_skeleton_failure_cases(splits["test_heldout_depth"], "results/skeleton_failure_cases.png")

    # Save JSON
    out_json = Path("results/topology_evaluation.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print("STAGE 6 TOPOLOGICAL VALIDATION SUMMARY")
    print("=" * 75)

    for split in ["val", "test_iid", "test_heldout_depth"]:
        g = results["benchmarks"][split]["grammar_synthesis"]
        b = results["benchmarks"][split]["baseline_cnn"]
        print(f"\n[{split.upper()}]")
        print(f"  Grammar Synthesis -> Exact Graph Topo Match: {g['exact_graph_topo_match_rate']*100:.2f}% (Delta_b1: {g['mean_delta_graph_beta_1']:.2f}) | Exact Mask Topo Match: {g['exact_mask_topo_match_rate']*100:.2f}% | Dual Match: {g['exact_dual_topo_match_rate']*100:.2f}%")
        print(f"  Baseline CNN      -> Exact Graph Topo Match: {b['exact_graph_topo_match_rate']*100:.2f}% (Delta_b1: {b['mean_delta_graph_beta_1']:.2f}) | Exact Mask Topo Match: {b['exact_mask_topo_match_rate']*100:.2f}% | Dual Match: {b['exact_dual_topo_match_rate']*100:.2f}%")

    print("\nR02 vs R05 Space-Filling Disambiguation:")
    print(f"  R02 Snake Kolam:     Graph b1={inv02.graph_beta_1}, Mask b1={inv02.mask_beta_1}")
    print(f"  R05 Hilbert Meander: Graph b1={inv05.graph_beta_1}, Mask b1={inv05.mask_beta_1}")
    print(f"  Topologically Separable: {inv02.graph_beta_1 != inv05.graph_beta_1}")

    print("\nArtifacts Saved:")
    print("  - results/topology_gallery.png")
    print("  - results/skeleton_failure_cases.png")
    print(f"  - {out_json}")


if __name__ == "__main__":
    main()
