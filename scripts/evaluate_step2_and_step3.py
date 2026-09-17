"""Script to run comparison on both full image (with dots) and stroke-only mask, and generate Step 3 difference images."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from kolam_r.generator import KolamGenerator
from kolam_r.schema import KolamParams
from kolam_r.topology.skeleton import skeletonize_zhang_suen
from kolam_r.topology.graph_extractor import extract_skeleton_graph
from kolam_r.curvefit.fit_curves import fit_skeleton_graph_curves
from kolam_r.curvefit.rasterize_curves import rasterize_curves


def main():
    generator = KolamGenerator()
    results_dir = Path("results/step3_discrepancies")
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("R01_d1_D4", KolamParams(production_rule_id="R01", recursion_depth=1, symmetry="D4", angle=45.0, grid_size=5, motif="M1")),
        ("R01_d2_D4", KolamParams(production_rule_id="R01", recursion_depth=2, symmetry="D4", angle=45.0, grid_size=5, motif="M1")),
        ("R02_d1_D4", KolamParams(production_rule_id="R02", recursion_depth=1, symmetry="D4", angle=90.0, grid_size=5, motif="M1")),
        ("R03_d1_D4", KolamParams(production_rule_id="R03", recursion_depth=1, symmetry="D4", angle=45.0, grid_size=5, motif="M1")),
        ("R05_d1_C4", KolamParams(production_rule_id="R05", recursion_depth=1, symmetry="C4", angle=90.0, grid_size=5, motif="M1")),
    ]

    print("=== STEP 2 TABLE: STROKE GRAPH TOPOLOGY (|Curves| == |E|) ===")
    header = f"{'Pattern':<12} | {'|V|_orig':<8} | {'|E|_orig':<8} | {'Curves':<8} | {'|C|==|E|':<8} | {'b0_orig':<7} | {'b1_orig':<7} | {'|V|_curve':<9} | {'|E|_curve':<9} | {'b0_curve':<8} | {'b1_curve':<8}"
    print(header)
    print("-" * len(header))

    for name, params in configs:
        res = generator.generate(params)
        orig_img = res.image_256

        # Extract skeleton from stroke binary mask (value > 200 isolates white strokes from gray dots)
        skel_orig = skeletonize_zhang_suen(orig_img > 200)
        g_orig = extract_skeleton_graph(skel_orig)
        v_orig = len(g_orig.vertices)
        e_orig = len(g_orig.edges)
        b0_orig, b1_orig = g_orig.compute_betti_numbers()

        # Fit curves
        curves = fit_skeleton_graph_curves(g_orig, linearity_threshold=0.5)
        num_curves = len(curves)
        equality = (num_curves == e_orig)

        # Rasterize curves back
        curve_img = rasterize_curves(curves, image_shape=(256, 256), stroke_width=2)

        # Extract skeleton from curve rasterization
        skel_curve = skeletonize_zhang_suen(curve_img > 100)
        g_curve = extract_skeleton_graph(skel_curve)
        v_curve = len(g_curve.vertices)
        e_curve = len(g_curve.edges)
        b0_curve, b1_curve = g_curve.compute_betti_numbers()

        print(f"{name:<12} | {v_orig:<8} | {e_orig:<8} | {num_curves:<8} | {str(equality):<8} | {b0_orig:<7} | {b1_orig:<7} | {v_curve:<9} | {e_curve:<9} | {b0_curve:<8} | {b1_curve:<8}")

        # Step 3: Produce Difference Image with marked discrepancy locations
        # RGB Difference Canvas:
        # Gray = unchanged background
        # White = matching stroke pixels
        # Green = present in original skeleton but missing in curve skeleton
        # Red = present in curve skeleton but missing in original skeleton
        h, w = 256, 256
        diff_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Base visualization of original strokes (dimmed)
        diff_rgb[orig_img > 200] = [40, 40, 40]

        # Skeletons
        both_skel = (skel_orig > 0) & (skel_curve > 0)
        orig_only = (skel_orig > 0) & (skel_curve == 0)
        curve_only = (skel_orig == 0) & (skel_curve > 0)

        diff_rgb[both_skel] = [200, 200, 200]  # Overlap in gray/white
        diff_rgb[orig_only] = [0, 255, 0]      # Green = original only
        diff_rgb[curve_only] = [255, 50, 50]   # Red = curve only

        diff_pil = Image.fromarray(diff_rgb)
        draw = ImageDraw.Draw(diff_pil)

        # If b0 or b1 differ, locate the topological discrepancy regions and draw yellow circles
        # Discrepancies occur where junction count or cycle branches differ
        # We find clusters of mismatch pixels (curve_only | orig_only) near vertices
        if (b0_orig != b0_curve) or (b1_orig != b1_curve):
            # Locate differing junction vertices
            for v_idx, (r, c) in enumerate(g_curve.vertices):
                # Check if this vertex has deg >= 3 (junction) that was not in g_orig
                deg = g_curve.degrees.get(v_idx, 0)
                if deg >= 3:
                    # Check distance to nearest original vertex
                    dists = [np.hypot(r - ov[0], c - ov[1]) for ov in g_orig.vertices]
                    if dists and min(dists) > 2.0:
                        # New junction created by curve rasterization aliasing!
                        draw.ellipse([c - 6, r - 6, c + 6, r + 6], outline="yellow", width=2)

        diff_path = results_dir / f"{name}_diff.png"
        diff_pil.save(diff_path)


if __name__ == "__main__":
    main()
