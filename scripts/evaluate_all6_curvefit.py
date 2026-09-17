"""Evaluate true continuous curvefit module across all 6 canonical rules R01-R06.
Uses clean stroke masks (no dot grid pollution) and reports genuine empirical metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kolam_r.generator import KolamGenerator
from app.prototype_app import CANONICAL_BENCHMARKS
from kolam_r.topology.skeleton import skeletonize_zhang_suen
from kolam_r.topology.graph_extractor import extract_skeleton_graph
from kolam_r.curvefit.fit_curves import fit_skeleton_graph_curves
from kolam_r.curvefit.rasterize_curves import rasterize_curves
from app.prototype_app import compute_ssim, compute_iou


def main():
    gen = KolamGenerator()

    header = f"{'RULE':<5} | {'CANONICAL NAME':<18} | {'V_orig':<6} | {'E_orig':<6} | {'Curves':<6} | {'b0_orig':<7} | {'b1_orig':<7} | {'b0_fit':<6} | {'b1_fit':<6} | {'IoU':<6} | {'SSIM':<6}"
    print(header)
    print("-" * len(header))

    for rid in ["R01", "R02", "R03", "R04", "R05", "R06"]:
        p = CANONICAL_BENCHMARKS[rid]
        res = gen.generate(p)
        orig_img = res.image_256

        # 1. Clean stroke skeleton (dots excluded: threshold > 200)
        skel_orig = skeletonize_zhang_suen(orig_img > 200)
        g_orig = extract_skeleton_graph(skel_orig)
        v_orig = len(g_orig.vertices)
        e_orig = len(g_orig.edges)
        b0_orig, b1_orig = g_orig.compute_betti_numbers()

        # 2. Fit continuous parametric curves (Step 1 module)
        curves = fit_skeleton_graph_curves(g_orig, linearity_threshold=0.5)

        # 3. Rasterize curves back to canvas
        recon_img = rasterize_curves(curves, image_shape=(256, 256), stroke_width=2)

        # 4. Extract topology of reconstructed curve image
        skel_fit = skeletonize_zhang_suen(recon_img > 100)
        g_fit = extract_skeleton_graph(skel_fit)
        b0_fit, b1_fit = g_fit.compute_betti_numbers()

        # 5. Compute real image metrics against clean original stroke mask
        clean_orig_stroke = (orig_img > 200).astype("uint8") * 255
        iou = compute_iou(clean_orig_stroke, recon_img)
        ssim = compute_ssim(clean_orig_stroke, recon_img)

        from kolam_r.lsystem.rules import RULE_REGISTRY

        rule_obj = RULE_REGISTRY[rid]
        print(f"{rid:<5} | {rule_obj.name:<18} | {v_orig:<6} | {e_orig:<6} | {len(curves):<6} | {b0_orig:<7} | {b1_orig:<7} | {b0_fit:<6} | {b1_fit:<6} | {iou:.4f} | {ssim:.4f}")


if __name__ == "__main__":
    main()
