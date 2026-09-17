"""Step 2 Evaluation Script: Apply curve fitting to real Stage 1 Kolam patterns and log metrics."""

from __future__ import annotations

import numpy as np
from PIL import Image

from kolam_r.generator import KolamGenerator
from kolam_r.schema import KolamParams
from kolam_r.topology.skeleton import skeletonize_zhang_suen
from kolam_r.topology.graph_extractor import extract_skeleton_graph
from kolam_r.curvefit.fit_curves import fit_skeleton_graph_curves
from kolam_r.curvefit.rasterize_curves import rasterize_curves


def run_evaluation():
    generator = KolamGenerator()

    configs = [
        ("R01_d1_D4", KolamParams(production_rule_id="R01", recursion_depth=1, symmetry="D4", angle=45.0, grid_size=5, motif="M1")),
        ("R01_d2_D4", KolamParams(production_rule_id="R01", recursion_depth=2, symmetry="D4", angle=45.0, grid_size=5, motif="M1")),
        ("R02_d1_D4", KolamParams(production_rule_id="R02", recursion_depth=1, symmetry="D4", angle=90.0, grid_size=5, motif="M1")),
        ("R03_d1_D4", KolamParams(production_rule_id="R03", recursion_depth=1, symmetry="D4", angle=45.0, grid_size=5, motif="M1")),
        ("R05_d1_C4", KolamParams(production_rule_id="R05", recursion_depth=1, symmetry="C4", angle=90.0, grid_size=5, motif="M1")),
    ]

    header = f"{'Pattern':<12} | {'|V|_orig':<8} | {'|E|_orig':<8} | {'Curves':<8} | {'|Curves|==|E|':<13} | {'b0_orig':<7} | {'b1_orig':<7} | {'|V|_curve':<9} | {'|E|_curve':<9} | {'b0_curve':<8} | {'b1_curve':<8}"
    print(header)
    print("-" * len(header))

    results = []
    for name, params in configs:
        res = generator.generate(params)
        orig_img = res.image_256

        # 1. Extract skeleton & graph from original discrete rasterization
        skel_orig = skeletonize_zhang_suen(orig_img > 30)
        g_orig = extract_skeleton_graph(skel_orig)
        v_orig = len(g_orig.vertices)
        e_orig = len(g_orig.edges)
        b0_orig, b1_orig = g_orig.compute_betti_numbers()

        # 2. Fit parametric curves
        curves = fit_skeleton_graph_curves(g_orig, linearity_threshold=0.5)
        num_curves = len(curves)
        equality = (num_curves == e_orig)

        # 3. Rasterize curves back at the same resolution (256x256) with matching stroke width (width=2)
        curve_img = rasterize_curves(curves, image_shape=(256, 256), stroke_width=2)

        # 4. Extract skeleton & graph from the curve-fit rasterization
        skel_curve = skeletonize_zhang_suen(curve_img > 30)
        g_curve = extract_skeleton_graph(skel_curve)
        v_curve = len(g_curve.vertices)
        e_curve = len(g_curve.edges)
        b0_curve, b1_curve = g_curve.compute_betti_numbers()

        line = f"{name:<12} | {v_orig:<8} | {e_orig:<8} | {num_curves:<8} | {str(equality):<13} | {b0_orig:<7} | {b1_orig:<7} | {v_curve:<9} | {e_curve:<9} | {b0_curve:<8} | {b1_curve:<8}"
        print(line)
        results.append({
            "name": name,
            "v_orig": v_orig, "e_orig": e_orig, "num_curves": num_curves,
            "b0_orig": b0_orig, "b1_orig": b1_orig,
            "v_curve": v_curve, "e_curve": e_curve,
            "b0_curve": b0_curve, "b1_curve": b1_curve,
            "orig_img": orig_img, "curve_img": curve_img,
            "skel_orig": skel_orig, "skel_curve": skel_curve,
        })

    return results


if __name__ == "__main__":
    run_evaluation()
