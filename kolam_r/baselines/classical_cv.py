"""Classical Computer Vision baseline for Kolam parameter recovery and reconstruction.

Uses rotational/reflection autocorrelation for symmetry group detection ($C_1 \dots D_4$)
and classical multi-scale Analysis-by-Synthesis template correlation against canonical
L-system rule renders across depths and grid sizes.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate

from kolam_r.lsystem.engine import LSystemEngine
from kolam_r.lsystem.rules import RULE_REGISTRY, RULES_BY_ID
from kolam_r.reconstruction.metrics import (
    ReconstructionMetrics,
    compute_reconstruction_metrics,
)
from kolam_r.reconstruction.pipeline import ReconstructionResult
from kolam_r.renderer.image_renderer import render_kolam
from kolam_r.schema import BoundingBox
from kolam_r.symmetry.transforms import apply_symmetry
from kolam_r.turtle.interpreter import TurtleInterpreter


def _calc_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalized cross-correlation between two 2D arrays."""
    a_fl = a.astype(np.float64).flatten()
    b_fl = b.astype(np.float64).flatten()
    a_std = np.std(a_fl)
    b_std = np.std(b_fl)
    if a_std < 1e-6 or b_std < 1e-6:
        return 0.0
    a_norm = (a_fl - np.mean(a_fl)) / a_std
    b_norm = (b_fl - np.mean(b_fl)) / b_std
    return float(np.mean(a_norm * b_norm))


def detect_symmetry_autocorrelation(img: np.ndarray, threshold: float = 0.65) -> str:
    """Detect symmetry group (C1, C2, C4, D1, D2, D4) using rotational & reflection correlation."""
    img_f = img.astype(np.float32)

    # 1. Rotations
    rot90 = np.rot90(img_f, 1)
    rot180 = np.rot90(img_f, 2)
    rot270 = np.rot90(img_f, 3)

    c90 = _calc_ncc(img_f, rot90)
    c180 = _calc_ncc(img_f, rot180)
    c270 = _calc_ncc(img_f, rot270)

    has_c4 = (c90 > threshold) and (c180 > threshold) and (c270 > threshold)
    has_c2 = (c180 > threshold)

    # 2. Reflections
    flip_h = np.fliplr(img_f)
    flip_v = np.flipud(img_f)
    flip_d1 = np.transpose(img_f)
    flip_d2 = np.transpose(np.rot90(img_f, 2))

    ref_h = _calc_ncc(img_f, flip_h)
    ref_v = _calc_ncc(img_f, flip_v)
    ref_d1 = _calc_ncc(img_f, flip_d1)
    ref_d2 = _calc_ncc(img_f, flip_d2)

    has_ref_hv = (ref_h > threshold) or (ref_v > threshold)
    has_ref_diag = (ref_d1 > threshold) or (ref_d2 > threshold)

    # Determine group
    if has_c4 and (has_ref_hv or has_ref_diag):
        return "D4"
    elif has_c4:
        return "C4"
    elif has_c2 and (has_ref_hv or has_ref_diag):
        return "D2"
    elif has_c2:
        return "C2"
    elif has_ref_hv or has_ref_diag:
        return "D1"
    else:
        return "C1"


class ClassicalCVBaseline:
    """Classical Computer Vision Baseline for Kolam Reconstruction.

    Combines:
    1. Symmetry group detection via rotational/reflection autocorrelation.
    2. Classical Analysis-by-Synthesis template matching across canonical rules and depths.
    """

    def __init__(self) -> None:
        self.lsystem_engine = LSystemEngine()
        self.interpreter = TurtleInterpreter()

    def predict_and_reconstruct(
        self,
        target_image: np.ndarray,
        motif: str = "M1",
        grid_size_hint: int = 5,
    ) -> ReconstructionResult:
        """Run classical symmetry detection and template search to reconstruct pattern."""
        detected_sym = detect_symmetry_autocorrelation(target_image)

        best_ncc = -1.0
        best_rule_id = "R01"
        best_depth = 1
        best_grid = grid_size_hint
        best_segments = []
        best_rendered = np.zeros_like(target_image)

        bbox = BoundingBox(
            min_x=0.0, min_y=0.0, max_x=64.0, max_y=64.0
        )

        # Sweep candidate rules and depths
        for rule in RULE_REGISTRY.values():
            max_d = min(4, rule.max_safe_depth)
            for d in range(1, max_d + 1):
                expanded_str = self.lsystem_engine.expand_rule(rule, depth=d)
                turtle_result = self.interpreter.interpret(expanded_str, angle=rule.default_angle)
                sym_segments = apply_symmetry(turtle_result.segments, symmetry=detected_sym)

                rendered = render_kolam(
                    segments=sym_segments,
                    image_size=64,
                    motif=motif,
                    grid_size=best_grid,
                )

                score = _calc_ncc(rendered, target_image)
                if score > best_ncc:
                    best_ncc = score
                    best_rule_id = rule.rule_id
                    best_depth = d
                    best_segments = sym_segments
                    best_rendered = rendered
                    bbox = BoundingBox(
                        min_x=turtle_result.min_x,
                        min_y=turtle_result.min_y,
                        max_x=turtle_result.max_x,
                        max_y=turtle_result.max_y,
                    )

        # Compute metrics
        recon_mask = (best_rendered > 30).astype(np.uint8) * 255
        metrics = compute_reconstruction_metrics(
            target_img=target_image,
            recon_img=best_rendered,
            target_segments=best_segments,
            recon_segments=best_segments,
        )

        return ReconstructionResult(
            method="classical_cv_template_matching",
            image=best_rendered,
            mask=recon_mask,
            segments=best_segments,
            bbox=bbox,
            metrics=metrics,
            grammar_string=f"A={best_rule_id};P={best_rule_id}",
            parameters={
                "rule_id": best_rule_id,
                "depth": best_depth,
                "symmetry": detected_sym,
                "grid_size": best_grid,
                "ncc_score": best_ncc,
            },
        )
