"""Analysis-by-Synthesis search metric sensitivity ablation for Protocol B."""

from __future__ import annotations

import numpy as np

from kolam_r.grammar.executor import ParsedGrammar
from kolam_r.lsystem.rules import ProductionRule
from kolam_r.reconstruction.metrics import (
    compute_iou,
    compute_mse,
    compute_ncc,
    compute_ssim,
)
from kolam_r.renderer.image_renderer import render_kolam
from kolam_r.symmetry.transforms import apply_symmetry
from kolam_r.turtle.interpreter import TurtleInterpreter


def search_depth_multi_metric(
    parsed: ParsedGrammar,
    target_image: np.ndarray,
    angle: float,
    symmetry: str = "C1",
    grid_size: int = 5,
    motif: str = "M1",
    max_safe_depth: int = 4,
    search_metric: str = "ncc",  # "ncc", "ssim", "iou", "mse"
) -> tuple[int, float]:
    """Sweep candidate depths [1..max_safe_depth] using specified similarity metric."""
    if not parsed.is_valid:
        return 1, 0.0

    interp = TurtleInterpreter()
    rule = ProductionRule(
        rule_id="DYNAMIC",
        name="Dynamic Grammar",
        axiom=parsed.axiom,
        productions=parsed.productions,
        default_angle=angle,
        description="Dynamic grammar for depth search",
        max_safe_depth=max_safe_depth,
        connectivity="closed",
        source="Synthesized",
    )

    from kolam_r.lsystem.engine import LSystemEngine
    engine = LSystemEngine()

    best_d = 1
    best_score = -float("inf") if search_metric != "mse" else float("inf")

    for d in range(1, max_safe_depth + 1):
        expanded = engine.expand_rule(rule, depth=d)
        res = interp.interpret(expanded, angle=angle)
        sym_segs = apply_symmetry(res.segments, symmetry=symmetry)

        candidate = render_kolam(
            segments=sym_segs,
            image_size=64,
            motif=motif,
            grid_size=grid_size,
        )

        if search_metric == "ncc":
            score = compute_ncc(candidate, target_image)
            if score > best_score:
                best_score = score
                best_d = d
        elif search_metric == "ssim":
            score = compute_ssim(candidate, target_image)
            if score > best_score:
                best_score = score
                best_d = d
        elif search_metric == "iou":
            score = compute_iou(candidate, target_image)
            if score > best_score:
                best_score = score
                best_d = d
        elif search_metric == "mse":
            score = compute_mse(candidate, target_image)
            if score < best_score:
                best_score = score
                best_d = d

    return best_d, float(best_score)
