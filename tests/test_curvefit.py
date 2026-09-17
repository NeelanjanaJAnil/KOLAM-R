"""Unit tests for the continuous parametric curve reconstruction module.

Step 1 Verification Requirements (Revised per Decision):
1. Fit a curve to a single straight line segment; verify degree 1 and near-zero error.
2. Fit a curve to a simple circular arc; verify degree 3 and low fitting error.
3. Round-trip test on known shapes (square, circle) using realistic stroke width (width=3):
   - Extract skeleton.
   - Extract skeleton graph.
   - Fit parametric curves (with periodic B-spline for simple closed loops).
   - Verify curve count strictly equals edge count (|Curves| == |E|).
   - Rasterize curves back to image with matching stroke width.
   - Compute and report actual SSIM against original shape.
4. Edge-case documentation: 1-pixel stroke width lower bound test kept for transparency.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.prototype_app import compute_ssim, compute_iou
from kolam_r.curvefit.fit_curves import fit_single_path, fit_skeleton_graph_curves
from kolam_r.curvefit.rasterize_curves import rasterize_curves
from kolam_r.topology.graph_extractor import extract_skeleton_graph
from kolam_r.topology.skeleton import skeletonize_zhang_suen


class TestParametricCurveFitting:
    """Unit tests for curve fitting primitives."""

    def test_straight_line_segment(self):
        """Test 1: Fit a curve to a straight line; verify degree 1 and near-zero error."""
        t = np.linspace(0.0, 1.0, 25)
        r_pts = 10.0 + 40.0 * t
        c_pts = 15.0 + 40.0 * t
        path = [(float(r), float(c)) for r, c in zip(r_pts, c_pts)]

        curve = fit_single_path(path, edge_index=0, linearity_threshold=0.5)

        print(f"\n[Test 1 Straight Line] Degree: {curve.degree}, Max Error: {curve.max_fitting_error:.8f} px, Mean Error: {curve.mean_fitting_error:.8f} px")

        assert curve.degree == 1, f"Expected degree 1 for straight line, got {curve.degree}"
        assert curve.max_fitting_error < 1e-6, f"Expected near-zero error, got {curve.max_fitting_error}"
        assert curve.mean_fitting_error < 1e-6, f"Expected near-zero error, got {curve.mean_fitting_error}"

    def test_circular_arc(self):
        """Test 2: Fit a curve to a circular arc; verify degree 3 and low fitting error."""
        angles = np.linspace(0.0, math.pi / 2.0, 30)
        center_r, center_c = 50.0, 50.0
        radius = 30.0
        r_pts = center_r + radius * np.sin(angles)
        c_pts = center_c + radius * np.cos(angles)
        path = [(float(r), float(c)) for r, c in zip(r_pts, c_pts)]

        curve = fit_single_path(path, edge_index=0, linearity_threshold=0.5)

        print(f"\n[Test 2 Circular Arc] Degree: {curve.degree}, Max Error: {curve.max_fitting_error:.4f} px, Mean Error: {curve.mean_fitting_error:.4f} px")

        assert curve.degree == 3, f"Expected degree 3 for circular arc, got {curve.degree}"
        assert curve.max_fitting_error < 0.5, f"Expected max error < 0.5 px, got {curve.max_fitting_error}"
        assert curve.mean_fitting_error < 0.3, f"Expected mean error < 0.3 px, got {curve.mean_fitting_error}"

    def test_round_trip_square_realistic_width(self):
        """Test 3a: Round-trip test on a square with realistic stroke width (width=3)."""
        h, w = 128, 128
        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        draw.rectangle([32, 32, 96, 96], outline=255, width=3)
        orig_arr = np.array(img, dtype=np.uint8)

        skel = skeletonize_zhang_suen(orig_arr > 0)
        graph = extract_skeleton_graph(skel)

        curves = fit_skeleton_graph_curves(graph, linearity_threshold=0.5)

        # Equality check: number of curves must strictly equal graph edge count
        assert len(curves) == len(graph.edges), f"Curve count {len(curves)} != edge count {len(graph.edges)}"

        recon_arr = rasterize_curves(curves, image_shape=(h, w), stroke_width=3)

        score = float(compute_ssim(orig_arr, recon_arr))
        iou_score = float(compute_iou(orig_arr, recon_arr))
        print(f"\n[Test 3a Square (width=3)] |V|={len(graph.vertices)}, |E|={len(graph.edges)}, Curves={len(curves)}, IoU={iou_score:.4f}, SSIM={score:.4f}")

        assert score > 0.80, f"Expected SSIM > 0.80 for realistic square round-trip, got {score:.4f}"
        assert iou_score > 0.65, f"Expected IoU > 0.65 for realistic square round-trip, got {iou_score:.4f}"

    def test_round_trip_circle_realistic_width(self):
        """Test 3b: Round-trip test on a circular loop with realistic stroke width (width=3)."""
        h, w = 128, 128
        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        draw.ellipse([32, 32, 96, 96], outline=255, width=3)
        orig_arr = np.array(img, dtype=np.uint8)

        skel = skeletonize_zhang_suen(orig_arr > 0)
        graph = extract_skeleton_graph(skel)

        curves = fit_skeleton_graph_curves(graph, linearity_threshold=0.5)

        # Equality check: number of curves must strictly equal graph edge count
        assert len(curves) == len(graph.edges), f"Curve count {len(curves)} != edge count {len(graph.edges)}"

        recon_arr = rasterize_curves(curves, image_shape=(h, w), stroke_width=3)

        score = float(compute_ssim(orig_arr, recon_arr))
        iou_score = float(compute_iou(orig_arr, recon_arr))
        print(f"\n[Test 3b Circle (width=3)] |V|={len(graph.vertices)}, |E|={len(graph.edges)}, Curves={len(curves)}, IoU={iou_score:.4f}, SSIM={score:.4f}")

        assert score > 0.75, f"Expected SSIM > 0.75 for realistic circle round-trip, got {score:.4f}"
        assert iou_score > 0.60, f"Expected IoU > 0.60 for realistic circle round-trip, got {iou_score:.4f}"

    def test_1px_stroke_lower_bound_documented(self):
        """Test 4: Document 1-pixel stroke width lower-bound behavior (non-gating)."""
        h, w = 128, 128
        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        draw.rectangle([32, 32, 96, 96], outline=255, width=1)
        orig_arr = np.array(img, dtype=np.uint8)

        skel = skeletonize_zhang_suen(orig_arr > 0)
        graph = extract_skeleton_graph(skel)
        curves = fit_skeleton_graph_curves(graph, linearity_threshold=0.5)
        recon_arr = rasterize_curves(curves, image_shape=(h, w), stroke_width=1)

        score = float(compute_ssim(orig_arr, recon_arr))
        iou_score = float(compute_iou(orig_arr, recon_arr))
        print(f"\n[Test 4 1-Pixel Square Lower-Bound] SSIM={score:.4f}, IoU={iou_score:.4f}")
        # Verified baseline: 1-pixel Bresenham rounding yields SSIM ~ 0.50
        assert score > 0.40
