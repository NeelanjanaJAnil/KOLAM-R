"""Unit tests for multi-fidelity reconstruction metrics."""

from __future__ import annotations

import numpy as np
import pytest

from kolam_r.reconstruction.metrics import (
    compute_chamfer_distance,
    compute_dice_coefficient,
    compute_iou,
    compute_mse,
    compute_ncc,
    compute_psnr,
    compute_reconstruction_metrics,
    compute_ssim,
)
from kolam_r.turtle.interpreter import LineSegment


class TestReconstructionMetrics:
    """Tests for metric calculations."""

    def test_identity_metrics(self):
        """Identical images must yield perfect scores: MSE=0, PSNR=100, SSIM=1, NCC=1, IoU=1, Dice=1."""
        img = np.zeros((64, 64), dtype=np.uint8)
        img[20:44, 20:44] = 255

        assert compute_mse(img, img) == 0.0
        assert compute_psnr(img, img) >= 99.0
        assert compute_ssim(img, img) == pytest.approx(1.0, abs=1e-3)
        assert compute_ncc(img, img) == pytest.approx(1.0, abs=1e-3)
        assert compute_iou(img, img) == 1.0
        assert compute_dice_coefficient(img, img) == 1.0

    def test_disjoint_images(self):
        """Completely non-overlapping foreground patterns must yield IoU=0, Dice=0."""
        img1 = np.zeros((64, 64), dtype=np.uint8)
        img2 = np.zeros((64, 64), dtype=np.uint8)
        img1[10:20, 10:20] = 255
        img2[40:50, 40:50] = 255

        assert compute_iou(img1, img2) == 0.0
        assert compute_dice_coefficient(img1, img2) == 0.0

    def test_partial_overlap_iou_dice(self):
        """50% overlapping boxes."""
        img1 = np.zeros((64, 64), dtype=np.uint8)
        img2 = np.zeros((64, 64), dtype=np.uint8)
        img1[10:30, 10:30] = 255  # 400 pixels
        img2[10:30, 20:40] = 255  # 400 pixels, overlap = 200 pixels

        iou = compute_iou(img1, img2)
        dice = compute_dice_coefficient(img1, img2)

        # Intersection = 200, Union = 600 -> IoU = 1/3
        assert iou == pytest.approx(1.0 / 3.0, abs=1e-2)
        # Dice = 2 * 200 / (400 + 400) = 0.5
        assert dice == pytest.approx(0.5, abs=1e-2)

    def test_chamfer_distance_identical_segments(self):
        """Identical line segments must yield Chamfer distance of 0.0."""
        segs = [
            LineSegment(0.0, 0.0, 10.0, 0.0),
            LineSegment(10.0, 0.0, 10.0, 10.0),
        ]
        d = compute_chamfer_distance(segs, segs, num_samples=100)
        assert d == pytest.approx(0.0, abs=1e-3)

    def test_chamfer_distance_shifted_segments(self):
        """Segments shifted by delta=5.0 should have Chamfer distance ~10.0 (sum of both directions)."""
        segs1 = [LineSegment(0.0, 0.0, 10.0, 0.0)]
        segs2 = [LineSegment(0.0, 5.0, 10.0, 5.0)]

        d = compute_chamfer_distance(segs1, segs2, num_samples=100)
        assert d == pytest.approx(10.0, abs=0.5)

    def test_full_reconstruction_metrics_container(self):
        """Verify compute_reconstruction_metrics returns a valid populated dataclass."""
        img1 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

        metrics = compute_reconstruction_metrics(img1, img2)
        d = metrics.to_dict()

        assert "mse" in d
        assert "psnr" in d
        assert "ssim" in d
        assert "ncc" in d
        assert "iou" in d
        assert "dice" in d
        assert "chamfer_distance" in d
