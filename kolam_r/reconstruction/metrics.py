"""Multi-fidelity reconstruction metrics suite for Kolam patterns.

Computes mathematical fidelity across 3 representations:
1. Pixel Intensity: MSE, PSNR, SSIM, NCC
2. Binary Overlap: IoU, Dice Coefficient
3. Vector Geometry: Bidirectional Point Chamfer Distance
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from kolam_r.turtle.interpreter import LineSegment


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Container for all forward synthesis fidelity metrics."""

    mse: float
    psnr: float
    ssim: float
    ncc: float
    iou: float
    dice: float
    chamfer_distance: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return asdict(self)


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Squared Error between two uint8/float grayscale images."""
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def compute_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = compute_mse(img1, img2)
    if mse < 1e-10:
        return 100.0  # Perfect reconstruction cap
    return float(10.0 * math.log10((max_val ** 2) / mse))


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    max_val: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
    sigma: float = 1.5,
) -> float:
    """Compute Structural Similarity Index (SSIM) using Gaussian window."""
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    # Gaussian filtered means
    mu1 = gaussian_filter(a, sigma=sigma, mode="reflect")
    mu2 = gaussian_filter(b, sigma=sigma, mode="reflect")

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # Variances and covariance
    sigma1_sq = gaussian_filter(a * a, sigma=sigma, mode="reflect") - mu1_sq
    sigma2_sq = gaussian_filter(b * b, sigma=sigma, mode="reflect") - mu2_sq
    sigma12 = gaussian_filter(a * b, sigma=sigma, mode="reflect") - mu1_mu2

    # SSIM map
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
    )
    return float(np.mean(ssim_map))


def compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Normalized Cross-Correlation (NCC) in [-1.0, 1.0]."""
    a = img1.astype(np.float64).flatten()
    b = img2.astype(np.float64).flatten()

    std_a = np.std(a)
    std_b = np.std(b)

    if std_a < 1e-6 or std_b < 1e-6:
        # If both are uniform constant images and equal, NCC = 1.0, else 0.0
        return 1.0 if np.allclose(a, b) else 0.0

    a_norm = (a - np.mean(a)) / (std_a + 1e-12)
    b_norm = (b - np.mean(b)) / (std_b + 1e-12)

    return float(np.mean(a_norm * b_norm))


def compute_iou(
    img1: np.ndarray, img2: np.ndarray, threshold: float = 30.0
) -> float:
    """Compute Intersection-over-Union (IoU) of binarized foreground patterns."""
    m1 = img1 > threshold
    m2 = img2 > threshold

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 1.0 if m1.sum() == 0 and m2.sum() == 0 else 0.0

    return float(intersection / union)


def compute_dice_coefficient(
    img1: np.ndarray, img2: np.ndarray, threshold: float = 30.0
) -> float:
    """Compute Dice similarity coefficient (F1 score) of binarized patterns."""
    m1 = img1 > threshold
    m2 = img2 > threshold

    intersection = np.logical_and(m1, m2).sum()
    total_area = m1.sum() + m2.sum()

    if total_area == 0:
        return 1.0

    return float(2.0 * intersection / total_area)


def _sample_points_from_segments(
    segments: list[LineSegment], num_points: int = 500
) -> np.ndarray:
    """Sample K points uniformly along a set of line segments."""
    if not segments:
        return np.zeros((1, 2), dtype=np.float64)

    lengths = [s.length for s in segments]
    total_len = sum(lengths)

    if total_len < 1e-6:
        return np.array([[segments[0].x1, segments[0].y1]], dtype=np.float64)

    points = []
    # Distribute points proportionally to segment lengths
    for seg, length in zip(segments, lengths):
        n_pts = max(1, int(round(num_points * (length / total_len))))
        t = np.linspace(0.0, 1.0, n_pts, endpoint=True)
        xs = seg.x1 + t * (seg.x2 - seg.x1)
        ys = seg.y1 + t * (seg.y2 - seg.y1)
        for x, y in zip(xs, ys):
            points.append([x, y])

    return np.array(points, dtype=np.float64)


def compute_chamfer_distance(
    segments1: list[LineSegment] | np.ndarray,
    segments2: list[LineSegment] | np.ndarray,
    num_samples: int = 500,
) -> float:
    """Compute bidirectional Chamfer distance between two point sets or segment lists.

    Args:
        segments1: Ground truth LineSegment list or (N, 2) coordinates.
        segments2: Predicted LineSegment list or (M, 2) coordinates.
        num_samples: Number of points to sample along line segments.

    Returns:
        Average Euclidean nearest-neighbor distance (in canvas coordinate units).
    """
    if isinstance(segments1, list):
        pts1 = _sample_points_from_segments(segments1, num_samples)
    else:
        pts1 = segments1.astype(np.float64)

    if isinstance(segments2, list):
        pts2 = _sample_points_from_segments(segments2, num_samples)
    else:
        pts2 = segments2.astype(np.float64)

    if len(pts1) == 0 or len(pts2) == 0:
        return 100.0  # Max penalty

    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)

    # Distances from pts1 to nearest in pts2
    d1_to_2, _ = tree2.query(pts1)
    # Distances from pts2 to nearest in pts1
    d2_to_1, _ = tree1.query(pts2)

    return float(np.mean(d1_to_2) + np.mean(d2_to_1))


def compute_reconstruction_metrics(
    target_img: np.ndarray,
    recon_img: np.ndarray,
    target_segments: list[LineSegment] | None = None,
    recon_segments: list[LineSegment] | None = None,
) -> ReconstructionMetrics:
    """Compute full suite of reconstruction fidelity metrics."""
    mse = compute_mse(target_img, recon_img)
    psnr = compute_psnr(target_img, recon_img)
    ssim = compute_ssim(target_img, recon_img)
    ncc = compute_ncc(target_img, recon_img)
    iou = compute_iou(target_img, recon_img)
    dice = compute_dice_coefficient(target_img, recon_img)

    if target_segments is not None and recon_segments is not None and len(target_segments) > 0 and len(recon_segments) > 0:
        chamfer = compute_chamfer_distance(target_segments, recon_segments)
    else:
        # Fallback: compute chamfer distance on non-zero pixel coordinates
        pts_target = np.argwhere(target_img > 30)
        pts_recon = np.argwhere(recon_img > 30)
        chamfer = compute_chamfer_distance(pts_target, pts_recon)

    return ReconstructionMetrics(
        mse=mse,
        psnr=psnr,
        ssim=ssim,
        ncc=ncc,
        iou=iou,
        dice=dice,
        chamfer_distance=chamfer,
    )
