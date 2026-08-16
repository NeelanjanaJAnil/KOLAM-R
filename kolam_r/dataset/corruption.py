"""Evaluation-only corruption suite for KOLAM-R.

Defines 5 realistic visual corruptions strictly for testing model robustness:
    1. Gaussian noise
    2. Small rotation
    3. Affine distortion
    4. Gaussian blur
    5. Stroke dropout (patch erasure)

SCIENTIFIC RULE: These corruptions are strictly for evaluation benchmarks
and must NEVER be used on the clean training set.
"""

from __future__ import annotations

import math
import random
from enum import Enum
from typing import Literal

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


class CorruptionType(str, Enum):
    """Supported evaluation-only corruption types."""

    GAUSSIAN_NOISE = "gaussian_noise"
    ROTATION = "rotation"
    AFFINE = "affine"
    BLUR = "blur"
    STROKE_DROPOUT = "stroke_dropout"


def apply_gaussian_noise(
    image: np.ndarray,
    severity: float = 0.15,
    seed: int | None = None,
) -> np.ndarray:
    """Add zero-mean additive Gaussian noise to a grayscale image.

    Args:
        image: Grayscale image as uint8 array of shape (H, W).
        severity: Noise standard deviation relative to 255 (default: 0.15).
        seed: Optional RNG seed.

    Returns:
        Corrupted uint8 image array.
    """
    rng = np.random.default_rng(seed)
    sigma = severity * 255.0
    noise = rng.normal(0, sigma, image.shape)
    corrupted = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return corrupted


def apply_rotation(
    image: np.ndarray,
    angle_deg: float | None = None,
    max_angle: float = 12.0,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a small in-plane rotation to simulate misaligned photography.

    Args:
        image: Grayscale image as uint8 array of shape (H, W).
        angle_deg: Specific rotation angle in degrees. If None, sampled uniformly in [-max_angle, max_angle].
        max_angle: Maximum absolute rotation angle in degrees.
        seed: Optional RNG seed.

    Returns:
        Corrupted uint8 image array of original dimensions.
    """
    if angle_deg is None:
        rng = random.Random(seed)
        angle_deg = rng.uniform(-max_angle, max_angle)

    pil_img = Image.fromarray(image, mode="L")
    rotated = pil_img.rotate(
        angle_deg,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=0,
    )
    return np.array(rotated, dtype=np.uint8)


def apply_affine_distortion(
    image: np.ndarray,
    shear_range: float = 0.15,
    scale_range: tuple[float, float] = (0.9, 1.1),
    seed: int | None = None,
) -> np.ndarray:
    """Apply affine shear and scale distortion to simulate perspective tilt.

    Args:
        image: Grayscale image as uint8 array of shape (H, W).
        shear_range: Maximum shear factor along x and y.
        scale_range: Range for random scaling factor.
        seed: Optional RNG seed.

    Returns:
        Corrupted uint8 image array of original dimensions.
    """
    rng = random.Random(seed)
    shear_x = rng.uniform(-shear_range, shear_range)
    shear_y = rng.uniform(-shear_range, shear_range)
    scale_x = rng.uniform(scale_range[0], scale_range[1])
    scale_y = rng.uniform(scale_range[0], scale_range[1])

    h, w = image.shape
    cx, cy = w / 2.0, h / 2.0

    # Transformation matrix mapping output to input
    # x_in = (x_out - cx) / scale_x - shear_x * (y_out - cy) + cx
    # y_in = (y_out - cy) / scale_y - shear_y * (x_out - cx) + cy
    a = 1.0 / scale_x
    b = -shear_x
    c = cx - a * cx - b * cy
    d = -shear_y
    e = 1.0 / scale_y
    f = cy - d * cx - e * cy

    pil_img = Image.fromarray(image, mode="L")
    transformed = pil_img.transform(
        (w, h),
        Image.Transform.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )
    return np.array(transformed, dtype=np.uint8)


def apply_gaussian_blur(
    image: np.ndarray,
    radius: float = 1.2,
) -> np.ndarray:
    """Apply spatial Gaussian blur to simulate out-of-focus capture.

    Args:
        image: Grayscale image as uint8 array of shape (H, W).
        radius: Blur kernel radius (default: 1.2).

    Returns:
        Corrupted uint8 image array.
    """
    pil_img = Image.fromarray(image, mode="L")
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred, dtype=np.uint8)


def apply_stroke_dropout(
    image: np.ndarray,
    num_patches: int = 3,
    max_patch_size: int = 14,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate partial stroke erasure or chalk occlusion via cutout patches.

    Args:
        image: Grayscale image as uint8 array of shape (H, W).
        num_patches: Number of erasure patches to apply.
        max_patch_size: Maximum width/height of each erasure box.
        seed: Optional RNG seed.

    Returns:
        Corrupted uint8 image array.
    """
    rng = random.Random(seed)
    corrupted = image.copy()
    h, w = image.shape

    # Find foreground pixels to place dropouts strategically over strokes
    fg_y, fg_x = np.where(image > 50)
    if len(fg_y) == 0:
        return corrupted

    for _ in range(num_patches):
        idx = rng.randint(0, len(fg_y) - 1)
        cy, cx = fg_y[idx], fg_x[idx]
        pw = rng.randint(6, max_patch_size)
        ph = rng.randint(6, max_patch_size)

        y1 = max(0, cy - ph // 2)
        y2 = min(h, cy + ph // 2)
        x1 = max(0, cx - pw // 2)
        x2 = min(w, cx + pw // 2)

        corrupted[y1:y2, x1:x2] = 0

    return corrupted


def apply_corruption(
    image: np.ndarray,
    corruption_type: CorruptionType | str,
    severity: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a specified corruption type to an image.

    Args:
        image: Input grayscale image as uint8 array (64, 64).
        corruption_type: Name or enum of the corruption.
        severity: Scaling factor for corruption intensity (1.0 = default).
        seed: Optional seed for reproducibility.

    Returns:
        Corrupted image as uint8 array (64, 64).
    """
    ctype = CorruptionType(corruption_type)

    if ctype == CorruptionType.GAUSSIAN_NOISE:
        return apply_gaussian_noise(image, severity=0.15 * severity, seed=seed)
    elif ctype == CorruptionType.ROTATION:
        return apply_rotation(image, max_angle=12.0 * severity, seed=seed)
    elif ctype == CorruptionType.AFFINE:
        return apply_affine_distortion(
            image,
            shear_range=0.15 * severity,
            scale_range=(1.0 - 0.1 * severity, 1.0 + 0.1 * severity),
            seed=seed,
        )
    elif ctype == CorruptionType.BLUR:
        return apply_gaussian_blur(image, radius=1.2 * severity)
    elif ctype == CorruptionType.STROKE_DROPOUT:
        return apply_stroke_dropout(
            image,
            num_patches=int(3 * severity),
            max_patch_size=int(14 * severity),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
