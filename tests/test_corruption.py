"""Tests for evaluation-only corruption suite."""

import numpy as np
import pytest

from kolam_r.dataset.corruption import (
    CorruptionType,
    apply_affine_distortion,
    apply_corruption,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_rotation,
    apply_stroke_dropout,
)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a synthetic 64x64 test image with a white square in center."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[20:44, 20:44] = 255
    return img


class TestCorruptions:
    """Tests for individual corruption algorithms."""

    def test_gaussian_noise(self, sample_image):
        """Gaussian noise should alter pixel values while preserving shape and dtype."""
        corrupted = apply_gaussian_noise(sample_image, severity=0.15, seed=42)
        assert corrupted.shape == (64, 64)
        assert corrupted.dtype == np.uint8
        assert not np.array_equal(corrupted, sample_image)
        # Background should have noise (non-zero)
        assert np.mean(corrupted[:10, :10]) > 0

    def test_rotation(self, sample_image):
        """Rotation should preserve shape, dtype, and bounds."""
        corrupted = apply_rotation(sample_image, angle_deg=10.0)
        assert corrupted.shape == (64, 64)
        assert corrupted.dtype == np.uint8
        assert np.any(corrupted > 0)

    def test_affine(self, sample_image):
        """Affine distortion should transform image geometry."""
        corrupted = apply_affine_distortion(sample_image, seed=42)
        assert corrupted.shape == (64, 64)
        assert corrupted.dtype == np.uint8
        assert np.any(corrupted > 0)

    def test_gaussian_blur(self, sample_image):
        """Blur should soften sharp edges."""
        corrupted = apply_gaussian_blur(sample_image, radius=1.5)
        assert corrupted.shape == (64, 64)
        assert corrupted.dtype == np.uint8
        # Blurred edge pixel should be intermediate value
        assert 0 < corrupted[19, 32] < 255

    def test_stroke_dropout(self, sample_image):
        """Stroke dropout should zero out foreground regions."""
        corrupted = apply_stroke_dropout(sample_image, num_patches=3, seed=42)
        assert corrupted.shape == (64, 64)
        assert corrupted.dtype == np.uint8
        # Foreground pixel count should be strictly less
        assert np.sum(corrupted > 0) < np.sum(sample_image > 0)

    def test_apply_corruption_dispatcher(self, sample_image):
        """Universal dispatcher must handle all enum types correctly."""
        for ctype in CorruptionType:
            corrupted = apply_corruption(sample_image, ctype, seed=42)
            assert corrupted.shape == (64, 64)
            assert corrupted.dtype == np.uint8
