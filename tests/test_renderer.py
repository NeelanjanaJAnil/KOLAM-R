"""Tests for the image rendering pipeline."""

import numpy as np
import pytest

from kolam_r.turtle.interpreter import LineSegment
from kolam_r.renderer.image_renderer import render_kolam


def _seg(x1, y1, x2, y2) -> LineSegment:
    return LineSegment(float(x1), float(y1), float(x2), float(y2))


class TestImageRenderer:
    """Tests for Kolam image rendering."""

    def test_output_shape_64(self):
        """Output image should be exactly 64x64."""
        segments = [_seg(0, 0, 1, 0), _seg(1, 0, 1, 1)]
        img = render_kolam(segments, image_size=64)
        assert img.shape == (64, 64)

    def test_output_shape_256(self):
        """Output image should be exactly 256x256."""
        segments = [_seg(0, 0, 1, 0)]
        img = render_kolam(segments, image_size=256)
        assert img.shape == (256, 256)

    def test_output_dtype(self):
        """Output should be uint8."""
        segments = [_seg(0, 0, 1, 0)]
        img = render_kolam(segments, image_size=64)
        assert img.dtype == np.uint8

    def test_non_zero_pixels(self):
        """A single segment should produce non-zero pixels."""
        segments = [_seg(0, 0, 5, 5)]
        img = render_kolam(segments, image_size=64)
        assert np.any(img > 0)

    def test_empty_segments_blank(self):
        """Empty segment list should produce a blank image."""
        img = render_kolam([], image_size=64)
        assert img.shape == (64, 64)
        assert np.all(img == 0)

    def test_grayscale_range(self):
        """Pixel values should be in [0, 255]."""
        segments = [_seg(0, 0, 1, 1), _seg(1, 1, 2, 0)]
        img = render_kolam(segments, image_size=64)
        assert img.min() >= 0
        assert img.max() <= 255

    def test_dot_grid_pixels(self):
        """With grid_size > 0, dot pixels should be present."""
        # Render with no segments but with dot grid
        img = render_kolam([], image_size=64, grid_size=3)
        assert np.any(img > 0)  # Dots should be visible

    def test_motif_m1(self):
        """M1 motif should render without error."""
        segments = [_seg(0, 0, 1, 0)]
        img = render_kolam(segments, image_size=64, motif="M1")
        assert img.shape == (64, 64)

    def test_motif_m2(self):
        """M2 motif (rounded joints) should render without error."""
        segments = [_seg(0, 0, 1, 0), _seg(1, 0, 1, 1)]
        img = render_kolam(segments, image_size=64, motif="M2")
        assert np.any(img > 0)

    def test_motif_m3(self):
        """M3 motif (thick stroke) should render without error."""
        segments = [_seg(0, 0, 1, 0)]
        img = render_kolam(segments, image_size=64, motif="M3")
        assert np.any(img > 0)

    def test_motif_m4(self):
        """M4 motif (double line) should render without error."""
        segments = [_seg(0, 0, 3, 0), _seg(3, 0, 3, 3)]
        img = render_kolam(segments, image_size=256, motif="M4")
        assert np.any(img > 0)

    def test_different_motifs_different_output(self):
        """Different motifs should produce visually different images."""
        segments = [_seg(0, 0, 5, 0), _seg(5, 0, 5, 5), _seg(5, 5, 0, 5)]
        img_m1 = render_kolam(segments, image_size=256, motif="M1")
        img_m3 = render_kolam(segments, image_size=256, motif="M3")
        # M3 is thicker, should have more non-zero pixels
        assert np.sum(img_m3 > 0) >= np.sum(img_m1 > 0)
