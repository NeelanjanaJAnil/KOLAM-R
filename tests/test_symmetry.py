"""Tests for symmetry group transformations."""

import math

import pytest

from kolam_r.turtle.interpreter import LineSegment
from kolam_r.symmetry.transforms import apply_symmetry, get_group_order


def _seg(x1, y1, x2, y2) -> LineSegment:
    """Shorthand for creating a LineSegment."""
    return LineSegment(float(x1), float(y1), float(x2), float(y2))


class TestSymmetryTransforms:
    """Tests for symmetry group transformations."""

    def test_c1_identity(self):
        """C1 should return segments unchanged."""
        segments = [_seg(0, 0, 1, 0)]
        result = apply_symmetry(segments, "C1")
        assert len(result) == 1
        assert abs(result[0].x2 - result[0].x1 - 1.0) < 1e-6

    def test_c2_doubles(self):
        """C2 should produce ~2x segments (minus deduplication)."""
        segments = [_seg(1, 0, 2, 0)]  # Off-center to avoid overlap
        result = apply_symmetry(segments, "C2")
        assert len(result) == 2

    def test_c4_quadruples(self):
        """C4 should produce 4 rotated copies of an off-center segment."""
        segments = [_seg(1, 0, 2, 0)]  # Clearly off-center
        result = apply_symmetry(segments, "C4")
        assert len(result) == 4

    def test_d4_eight_copies(self):
        """D4 should produce 8 copies of a general-position segment."""
        # Use a segment that's not on any symmetry axis
        segments = [_seg(1, 0.5, 2, 0.5)]
        result = apply_symmetry(segments, "D4")
        assert len(result) == 8

    def test_d1_reflection(self):
        """D1 should produce 2 copies via reflection."""
        segments = [_seg(1, 1, 2, 1)]
        result = apply_symmetry(segments, "D1")
        assert len(result) == 2

    def test_d2_four_copies(self):
        """D2 should produce 4 copies."""
        segments = [_seg(1, 1, 2, 1)]
        result = apply_symmetry(segments, "D2")
        assert len(result) == 4

    def test_empty_segments(self):
        """Empty input should return empty output."""
        result = apply_symmetry([], "C4")
        assert result == []

    def test_c4_single_horizontal_produces_four_orientations(self):
        """A horizontal segment under C4 should produce 4 oriented segments."""
        segments = [_seg(1, 0, 2, 0)]
        result = apply_symmetry(segments, "C4", deduplicate=True)
        # Should have segments in 4 different orientations
        angles = set()
        for seg in result:
            dx = seg.x2 - seg.x1
            dy = seg.y2 - seg.y1
            angle = round(math.degrees(math.atan2(dy, dx)) % 360, 1)
            angles.add(angle)
        assert len(angles) == 4

    def test_deduplication(self):
        """Centered segment under C2 should deduplicate to fewer segments."""
        # A segment through origin: C2 rotation maps it to itself (reversed)
        segments = [_seg(-1, 0, 1, 0)]
        result = apply_symmetry(segments, "C2", deduplicate=True)
        assert len(result) == 1  # 180° rotation of (-1,0)-(1,0) gives (1,0)-(-1,0) = same segment reversed

    def test_no_deduplication(self):
        """Without deduplication, all copies should be returned."""
        segments = [_seg(-1, 0, 1, 0)]
        result = apply_symmetry(segments, "C2", deduplicate=False)
        assert len(result) == 2

    def test_group_order(self):
        """Group orders should be correct."""
        assert get_group_order("C1") == 1
        assert get_group_order("C2") == 2
        assert get_group_order("C4") == 4
        assert get_group_order("D1") == 2
        assert get_group_order("D2") == 4
        assert get_group_order("D4") == 8

    def test_invalid_symmetry(self):
        """Invalid symmetry class should raise ValueError."""
        with pytest.raises(ValueError):
            apply_symmetry([_seg(0, 0, 1, 0)], "C3")
