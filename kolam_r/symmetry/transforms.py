"""Symmetry group transformations for Kolam patterns.

Applies mathematical symmetry group actions (C1, C2, C4, D1, D2, D4)
to line segments, replicating a base pattern according to the
group's generators.

All transformations are relative to the origin (0, 0), which should
be the center of the pattern's bounding box.
"""

from __future__ import annotations

import math
from typing import Literal

from kolam_r.turtle.interpreter import LineSegment


# Type alias for symmetry class names
SymmetryClass = Literal["C1", "C2", "C4", "D1", "D2", "D4"]


def _rotate_point(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    """Rotate a point around the origin by angle_deg degrees (counter-clockwise)."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)


def _reflect_point(
    x: float, y: float, axis: str
) -> tuple[float, float]:
    """Reflect a point across the specified axis.

    Args:
        x, y: Point coordinates.
        axis: One of 'horizontal' (y-axis, x -> -x),
              'vertical' (x-axis, y -> -y),
              'diagonal_main' (y=x line, swap x,y),
              'diagonal_anti' (y=-x line, swap and negate).
    """
    if axis == "horizontal":
        return (-x, y)
    elif axis == "vertical":
        return (x, -y)
    elif axis == "diagonal_main":
        return (y, x)
    elif axis == "diagonal_anti":
        return (-y, -x)
    else:
        raise ValueError(f"Unknown reflection axis: {axis}")


def _transform_segment(
    seg: LineSegment,
    rotation_deg: float = 0.0,
    reflect_axis: str | None = None,
) -> LineSegment:
    """Apply rotation and/or reflection to a line segment."""
    x1, y1, x2, y2 = seg.x1, seg.y1, seg.x2, seg.y2

    # Apply reflection first, then rotation
    if reflect_axis is not None:
        x1, y1 = _reflect_point(x1, y1, reflect_axis)
        x2, y2 = _reflect_point(x2, y2, reflect_axis)

    if abs(rotation_deg) > 1e-9:
        x1, y1 = _rotate_point(x1, y1, rotation_deg)
        x2, y2 = _rotate_point(x2, y2, rotation_deg)

    return LineSegment(x1, y1, x2, y2)


def _round_segment(seg: LineSegment, decimals: int = 6) -> tuple:
    """Round segment coordinates for deduplication."""
    return (
        round(seg.x1, decimals),
        round(seg.y1, decimals),
        round(seg.x2, decimals),
        round(seg.y2, decimals),
    )


def _center_segments(
    segments: list[LineSegment],
) -> tuple[list[LineSegment], float, float]:
    """Translate segments so their bounding box is centered at the origin.

    Returns:
        Tuple of (centered_segments, center_x, center_y) where center_x/y
        are the original center coordinates.
    """
    if not segments:
        return segments, 0.0, 0.0

    all_x = [s.x1 for s in segments] + [s.x2 for s in segments]
    all_y = [s.y1 for s in segments] + [s.y2 for s in segments]
    cx = (min(all_x) + max(all_x)) / 2.0
    cy = (min(all_y) + max(all_y)) / 2.0

    centered = [
        LineSegment(s.x1 - cx, s.y1 - cy, s.x2 - cx, s.y2 - cy)
        for s in segments
    ]
    return centered, cx, cy


def apply_symmetry(
    segments: list[LineSegment],
    symmetry: SymmetryClass,
    deduplicate: bool = True,
    center: tuple[float, float] | None = (0.0, 0.0),
) -> list[LineSegment]:
    """Apply symmetry group transformations to a set of line segments.

    Transformations (rotations and reflections) are performed relative to
    the given center point (default (0, 0)).

    Args:
        segments: Base pattern line segments.
        symmetry: Symmetry class name.
        deduplicate: If True, remove duplicate segments (after rounding).
        center: Center of symmetry (cx, cy). If (0.0, 0.0), transforms are
            applied directly around the origin.

    Returns:
        List of line segments after applying all symmetry transforms.
    """
    if not segments:
        return []

    if center is not None:
        cx, cy = center
        centered = [
            LineSegment(s.x1 - cx, s.y1 - cy, s.x2 - cx, s.y2 - cy)
            for s in segments
        ]
    else:
        cx, cy = 0.0, 0.0
        centered = segments

    # Define group actions as (rotation_deg, reflect_axis) pairs
    actions: list[tuple[float, str | None]] = _get_group_actions(symmetry)

    # Apply all group actions
    all_segments: list[LineSegment] = []
    seen: set[tuple] = set()

    for rotation_deg, reflect_axis in actions:
        for seg in centered:
            transformed = _transform_segment(seg, rotation_deg, reflect_axis)
            if deduplicate:
                key = _round_segment(transformed)
                # Also check the reversed segment
                key_rev = (key[2], key[3], key[0], key[1])
                if key in seen or key_rev in seen:
                    continue
                seen.add(key)
            all_segments.append(transformed)

    # Translate back if non-zero center was used
    if center is not None and (cx != 0.0 or cy != 0.0):
        return [
            LineSegment(s.x1 + cx, s.y1 + cy, s.x2 + cx, s.y2 + cy)
            for s in all_segments
        ]
    return all_segments


def _get_group_actions(
    symmetry: SymmetryClass,
) -> list[tuple[float, str | None]]:
    """Return the list of (rotation, reflection) pairs for a symmetry group.

    Each tuple represents one group element as (rotation_degrees, reflect_axis).
    """
    if symmetry == "C1":
        # Trivial group: identity only
        return [(0.0, None)]

    elif symmetry == "C2":
        # Cyclic group of order 2: identity + 180° rotation
        return [
            (0.0, None),
            (180.0, None),
        ]

    elif symmetry == "C4":
        # Cyclic group of order 4: rotations by 0°, 90°, 180°, 270°
        return [
            (0.0, None),
            (90.0, None),
            (180.0, None),
            (270.0, None),
        ]

    elif symmetry == "D1":
        # Dihedral group of order 2: identity + vertical reflection
        return [
            (0.0, None),
            (0.0, "horizontal"),
        ]

    elif symmetry == "D2":
        # Dihedral group of order 4: identity + 180° rotation + 2 reflections
        return [
            (0.0, None),
            (180.0, None),
            (0.0, "horizontal"),
            (0.0, "vertical"),
        ]

    elif symmetry == "D4":
        # Dihedral group of order 8: 4 rotations + 4 reflections
        return [
            (0.0, None),
            (90.0, None),
            (180.0, None),
            (270.0, None),
            (0.0, "horizontal"),
            (0.0, "vertical"),
            (0.0, "diagonal_main"),
            (0.0, "diagonal_anti"),
        ]

    else:
        raise ValueError(f"Unknown symmetry class: {symmetry}")


def get_group_order(symmetry: SymmetryClass) -> int:
    """Return the order (number of elements) of the symmetry group."""
    orders = {"C1": 1, "C2": 2, "C4": 4, "D1": 2, "D2": 4, "D4": 8}
    return orders[symmetry]
