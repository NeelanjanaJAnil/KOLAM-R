"""Geometry extraction module for KOLAM-R.

Extracts ordered continuous stroke chains and path coordinates from
turtle line segments without rasterization distortion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from kolam_r.turtle.interpreter import LineSegment


@dataclass
class ContinuousStroke:
    """An ordered continuous polygonal stroke path (x_i, y_i)."""

    stroke_id: int
    points: np.ndarray  # Shape: (N, 2)
    is_closed: bool

    @property
    def num_points(self) -> int:
        return len(self.points)

    @property
    def total_length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        diffs = np.diff(self.points, axis=0)
        return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

    @property
    def cumulative_distances(self) -> np.ndarray:
        """Cumulative arc length parameter s in [0, L]."""
        if len(self.points) < 2:
            return np.array([0.0])
        diffs = np.diff(self.points, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        return np.concatenate([[0.0], np.cumsum(dists)])

    @property
    def normalized_parameter(self) -> np.ndarray:
        """Normalized parameter t in [0, 1]."""
        s = self.cumulative_distances
        total = s[-1]
        if total <= 1e-9:
            return np.linspace(0.0, 1.0, len(s))
        return s / total


def extract_continuous_strokes(
    segments: list[LineSegment], tolerance: float = 1e-4
) -> list[ContinuousStroke]:
    """Chain ordered turtle line segments into continuous strokes."""
    if not segments:
        return []

    strokes: list[ContinuousStroke] = []
    curr_pts: list[tuple[float, float]] = [(segments[0].x1, segments[0].y1), (segments[0].x2, segments[0].y2)]

    for seg in segments[1:]:
        last_x, last_y = curr_pts[-1]
        # Check if this segment connects to the last point
        if math.hypot(seg.x1 - last_x, seg.y1 - last_y) <= tolerance:
            curr_pts.append((seg.x2, seg.y2))
        else:
            # End of previous stroke, start new stroke
            pts_arr = np.array(curr_pts, dtype=np.float64)
            is_closed = math.hypot(pts_arr[0, 0] - pts_arr[-1, 0], pts_arr[0, 1] - pts_arr[-1, 1]) <= tolerance
            strokes.append(ContinuousStroke(stroke_id=len(strokes), points=pts_arr, is_closed=is_closed))
            curr_pts = [(seg.x1, seg.y1), (seg.x2, seg.y2)]

    if curr_pts:
        pts_arr = np.array(curr_pts, dtype=np.float64)
        is_closed = math.hypot(pts_arr[0, 0] - pts_arr[-1, 0], pts_arr[0, 1] - pts_arr[-1, 1]) <= tolerance
        strokes.append(ContinuousStroke(stroke_id=len(strokes), points=pts_arr, is_closed=is_closed))

    return strokes
