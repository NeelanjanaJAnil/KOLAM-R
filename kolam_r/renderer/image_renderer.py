"""Image rendering pipeline for Kolam patterns.

Renders geometric line segments and dot grids to rasterized
grayscale images at specified resolutions.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

from kolam_r.turtle.interpreter import LineSegment, LoopMarker


# Motif rendering configurations
# M1: Basic thin stroke
# M2: Medium stroke with joint rounding
# M3: Thick stroke
# M4: Double-line stroke (parallel lines with gap)
MOTIF_CONFIGS = {
    "M1": {"line_width_64": 1, "line_width_256": 2, "joint": "none", "style": "single"},
    "M2": {"line_width_64": 1, "line_width_256": 3, "joint": "round", "style": "single"},
    "M3": {"line_width_64": 2, "line_width_256": 5, "joint": "none", "style": "single"},
    "M4": {"line_width_64": 1, "line_width_256": 2, "joint": "none", "style": "double"},
}


def _compute_transform(
    segments: list[LineSegment],
    image_size: int,
    padding: int,
    grid_size: int | None = None,
    dot_spacing: float = 1.0,
) -> tuple[float, float, float, float]:
    """Compute the affine transform parameters to map world coords to pixel coords.

    Returns:
        (scale, center_x, center_y, canvas_center) where:
        pixel_x = canvas_center + (world_x - center_x) * scale
        pixel_y = canvas_center - (world_y - center_y) * scale
    """
    canvas_center = image_size / 2.0
    if not segments:
        if grid_size is not None and grid_size > 0:
            span = max(1.0, (grid_size - 1) * dot_spacing)
            available = image_size - 2 * padding
            scale = available / span
            return scale, 0.0, 0.0, canvas_center
        return 1.0, 0.0, 0.0, canvas_center

    all_x = [s.x1 for s in segments] + [s.x2 for s in segments]
    all_y = [s.y1 for s in segments] + [s.y2 for s in segments]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    w_span = max_x - min_x
    h_span = max_y - min_y

    if w_span < 1e-9:
        w_span = 1.0
    if h_span < 1e-9:
        h_span = 1.0

    available = image_size - 2 * padding
    scale = available / max(w_span, h_span)

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    return scale, center_x, center_y, canvas_center


def _world_to_pixel(
    x: float,
    y: float,
    scale: float,
    center_x: float,
    center_y: float,
    canvas_center: float,
) -> tuple[float, float]:
    """Transform world coordinates to pixel coordinates."""
    px = canvas_center + (x - center_x) * scale
    py = canvas_center - (y - center_y) * scale  # Flip y for image coordinates
    return px, py


def render_kolam(
    segments: list[LineSegment],
    image_size: int = 64,
    padding: int = 4,
    motif: str = "M1",
    grid_size: int | None = None,
    dot_spacing: float = 1.0,
    background_color: int = 0,
    stroke_color: int = 255,
    dot_color: int = 128,
    loop_markers: list[LoopMarker] | None = None,
) -> np.ndarray:
    """Render Kolam line segments to a grayscale image.

    Pipeline:
        1. Compute world-to-pixel transform
        2. Draw line segments with motif styling
        3. Draw dot grid (if grid_size specified)
        4. Draw loop markers (if any)
        5. Return as numpy array

    Args:
        segments: Line segments from turtle interpretation.
        image_size: Output image dimension (square). Default 64.
        padding: Pixel padding around the pattern.
        motif: Motif identifier (M1–M4) for stroke styling.
        grid_size: If specified, draw an NxN dot grid.
        dot_spacing: Spacing between dots in world coordinates.
        background_color: Background pixel value (0=black).
        stroke_color: Stroke pixel value (255=white).
        dot_color: Dot pixel value (128=gray).
        loop_markers: Optional loop marker positions.

    Returns:
        Grayscale numpy array of shape (image_size, image_size), dtype uint8.
    """
    config = MOTIF_CONFIGS.get(motif, MOTIF_CONFIGS["M1"])

    # Select line width based on image size
    if image_size <= 64:
        line_width = config["line_width_64"]
        dot_radius = 1
    else:
        line_width = config["line_width_256"]
        dot_radius = 3

    # Create image
    img = Image.new("L", (image_size, image_size), background_color)
    draw = ImageDraw.Draw(img)

    if not segments and (grid_size is None or grid_size <= 0):
        return np.array(img, dtype=np.uint8)

    # Compute transform
    scale, center_x, center_y, canvas_center = _compute_transform(
        segments, image_size, padding, grid_size, dot_spacing
    )

    def to_pixel(x: float, y: float) -> tuple[float, float]:
        return _world_to_pixel(x, y, scale, center_x, center_y, canvas_center)

    # Draw dot grid first (behind strokes)
    if grid_size is not None and grid_size > 0:
        # Generate dot positions centered at origin in world space
        half = (grid_size - 1) * dot_spacing / 2.0
        for i in range(grid_size):
            for j in range(grid_size):
                dx = -half + i * dot_spacing
                dy = -half + j * dot_spacing
                px, py = to_pixel(dx, dy)
                draw.ellipse(
                    [
                        px - dot_radius,
                        py - dot_radius,
                        px + dot_radius,
                        py + dot_radius,
                    ],
                    fill=dot_color,
                )

    # Draw segments
    if config["style"] == "double":
        # Double-line: draw two parallel lines with a gap
        offset = max(1, line_width)
        for seg in segments:
            px1, py1 = to_pixel(seg.x1, seg.y1)
            px2, py2 = to_pixel(seg.x2, seg.y2)
            # Compute perpendicular offset
            dx = px2 - px1
            dy = py2 - py1
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            # Normal vector
            nx = -dy / length * offset
            ny = dx / length * offset
            draw.line(
                [(px1 + nx, py1 + ny), (px2 + nx, py2 + ny)],
                fill=stroke_color,
                width=line_width,
            )
            draw.line(
                [(px1 - nx, py1 - ny), (px2 - nx, py2 - ny)],
                fill=stroke_color,
                width=line_width,
            )
    else:
        # Single line
        for seg in segments:
            px1, py1 = to_pixel(seg.x1, seg.y1)
            px2, py2 = to_pixel(seg.x2, seg.y2)
            draw.line(
                [(px1, py1), (px2, py2)],
                fill=stroke_color,
                width=line_width,
            )

        # Joint rounding for M2
        if config["joint"] == "round":
            joint_radius = max(1, line_width)
            # Draw filled circles at segment endpoints for smooth joints
            points_seen: set[tuple[int, int]] = set()
            for seg in segments:
                for x, y in [(seg.x1, seg.y1), (seg.x2, seg.y2)]:
                    px, py = to_pixel(x, y)
                    key = (int(round(px)), int(round(py)))
                    if key not in points_seen:
                        points_seen.add(key)
                        draw.ellipse(
                            [
                                px - joint_radius,
                                py - joint_radius,
                                px + joint_radius,
                                py + joint_radius,
                            ],
                            fill=stroke_color,
                        )

    # Draw loop markers
    if loop_markers:
        loop_radius = max(2, int(dot_radius * 1.5))
        for marker in loop_markers:
            px, py = to_pixel(marker.x, marker.y)
            draw.ellipse(
                [
                    px - loop_radius,
                    py - loop_radius,
                    px + loop_radius,
                    py + loop_radius,
                ],
                outline=stroke_color,
                width=1,
            )

    return np.array(img, dtype=np.uint8)
