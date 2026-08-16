"""Equation-to-image reconstruction and geometric verification module.

Evaluates recovered analytical equations x_k(t), y_k(t) at discrete parameter intervals,
reconstructs the continuous Kolam stroke coordinates, and renders the raster image.
"""

from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw

from kolam_r.schema import GeometricRepresentation, BoundingBox
from kolam_r.renderer.image_renderer import _compute_transform, _world_to_pixel
from kolam_r.turtle.interpreter import LineSegment


def evaluate_equation_segments(
    geom_rep: GeometricRepresentation,
    samples_per_equation: int = 5,
) -> list[LineSegment]:
    """Sample geometric line segments directly from recovered analytical equations."""
    reconstructed_segments: list[LineSegment] = []

    for eq in geom_rep.equations:
        t_vals = np.linspace(eq.t_min, eq.t_max, samples_per_equation)
        x_vals = np.zeros_like(t_vals)
        for p, c in enumerate(eq.coefficients_x):
            x_vals += c * (t_vals ** p)

        y_vals = np.zeros_like(t_vals)
        for p, c in enumerate(eq.coefficients_y):
            y_vals += c * (t_vals ** p)

        for i in range(len(t_vals) - 1):
            reconstructed_segments.append(
                LineSegment(
                    x1=float(x_vals[i]),
                    y1=float(y_vals[i]),
                    x2=float(x_vals[i + 1]),
                    y2=float(y_vals[i + 1]),
                )
            )

    return reconstructed_segments


def render_equation_kolam(
    geom_rep: GeometricRepresentation,
    image_size: int = 256,
    padding: int = 16,
    line_width: int = 2,
    grid_size: int | None = None,
    dot_spacing: float = 1.0,
    background_color: int = 0,
    stroke_color: int = 255,
    dot_color: int = 128,
) -> np.ndarray:
    """Render reconstructed Kolam image directly by evaluating its mathematical equations."""
    segments = evaluate_equation_segments(geom_rep, samples_per_equation=2)
    img = Image.new("L", (image_size, image_size), background_color)
    draw = ImageDraw.Draw(img)

    scale, center_x, center_y, canvas_center = _compute_transform(
        segments=segments,
        image_size=image_size,
        padding=padding,
        grid_size=grid_size,
        dot_spacing=dot_spacing,
    )

    # Render dot grid if requested
    if grid_size is not None and grid_size > 0:
        dot_radius = max(1.0, 2.0 * scale * 0.08)
        offset = (grid_size - 1) * dot_spacing / 2.0
        for r in range(grid_size):
            for c in range(grid_size):
                wx = c * dot_spacing - offset
                wy = r * dot_spacing - offset
                px, py = _world_to_pixel(wx, wy, scale, center_x, center_y, canvas_center)
                draw.ellipse(
                    [px - dot_radius, py - dot_radius, px + dot_radius, py + dot_radius],
                    fill=dot_color,
                )

    # Draw mathematical equation stroke segments
    for seg in segments:
        p1 = _world_to_pixel(seg.x1, seg.y1, scale, center_x, center_y, canvas_center)
        p2 = _world_to_pixel(seg.x2, seg.y2, scale, center_x, center_y, canvas_center)
        draw.line([p1, p2], fill=stroke_color, width=line_width)

    return np.array(img, dtype=np.uint8)
