"""Rasterize continuous parametric curves onto a 2D pixel grid.

Evaluates each continuous parametric curve r_k(t) at sub-pixel steps and draws
continuous Bresenham line segments onto a discrete canvas of specified dimensions.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from kolam_r.curvefit.fit_curves import ParametricCurve


def rasterize_curves(
    curves: list[ParametricCurve],
    image_shape: tuple[int, int] = (256, 256),
    stroke_width: int = 1,
    samples_per_curve: int = 64,
) -> np.ndarray:
    """Rasterize a collection of parametric curves onto a binary/grayscale pixel array.

    Args:
        curves: List of ParametricCurve objects.
        image_shape: (height, width) of the output canvas.
        stroke_width: Pixel line width for rasterized strokes.
        samples_per_curve: Number of parameter t evaluations per curve.

    Returns:
        2D uint8 numpy array of shape (height, width) with values in {0, 255}.
    """
    h, w = image_shape
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)

    for curve in curves:
        t_vals = np.linspace(curve.t_min, curve.t_max, max(2, samples_per_curve))
        x_pts, y_pts = curve.evaluate(t_vals)

        # In image coordinates, row=x_pts, col=y_pts (or vice-versa depending on mapping).
        # In graph_extractor, vertices are (row, col). PIL Draw expects (col, row) = (x_pixel, y_pixel).
        pixel_points = []
        for r, c in zip(x_pts, y_pts):
            col_px = float(c)
            row_px = float(r)
            pixel_points.append((col_px, row_px))

        if len(pixel_points) >= 2:
            draw.line(pixel_points, fill=255, width=stroke_width)
        elif len(pixel_points) == 1:
            c, r = pixel_points[0]
            draw.point((c, r), fill=255)

    return np.array(img, dtype=np.uint8)
