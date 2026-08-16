"""Mathematical curve fitting and equation recovery engine.

Fits piecewise parametric curves x(t), y(t) with explicit analytical formulas,
polynomial coefficients, and statistical goodness-of-fit (R² and fitting error).
"""

from __future__ import annotations

import math
import numpy as np

from kolam_r.turtle.interpreter import LineSegment
from kolam_r.schema import EquationTerm, GeometricRepresentation


def _format_polynomial(coeffs: list[float], var: str = "t") -> str:
    """Format polynomial coefficients [c0, c1, c2, ...] into readable analytical expression."""
    terms = []
    for deg, c in enumerate(coeffs):
        if abs(c) < 1e-5:
            continue
        c_str = f"{c:.4f}"
        if deg == 0:
            terms.append(c_str)
        elif deg == 1:
            terms.append(f"{c_str}*{var}")
        else:
            terms.append(f"{c_str}*{var}^{deg}")

    if not terms:
        return "0.0"
    return " + ".join(terms).replace("+ -", "- ")


def fit_segments_piecewise_parametric(
    segments: list[LineSegment],
) -> GeometricRepresentation:
    """Fit exact piecewise parametric linear equations x_k(t), y_k(t) for all turtle stroke segments.
    
    Each line segment from (x1, y1) to (x2, y2) has the exact closed-form equation:
        x_k(t) = x1 + (x2 - x1)*t
        y_k(t) = y1 + (y2 - y1)*t
        where t in [0, 1]
    """
    equation_terms: list[EquationTerm] = []

    for idx, seg in enumerate(segments):
        dx = seg.x2 - seg.x1
        dy = seg.y2 - seg.y1

        coeffs_x = [float(seg.x1), float(dx)]
        coeffs_y = [float(seg.y1), float(dy)]

        term = EquationTerm(
            subpath_index=idx,
            expression_x=f"x(t) = {_format_polynomial(coeffs_x)}",
            expression_y=f"y(t) = {_format_polynomial(coeffs_y)}",
            t_min=0.0,
            t_max=1.0,
            degree=1,
            coefficients_x=coeffs_x,
            coefficients_y=coeffs_y,
            r_squared_x=1.0,
            r_squared_y=1.0,
            max_error=0.0,
        )
        equation_terms.append(term)

    # Determine if curve forms closed loops
    is_closed = False
    if segments:
        first_pt = (segments[0].x1, segments[0].y1)
        last_pt = (segments[-1].x2, segments[-1].y2)
        is_closed = math.hypot(first_pt[0] - last_pt[0], first_pt[1] - last_pt[1]) < 1e-4

    return GeometricRepresentation(
        representation_type="piecewise_parametric_linear",
        num_subpaths=len(equation_terms),
        equations=equation_terms,
        mean_fitting_error=0.0,
        max_fitting_error=0.0,
        is_closed_loop=is_closed,
    )
