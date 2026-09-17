"""Fit continuous parametric curves to skeleton graph branch paths.

Adheres strictly to the Step 0 design document and Step 1 decisions:
- Fits one parametric curve r_k(t) = [x_k(t), y_k(t)]^T, t in [0, 1] per skeleton graph edge (K = |E|).
- Uses cumulative chord-length parameterization.
- Selects degree d=1 (linear) if M <= 3 or maximum linear deviation <= 0.5 pixels.
- For open branch segments with curvature (M > 3, delta_lin > 0.5): fits a cubic polynomial
  with exact endpoint boundary conditions: r_k(0) = p_0, r_k(1) = p_{M-1}.
- For isolated closed simple loops (start == end, |V|=1, |E|=1): fits a periodic cubic B-spline,
  guaranteeing exact C^2 loop continuity and preserving K = |E| without inserting artificial vertices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import splprep, splev

from kolam_r.topology.graph_extractor import SkeletonGraph


@dataclass
class ParametricCurve:
    """A continuous parametric curve r(t) = [x(t), y(t)]^T for t in [0, 1]."""

    edge_index: int
    degree: int  # 1 (linear), 3 (cubic polynomial), or -3 (periodic cubic B-spline)
    coefficients_x: list[float] = field(default_factory=list)  # [c0, c1, c2, c3]
    coefficients_y: list[float] = field(default_factory=list)  # [c0, c1, c2, c3]
    spline_tck: tuple | None = None  # (t, c, k) representation for periodic splines
    t_min: float = 0.0
    t_max: float = 1.0
    is_periodic: bool = False
    num_sampled_points: int = 0
    max_fitting_error: float = 0.0
    mean_fitting_error: float = 0.0

    def evaluate(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate (x(t), y(t)) for parameter t in [0, 1]."""
        t_arr = np.asarray(t, dtype=float)

        if self.is_periodic and self.spline_tck is not None:
            # Evaluate periodic B-spline via scipy splev
            # Notice splev returns [r, c] matching splprep inputs
            r_eval, c_eval = splev(t_arr, self.spline_tck)
            return np.asarray(r_eval, dtype=float), np.asarray(c_eval, dtype=float)

        # Polynomial evaluation: sum(c_i * t^i)
        x_val = np.zeros_like(t_arr)
        y_val = np.zeros_like(t_arr)

        for p, c in enumerate(self.coefficients_x):
            x_val += c * (t_arr ** p)
        for p, c in enumerate(self.coefficients_y):
            y_val += c * (t_arr ** p)

        return x_val, y_val


def fit_single_path(
    path: list[tuple[float, float]],
    edge_index: int = 0,
    linearity_threshold: float = 0.5,
    is_closed_loop: bool = False,
) -> ParametricCurve:
    """Fit a single parametric curve to an ordered sequence of 2D coordinates.

    Args:
        path: List of (row, col) coordinates along the branch.
        edge_index: Index of the edge in the graph.
        linearity_threshold: Residual error threshold (in pixels) to distinguish linear from cubic.
        is_closed_loop: Whether this path forms an isolated closed simple loop.

    Returns:
        ParametricCurve instance with analytical coefficients and computed error.
    """
    pts = np.asarray(path, dtype=float)
    M = len(pts)

    if M == 0:
        return ParametricCurve(
            edge_index=edge_index,
            degree=1,
            coefficients_x=[0.0, 0.0],
            coefficients_y=[0.0, 0.0],
            num_sampled_points=0,
            max_fitting_error=0.0,
            mean_fitting_error=0.0,
        )

    if M == 1:
        return ParametricCurve(
            edge_index=edge_index,
            degree=1,
            coefficients_x=[float(pts[0, 0]), 0.0],
            coefficients_y=[float(pts[0, 1]), 0.0],
            num_sampled_points=1,
            max_fitting_error=0.0,
            mean_fitting_error=0.0,
        )

    # Check for closed loop condition: either explicitly flagged or endpoints match
    dist_endpoints = float(np.hypot(pts[0, 0] - pts[-1, 0], pts[0, 1] - pts[-1, 1]))
    is_loop = is_closed_loop or (dist_endpoints < 1e-4 and M > 4)

    # 1. Closed Simple Loops: Periodic Cubic B-Spline
    if is_loop and M >= 5:
        # Use periodic B-spline: preserves K = |E| without inserting artificial vertices
        try:
            # splprep requires unique interior points or strictly positive increments
            # Filter consecutive duplicate coordinates if any
            delta_steps = np.hypot(pts[1:, 0] - pts[:-1, 0], pts[1:, 1] - pts[:-1, 1])
            keep_mask = np.concatenate(([True], delta_steps > 1e-6))
            pts_clean = pts[keep_mask]

            # If closed, splprep expects endpoints to match
            if np.hypot(pts_clean[0, 0] - pts_clean[-1, 0], pts_clean[0, 1] - pts_clean[-1, 1]) > 1e-4:
                pts_clean = np.vstack([pts_clean, pts_clean[0]])

            tck, u_spline = splprep([pts_clean[:, 0], pts_clean[:, 1]], s=0, k=3, per=True)
            u_eval = np.linspace(0.0, 1.0, len(pts))
            r_fit, c_fit = splev(u_eval, tck)
            errs = np.hypot(pts[:, 0] - r_fit, pts[:, 1] - c_fit)

            return ParametricCurve(
                edge_index=edge_index,
                degree=3,
                spline_tck=tck,
                is_periodic=True,
                num_sampled_points=M,
                max_fitting_error=float(np.max(errs)),
                mean_fitting_error=float(np.mean(errs)),
            )
        except Exception:
            # Fall back to polynomial if spline condition degenerates
            pass

    # 2. Cumulative chord-length parameterization for open paths
    diffs = pts[1:] - pts[:-1]
    segment_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    total_length = float(np.sum(segment_lengths))

    if total_length < 1e-9:
        t_vals = np.linspace(0.0, 1.0, M)
    else:
        cum_dist = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        t_vals = cum_dist / total_length

    # 3. Check linear baseline between endpoints
    p0 = pts[0]
    p_end = pts[-1]
    lin_interp = (1.0 - t_vals[:, None]) * p0 + t_vals[:, None] * p_end
    deviations = np.hypot(pts[:, 0] - lin_interp[:, 0], pts[:, 1] - lin_interp[:, 1])
    delta_lin = float(np.max(deviations))

    # Decision rule:
    # If M <= 3 or delta_lin <= linearity_threshold: degree 1
    if M <= 3 or delta_lin <= linearity_threshold:
        coeffs_x = [float(p0[0]), float(p_end[0] - p0[0])]
        coeffs_y = [float(p0[1]), float(p_end[1] - p0[1])]
        eval_x = coeffs_x[0] + coeffs_x[1] * t_vals
        eval_y = coeffs_y[0] + coeffs_y[1] * t_vals
        errs = np.hypot(pts[:, 0] - eval_x, pts[:, 1] - eval_y)

        return ParametricCurve(
            edge_index=edge_index,
            degree=1,
            coefficients_x=coeffs_x,
            coefficients_y=coeffs_y,
            num_sampled_points=M,
            max_fitting_error=float(np.max(errs)),
            mean_fitting_error=float(np.mean(errs)),
        )

    # 4. Degree 3: Cubic polynomial with exact endpoint boundary conditions
    # r(t) = p0 + (p_end - p0)*t + k1*(t - t^2) + k2*(t^2 - t^3)
    internal_mask = (t_vals > 0.0) & (t_vals < 1.0)
    if np.sum(internal_mask) < 2:
        coeffs_x = [float(c) for c in np.polyfit(t_vals, pts[:, 0], deg=3)[::-1]]
        coeffs_y = [float(c) for c in np.polyfit(t_vals, pts[:, 1], deg=3)[::-1]]
    else:
        t_int = t_vals[internal_mask]
        target_x = pts[internal_mask, 0] - lin_interp[internal_mask, 0]
        target_y = pts[internal_mask, 1] - lin_interp[internal_mask, 1]

        basis_1 = t_int * (1.0 - t_int)
        basis_2 = (t_int ** 2) * (1.0 - t_int)
        A = np.column_stack((basis_1, basis_2))

        (k1_x, k2_x), _, _, _ = np.linalg.lstsq(A, target_x, rcond=None)
        (k1_y, k2_y), _, _, _ = np.linalg.lstsq(A, target_y, rcond=None)

        coeffs_x = [
            float(p0[0]),
            float((p_end[0] - p0[0]) + k1_x),
            float(-k1_x + k2_x),
            float(-k2_x),
        ]
        coeffs_y = [
            float(p0[1]),
            float((p_end[1] - p0[1]) + k1_y),
            float(-k1_y + k2_y),
            float(-k2_y),
        ]

    curve = ParametricCurve(
        edge_index=edge_index,
        degree=3,
        coefficients_x=coeffs_x,
        coefficients_y=coeffs_y,
        num_sampled_points=M,
    )
    eval_x, eval_y = curve.evaluate(t_vals)
    errs = np.hypot(pts[:, 0] - eval_x, pts[:, 1] - eval_y)
    curve.max_fitting_error = float(np.max(errs))
    curve.mean_fitting_error = float(np.mean(errs))

    return curve


def fit_skeleton_graph_curves(
    graph: SkeletonGraph,
    linearity_threshold: float = 0.5,
) -> list[ParametricCurve]:
    """Fit one continuous parametric curve per edge in the skeleton graph.

    Guarantees:
        len(curves) == len(graph.edges) == |E|
    """
    curves: list[ParametricCurve] = []

    for i, edge in enumerate(graph.edges):
        u, v = edge
        is_self_loop = (u == v)

        if i < len(graph.edge_paths) and len(graph.edge_paths[i]) >= 2:
            path = graph.edge_paths[i]
        else:
            path = [graph.vertices[u], graph.vertices[v]]

        curve = fit_single_path(
            path=path,
            edge_index=i,
            linearity_threshold=linearity_threshold,
            is_closed_loop=is_self_loop,
        )
        curves.append(curve)

    return curves
