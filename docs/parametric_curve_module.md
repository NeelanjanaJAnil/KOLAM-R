# Design Document: Continuous Parametric Curve Reconstruction Module

**Date:** 2026-08-24  
**Module Name:** `kolam_r.curvefit` (`fit_curves.py`, `rasterize_curves.py`)  
**Status:** Step 0 Design Specification

---

## 1. What Exactly is Being Fit?

### Input Representation
The input to the curve-fitting pipeline is a topological **Skeleton Graph** $G = (V, E)$ extracted from the medial axis skeleton of a Kolam rasterization (via `kolam_r/topology/graph_extractor.py`). 

In the skeleton graph:
- **Vertices ($V$):** Centroids of junction clusters ($\text{degree} \ge 3$), terminal endpoints ($\text{degree} = 1$), and seed points on isolated simple loops.
- **Edges / Branch Paths ($E$):** Each edge $e_k = (u, v) \in E$ corresponds to an ordered sequence of 2D pixel coordinates $P_k = [p_0, p_1, \dots, p_{M-1}]$, where $p_m = (x_m, y_m) \in \mathbb{R}^2$, traced along the regular degree-2 skeleton pixels between vertex $u$ and vertex $v$.

### Mathematical Formulation
For each edge $e_k \in E$ with ordered coordinate sequence $P_k$:
1. Parameterize $P_k$ by normalized cumulative chord length:
   $$s_0 = 0, \quad s_m = s_{m-1} + \|p_m - p_{m-1}\|_2, \quad t_m = \frac{s_m}{s_{M-1}} \in [0, 1]$$
2. Fit a continuous parametric curve:
   $$\mathbf{r}_k(t) = \begin{bmatrix} x_k(t) \\ y_k(t) \end{bmatrix}, \quad t \in [0, 1]$$

### Exact Criterion for Selecting Degree-1 (Linear) vs. Degree-3 (Cubic)
To prevent overfitting on straight strokes and avoid underfitting on curved loops, the degree selection for each path $P_k$ follows a deterministic, two-step threshold rule:

1. **Point Count Constraint:**
   - If $M \le 3$ (path has 3 or fewer points), degree is strictly **$d = 1$** (linear), as cubic fitting is mathematically underdetermined or trivial.
2. **Residual Linearity Tolerance Criterion:**
   - For paths with $M > 3$, first compute a baseline linear interpolation between the endpoints $p_0$ and $p_{M-1}$:
     $$\mathbf{r}_{\text{lin}}(t_m) = (1 - t_m) p_0 + t_m p_{M-1}$$
   - Calculate the maximum Euclidean deviation (residual linearity error):
     $$\delta_{\text{lin}} = \max_{0 \le m < M} \|p_m - \mathbf{r}_{\text{lin}}(t_m)\|_2$$
   - **Decision Rule:**
     - If $\delta_{\text{lin}} \le \mathbf{0.5\text{ pixels}}$: The segment is mathematically straight within pixel grid quantization error. Select **degree $d = 1$ (linear parametric)**.
     - If $\delta_{\text{lin}} > \mathbf{0.5\text{ pixels}}$: The segment exhibits genuine geometric curvature. Select **degree $d = 3$ (cubic parametric polynomial / spline)** with exact endpoint boundary conditions ($\mathbf{r}_k(0) = p_0, \mathbf{r}_k(1) = p_{M-1}$).

---

## 2. Derivation of Equation Count per Pattern

The number of parametric curves $K$ generated for a pattern is **strictly equal to the edge count $|E|$** of the extracted skeleton graph:
$$K = |E|$$

### Formula & Derivation:
- Let $G = (V, E)$ be the simplified junction graph extracted from the skeleton.
- Each graph edge $e \in E$ represents a single continuous stroke branch between junctions or endpoints.
- Therefore, the number of recovered parametric equations is derived directly from graph topology:
  $$K = |E| = |V| + \beta_1 - \beta_0$$
  where:
  - $|V|$ is the number of junction and endpoint vertices.
  - $\beta_1$ is the 1D homology rank (first Betti number, number of independent closed loops / cycle rank).
  - $\beta_0$ is the number of connected components.

*Note:* $K$ is not a fixed constant (e.g. 72). It is dynamically determined by the empirical topology of the specific Kolam pattern and its recursion depth / symmetry configuration.

---

## 3. Research Purpose of the Module

The primary research purpose of this module is:
> **To serve as a diagnostic and analytical instrument that quantifies the topological and structural fidelity lost or altered when a discrete rasterization (or L-system string rasterization) is translated into a continuous piecewise curve-fit representation.**

By comparing the original discrete skeleton graph $G_{\text{orig}}$ against the curve-fit rasterized skeleton graph $G_{\text{curve}}$, this module allows researchers to:
1. Isolate rasterization/discretization artifacts (such as junction merging, 1-pixel bridge formation, and sub-pixel aliasing) from true grammar-recovery errors.
2. Measure whether topological invariants ($\beta_0, \beta_1$) and cycle ranks are preserved under continuous curve approximations.
3. Establish empirical tolerances on spline/curve fitting errors before topological degradation occurs.

---

## 4. Explicit Comparison & Diagnostic Role

> **Explicit Confirmation:**  
> This module is strictly a **comparative diagnostic tool** and **geometric analysis layer**.  
> It does **NOT** replace, supersede, or alter the primary L-system grammar-recovery pipeline ($\text{Image} \to \text{Grammar} \to \text{Parametric Generator} \to \text{Validation}$).  
> All grammar inference, rule classification, and inverse learning remain grounded in the deterministic L-system representation.
