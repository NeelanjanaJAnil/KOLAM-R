"""Binary mask 2D Cubical Complex persistent homology using GUDHI.

Computes topological persistence diagrams, birth-death pairs, and
Betti numbers (mask_beta_0, mask_beta_1) from binary raster masks.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


@dataclass(frozen=True)
class CubicalPersistenceResult:
    """Persistent homology results from a 2D digital image cubical complex."""

    beta_0: int  # Number of connected foreground components
    beta_1: int  # Number of enclosed holes / cavities in foreground
    euler_characteristic: int  # chi = beta_0 - beta_1
    persistence_pairs_dim0: list[tuple[float, float]]
    persistence_pairs_dim1: list[tuple[float, float]]


def compute_mask_betti_gudhi(
    binary_mask: np.ndarray,
    threshold: float = 0.5,
) -> tuple[int, int]:
    """Compute (beta_0, beta_1) of foreground pixels via GUDHI Cubical Complex.

    Args:
        binary_mask: 2D array of shape (H, W) where foreground > 0.
        threshold: Binarization threshold.

    Returns:
        tuple of (mask_beta_0, mask_beta_1).
    """
    res = compute_cubical_persistence(binary_mask, threshold=threshold)
    return res.beta_0, res.beta_1


def compute_cubical_persistence(
    binary_mask: np.ndarray,
    threshold: float = 0.5,
) -> CubicalPersistenceResult:
    """Compute 2D Cubical Complex persistent homology on a binary mask.

    Foreground pixels (mask > 0) are assigned filtration value 0.0 (born at 0.0).
    Background pixels are assigned filtration value 1.0.
    At threshold 0.5, sublevel filtration isolates the topology of the foreground.
    """
    if not GUDHI_AVAILABLE:
        # Fallback if gudhi is not installed: simple connected component & hole counting
        from scipy.ndimage import label, binary_fill_holes
        mask = (binary_mask > threshold).astype(bool)
        lbl, num_b0 = label(mask)
        filled = binary_fill_holes(mask)
        holes = filled & (~mask)
        _, num_b1 = label(holes)
        return CubicalPersistenceResult(
            beta_0=int(num_b0),
            beta_1=int(num_b1),
            euler_characteristic=int(num_b0 - num_b1),
            persistence_pairs_dim0=[],
            persistence_pairs_dim1=[],
        )

    mask = (binary_mask > threshold).astype(np.float64)
    # Sublevel filtration: foreground=0.0, background=1.0
    filtration = np.where(mask > 0.5, 0.0, 1.0)

    cc = gudhi.CubicalComplex(
        dimensions=list(mask.shape),
        top_dimensional_cells=filtration.flatten(),
    )
    pers = cc.persistence()

    pairs_0 = []
    pairs_1 = []

    for dim, (b, d) in pers:
        if dim == 0:
            pairs_0.append((float(b), float(d)))
        elif dim == 1:
            pairs_1.append((float(b), float(d)))

    # Topological features present in the foreground mask (born at 0.0, persisting past 0.5)
    b0 = sum(1 for (b, d) in pairs_0 if b < 0.5 and d > 0.5)
    b1 = sum(1 for (b, d) in pairs_1 if b < 0.5 and d > 0.5)
    chi = b0 - b1

    return CubicalPersistenceResult(
        beta_0=b0,
        beta_1=b1,
        euler_characteristic=chi,
        persistence_pairs_dim0=pairs_0,
        persistence_pairs_dim1=pairs_1,
    )
