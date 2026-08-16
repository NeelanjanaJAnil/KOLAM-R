"""KOLAM-R Topological Validation Engine.

Provides dual topological representations:
1. Stroke Skeleton Graph Topology: Medial axis thinning -> Junction graph G = (V, E) -> graph beta_0, beta_1
2. Binary Mask Cubical Complex Persistent Homology: GUDHI 2D cubical complex -> mask beta_0, beta_1
"""

from kolam_r.topology.betti import (
    TopologicalInvariants,
    compute_graph_betti_numbers,
    compute_mask_betti_numbers,
    extract_all_topological_invariants,
)
from kolam_r.topology.cubical import (
    compute_cubical_persistence,
    compute_mask_betti_gudhi,
)
from kolam_r.topology.graph_extractor import (
    SkeletonGraph,
    extract_skeleton_graph,
)
from kolam_r.topology.skeleton import (
    skeletonize_zhang_suen,
)
from kolam_r.topology.validator import (
    TopologicalValidationReport,
    validate_reconstruction_topology,
)

__all__ = [
    "skeletonize_zhang_suen",
    "extract_skeleton_graph",
    "SkeletonGraph",
    "compute_mask_betti_gudhi",
    "compute_cubical_persistence",
    "compute_graph_betti_numbers",
    "compute_mask_betti_numbers",
    "extract_all_topological_invariants",
    "TopologicalInvariants",
    "TopologicalValidationReport",
    "validate_reconstruction_topology",
]
