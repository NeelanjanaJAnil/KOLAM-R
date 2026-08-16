"""Topological invariants calculation and data structures for KOLAM-R."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from kolam_r.topology.cubical import compute_mask_betti_gudhi
from kolam_r.topology.graph_extractor import extract_skeleton_graph
from kolam_r.topology.skeleton import skeletonize_zhang_suen


@dataclass(frozen=True)
class TopologicalInvariants:
    """Comprehensive topological summary of a pattern across dual representations."""

    # 1. Stroke Skeleton Graph Topology (1D graph cycle rank)
    graph_beta_0: int  # Connected stroke components
    graph_beta_1: int  # Independent stroke cycles (Eulerian loops)
    vertex_count: int
    edge_count: int
    degree_distribution: dict[int, int]

    # 2. Binary Raster Cubical Complex Persistence (2D hole topology)
    mask_beta_0: int  # Connected foreground pixel blobs
    mask_beta_1: int  # Enclosed raster cavities / background holes
    euler_characteristic: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_graph_betti_numbers(binary_mask: np.ndarray) -> tuple[int, int]:
    """Compute (graph_beta_0, graph_beta_1) from a binary mask via skeletonization."""
    skel = skeletonize_zhang_suen(binary_mask)
    graph = extract_skeleton_graph(skel)
    return graph.compute_betti_numbers()


def compute_mask_betti_numbers(binary_mask: np.ndarray) -> tuple[int, int]:
    """Compute (mask_beta_0, mask_beta_1) from a binary mask via GUDHI cubical complex."""
    return compute_mask_betti_gudhi(binary_mask)


def extract_all_topological_invariants(binary_mask: np.ndarray) -> TopologicalInvariants:
    """Extract both Graph and Mask topological invariants from a binary pattern."""
    mask = (binary_mask > 30).astype(np.uint8)

    # 1. Stroke Skeleton Graph
    skel = skeletonize_zhang_suen(mask)
    graph = extract_skeleton_graph(skel)
    g_b0, g_b1 = graph.compute_betti_numbers()

    # Degree distribution summary
    deg_counts: dict[int, int] = {}
    for deg in graph.degrees.values():
        deg_counts[deg] = deg_counts.get(deg, 0) + 1

    # 2. GUDHI Cubical Complex
    m_b0, m_b1 = compute_mask_betti_gudhi(mask)
    chi = m_b0 - m_b1

    return TopologicalInvariants(
        graph_beta_0=g_b0,
        graph_beta_1=g_b1,
        vertex_count=len(graph.vertices),
        edge_count=len(graph.edges),
        degree_distribution=deg_counts,
        mask_beta_0=m_b0,
        mask_beta_1=m_b1,
        euler_characteristic=chi,
    )
