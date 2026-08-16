"""Topological validation and comparative verification for KOLAM-R."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from kolam_r.topology.betti import (
    TopologicalInvariants,
    extract_all_topological_invariants,
)


@dataclass(frozen=True)
class TopologicalValidationReport:
    """Detailed topological comparison between target and reconstructed patterns."""

    target_invariants: TopologicalInvariants
    recon_invariants: TopologicalInvariants

    # Graph Topology Differences (1D stroke cycles)
    delta_graph_beta_0: int
    delta_graph_beta_1: int
    exact_graph_topo_match: bool

    # Mask Cubical Complex Differences (2D cavities)
    delta_mask_beta_0: int
    delta_mask_beta_1: int
    exact_mask_topo_match: bool

    # Strict Dual Topological Match
    exact_dual_topo_match: bool

    def to_dict(self) -> dict:
        return asdict(self)


def validate_reconstruction_topology(
    target_mask: np.ndarray,
    recon_mask: np.ndarray,
) -> TopologicalValidationReport:
    """Compare ground truth and reconstructed patterns under dual topological invariants."""
    tgt_inv = extract_all_topological_invariants(target_mask)
    rec_inv = extract_all_topological_invariants(recon_mask)

    d_g_b0 = abs(rec_inv.graph_beta_0 - tgt_inv.graph_beta_0)
    d_g_b1 = abs(rec_inv.graph_beta_1 - tgt_inv.graph_beta_1)
    g_match = (d_g_b0 == 0) and (d_g_b1 == 0)

    d_m_b0 = abs(rec_inv.mask_beta_0 - tgt_inv.mask_beta_0)
    d_m_b1 = abs(rec_inv.mask_beta_1 - tgt_inv.mask_beta_1)
    m_match = (d_m_b0 == 0) and (d_m_b1 == 0)

    dual_match = g_match and m_match

    return TopologicalValidationReport(
        target_invariants=tgt_inv,
        recon_invariants=rec_inv,
        delta_graph_beta_0=d_g_b0,
        delta_graph_beta_1=d_g_b1,
        exact_graph_topo_match=g_match,
        delta_mask_beta_0=d_m_b0,
        delta_mask_beta_1=d_m_b1,
        exact_mask_topo_match=m_match,
        exact_dual_topo_match=dual_match,
    )
