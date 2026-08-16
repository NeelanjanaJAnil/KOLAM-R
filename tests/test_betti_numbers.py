"""Mandatory Ground-Truth Unit Tests for Stage 6 Topological Invariants.

GATING CRITERION: These tests against known mathematical ground truth shapes
must pass 100% before evaluating any Kolam reconstructions.

Test Shapes:
  (a) One connected open curve: beta_0 = 1, beta_1 = 0
  (b) One closed loop:          beta_0 = 1, beta_1 = 1
  (c) Two disconnected loops:   beta_0 = 2, beta_1 = 2
  (d) Connected figure-8 loop:  beta_0 = 1, beta_1 = 2
"""

from __future__ import annotations

import numpy as np
import pytest

from kolam_r.topology.betti import (
    compute_graph_betti_numbers,
    compute_mask_betti_numbers,
    extract_all_topological_invariants,
)
from kolam_r.topology.cubical import compute_mask_betti_gudhi
from kolam_r.topology.graph_extractor import extract_skeleton_graph
from kolam_r.topology.skeleton import skeletonize_zhang_suen
from kolam_r.topology.validator import validate_reconstruction_topology


class TestMandatoryGroundTruthTopology:
    """Gating unit tests for topological invariant calculation."""

    def test_shape_a_open_curve(self):
        """Shape (a): Single connected open curve -> beta_0 = 1, beta_1 = 0."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20, 10:50] = 255  # Straight horizontal line

        # 1. Stroke Skeleton Graph
        skel = skeletonize_zhang_suen(mask)
        graph = extract_skeleton_graph(skel)
        g_b0, g_b1 = graph.compute_betti_numbers()

        assert g_b0 == 1, f"Expected graph beta_0=1, got {g_b0}"
        assert g_b1 == 0, f"Expected graph beta_1=0, got {g_b1}"

        # 2. GUDHI Cubical Complex
        m_b0, m_b1 = compute_mask_betti_gudhi(mask)
        assert m_b0 == 1, f"Expected mask beta_0=1, got {m_b0}"
        assert m_b1 == 0, f"Expected mask beta_1=0, got {m_b1}"

    def test_shape_b_closed_loop(self):
        """Shape (b): Single closed loop -> beta_0 = 1, beta_1 = 1."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[15:45, 15:45] = 255
        mask[22:38, 22:38] = 0  # Hollow square ring

        # 1. Stroke Skeleton Graph
        skel = skeletonize_zhang_suen(mask)
        graph = extract_skeleton_graph(skel)
        g_b0, g_b1 = graph.compute_betti_numbers()

        assert g_b0 == 1, f"Expected graph beta_0=1, got {g_b0}"
        assert g_b1 == 1, f"Expected graph beta_1=1, got {g_b1}"

        # 2. GUDHI Cubical Complex
        m_b0, m_b1 = compute_mask_betti_gudhi(mask)
        assert m_b0 == 1, f"Expected mask beta_0=1, got {m_b0}"
        assert m_b1 == 1, f"Expected mask beta_1=1, got {m_b1}"

    def test_shape_c_two_disconnected_loops(self):
        """Shape (c): Two separate disconnected loops -> beta_0 = 2, beta_1 = 2."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        # Loop 1
        mask[8:24, 8:24] = 255
        mask[12:20, 12:20] = 0
        # Loop 2
        mask[36:56, 36:56] = 255
        mask[42:50, 42:50] = 0

        # 1. Stroke Skeleton Graph
        skel = skeletonize_zhang_suen(mask)
        graph = extract_skeleton_graph(skel)
        g_b0, g_b1 = graph.compute_betti_numbers()

        assert g_b0 == 2, f"Expected graph beta_0=2, got {g_b0}"
        assert g_b1 == 2, f"Expected graph beta_1=2, got {g_b1}"

        # 2. GUDHI Cubical Complex
        m_b0, m_b1 = compute_mask_betti_gudhi(mask)
        assert m_b0 == 2, f"Expected mask beta_0=2, got {m_b0}"
        assert m_b1 == 2, f"Expected mask beta_1=2, got {m_b1}"

    def test_shape_d_figure_eight_connected_two_cycles(self):
        """Shape (d): Connected figure-8 with two independent cycles -> beta_0 = 1, beta_1 = 2."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        # Top ring
        mask[10:30, 20:44] = 255
        mask[14:26, 24:40] = 0
        # Bottom ring (sharing junction at y=29..30)
        mask[29:50, 20:44] = 255
        mask[33:46, 24:40] = 0

        # 1. Stroke Skeleton Graph
        skel = skeletonize_zhang_suen(mask)
        graph = extract_skeleton_graph(skel)
        g_b0, g_b1 = graph.compute_betti_numbers()

        assert g_b0 == 1, f"Expected graph beta_0=1, got {g_b0}"
        assert g_b1 == 2, f"Expected graph beta_1=2, got {g_b1}"

        # 2. GUDHI Cubical Complex
        m_b0, m_b1 = compute_mask_betti_gudhi(mask)
        assert m_b0 == 1, f"Expected mask beta_0=1, got {m_b0}"
        assert m_b1 == 2, f"Expected mask beta_1=2, got {m_b1}"

    def test_topological_validator_identity(self):
        """Identical patterns must yield delta_beta = 0 and exact_dual_topo_match = True."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[15:45, 15:45] = 255
        mask[22:38, 22:38] = 0

        report = validate_reconstruction_topology(mask, mask)
        assert report.delta_graph_beta_0 == 0
        assert report.delta_graph_beta_1 == 0
        assert report.delta_mask_beta_0 == 0
        assert report.delta_mask_beta_1 == 0
        assert report.exact_graph_topo_match is True
        assert report.exact_mask_topo_match is True
        assert report.exact_dual_topo_match is True
