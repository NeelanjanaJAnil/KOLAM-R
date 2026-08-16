"""Skeleton graph extraction for Kolam topology analysis.

Converts 1-pixel medial skeletons into topological junction graphs G = (V, E)
where vertices are endpoints, junctions, and loop seeds, and edges are
stroke branch paths.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SkeletonGraph:
    """Topological graph extracted from a 1-pixel medial axis skeleton."""

    vertices: list[tuple[float, float]]  # (row, col) coordinates of graph vertices
    edges: list[tuple[int, int]]         # (v_from_idx, v_to_idx)
    degrees: dict[int, int] = field(default_factory=dict)
    connected_components: int = 0
    cycle_rank_beta_1: int = 0

    def compute_betti_numbers(self) -> tuple[int, int]:
        """Compute (beta_0, beta_1) of this graph."""
        v_count = len(self.vertices)
        e_count = len(self.edges)
        b0 = self.connected_components
        # 1D Homology rank: beta_1 = |E| - |V| + beta_0
        b1 = max(0, e_count - v_count + b0)
        self.cycle_rank_beta_1 = b1
        return b0, b1


def _get_8_neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    """Return valid 8-neighborhood coordinate offsets."""
    nbrs = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                nbrs.append((nr, nc))
    return nbrs


def extract_skeleton_graph(skeleton: np.ndarray) -> SkeletonGraph:
    """Extract topological stroke graph G = (V, E) from a 1-pixel binary skeleton.

    Args:
        skeleton: 2D binary array of uint8 values in {0, 1}.

    Returns:
        SkeletonGraph with vertices, edges, and Betti numbers.
    """
    skel = (skeleton > 0).astype(np.uint8)
    h, w = skel.shape

    pixels = [tuple(p) for p in np.argwhere(skel > 0)]
    if not pixels:
        return SkeletonGraph(vertices=[], edges=[], degrees={}, connected_components=0, cycle_rank_beta_1=0)

    pixel_set = set(pixels)

    # 1. Identify pixel neighbor counts
    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in pixels:
        nbrs = [p for p in _get_8_neighbors(r, c, h, w) if p in pixel_set]
        neighbor_map[(r, c)] = nbrs

    # 2. Find Connected Components of the skeleton pixel set
    visited_pixels: set[tuple[int, int]] = set()
    components: list[set[tuple[int, int]]] = []

    for p in pixels:
        if p not in visited_pixels:
            comp = set()
            queue = deque([p])
            visited_pixels.add(p)
            while queue:
                curr = queue.popleft()
                comp.add(curr)
                for nbr in neighbor_map[curr]:
                    if nbr not in visited_pixels:
                        visited_pixels.add(nbr)
                        queue.append(nbr)
            components.append(comp)

    all_vertices: list[tuple[float, float]] = []
    all_edges: list[tuple[int, int]] = []

    # 3. Process each connected component independently
    for comp in components:
        special_pixels = [p for p in comp if len(neighbor_map[p]) != 2]

        if not special_pixels:
            # Isolated closed simple cycle (all pixels have degree 2)
            seed = next(iter(comp))
            v_idx = len(all_vertices)
            all_vertices.append((float(seed[0]), float(seed[1])))
            all_edges.append((v_idx, v_idx))
            continue

        junction_pixels = {p for p in comp if len(neighbor_map[p]) >= 3}
        endpoints = {p for p in comp if len(neighbor_map[p]) == 1}

        # Cluster adjacent junction pixels (deg >= 3)
        junction_clusters: list[set[tuple[int, int]]] = []
        visited_junc: set[tuple[int, int]] = set()
        for j in junction_pixels:
            if j not in visited_junc:
                cluster = set()
                q = deque([j])
                visited_junc.add(j)
                while q:
                    curr = q.popleft()
                    cluster.add(curr)
                    for nbr in neighbor_map[curr]:
                        if nbr in junction_pixels and nbr not in visited_junc:
                            visited_junc.add(nbr)
                            q.append(nbr)
                junction_clusters.append(cluster)

        # Map each special pixel to its vertex index
        pixel_to_v_idx: dict[tuple[int, int], int] = {}

        # Add junction cluster centroids as vertices
        for cluster in junction_clusters:
            v_idx = len(all_vertices)
            cr = float(np.mean([p[0] for p in cluster]))
            cc = float(np.mean([p[1] for p in cluster]))
            all_vertices.append((cr, cc))
            for p in cluster:
                pixel_to_v_idx[p] = v_idx

        # Add endpoints as vertices
        for ep in endpoints:
            v_idx = len(all_vertices)
            all_vertices.append((float(ep[0]), float(ep[1])))
            pixel_to_v_idx[ep] = v_idx

        # 4. Trace branch paths of regular degree-2 pixels
        visited_branch_steps: set[tuple[tuple[int, int], tuple[int, int]]] = set()

        # Iterate over all starting ports from special pixels
        for start_p, start_v in list(pixel_to_v_idx.items()):
            for first_step in neighbor_map[start_p]:
                step_key = (start_p, first_step)
                rev_step_key = (first_step, start_p)
                if step_key in visited_branch_steps or rev_step_key in visited_branch_steps:
                    continue

                if first_step in pixel_to_v_idx:
                    # Direct step between two vertex pixels
                    end_v = pixel_to_v_idx[first_step]
                    if start_v != end_v:
                        all_edges.append((start_v, end_v))
                    visited_branch_steps.add(step_key)
                    visited_branch_steps.add(rev_step_key)
                    continue

                # Trace chain of degree-2 path pixels
                prev = start_p
                curr = first_step
                path_len = 1
                visited_branch_steps.add(step_key)
                visited_branch_steps.add(rev_step_key)

                while curr not in pixel_to_v_idx:
                    nbrs = [n for n in neighbor_map[curr] if n != prev]
                    if not nbrs:
                        break
                    next_p = nbrs[0]
                    visited_branch_steps.add((curr, next_p))
                    visited_branch_steps.add((next_p, curr))
                    prev = curr
                    curr = next_p
                    path_len += 1

                if curr in pixel_to_v_idx:
                    end_v = pixel_to_v_idx[curr]
                    if start_v != end_v or path_len > 3:
                        all_edges.append((start_v, end_v))

    # Compute node degrees
    degrees: dict[int, int] = {i: 0 for i in range(len(all_vertices))}
    for u, v in all_edges:
        degrees[u] = degrees.get(u, 0) + (2 if u == v else 1)
        if u != v:
            degrees[v] = degrees.get(v, 0) + 1

    graph = SkeletonGraph(
        vertices=all_vertices,
        edges=all_edges,
        degrees=degrees,
        connected_components=len(components),
    )
    graph.compute_betti_numbers()
    return graph
