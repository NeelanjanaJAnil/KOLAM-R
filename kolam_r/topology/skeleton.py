"""Morphological skeletonization for Kolam binary patterns.

Implements the Zhang-Suen parallel thinning algorithm to extract
1-pixel wide medial axis skeletons while preserving 8-connectivity
and topological loops.
"""

from __future__ import annotations

import numpy as np


def skeletonize_zhang_suen(binary_mask: np.ndarray) -> np.ndarray:
    """Extract 1-pixel medial axis skeleton using Zhang-Suen thinning.

    Args:
        binary_mask: 2D array of shape (H, W) where non-zero values are foreground.

    Returns:
        2D binary array of shape (H, W) with uint8 values in {0, 1}.
    """
    img = (binary_mask > 0).astype(np.uint8)
    h, w = img.shape

    # Pad with 1 pixel of background to avoid boundary indexing checks
    padded = np.pad(img, 1, mode="constant", constant_values=0)

    changing = True
    while changing:
        changing = False

        # Step 1
        p2 = padded[:-2, 1:-1]
        p3 = padded[:-2, 2:]
        p4 = padded[1:-1, 2:]
        p5 = padded[2:, 2:]
        p6 = padded[2:, 1:-1]
        p7 = padded[2:, :-2]
        p8 = padded[1:-1, :-2]
        p9 = padded[:-2, :-2]
        p1 = padded[1:-1, 1:-1]

        # Number of non-zero neighbors B(P1)
        b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

        # Number of 0-to-1 transitions in the ordered sequence P2, P3, ..., P9, P2
        transitions = (
            ((p2 == 0) & (p3 == 1)).astype(int)
            + ((p3 == 0) & (p4 == 1)).astype(int)
            + ((p4 == 0) & (p5 == 1)).astype(int)
            + ((p5 == 0) & (p6 == 1)).astype(int)
            + ((p6 == 0) & (p7 == 1)).astype(int)
            + ((p7 == 0) & (p8 == 1)).astype(int)
            + ((p8 == 0) & (p9 == 1)).astype(int)
            + ((p9 == 0) & (p2 == 1)).astype(int)
        )

        step1_del = (
            (p1 == 1)
            & (b >= 2)
            & (b <= 6)
            & (transitions == 1)
            & (p2 * p4 * p6 == 0)
            & (p4 * p6 * p8 == 0)
        )

        if np.any(step1_del):
            padded[1:-1, 1:-1][step1_del] = 0
            changing = True

        # Step 2
        p2 = padded[:-2, 1:-1]
        p3 = padded[:-2, 2:]
        p4 = padded[1:-1, 2:]
        p5 = padded[2:, 2:]
        p6 = padded[2:, 1:-1]
        p7 = padded[2:, :-2]
        p8 = padded[1:-1, :-2]
        p9 = padded[:-2, :-2]
        p1 = padded[1:-1, 1:-1]

        b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

        transitions = (
            ((p2 == 0) & (p3 == 1)).astype(int)
            + ((p3 == 0) & (p4 == 1)).astype(int)
            + ((p4 == 0) & (p5 == 1)).astype(int)
            + ((p5 == 0) & (p6 == 1)).astype(int)
            + ((p6 == 0) & (p7 == 1)).astype(int)
            + ((p7 == 0) & (p8 == 1)).astype(int)
            + ((p8 == 0) & (p9 == 1)).astype(int)
            + ((p9 == 0) & (p2 == 1)).astype(int)
        )

        step2_del = (
            (p1 == 1)
            & (b >= 2)
            & (b <= 6)
            & (transitions == 1)
            & (p2 * p4 * p8 == 0)
            & (p2 * p6 * p8 == 0)
        )

        if np.any(step2_del):
            padded[1:-1, 1:-1][step2_del] = 0
            changing = True

    return padded[1:-1, 1:-1].astype(np.uint8)
