"""Visual comparison and gallery generator for KOLAM-R reconstructions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from kolam_r.reconstruction.pipeline import ReconstructionResult


class ReconstructionVisualizer:
    """Renders publication-grade visual side-by-side comparisons."""

    @staticmethod
    def render_comparison_row(
        target_img: np.ndarray,
        baseline_res: ReconstructionResult,
        grammar_res: ReconstructionResult,
        ax_row: list[plt.Axes],
        row_title: str = "",
    ) -> None:
        """Render a single 4-panel comparison row: Target | CNN | Grammar | Error."""
        diff = np.abs(target_img.astype(np.float32) - grammar_res.image.astype(np.float32))

        # Target Image
        ax_row[0].imshow(target_img, cmap="gray", vmin=0, vmax=255)
        ax_row[0].set_title(f"Target: {row_title}", fontsize=9, pad=4)
        ax_row[0].axis("off")

        # Stage 3 CNN Baseline
        b_psnr = f"{baseline_res.metrics.psnr:.1f}dB" if baseline_res.metrics else ""
        b_ssim = f"SSIM:{baseline_res.metrics.ssim:.2f}" if baseline_res.metrics else ""
        ax_row[1].imshow(baseline_res.image, cmap="gray", vmin=0, vmax=255)
        ax_row[1].set_title(f"CNN Recon ({b_psnr}, {b_ssim})", fontsize=8, pad=4)
        ax_row[1].axis("off")

        # Stage 4 Grammar Program Synthesis
        g_psnr = f"{grammar_res.metrics.psnr:.1f}dB" if grammar_res.metrics else ""
        g_ssim = f"SSIM:{grammar_res.metrics.ssim:.2f}" if grammar_res.metrics else ""
        ax_row[2].imshow(grammar_res.image, cmap="gray", vmin=0, vmax=255)
        ax_row[2].set_title(f"Grammar Recon ({g_psnr}, {g_ssim})", fontsize=8, pad=4, color="darkgreen")
        ax_row[2].axis("off")

        # Difference / Error Map
        im = ax_row[3].imshow(diff, cmap="inferno", vmin=0, vmax=255)
        g_iou = f"IoU:{grammar_res.metrics.iou:.2f}" if grammar_res.metrics else ""
        ax_row[3].set_title(f"|Error Map| ({g_iou})", fontsize=8, pad=4)
        ax_row[3].axis("off")

    @classmethod
    def render_gallery(
        cls,
        sample_tuples: list[tuple[np.ndarray, ReconstructionResult, ReconstructionResult, str]],
        output_path: str | Path,
        main_title: str = "KOLAM-R Forward Synthesis Reconstruction Gallery",
    ) -> None:
        """Render a multi-row visual comparison gallery."""
        num_rows = len(sample_tuples)
        if num_rows == 0:
            return

        fig, axes = plt.subplots(
            num_rows, 4, figsize=(12, 3.0 * num_rows), dpi=200, constrained_layout=True
        )

        if num_rows == 1:
            axes = np.expand_dims(axes, 0)

        for row_idx, (target, b_res, g_res, title) in enumerate(sample_tuples):
            cls.render_comparison_row(target, b_res, g_res, axes[row_idx], row_title=title)

        fig.suptitle(main_title, fontsize=13, fontweight="bold")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
