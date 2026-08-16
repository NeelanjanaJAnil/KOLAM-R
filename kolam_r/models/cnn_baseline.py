"""Multi-task CNN Baseline for direct Kolam parameter prediction.

Backbone: 4-block convolutional network with residual connections,
batch normalization, and global average pooling.
Heads: 5 categorical classification heads + 1 continuous angle regression head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kolam_r.schema import VALID_MOTIFS, VALID_RULES, VALID_SYMMETRIES


class ConvBlock(nn.Module):
    """Standard convolutional block: Conv -> BN -> LeakyReLU -> Conv -> BN -> LeakyReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        # Shortcut for residual connection if channel dims change
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + res)
        out = self.pool(out)
        return out


class KolamCNNBaseline(nn.Module):
    """Multi-task convolutional neural network for direct Kolam parameter prediction."""

    def __init__(
        self,
        num_rules: int = len(VALID_RULES),           # 6
        num_symmetries: int = len(VALID_SYMMETRIES),  # 6
        num_motifs: int = len(VALID_MOTIFS),          # 4
        num_depths: int = 4,                          # depths 1, 2, 3, 4
        num_grid_sizes: int = 4,                      # grid sizes 3, 5, 7, 9
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        # Feature Backbone: (B, 1, 64, 64) -> (B, 256, 4, 4)
        self.layer1 = ConvBlock(1, 32)    # -> (B, 32, 32, 32)
        self.layer2 = ConvBlock(32, 64)   # -> (B, 64, 16, 16)
        self.layer3 = ConvBlock(64, 128)  # -> (B, 128, 8, 8)
        self.layer4 = ConvBlock(128, hidden_dim)  # -> (B, 256, 4, 4)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_shared = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Multi-Task Prediction Heads
        self.head_rule = nn.Linear(hidden_dim, num_rules)
        self.head_symmetry = nn.Linear(hidden_dim, num_symmetries)
        self.head_motif = nn.Linear(hidden_dim, num_motifs)
        self.head_depth = nn.Linear(hidden_dim, num_depths)
        self.head_grid_size = nn.Linear(hidden_dim, num_grid_sizes)
        self.head_angle = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Outputs normalized angle in [0, 1] (representing 0° to 180°)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 1, 64, 64).

        Returns:
            Dictionary containing prediction logits for categorical heads and
            normalized continuous prediction for the angle head.
        """
        feats = self.layer1(x)
        feats = self.layer2(feats)
        feats = self.layer3(feats)
        feats = self.layer4(feats)

        pooled = self.global_pool(feats)
        shared = self.fc_shared(pooled)

        logits_rule = self.head_rule(shared)
        logits_sym = self.head_symmetry(shared)
        logits_motif = self.head_motif(shared)
        logits_depth = self.head_depth(shared)
        logits_grid = self.head_grid_size(shared)
        pred_angle_norm = self.head_angle(shared).squeeze(-1)  # shape (B,)

        return {
            "logits_rule": logits_rule,
            "logits_symmetry": logits_sym,
            "logits_motif": logits_motif,
            "logits_depth": logits_depth,
            "logits_grid_size": logits_grid,
            "pred_angle_norm": pred_angle_norm,
            # Convenient aliases
            "rule_id": logits_rule,
            "symmetry": logits_sym,
            "motif": logits_motif,
            "recursion_depth": logits_depth,
            "grid_size": logits_grid,
            "angle": pred_angle_norm * 180.0,
        }


# Type alias for consistency
MultiTaskCNN = KolamCNNBaseline
