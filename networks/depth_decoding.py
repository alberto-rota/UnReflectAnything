# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#


import torch
import torch.nn as nn
import transformers
from utilities import *
from networks.base import MONO3DModel
import numpy as np
import cv2
from typing import List, Dict, Tuple

size2embeddim = {"tiny": 256, "small": 384, "base": 768, "large": 1024}


class DepthWarp(nn.Module):
    def __init__(self, hidden_channels=16):
        super(DepthWarp, self).__init__()
        self.adjust_net = nn.Sequential(
            nn.Conv2d(
                3, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect"
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels, 1, kernel_size=3, padding=1, padding_mode="reflect"
            ),
        )
        nn.init.normal_(self.adjust_net[-1].weight)
        nn.init.constant_(self.adjust_net[-1].bias, 0)

    def forward(self, depth):
        depth = depth.unsqueeze(1)
        batch_size, _, H, W = depth.size()
        # Create normalized coordinate grid
        y = torch.linspace(-1, 1, steps=H, device=depth.device)
        x = torch.linspace(-1, 1, steps=W, device=depth.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch_size, -1, H, W)
        # Concatenate depth and coordinates
        input_features = torch.cat([depth, coords], dim=1)
        # Compute adjustment map
        adjustment = self.adjust_net(input_features)
        # Adjust the depth map
        adjusted_depth = depth + adjustment
        return adjusted_depth


class TrueDepthTransformation(nn.Module):
    def __init__(self):
        super(TrueDepthTransformation, self).__init__()
        # Initialize d_min and d_max as learnable parameters
        self.d_min = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.d_max = nn.Parameter(torch.tensor(0.40, dtype=torch.float32))

    def forward(self, inverse_depth):
        # Ensure d_min and d_max are positive to avoid invalid depth calculations
        d_min_clamped = torch.clamp(self.d_min, min=1e-6)
        d_max_clamped = torch.clamp(self.d_max, min=1e-6)

        # Compute intermediate values
        inv_d_min = 1.0 / d_min_clamped
        inv_d_max = 1.0 / d_max_clamped

        # Compute True Depth using differentiable operations
        inv_depth = inverse_depth * (inv_d_min - inv_d_max) + inv_d_max
        true_depth = 1.0 / inv_depth  # Reciprocal of the computed inverse depth

        return true_depth


class DPT_Predictor(MONO3DModel):
    def __init__(self, backbone_brand="intel", size="large", out_h=384, out_w=384):
        """
        Initializes a DPT_DepthEstimator object.

        Args:
            pretrained_backbone (bool): Whether to use a pretrained backbone. Defaults to False.
            pretrained_neck (bool): Whether to use a pretrained neck. Defaults to False.
            pretrained_head (bool): Whether to use a pretrained head. Defaults to False.
        """
        super(DPT_Predictor, self).__init__()
        if backbone_brand == "intel":
            backbone_brand = "Intel"
        self.model = nn.Sequential(
            transformers.DPTForDepthEstimation.from_pretrained(
                f"{backbone_brand}/dpt-{size}"
            ).neck,
            transformers.DPTForDepthEstimation.from_pretrained(
                f"{backbone_brand}/dpt-{size}"
            ).head,
        )
        self.out_h = out_h
        self.out_w = out_w
        self.EPS = 1e-6
        self.size = size
        self.resize = resizeTransform(self.out_h, self.out_w)
        self.eps = 1e-6
        # self.depth_warp = DepthWarp()
        # self.true_depth = TrueDepthTransformation()

    def forward(self, backbone_out: torch.Tensor, out_indices) -> torch.Tensor:
        """
        Forward pass of the DPT_DepthEstimator.

        Args:
            source (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        backbone_out = list(backbone_out)
        # HIDDEN_LEVELS_CONNECTED = [5, 11, 17, 24]
        hidden_states_for_dpt = [
            backbone_out[i if "beit" in self.size or "swin" in self.size else depth]
            for i, depth in enumerate(out_indices)
        ]
        inverse_depth = self.model(hidden_states_for_dpt)
        median = torch.median(inverse_depth.view(inverse_depth.shape[0], -1), dim=1)[0]
        # Reshape for broadcasting
        median = median.view(-1, 1, 1)
        rescaled_inverse_depth = inverse_depth / (median + self.eps)
        # inverse_depth = self.depth_warp(inverse_depth)
        # inverse_depth = self.true_depth(inverse_depth)
        # return rescaled_inverse_depth
        return self.resize(rescaled_inverse_depth)
