# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#
# %%
import torch
import torch.nn as nn
import transformers
from utilities import *
from networks.base import MONO3DModel
from transformers import logging
import warnings
import networks.backbones_fast_preprocessors as bp

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="kornia.feature.lightglue"
)
import kornia

logging.set_verbosity_error()
size2embeddim = {"tiny": 256, "small": 384, "base": 768, "beit-base-384": 768,"large": 1024}


class DINOv2_Intel(MONO3DModel):
    def __init__(self, size="large", intermediate_states=True):
        """
        Initializes a DPT_DepthEstimator object.

        Args:
            pretrained_backbone (bool): Whether to use a pretrained backbone. Defaults to False.
            pretrained_neck (bool): Whether to use a pretrained neck. Defaults to False.
            pretrained_head (bool): Whether to use a pretrained head. Defaults to False.
        """
        super(DINOv2_Intel, self).__init__()
        self.model = (
            transformers.DPTForDepthEstimation.from_pretrained(
                f"Intel/dpt-{size}"
            ).backbone
            if "swin" in size or "beit" in size
            else transformers.DPTForDepthEstimation.from_pretrained(
                f"Intel/dpt-{size}"
            ).dpt
        )
        self.preprocess = bp.BackbonePreprocessor(
            transformers.AutoImageProcessor.from_pretrained(
                f"Intel/dpt-{size}", do_rescale=False
            )
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.intermediate_states = intermediate_states
        if self.intermediate_states:
            self.model.config.output_hidden_states = True
        self.size = size
        if not hasattr(self.model.encoder, "has_relative_position_bias"):
            self.model.encoder.has_relative_position_bias = False

    def get_feature_dim(self):
        return size2embeddim[self.size]

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DPT_DepthEstimator.

        Args:
            source (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        h, w = source.shape[-2:]
        # CHeck if source is all nans
        source = self.preprocess(source)
        out = self.model(source)  # ["last_hidden_state"].unsqueeze(1)
        if self.intermediate_states:
            if "beit" in self.size:
                return {
                    "last_hidden_state": out["feature_maps"][-1],
                    "feature_maps": out["feature_maps"],
                }
            return out
        return out["last_hidden_state"]


class DINOv2_Facebook(MONO3DModel):
    def __init__(self, size="large", intermediate_states=True):
        """
        Initializes a DPT_DepthEstimator object.

        Args:
            pretrained_backbone (bool): Whether to use a pretrained backbone. Defaults to False.
            pretrained_neck (bool): Whether to use a pretrained neck. Defaults to False.
            pretrained_head (bool): Whether to use a pretrained head. Defaults to False.
        """
        super(DINOv2_Facebook, self).__init__()
        self.model = transformers.DPTForDepthEstimation.from_pretrained(
            f"facebook/dpt-{size}"
        ).backbone
        for p in self.model.parameters():
            p.requires_grad = False
        self.preprocess = transformers.AutoImageProcessor.from_pretrained(
            f"facebook/dpt-{size}", do_rescale=False
        )
        self.intermediate_states = intermediate_states
        if self.intermediate_states:
            self.model.config.output_hidden_states = True
        self.size = size

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DPT_DepthEstimator.

        Args:
            source (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        h, w = source.shape[-2:]
        source = self.preprocess(images=source.clamp(0, 1), return_tensors="pt")[
            "pixel_values"
        ].to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        out = self.model(source)  # ["last_hidden_state"].unsqueeze(1)
        if self.intermediate_states:
            out["last_hidden_state"] = out["hidden_states"][-1]
            return out
        return out["last_hidden_state"]


class LoFTR(nn.Module):
    def __init__(self, coarse_only=True):
        super(LoFTR, self).__init__()
        try:
            self.model = kornia.feature.LoFTR(
                pretrained="indoor", coarse_only=coarse_only
            )
            self.random_embeddings = False
        except:
            # This will get called when using the backbone directly from kornia (not the modified version).
            # It will be the case when running unit tests where kornia is installed from requirements.txt
            # In this case, we will use the default pretrained model and return random embeddigns
            self.model = kornia.feature.LoFTR(pretrained="indoor")
            self.random_embeddings = True
            print("> Using original unmodified LoFTR --> Use this only for unit tests")
        self.coarse_only = coarse_only
        self.loftr2tensor = LoFTR2Tensor()

    def forward(self, framestack):
        loftrout = self.model(
            {
                "image0": framestack[:, 0].mean(dim=1, keepdim=True),
                "image1": framestack[:, 1].mean(dim=1, keepdim=True),
            }
        )
        if self.random_embeddings:
            return torch.rand(framestack.shape[0], 2, 256, 48, 48)
        if self.coarse_only:
            return torch.stack(
                [
                    embedding2chw(
                        loftrout[0],
                        aspect_ratio=framestack[:, 0].shape[-2]
                        / framestack[:, 0].shape[-1],
                    ),
                    embedding2chw(
                        loftrout[1],
                        aspect_ratio=framestack[:, 0].shape[-2]
                        / framestack[:, 0].shape[-1],
                    ),
                ],
                dim=1,
            )
        else:
            return self.loftr2tensor(loftrout)


class LoFTR_DINO(nn.Module):
    def __init__(self, coarse_only=True):
        super(LoFTR_DINO, self).__init__()
        try:
            self.model = kornia.feature.LoFTR_DINO(
                pretrained="indoor", coarse_only=coarse_only
            )
            self.random_embeddings = False
        except:
            # This will get called when using the backbone directly from kornia (not the modified version).
            # It will be the case when running unit tests where kornia is installed from requirements.txt
            # In this case, we will use the default pretrained model and return random embeddigns
            self.model = kornia.feature.LoFTR(pretrained="indoor")
            self.random_embeddings = True
            print("> Using original unmodified LoFTR --> Use this only for unit tests")
        self.coarse_only = coarse_only
        self.loftr2tensor = LoFTR2Tensor()

    def forward(self, framestack):
        loftrout = self.model(
            {
                "image0": framestack[:, 0].mean(dim=1, keepdim=True),
                "image1": framestack[:, 1].mean(dim=1, keepdim=True),
            }
        )
        if self.random_embeddings:
            return torch.rand(framestack.shape[0], 2, 256, 48, 48)
        if self.coarse_only:
            return torch.stack(
                [
                    embedding2chw(
                        loftrout[0],
                        aspect_ratio=framestack[:, 0].shape[-2]
                        / framestack[:, 0].shape[-1],
                    ),
                    embedding2chw(
                        loftrout[1],
                        aspect_ratio=framestack[:, 0].shape[-2]
                        / framestack[:, 0].shape[-1],
                    ),
                ],
                dim=1,
            )
        else:
            return self.loftr2tensor(loftrout)


def embeddings2fmaps(x):
    featuremap_dim = int((x.shape[1] - 1) ** 0.5)
    featuremap_depth = int(x.shape[-1])
    return x.permute(0, 2, 1)[..., 1:].reshape(
        (-1, featuremap_depth, featuremap_dim, featuremap_dim)
    )


import torch
import torch.nn as nn


class LoFTR2Tensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loftr_output):
        """
        Transform LoFTR output dict into batched tensors with padding and mask.

        Args:
            loftr_output (dict): Dictionary containing:
                - keypoints0: (N, 2) tensor of source keypoints
                - keypoints1: (N, 2) tensor of target keypoints
                - confidence: (N,) tensor of confidence scores
                - batch_indexes: (N,) tensor of batch indices

        Returns:
            tuple:
                - source_points: (B, N, 2) tensor of source keypoints
                - target_points: (B, N, 2) tensor of target keypoints
                - scores: (B, N) tensor of confidence scores
                - valid_mask: (B, N) boolean tensor indicating valid (non-padded) points
        """
        # Get number of batches and points per batch
        batch_size = loftr_output["batch_indexes"].max().item() + 1
        points_per_batch = torch.bincount(
            loftr_output["batch_indexes"], minlength=batch_size
        )
        max_points = points_per_batch.max().item()
        device = loftr_output["keypoints0"].device

        # Initialize output tensors
        source_points = torch.zeros((batch_size, max_points, 2), device=device)
        target_points = torch.zeros((batch_size, max_points, 2), device=device)
        scores = torch.zeros((batch_size, max_points), device=device)
        valid_mask = torch.zeros(
            (batch_size, max_points), dtype=torch.bool, device=device
        )

        # Current position in the flat tensors for each batch
        start_idx = 0

        # Fill tensors batch by batch
        for batch_idx in range(batch_size):
            num_points = points_per_batch[batch_idx].item()
            if num_points == 0:
                continue

            # Get slice of points for this batch
            end_idx = start_idx + num_points
            batch_slice = slice(start_idx, end_idx)

            # Fill tensors
            source_points[batch_idx, :num_points] = loftr_output["keypoints0"][
                batch_slice
            ]
            target_points[batch_idx, :num_points] = loftr_output["keypoints1"][
                batch_slice
            ]
            scores[batch_idx, :num_points] = loftr_output["confidence"][batch_slice]
            valid_mask[batch_idx, :num_points] = True

            start_idx = end_idx

        return source_points, target_points, scores  # , valid_mask
