# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

import torch
import torch.nn as nn

from utilities import *
from networks.base import MONO3DModel
import networks.backbones as backbones
import networks.depth_decoding as depth_decoding
import networks.odometry_decoding as odometry_decoding
import networks.dfe as dfe
import tempfile
import warnings
import projections as proj
import copy

torch.autograd.set_detect_anomaly(True)

size2embeddim = {"small": 384, "base": 768, "large": 1024}


class MONO3D_E2E(MONO3DModel):
    def __init__(
        self,
        backbone_brand="intel",
        size="large",
        with_embeddings=False,
        loftr_coarse=False,
    ):
        super(MONO3D_E2E, self).__init__()

        # Initialize parameters
        self.backbone_brand = backbone_brand
        self.size = size
        self.with_embeddings = with_embeddings
        self.loftr_coarse = loftr_coarse

        # Initialize backbone
        self.backbone = getattr(backbones, f"DINOv2_{backbone_brand.capitalize()}")(
            size=size
        )

        # Initialize LoFTR
        self.loftr = backbones.LoFTR(coarse_only=loftr_coarse)
        for p in self.loftr.parameters():
            p.requires_grad = False

        # Initialize depth predictor
        self.depthpredictor = depth_decoding.DPT_Predictor(
            backbone_brand=backbone_brand, size=size, out_h=384, out_w=384
        )
        for n, p in self.depthpredictor.named_parameters():
            if "warp" not in n.lower():
                p.requires_grad = False

        # Initialize other components
        self.fundamentalpredictor = odometry_decoding.FUND_Predictor(
            size="dino", loftr_coarse=loftr_coarse
        )
        self.patchsize_resampler = torchvision.transforms.Resize((384 // 8, 384 // 8))
        self.fundamental2pose = proj.Fundamental2Pose()

        # Extract backbone output indices
        self.backbone_out_indices = self.backbone.model.config.to_dict()[
            (
                "out_indices"
                if "swin" in size or "beit" in size or "facebook" in backbone_brand
                else "backbone_out_indices"
            )
        ]

        # Optionally remove components
        if self.with_embeddings:
            del self.loftr, self.backbone

    def forward(self, framestack, intrinsics):
        """Forward pass for MONO3D_E2E model."""
        source = framestack[:, 0]  # Source image
        target = framestack[:, -1]  # Target image

        if not self.with_embeddings:
            with torch.no_grad():
                loftrout = self.loftr(framestack)
                source_features_dino = self._extract_dino_features(source)
                target_features_dino = self._extract_dino_features(target)

                if self.loftr_coarse:
                    patchfeatures = self._prepare_patch_features(
                        loftrout, source_features_dino, target_features_dino
                    )
                else:
                    patchfeatures = loftrout
        else:
            source_features_dino = torch.unbind(source, dim=1)
            target_features_dino = torch.unbind(target, dim=1)

        depth_output = self._compute_depth(source_features_dino)
        odom_output = self.fundamentalpredictor(patchfeatures)
        pose = self.fundamental2pose(
            odom_output["fundamental"],
            intrinsics,
            source_features_dino[-1][:, 0],
            target_features_dino[-1][:, 0],
        )
        odom_output["camera_pose"] = pose

        mono3doutput = {"inverse_depth": depth_output} | odom_output
        mono3doutput["source_embedding"] = self._extract_embeddings(
            source_features_dino
        )
        mono3doutput["target_embedding"] = self._extract_embeddings(
            target_features_dino
        )

        return mono3doutput


class ScaleRegressor(nn.Module):
    def __init__(self, embed_dim):
        super(ScaleRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.ReLU(),  # This ensures the output is non-negative.
        )

    def forward(self, cls1, cls2):
        # Extract CLS tokens: shape (B, E)
        # Concatenate along the feature dimension: shape (B, 2E)
        combined_features = torch.cat([cls1, cls2], dim=-1)
        scale = self.fc(combined_features)  # shape (B, 1)
        return scale.squeeze(-1)  # shape (B,)


class SpatialScaleRegressor(nn.Module):
    def __init__(self, embed_dim=768, conv_channels=64, num_conv_layers=3):
        """
        Args:
            embed_dim (int): The embedding dimension of the patch tokens.
            conv_channels (int): Number of channels for the convolutional layers.
            num_conv_layers (int): How many conv layers to apply.
        """
        super(SpatialScaleRegressor, self).__init__()
        # When fusing two images, we concatenate their spatial maps.
        # So the input channel dimension will be 2 * embed_dim.
        in_channels = 2 * embed_dim

        layers = []
        current_channels = in_channels
        for i in range(num_conv_layers):
            layers.append(
                nn.Conv2d(current_channels, conv_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.BatchNorm2d(conv_channels))
            layers.append(nn.ReLU(inplace=True))
            current_channels = conv_channels
        self.conv = nn.Sequential(*layers)

        # Global average pooling will produce a feature vector of length `conv_channels`.
        self.fc = nn.Sequential(
            nn.Linear(conv_channels, conv_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(conv_channels // 2, 1),
            nn.ReLU(),  # Guarantees non-negative output
        )

    def forward(self, feat1, feat2):
        """
        Args:
            feat1 (torch.Tensor): Spatial feature map from image 1 with shape (B, D, H, W).
            feat2 (torch.Tensor): Spatial feature map from image 2 with shape (B, D, H, W).
        Returns:
            scale (torch.Tensor): Predicted translation scale of shape (B, 1).
        """
        # Fuse the two feature maps along the channel dimension.
        fused = torch.cat([feat1, feat2], dim=1)  # shape: (B, 2D, H, W)
        x = self.conv(fused)  # shape: (B, conv_channels, H, W)
        # Global average pooling: compute the mean over spatial dimensions.
        x = x.mean(dim=[2, 3])  # shape: (B, conv_channels)
        scale = self.fc(x)  # shape: (B, 1)
        return scale.squeeze(1)


class MatcherModel_E2E(MONO3DModel):
    def __init__(
        self,
        backbone_brand="intel",
        size="large",
        with_embeddings=False,
        resampled_patch_size=16,
    ):
        super(MatcherModel_E2E, self).__init__()

        # Initialize parameters
        self.backbone_brand = backbone_brand
        self.size = size
        self.with_embeddings = with_embeddings
        self.resampled_patch_size = resampled_patch_size

        # Initialize backbone
        self.backbone = getattr(backbones, f"DINOv2_{backbone_brand.capitalize()}")(
            size=size
        )
        self.lastvitlayer = copy.deepcopy(self.backbone.model.encoder.layer[-1])
        for p in self.lastvitlayer.parameters():
            p.requires_grad = True

        def reinit_weights(m):
            if isinstance(m, nn.Linear):  # or whatever layer type you're targeting
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Reinitialize just the final layer
        self.lastvitlayer.apply(
            reinit_weights
        )  # assuming 'fc' is your final layer name
        # Initialize depth predictor
        self.depthpredictor = depth_decoding.DPT_Predictor(
            backbone_brand=backbone_brand, size=size, out_h=384, out_w=384
        )
        for n, p in self.depthpredictor.named_parameters():
            if "warp" not in n.lower():
                p.requires_grad = False

        # Initialize other components
        # self.fundamentalpredictor = odometry_decoding.FUND_Predictor(
        #     size="dino", loftr_coarse=loftr_coarse
        # )
        self.patchsize_resampler = lambda x: nn.functional.interpolate(
            x,
            size=(384 // self.resampled_patch_size, 384 // self.resampled_patch_size),
            mode="bilinear",
            align_corners=False,
        )
        # self.fundamental2pose = proj.Fundamental2Pose()

        # Extract backbone output indices
        self.backbone_out_indices = self.backbone.model.config.to_dict()[
            (
                "out_indices"
                if "swin" in size or "beit" in size or "facebook" in backbone_brand
                else "backbone_out_indices"
            )
        ]

        # Optionally remove components
        if self.with_embeddings:
            del self.loftr, self.backbone

    def forward(self, framestack):
        """Forward pass for MONO3D_E2E model."""
        source = framestack[:, 0]  # Source image
        target = framestack[:, -1]  # Target image

        if not self.with_embeddings:
            with torch.no_grad():
                # loftrout = self.loftr(framestack)
                source_features_dino = self._extract_dino_features(source)
                target_features_dino = self._extract_dino_features(target)

                # if self.loftr_coarse:
                #     patchfeatures = self._prepare_patch_features(
                #         loftrout, source_features_dino, target_features_dino
                #     )
                # else:
                # patchfeatures = loftrout
        else:
            source_features_dino = torch.unbind(source, dim=1)
            target_features_dino = torch.unbind(target, dim=1)

        source_features_matcher = self.lastvitlayer(source_features_dino[-1])[0]
        target_features_matcher = self.lastvitlayer(target_features_dino[-1])[0]
        mono3doutput = {
            "source_embedding": source_features_dino[-1][:, 1:, :].permute(0, 2, 1),
            "target_embedding": target_features_dino[-1][:, 1:, :].permute(0, 2, 1),
            "source_embedding_match": source_features_matcher[:, 1:, :].permute(
                0, 2, 1
            ),
            "target_embedding_match": target_features_matcher[:, 1:, :].permute(
                0, 2, 1
            ),
            "source_cls": source_features_dino[-1][:, 0, :],
            "target_cls": target_features_dino[-1][:, 0, :],
        }

        # Apply differentiable resize if resampled_patch_size is different than 16
        if self.resampled_patch_size != 16:
            for key in [
                "source_embedding",
                "target_embedding",
                "source_embedding_match",
                "target_embedding_match",
            ]:
                mono3doutput[key] = chw2embedding(
                    self.patchsize_resampler(embedding2chw(mono3doutput[key], False))
                )

        return mono3doutput

    def _extract_dino_features(self, image):
        """Extract features using the DINO backbone."""
        return self.backbone(image)[
            (
                "feature_maps"
                if "swin" in self.size or "beit" in self.size
                else "hidden_states"
            )
        ]

    def _prepare_patch_features(
        self, loftrout, source_features_dino, target_features_dino
    ):
        """Prepare patch features for pose estimation."""
        return torch.stack(
            [
                self.patchsize_resampler(
                    embedding2chw(source_features_dino[-1][:, 1:])
                ),
                self.patchsize_resampler(
                    embedding2chw(target_features_dino[-1][:, 1:])
                ),
            ],
            dim=1,
        )

    def _compute_depth(self, source_features_dino):
        """Compute depth maps from source features."""
        depthstack = self.depthpredictor(
            source_features_dino,
            self.backbone_out_indices,
        )
        return depthstack.view(-1, 1, depthstack.shape[-2], depthstack.shape[-1])

    def _extract_embeddings(self, features, loftr_shape=True):
        """Extract embeddings for a single image."""
        if loftr_shape:
            return self.patchsize_resampler(embedding2chw(features[-1][:, 1:]))
        else:
            return embedding2chw(features[-1][:, 1:])

    def depth(self, frames):
        """Inference-only method to return depth maps for one or more images or frame stacks."""

        source_features_dino = self._extract_dino_features(frames[:, 0])
        return self._compute_depth(source_features_dino)

    def embed(self, *frames, mode="chw", loftr_shape=False):
        """Inference-only method to return embeddings for one or more frames."""
        assert mode in ["chw", "seq"], "Invalid mode. Must be 'chw' or 'seq'."
        embs = [
            self._extract_embeddings(self._extract_dino_features(image), loftr_shape)
            for image in frames
        ]
        embs = [chw2embedding(emb) if mode == "seq" else emb for emb in embs]
        if len(embs) == 1:
            return embs[0]
        return embs
