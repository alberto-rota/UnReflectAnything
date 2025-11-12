from typing import Dict, List, Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from models_utils import pixel_mask_to_patch_mask

from logger import get_logger

logger = get_logger(__name__).set_context("MODEL")


def _is_instance_or_cfg(x, cls):
    """Return 'instance' if x is an instance of cls, 'cfg' if dict, else raise."""
    if isinstance(x, cls):
        return "instance"
    if isinstance(x, dict):
        return "cfg"
    raise TypeError(f"Expected {cls.__name__} instance or dict config, got {type(x)}.")


def _build(component, cls):
    """
    Build a component given either an instance of `cls` or a config dict
    with kwargs for cls(**config_dict).
    """
    kind = _is_instance_or_cfg(component, cls)
    if kind == "instance":
        return component
    return cls(component)  # Pass dict as single argument for config-based constructors


class DINOv3(nn.Module):
    """
    Configurable DINOv3 model with flexible return optionas.
    Supports returning hidden states and feature maps in various formats.
    """

    def __init__(self, config):
        """
        Initialize DINOv3 model with configuration.

        Args:
            config: dict containing:
                - model_name: str, DINOv3 model name (default: "facebook/dinov3-vitb16-pretrain-lvd1689m")
                - image_size: int, input image size (default: 896). The input image will be preprocessed to have this size (square)
                - freeze_backbone: bool, whether to freeze DINOv3 parameters (default: True)
                - return_last_hidden_state: bool, return last hidden state (default: True)
                - return_all_hidden_states: bool, return all hidden states (default: False)
                - return_selected_layers: list[int], specific layer indices to return (default: None)
                - return_as_feature_maps: bool, reshape patch tokens to spatial format (default: False)
                - return_cls_token: bool, return CLS token (default: False)
                - return_register_tokens: bool, return register tokens (default: False)
        """
        super().__init__()

        # Configuration with defaults
        self.config = {
            "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "image_size": 896,
            "freeze_backbone": True,
            "return_last_hidden_state": True,
            "return_all_hidden_states": False,
            "return_selected_layers": None,
            "return_patch_tokens_only": True,
            "return_as_feature_maps": False,
            "return_cls_token": False,
            "return_register_tokens": False,
            **config,  # Override defaults with user config
        }

        # DINOv3 backbone
        self.dinov3 = AutoModel.from_pretrained(self.config["model_name"])
        self.processor = AutoImageProcessor.from_pretrained(self.config["model_name"])
        self.processor.size = {
            "height": self.config["image_size"],
            "width": self.config["image_size"],
        }

        # Freeze parameters if requested
        if self.config["freeze_backbone"]:
            for param in self.dinov3.parameters():
                param.requires_grad = False

        # Model properties
        self.feature_dim = self.dinov3.config.hidden_size  # 768 for ViT-B/16
        self.patch_size = self.dinov3.config.patch_size  # 16 for DINOv3
        self.dinov3.config.image_size = self.config["image_size"]

    def get_patch_spatial_dims(self, input_height, input_width):
        """Calculate spatial dimensions of patch features based on input size."""
        patch_h = input_height // self.patch_size
        patch_w = input_width // self.patch_size
        return patch_h, patch_w

    def extract_cls_token(self, hidden_states):
        """
        Extract CLS token from hidden states.

        Args:
            hidden_states: [B, N_tokens, feature_dim] - Raw hidden states from DINOv3

        Returns:
            [B, feature_dim] - CLS token features
        """
        return hidden_states[:, 0]  # CLS token is at index 0

    def extract_register_tokens(self, hidden_states):
        """
        Extract register tokens from hidden states.

        Args:
            hidden_states: [B, N_tokens, feature_dim] - Raw hidden states from DINOv3

        Returns:
            [B, 4, feature_dim] - Register token features (4 register tokens)
        """
        return hidden_states[:, 1:5]  # Register tokens are at indices 1-4

    def tokens_to_feature_maps(self, hidden_states, batch_size, patch_h, patch_w):
        """
        Convert patch tokens to spatial feature maps.

        Args:
            hidden_states: [B, N_tokens, feature_dim] - Raw hidden states from DINOv3
            batch_size: int - Batch size
            patch_h, patch_w: int - Spatial dimensions of patches

        Returns:
            [B, feature_dim, patch_h, patch_w] - Spatial feature maps
        """
        # Remove CLS token (index 0) and register tokens (indices 1-4)
        # DINOv3 has 1 CLS token + 4 register tokens + patch tokens
        patch_tokens = hidden_states[:, 5:]  # [B, N_patches, feature_dim]

        # Reshape to spatial format
        patch_tokens = patch_tokens.transpose(1, 2)  # [B, feature_dim, N_patches]
        feature_maps = patch_tokens.view(batch_size, self.feature_dim, patch_h, patch_w)

        return feature_maps

    def forward(self, rgb_image):
        """
        Forward pass with configurable return options.

        Args:
            rgb_image: [B, 3, H, W] - Input RGB images (should be preprocessed for DINOv3)
                      Can be any size as long as H and W are divisible by patch_size

        Returns:
            dict containing requested outputs based on config:
                - 'last_hidden_state': [B, N_tokens, feature_dim] or [B, feature_dim, patch_h, patch_w]
                - 'cls_token': [B, feature_dim] - CLS token features
                - 'register_tokens': [B, 4, feature_dim] - Register token features (4 tokens)
                - 'all_hidden_states': List of [B, N_tokens, feature_dim] or [B, feature_dim, patch_h, patch_w]
                - 'selected_hidden_states': List of selected layer outputs
        """
        batch_size, _, input_h, input_w = rgb_image.shape

        # Ensure input dimensions are compatible with patch size
        assert input_h % self.patch_size == 0, (
            f"Height {input_h} must be divisible by patch size {self.patch_size}"
        )
        assert input_w % self.patch_size == 0, (
            f"Width {input_w} must be divisible by patch size {self.patch_size}"
        )

        # Calculate patch spatial dimensions
        patch_h, patch_w = self.get_patch_spatial_dims(input_h, input_w)

        # Get DINOv3 outputs
        need_all_hidden_states = (
            self.config["return_all_hidden_states"]
            or self.config["return_selected_layers"] is not None
        )

        outputs = self.dinov3(rgb_image, output_hidden_states=need_all_hidden_states)

        # Prepare return dictionary
        result = {}

        # Return last hidden state
        if self.config["return_last_hidden_state"]:
            last_hidden = outputs.last_hidden_state  # [B, N_tokens, feature_dim]
            if self.config["return_as_feature_maps"]:
                last_hidden = self.tokens_to_feature_maps(
                    last_hidden, batch_size, patch_h, patch_w
                )
            result["last_hidden_state"] = last_hidden

        # Return CLS token
        if self.config["return_cls_token"]:
            cls_token = self.extract_cls_token(
                outputs.last_hidden_state
            )  # [B, feature_dim]
            result["cls_token"] = cls_token

        # Return register tokens
        if self.config["return_register_tokens"]:
            register_tokens = self.extract_register_tokens(
                outputs.last_hidden_state
            )  # [B, 4, feature_dim]
            result["register_tokens"] = register_tokens

        # Return all hidden states
        if self.config["return_all_hidden_states"]:
            all_hidden = outputs.hidden_states  # Tuple of [B, N_tokens, feature_dim]
            if self.config["return_patch_tokens_only"]:
                all_hidden = [h[:, 5:] for h in all_hidden]
            if self.config["return_as_feature_maps"]:
                all_hidden = [
                    self.tokens_to_feature_maps(h, batch_size, patch_h, patch_w)
                    for h in all_hidden
                ]
            result["all_hidden_states"] = all_hidden

        # Return selected hidden states
        if self.config["return_selected_layers"] is not None:
            selected_layers = self.config["return_selected_layers"]
            all_hidden = outputs.hidden_states
            if self.config["return_patch_tokens_only"]:
                all_hidden = [h[:, 5:] for h in all_hidden]
            selected_hidden = [all_hidden[i] for i in selected_layers]
            if self.config["return_as_feature_maps"]:
                selected_hidden = [
                    self.tokens_to_feature_maps(h, batch_size, patch_h, patch_w)
                    for h in selected_hidden
                ]
            result["selected_hidden_states"] = selected_hidden

        return result

    def preprocess_image(self, image_tensor):
        """
        Preprocess image for DINOv3 using the proper processor.

        Args:
            image_tensor: [B, 3, H, W] - Raw image tensor with values in [0, 1]

        Returns:
            [B, 3, H, W] - Preprocessed tensor ready for DINOv3
        """
        import PIL.Image

        # Vectorized conversion to [0, 255] range and permute dimensions
        # [B, 3, H, W] -> [B, H, W, 3]
        img_batch = (image_tensor * 255).byte().permute(0, 2, 3, 1)  # [B, H, W, 3]

        # Convert entire batch to numpy for PIL processing
        img_numpy = img_batch.cpu().numpy()  # [B, H, W, 3]

        # Create PIL images from the entire batch
        pil_images = [
            PIL.Image.fromarray(img_numpy[i], mode="RGB")
            for i in range(img_numpy.shape[0])
        ]

        # Process entire batch at once using the processor
        processed = self.processor(images=pil_images, return_tensors="pt")
        processed_batch = processed["pixel_values"]  # [B, 3, H, W]

        return processed_batch.to(image_tensor.device)


class DPTReassembleLayer(nn.Module):
    """
    Reassemble layer to convert transformer tokens to spatial feature maps.
    Handles projection, spatial rearrangement, and upsampling/downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: float,
        readout_type: str = "ignore",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.readout_type = readout_type

        # Readout projection if using "project" method
        if readout_type == "project":
            self.readout_project = nn.Sequential(
                nn.Linear(2 * in_channels, in_channels), nn.GELU()
            )

        # Channel projection
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Spatial resampling based on scale factor
        if scale_factor == 4.0:
            # 4x upsampling: 24x24 -> 96x96
            self.resample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=8,
                stride=4,
                padding=2,
                bias=True,
            )
        elif scale_factor == 2.0:
            # 2x upsampling: 24x24 -> 48x48
            self.resample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            )
        elif scale_factor == 1.0:
            # No resampling: 24x24 -> 24x24
            self.resample = nn.Identity()
        elif scale_factor == 0.5:
            # 0.5x downsampling: 24x24 -> 12x12
            self.resample = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            )

    def forward(
        self, hidden_state: torch.Tensor, patch_h: int, patch_w: int
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: [B, N_tokens, feature_dim] - includes CLS + register + patch tokens
            patch_h, patch_w: Spatial dimensions of patches
        Returns:
            [B, out_channels, H', W'] - Spatial feature map
        """
        batch_size = hidden_state.shape[0]

        # Reshape to spatial format
        patch_tokens = hidden_state.transpose(1, 2)  # [B, feature_dim, patch_h*patch_w]
        patch_tokens = patch_tokens.reshape(
            batch_size, self.in_channels, patch_h, patch_w
        )  # [B, feature_dim, patch_h, patch_w]

        # Project channels
        patch_tokens = self.proj(patch_tokens)  # [B, out_channels, patch_h, patch_w]

        # Resample spatial resolution
        output = self.resample(patch_tokens)  # [B, out_channels, H', W']

        return output


class DPTFeatureFusionBlock(nn.Module):
    """
    Feature fusion block based on RefineNet architecture.
    Combines features from different scales using residual connections.
    """

    def __init__(self, in_channels: int, out_channels: int = 256, use_bn: bool = False):
        super().__init__()

        self.residual_conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=not use_bn
        )
        self.residual_conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn
            ),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn
            ),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
        )

        self.relu = nn.ReLU(inplace=True)

        # Output projection
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]
        Returns:
            [B, out_channels, H, W]
        """
        # Project to output channels
        residual = self.residual_conv1(x)  # [B, out_channels, H, W]

        # Residual refinement
        out = self.residual_conv2(residual)  # [B, out_channels, H, W]
        out = self.relu(out + residual)  # [B, out_channels, H, W]

        # Final projection
        out = self.out_conv(out)  # [B, out_channels, H, W]

        return out


class DPT_Decoder(nn.Module):
    """
    DPT decoder adapted for RGB output from DINOv3 features.
    Implements multi-scale reassembly, progressive fusion, and RGB prediction head.

    Dropout:
        Controlled by config["dropout"] (float in [0,1]). Applied as 2D dropout
        after each fusion addition and between stages in the head to regularize
        training. Shapes are preserved; tensor dims are:
        - inputs hidden_states: List[4] of [B, N_tokens, C]
        - output: [B, C_out, H, W]
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        # Default configuration
        default_config = {
            "feature_dim": 768,  # DINOv3 ViT-B feature dimension
            "reassemble_out_channels": [96, 192, 384, 768],  # Neck hidden sizes
            "reassemble_factors": [4.0, 2.0, 1.0, 0.5],  # Spatial scale factors
            "fusion_hidden_size": 256,
            "readout_type": "ignore",  # 'ignore', 'add', or 'project'
            "use_bn": False,
            "dropout": 0.0,
            "output_image_size": (448, 448),  # If None, maintains input size
            "output_channels": 3,  # Set to 4 for RGBA output
        }

        self.config = {**default_config, **(config or {})}
        self.out_image_size = self.config["output_image_size"]
        # Create reassemble layers for multi-scale feature extraction
        self.reassemble_layers = nn.ModuleList(
            [
                DPTReassembleLayer(
                    in_channels=self.config["feature_dim"],
                    out_channels=out_ch,
                    scale_factor=scale,
                    readout_type=self.config["readout_type"],
                )
                for out_ch, scale in zip(
                    self.config["reassemble_out_channels"],
                    self.config["reassemble_factors"],
                )
            ]
        )

        # Create fusion blocks for progressive feature combination
        fusion_in_channels = self.config["reassemble_out_channels"]
        self.fusion_blocks = nn.ModuleList(
            [
                DPTFeatureFusionBlock(
                    in_channels=ch,
                    out_channels=self.config["fusion_hidden_size"],
                    use_bn=self.config["use_bn"],
                )
                for ch in fusion_in_channels
            ]
        )

        # Dropout used after fusion additions and inside head stages
        p = float(self.config.get("dropout", 0.0))
        self.drop2d = nn.Dropout2d(p) if p and p > 0.0 else nn.Identity()

        # RGB prediction head
        self.rgb_head = nn.Sequential(
            # First stage: 256 -> 128 channels with spatial refinement
            nn.Conv2d(self.config["fusion_hidden_size"], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
            self.drop2d,
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # 192x192 -> 384x384
            # Second stage: 128 -> 64 channels with feature refinement
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
            self.drop2d,
            # Third stage: 64 -> 32 channels
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
            self.drop2d,
            # Final RGB projection
            nn.Conv2d(32, self.config["output_channels"], kernel_size=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(
        self,
        hidden_states: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through DPT decoder.

        Args:
            hidden_states: List of 4 tensors from DINOv3 selected layers
                          Each tensor: [B, N_tokens, feature_dim] where N_tokens = 5 + (H/16) * (W/16)
            input_height: Original input image height
            input_width: Original input image width

        Returns:
            rgb_output: [B, 3, H, W] - RGB image in [0, 1] range
        """
        input_height, input_width = self.out_image_size
        # Calculate patch grid dimensions
        patch_h = input_height // 16  # DINOv3 uses patch_size=16
        patch_w = input_width // 16
        # Apply reassemble layers to create multi-scale feature maps
        reassembled_features = []
        for i, (hidden_state, reassemble) in enumerate(
            zip(hidden_states, self.reassemble_layers)
        ):
            feature_map = reassemble(hidden_state, patch_h, patch_w)
            reassembled_features.append(feature_map)

        # Expected spatial dimensions after reassembly (for 384x384 input):
        # Stage 0: [B, 96, 96, 96]   (4x upsampling from 24x24)
        # Stage 1: [B, 192, 48, 48]  (2x upsampling from 24x24)
        # Stage 2: [B, 384, 24, 24]  (no resampling)
        # Stage 3: [B, 768, 12, 12]  (0.5x downsampling from 24x24)

        # Progressive fusion from smallest to largest scale
        # Start with smallest scale (stage 3)
        fused = self.fusion_blocks[3](reassembled_features[3])  # [B, 256, 12, 12]
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # [B, 256, 24, 24]

        # Add stage 2 features
        fused = fused + self.fusion_blocks[2](
            reassembled_features[2]
        )  # [B, 256, 24, 24]
        fused = self.drop2d(fused)
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # [B, 256, 48, 48]

        # Add stage 1 features
        fused = fused + self.fusion_blocks[1](
            reassembled_features[1]
        )  # [B, 256, 48, 48]
        fused = self.drop2d(fused)
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # [B, 256, 96, 96]

        # Add stage 0 features
        fused = fused + self.fusion_blocks[0](
            reassembled_features[0]
        )  # [B, 256, 96, 96]
        fused = self.drop2d(fused)
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # [B, 256, 192, 192]

        # Apply RGB head
        rgb_output = self.rgb_head(fused)  # [B, 3, 384, 384]

        # Resize to original input size if needed
        if self.config["output_image_size"] or (
            rgb_output.shape[-2:] != (input_height, input_width)
        ):
            target_size = self.config["output_image_size"] or (
                input_height,
                input_width,
            )
            rgb_output = F.interpolate(
                rgb_output, size=target_size, mode="bilinear", align_corners=True
            )

        return rgb_output


class UnReflect_Model(nn.Module):
    """
    RGB with flexible DPT decoders.

    Inputs (forward):
      batch["rgb"] : (B,3,H,W) in [0,1]

    Returns:
      {
        "decoder_name": (B,C,H,W) for each configured decoder
        "tokens": {
           "rgb": (B,N,C),
        }
      }
    """

    def __init__(
        self,
        # 1) RGB encoder (DINOv3) — instance or config dict
        dinov3,
        # 2) Flexible decoders — dict of decoder_name -> DPT_Decoder config/instance
        decoders=None,  # Dict[str, DPT_Decoder instance or dict config]
        # Legacy support for backward compatibility (will be deprecated)
        spec_decoder=None,  # DPT_Decoder instance or dict
        diffuse_decoder=None,  # DPT_Decoder instance or dict
        highlight_decoder=None,  # DPT_Decoder instance or dict
        # Optional: if your DINO wrapper needs these hints
        patch_size: int = 16,
        **kwargs,
    ):
        super().__init__()

        # ---- RGB (DINOv3) ----
        # Accept either an instance or a DINOv3(**cfg) dict
        self.dinov3 = _build(dinov3, DINOv3)

        self.image_size = self.dinov3.config["image_size"]
        self.patch_size = patch_size
        self.embed_dim = self.dinov3.feature_dim

        # ---- Decoders (DPT_Decoder) ----
        # Handle flexible decoder configuration with legacy support
        def build_dpt(dec, decoder_name=None):
            """Build a decoder from config or instance (standard or FiLM-conditioned).

            Accepts either an instantiated `DPT_Decoder`/`FiLMConditionedDPT` or a
            config dict. When a config dict is provided, if it contains
            `use_film` (or `USE_FILM`) set to True, a `FiLMConditionedDPT` is
            created; otherwise a standard `DPT_Decoder` is created. The config
            dict is passed as a single dictionary argument to the decoder's
            constructor, augmented with the resolved `feature_dim`.

            If `from_pretrained` (or `FROM_PRETRAINED`) is set and not empty,
            the decoder weights are loaded from that path and the decoder is frozen.

            Args:
                dec: Decoder instance or config dict
                decoder_name: Optional decoder name for prefix stripping when loading weights
            """
            if isinstance(dec, (DPT_Decoder, FiLMConditionedDPT)):
                return dec
            if isinstance(dec, dict):
                # Extract pretrained path before building decoder
                pretrained_path = dec.get(
                    "from_pretrained", dec.get("FROM_PRETRAINED", "")
                )
                # Determine whether to build FiLM-conditioned or standard decoder
                use_film = bool(dec.get("use_film", dec.get("USE_FILM", False)))
                # Build config dict for the decoder class
                config = {
                    "feature_dim": self.embed_dim,
                    **dec,
                }
                # Remove the control flags from the config passed into the module
                config.pop("use_film", None)
                config.pop("USE_FILM", None)
                config.pop("from_pretrained", None)
                config.pop("FROM_PRETRAINED", None)

                # Create decoder instance
                decoder = (
                    FiLMConditionedDPT(config) if use_film else DPT_Decoder(config)
                )

                # Load pretrained weights and freeze if path is specified and not empty
                if pretrained_path and pretrained_path != "":
                    if not os.path.exists(pretrained_path):
                        raise FileNotFoundError(
                            f"Pretrained decoder weights not found at: {pretrained_path}"
                        )

                    # Load checkpoint (handle both raw state_dict and checkpoint formats)
                    checkpoint = torch.load(
                        pretrained_path, map_location="cpu", weights_only=False
                    )

                    # Extract state dict (handle both formats)
                    state_dict = checkpoint
                    if isinstance(checkpoint, dict):
                        # Try common checkpoint keys
                        state_dict = (
                            checkpoint.get("model_state_dict")
                            or checkpoint.get("state_dict")
                            or checkpoint
                        )

                    # Strip common prefixes if state dict was saved as part of larger model
                    # Handle cases like "decoders.diffuse.weight" -> "weight" or "decoder.weight" -> "weight"
                    if (
                        isinstance(state_dict, dict)
                        and decoder_name
                        and len(state_dict) > 0
                    ):
                        sample_key = next(iter(state_dict.keys()))
                        if "." in sample_key:
                            # Try to find decoder-specific prefix pattern
                            prefix_options = [
                                f"decoders.{decoder_name}.",
                                f"{decoder_name}.",
                                "decoder.",
                            ]
                            for prefix in prefix_options:
                                # Check if any keys start with this prefix
                                matching_keys = [
                                    k for k in state_dict.keys() if k.startswith(prefix)
                                ]
                                if matching_keys:
                                    # Strip decoder prefix from matching keys, keep others as-is
                                    stripped_dict = {}
                                    for k, v in state_dict.items():
                                        if k.startswith(prefix):
                                            stripped_dict[k[len(prefix) :]] = v
                                        else:
                                            stripped_dict[k] = v
                                    state_dict = stripped_dict
                                    break

                    # Load weights with strict=False to handle partial matches
                    missing_keys, unexpected_keys = decoder.load_state_dict(
                        state_dict, strict=False
                    )
                    if missing_keys:
                        import warnings

                        warnings.warn(
                            f"Some keys were missing when loading pretrained decoder from {pretrained_path}: {missing_keys[: min(5, len(missing_keys))]}..."
                        )

                    # Freeze all decoder parameters
                    for param in decoder.parameters():
                        param.requires_grad = False
                    decoder.eval()  # Set to eval mode for frozen decoder
                    logger.info(
                        f"Loaded pre-trained decoder weights from {pretrained_path}"
                    )
                return decoder
            raise TypeError(
                "Decoder must be DPT_Decoder/FiLMConditionedDPT instance or dict."
            )

        self.decoder_names = list(decoders.keys())
        self.decoders = nn.ModuleDict()
        for decoder_name, decoder_config in decoders.items():
            self.decoders[decoder_name] = build_dpt(
                decoder_config, decoder_name=decoder_name
            )

    def _rgb_tokens(self, rgb_preproc):
        """Extract DINOv3 tokens and infer (Hp, Wp) if wrapper doesn’t return them."""
        with torch.no_grad():
            out = self.dinov3(rgb_preproc)
        tokens = out.get("last_hidden_state", out.get("tokens"))
        if tokens is None:
            raise KeyError(
                "DINOv3 wrapper must return 'last_hidden_state' or 'tokens'."
            )
        Hp = self.image_size // self.patch_size
        Wp = self.image_size // self.patch_size
        return tokens, (Hp, Wp)

    def forward(self, model_input_dict):
        # 1) RGB → DINO tokens

        rgb_in = self.dinov3.preprocess_image(model_input_dict["rgb"])
        rgb_tokens = self.dinov3(rgb_in)["selected_hidden_states"]

        # 6) Decode with flexible decoder heads
        outputs = {}
        for decoder_name in self.decoder_names:
            decoder_output = self.decoders[decoder_name](rgb_tokens)
            outputs[decoder_name] = decoder_output

        # Optional: Add tokens for debugging/analysis
        # outputs.update({
        #     "rgb_tokens": rgb_tokens,
        # })

        return outputs


class RGBDistillDecomposer(UnReflect_Model):
    def __init__(self):
        super().__init__()


# ---- 1) A FiLM-enabled DPT that can be used for the diffuse/spec decoders ----
class FiLMConditionedDPT(DPT_Decoder):
    """
    Drop-in replacement for DPT_Decoder that accepts a spatial mask (B,1,H,W)
    (e.g., predicted highlight map) and applies FiLM at each fusion stage.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._mask = None
        self._mask_enabled = False

        # For each fusion stage we predict [gamma, beta] from [mask, distance]
        self.film = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(2, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, self.config["fusion_hidden_size"] * 2, 1),
                )
                for _ in range(4)
            ]
        )

    @torch.no_grad()
    def _build_mask_pyr(self, m: torch.Tensor):
        """
        m: (B,1,H,W) in [0,1]. Produces a pyramid of [mask, distance] at stage sizes.
        Stage spatial sizes after fusion are: 12, 24, 48, 96 (before last upsample).
        """
        H, W = self.out_image_size
        # Cheap distance proxy (replace with proper EDT if you have it)
        inv = 1.0 - m
        d = inv
        for _ in range(3):
            d = F.avg_pool2d(d, 3, 1, 1)
        d = 1.0 - d
        d = d / (d.amax(dim=(-1, -2), keepdim=True) + 1e-6)

        # Spatial sizes to match fused feature maps at each fusion stage:
        # stage3 -> (H/32, W/32), stage2 -> (H/16, W/16), stage1 -> (H/8, W/8), stage0 -> (H/4, W/4)
        sizes = [
            (H // 32, W // 32),  # e.g., 14x14 when H=W=448 (12x12 when H=W=384)
            (H // 16, W // 16),  # e.g., 28x28 (24x24)
            (H // 8, W // 8),  # e.g., 56x56 (48x48)
            (H // 4, W // 4),  # e.g., 112x112 (96x96)
            # (H // 2,  W // 2),   # e.g., 112x112 (96x96)
        ]

        pyr = []
        for hh, ww in sizes:
            mm = F.interpolate(m, size=(hh, ww), mode="nearest")
            dd = F.interpolate(d, size=(hh, ww), mode="bilinear", align_corners=True)
            pyr.append(torch.cat([mm, dd], dim=1))  # (B,2,hh,ww)
        return pyr

    def set_mask(self, mask: torch.Tensor | None):
        """
        mask: (B,1,H,W) in [0,1] (soft is fine). Set to None to disable FiLM.
        """
        self._mask = mask
        self._mask_enabled = mask is not None

    def _apply_film(self, x: torch.Tensor, ab_layer: nn.Module, cond: torch.Tensor):
        ab = ab_layer(cond)  # (B, 2C, H, W)
        gamma, beta = torch.chunk(ab, 2, dim=1)
        return gamma * x + beta

    def forward(self, hidden_states, return_mask: bool = True):
        # Standard reassembly part (unchanged)
        H, W = self.out_image_size
        ph, pw = H // 16, W // 16

        feats = []
        for hs, reas in zip(hidden_states, self.reassemble_layers):
            feats.append(
                reas(hs, ph, pw)
            )  # sizes: [96,96], [48,48], [24,24], [12,12] in channels [96,192,384,768]

        # Prepare mask pyramid if available
        mask_pyr = (
            self._build_mask_pyr(self._mask) if self._mask_enabled else [None] * 4
        )

        # Fusion with FiLM at every stage
        fused = self.fusion_blocks[3](feats[3])  # (B,256,12,12)
        if mask_pyr[0] is not None:
            fused = self._apply_film(fused, self.film[3], mask_pyr[0])

        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # -> 24

        fused = fused + self.fusion_blocks[2](feats[2])  # (B,256,24,24)
        if mask_pyr[1] is not None:
            fused = self._apply_film(fused, self.film[2], mask_pyr[1])
        fused = self.drop2d(fused)
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # -> 48

        fused = fused + self.fusion_blocks[1](feats[1])  # (B,256,48,48)
        if mask_pyr[2] is not None:
            fused = self._apply_film(fused, self.film[1], mask_pyr[2])
        fused = self.drop2d(fused)
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # -> 96

        fused = fused + self.fusion_blocks[0](feats[0])  # (B,256,96,96)
        if mask_pyr[3] is not None:
            fused = self._apply_film(fused, self.film[0], mask_pyr[3])
        fused = self.drop2d(fused)
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=True
        )  # -> 192

        out = self.rgb_head(fused)  # (B,C,H,W) after final resize
        if self.config["output_image_size"] or (out.shape[-2:] != (H, W)):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)
        if return_mask:
            return out, mask_pyr
        return out


# ---- 2) Wiring in RGBDistillDecomposer: run highlight first, then condition diffuse/spec ----
class UnReflect_Model_FiLMConditioned(UnReflect_Model):
    """
    Assumes decoders dict contains at least:
      - "highlight": DPT_Decoder(output_channels=1) that predicts soft mask in [0,1]
      - "diffuse":   FiLMConditionedDPT(...)        that will be conditioned by the mask
    You can add "specular" similarly if desired.
    """

    def forward(self, model_input_dict):
        x = model_input_dict["rgb"]  # (B,3,H,W) in [0,1]
        rgb_in = self.dinov3.preprocess_image(x)
        tokens_list = self.dinov3(rgb_in)[
            "selected_hidden_states"
        ]  # List[4] of [B,N_p,C]

        outputs = {}

        # 1) Predict highlight mask (soft), from highlight head
        if "highlight" not in self.decoders:
            raise KeyError("decoders must include a 'highlight' head for Option A.")
        hl_logits = self.decoders["highlight"](
            tokens_list
        )  # (B,1,H,W) in [0,1] (Sigmoid in head)
        mask_soft = hl_logits.clamp(
            0, 1
        )  # treat as soft attention; no hard threshold here
        outputs["highlight"] = mask_soft

        # 2) Condition other heads with FiLM if supported
        for name, dec in self.decoders.items():
            if name == "highlight":
                continue
            if hasattr(dec, "set_mask"):
                dec.set_mask(mask_soft)
            outputs[name] = dec(tokens_list)
            # outputs["mask_pyr"] = mask_pyr
        return outputs


# ---- 1) Tiny token-inpainter (works on patch tokens) ----
class _TinyMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hid, dim)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class _TransformerBlk(nn.Module):
    def __init__(self, dim=768, heads=12, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = _TinyMLP(dim, 4.0, drop)

    def forward(self, x, attn_bias=None):
        q = k = v = self.n1(x)
        if attn_bias is None:
            a, _ = self.attn(q, k, v, need_weights=False)
        else:
            # attn_bias: (B, N, N) added to logits; implement via attn_mask per batch
            # Fallback: no bias (PyTorch's MHA doesn't support per-batch logit bias directly)
            a, _ = self.attn(q, k, v, need_weights=False)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x


class TokenInpainter(nn.Module):
    """
    Completes masked patch tokens from context.
    Input:  T  = (B, N, C) tokens for a selected DINO layer
            pm = (B, N)    boolean mask at patch resolution (True = masked/hole)
    Output: X  = (B, N, C) refined tokens; we will take X at masked positions
    """

    def __init__(self, dim=768, depth=4, heads=16, drop=0.0, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TransformerBlk(dim, heads, drop) for _ in range(depth)]
        )
        self.out_proj = nn.Linear(dim, dim, bias=True)
        # Learnable mask token to seed missing positions (randomly initialized with truncated normal, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Learnable indicator for masked positions (helps network identify holes)
        self.mask_indicator = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_indicator, std=0.02)
        # Enable fixed 2D sinusoidal positional encodings
        self.use_positional_encoding = True

    def forward(self, T: torch.Tensor, pm_bool: torch.Tensor):
        # T: [B, N, C], pm_bool: [B, N] (True = hole)
        B, N, C = T.shape
        mask = pm_bool.unsqueeze(-1)                     # [B, N, 1]
        mask_tok = self.mask_token                      # [1, 1, C]
        # Seed masked positions with learned token, keep context as-is
        T_seed = torch.where(mask, mask_tok.expand(B, N, C), T)

        # Add positional encodings (crucial for spatial reasoning in attention)
        if self.use_positional_encoding:
            hw = int(N ** 0.5)
            if hw * hw != N:
                hw = int(round(N ** 0.5))
            pos = self._build_2d_sincos_pos_embed(hw, hw, C, T_seed.device)  # [1,N,C]
            T_seed = T_seed + pos

        # Add explicit masked-position indicator
        T_seed = T_seed + torch.where(
            mask,
            self.mask_indicator.expand(B, N, C),
            torch.zeros_like(T_seed),
        )
        X = T_seed
        for blk in self.blocks:
            X = blk(X)
        # Expected shape after projection: [B, N, C]
        return self.out_proj(X)

    @staticmethod
    def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
        """
        Create 2D sinusoidal positional embeddings of shape [1, h*w, dim].
        """
        assert dim % 2 == 0, "positional dim must be even"
        half = dim // 2
        emb_h = TokenInpainter._build_1d_sincos_embed(half, h, device)  # [h, half]
        emb_w = TokenInpainter._build_1d_sincos_embed(half, w, device)  # [w, half]
        emb_h = emb_h[:, None, :].expand(h, w, half)
        emb_w = emb_w[None, :, :].expand(h, w, half)
        pos = torch.cat([emb_h, emb_w], dim=-1).reshape(1, h * w, dim)
        return pos

    @staticmethod
    def _build_1d_sincos_embed(dim: int, length: int, device: torch.device) -> torch.Tensor:
        assert dim % 2 == 0, "1D pos dim must be even"
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)  # [L,1]
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-(torch.log(torch.tensor(10000.0, device=device))))
            / (dim // 2)
        )  # [dim/2]
        angles = positions * div_term  # [L, dim/2]
        emb = torch.empty((length, dim), device=device, dtype=torch.float32)
        emb[:, 0::2] = torch.sin(angles)
        emb[:, 1::2] = torch.cos(angles)
        return emb


class UnReflect_Model_TokenInpainter(UnReflect_Model):
    """
    Assumes decoders dict contains at least:
      - "highlight": DPT_Decoder(output_channels=1) that predicts a soft mask in [0,1]
      - "diffuse":   (regular) DPT_Decoder that expects standard DINO tokens
    Keeps DPT decoders intact; completes tokens first.
    """

    def __init__(
        self,
        dinov3,
        decoders,
        patch_size: int = 16,
        token_inpainter_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            dinov3=dinov3, decoders=decoders, patch_size=patch_size, **kwargs
        )
        dim = self.embed_dim
        self.token_inpaint = TokenInpainter(dim=dim)
        
    def extract_tokens(self, image):
        rgb_in = self.dinov3.preprocess_image(image)
        tokens_list = self.dinov3(rgb_in)[
            "selected_hidden_states"
        ] 
        return tokens_list
    
    def forward(self, model_input_dict):
        x = model_input_dict["rgb"]  # (B,3,H,W)
        rgb_in = self.dinov3.preprocess_image(x)
        tokens_list = self.dinov3(rgb_in)[
            "selected_hidden_states"
        ]  # List[4] of (B,N,C), PATCH TOKENS ONLY

        outputs = {}

        ### FIRST: Predict soft highlight mask in image space
        if "highlight" not in self.decoders:
            raise KeyError("decoders must include a 'highlight' head for Option B.")
        hl_soft = self.decoders["highlight"](tokens_list)  # (B,1,H,W) in [0,1]
        outputs["highlight"] = hl_soft

        ### SECOND: Construct patch-level mask - From the prediction or override from GT if provided
        if "inpaint_mask_override" in model_input_dict:
            patchmask_bool = pixel_mask_to_patch_mask(
                model_input_dict["inpaint_mask_override"],
                patch_size=self.patch_size,
                threshold=0.1,
                invert=False,
            )
        else:
            patchmask_bool = pixel_mask_to_patch_mask(
                outputs["highlight"],
                patch_size=self.patch_size,
                threshold=0.1,
                invert=False,
            )
            

        # patchmask_bool = 1 : MUST IMPAINT THE TOKEN
        # patchmask_bool = 0 : IS TEACHER TOKEN
        outputs["patch_mask"] = patchmask_bool

        ### THIRD: Inpaint the tokens in the mask
        completed_tokens = []
        for n, T in enumerate(tokens_list):  # (B,N,C)
            T_inpainted = self.token_inpaint(T, torch.logical_not(patchmask_bool))  # refined all tokens
            
            # ### ! REMOVE - DEBUG ONLY
            # if "diffuse_tokens" in model_input_dict:
            #     T_inpainted = model_input_dict["diffuse_tokens"][n]
            # ###                                                                    
            
            # keep teacher tokens on context; use predicted tokens on masked patches
            T_comp = torch.where(
                patchmask_bool.unsqueeze(-1), T_inpainted, T
            )  # (B,N,C)
            completed_tokens.append(T_comp)

        # outputs["tokens_teacher"] = tokens_list
        outputs["tokens_inpainted"] = T_inpainted
        outputs["tokens_completed"] = completed_tokens
        
        # 4) Decode with completed tokens (do NOT pass mask to decoder)
        for name, dec in self.decoders.items():
            if name == "highlight":
                continue
            outputs[name] = dec(completed_tokens)

        return outputs
