import importlib
import inspect
import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from logger import get_logger
from models_utils import pixel_mask_to_patch_mask

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


class DINOv3_ConvNext(nn.Module):
    """
    Configurable DINOv3 ConvNeXt model with flexible return options.
    Supports returning hidden states and feature maps in various formats.
    ConvNeXt models output spatial feature maps directly (not token sequences).
    """

    def __init__(self, config):
        """
        Initialize DINOv3 ConvNeXt model with configuration.

        Args:
            config: dict containing:
                - model_name: str, DINOv3 ConvNeXt model name (default: "facebook/dinov3-convnext-large-pretrain-lvd1689m")
                - image_size: int, input image size (default: 896). The input image will be preprocessed to have this size (square)
                - freeze_backbone: bool, whether to freeze DINOv3 parameters (default: True)
                - return_last_hidden_state: bool, return last hidden state (default: True)
                - return_all_hidden_states: bool, return all hidden states (default: False)
                - return_selected_layers: list[int], specific layer indices to return (default: None)
                - return_as_feature_maps: bool, return as spatial feature maps (default: True for ConvNeXt)
                - return_cls_token: bool, return CLS token (default: False). For ConvNeXt, uses global average pooling
                - return_register_tokens: bool, return register tokens (default: False). Not available for ConvNeXt, returns empty tensor
        """
        super().__init__()

        # Configuration with defaults
        self.config = {
            "model_name": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            "image_size": 896,
            "freeze_backbone": True,
            "return_last_hidden_state": True,
            "return_all_hidden_states": False,
            "return_selected_layers": [1,2,3,4],
            "return_patch_tokens_only": True,
            "return_as_feature_maps": True,  # Default True for ConvNeXt (native format)
            "return_cls_token": False,
            "return_register_tokens": False,
            **config,  # Override defaults with user config
        }
        # Override defaults with user config
        self.config["return_selected_layers"] = [1,2,3,4]
        # DINOv3 ConvNeXt backbone
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

        # Model properties - ConvNeXt outputs spatial feature maps
        # Get feature dimension from model config
        self.feature_dim = getattr(self.dinov3.config, "hidden_size", None) or getattr(
            self.dinov3.config, "embed_dim", None
        )
        if self.feature_dim is None:
            # Fallback: try to infer from model structure
            # ConvNeXt models typically have hidden_sizes in config
            if hasattr(self.dinov3.config, "hidden_sizes"):
                self.feature_dim = self.dinov3.config.hidden_sizes[-1]
            else:
                raise ValueError(
                    "Could not determine feature_dim from model config. Please specify manually."
                )

        # ConvNeXt uses patch size similar to ViT, typically 4 or 16
        # DINOv3 ConvNeXt uses patch_size=4 based on the architecture
        self.patch_size = getattr(self.dinov3.config, "patch_size", 4)
        self.dinov3.config.image_size = self.config["image_size"]

    def get_patch_spatial_dims(self, input_height, input_width):
        """Calculate spatial dimensions of feature maps based on input size."""
        # ConvNeXt typically downsamples by a factor related to patch_size
        # For DINOv3 ConvNeXt, the final feature map is typically H/patch_size x W/patch_size
        patch_h = input_height // self.patch_size
        patch_w = input_width // self.patch_size
        return patch_h, patch_w

    def feature_maps_to_tokens(self, feature_maps):
        """
        Convert spatial feature maps to token sequence format for API compatibility.
        Handles both [B, C, H, W] and [B, N, C] input formats.

        Args:
            feature_maps: [B, C, H, W] or [B, N, C] - Spatial feature maps or tokens from ConvNeXt

        Returns:
            [B, N, C] - Token-like sequence format
        """
        if len(feature_maps.shape) == 4:
            # Input is [B, C, H, W] - convert to tokens
            B, C, H, W = feature_maps.shape
            # Flatten spatial dimensions and transpose to [B, N, C]
            tokens = feature_maps.flatten(2).transpose(1, 2)  # [B, H*W, C]
            return tokens
        elif len(feature_maps.shape) == 3:
            # Input is already [B, N, C] - return as-is
            return feature_maps
        else:
            raise ValueError(
                f"Unexpected feature_maps shape: {feature_maps.shape}. Expected 3D [B, N, C] or 4D [B, C, H, W]"
            )

    def extract_cls_token(self, feature_maps):
        """
        Extract CLS-like token from feature maps using global average pooling.
        ConvNeXt doesn't have CLS tokens, so we use global average pooling.
        Handles both [B, C, H, W] and [B, N, C] input formats.

        Args:
            feature_maps: [B, C, H, W] or [B, N, C] - Spatial feature maps or tokens from ConvNeXt

        Returns:
            [B, C] - Global pooled features (CLS-like token)
        """
        if len(feature_maps.shape) == 4:
            # Input is [B, C, H, W] - use global average pooling
            return F.adaptive_avg_pool2d(feature_maps, (1, 1)).squeeze(-1).squeeze(-1)
        elif len(feature_maps.shape) == 3:
            # Input is [B, N, C] - use mean pooling over sequence dimension
            return feature_maps.mean(dim=1)  # [B, C]
        else:
            raise ValueError(
                f"Unexpected feature_maps shape: {feature_maps.shape}. Expected 3D [B, N, C] or 4D [B, C, H, W]"
            )

    def extract_register_tokens(self, feature_maps):
        """
        Extract register-like tokens from feature maps.
        ConvNeXt doesn't have register tokens, so we sample 4 corner locations or first 4 tokens.
        Handles both [B, C, H, W] and [B, N, C] input formats.

        Args:
            feature_maps: [B, C, H, W] or [B, N, C] - Spatial feature maps or tokens from ConvNeXt

        Returns:
            [B, 4, C] - Sampled corner features (for API compatibility)
        """
        if len(feature_maps.shape) == 4:
            # Input is [B, C, H, W] - sample 4 corner locations
            B, C, H, W = feature_maps.shape
            # Vectorized sampling of 4 corner locations: top-left, top-right, bottom-left, bottom-right
            corners = torch.stack(
                [
                    feature_maps[:, :, 0, 0],      # top-left
                    feature_maps[:, :, 0, W - 1],   # top-right
                    feature_maps[:, :, H - 1, 0],   # bottom-left
                    feature_maps[:, :, H - 1, W - 1], # bottom-right
                ],
                dim=1,
            )  # [B, 4, C]
            return corners
        elif len(feature_maps.shape) == 3:
            # Input is [B, N, C] - take first 4 tokens or pad if needed
            B, N, C = feature_maps.shape
            if N >= 4:
                return feature_maps[:, :4, :]  # [B, 4, C]
            else:
                # Pad with zeros if we have fewer than 4 tokens
                padding = torch.zeros(B, 4 - N, C, device=feature_maps.device, dtype=feature_maps.dtype)
                return torch.cat([feature_maps, padding], dim=1)  # [B, 4, C]
        else:
            raise ValueError(
                f"Unexpected feature_maps shape: {feature_maps.shape}. Expected 3D [B, N, C] or 4D [B, C, H, W]"
            )

    def forward(self, rgb_image):
        """
        Forward pass with configurable return options.

        Args:
            rgb_image: [B, 3, H, W] - Input RGB images (should be preprocessed for DINOv3)
                      Can be any size as long as H and W are divisible by patch_size

        Returns:
            dict containing requested outputs based on config:
                - 'last_hidden_state': [B, C, H', W'] (if return_as_feature_maps=True) 
                                      or [B, H'*W', C] (if return_as_feature_maps=False)
                - 'cls_token': [B, C] - Global pooled features (CLS-like token)
                - 'register_tokens': [B, 4, C] - Sampled corner features (for compatibility)
                - 'all_hidden_states': List of feature maps or token sequences
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

        # Calculate expected spatial dimensions
        patch_h, patch_w = self.get_patch_spatial_dims(input_h, input_w)

        # Get DINOv3 ConvNeXt outputs
        need_all_hidden_states = (
            self.config["return_all_hidden_states"]
            or self.config["return_selected_layers"] is not None
        )

        outputs = self.dinov3(rgb_image, output_hidden_states=need_all_hidden_states)

        # ConvNeXt outputs are spatial feature maps: [B, C, H', W']
        # The actual spatial dimensions may differ from patch_h/patch_w due to downsampling
        # We'll use the actual output dimensions

        # Prepare return dictionary
        result = {}

        # Get last hidden state - could be [B, C, H', W'] or [B, N, C] depending on model
        if not hasattr(outputs, "last_hidden_state") or outputs.last_hidden_state is None:
            raise ValueError(
                "Model did not return last_hidden_state. Check model output structure."
            )
        last_hidden = outputs.last_hidden_state
        
        # Detect the format of the output
        is_spatial = len(last_hidden.shape) == 4  # [B, C, H, W]
        is_tokens = len(last_hidden.shape) == 3   # [B, N, C]

        # Return last hidden state
        if self.config["return_last_hidden_state"]:
            if self.config["return_as_feature_maps"]:
                if is_spatial:
                    result["last_hidden_state"] = last_hidden  # [B, C, H', W']
                elif is_tokens:
                    # Convert tokens to spatial format if needed
                    # This requires knowing the spatial dimensions, which we can infer
                    B, N, C = last_hidden.shape
                    # Try to infer spatial dimensions from input size
                    # For ConvNeXt, typically N = (H/patch_size) * (W/patch_size)
                    # We'll reshape assuming square patches
                    spatial_size = int(N ** 0.5)
                    if spatial_size * spatial_size == N:
                        # Perfect square - reshape to spatial
                        result["last_hidden_state"] = last_hidden.transpose(1, 2).view(B, C, spatial_size, spatial_size)
                    else:
                        # Not a perfect square, keep as tokens but warn
                        result["last_hidden_state"] = last_hidden
                else:
                    result["last_hidden_state"] = last_hidden
            else:
                # Convert to token-like format for compatibility
                result["last_hidden_state"] = self.feature_maps_to_tokens(
                    last_hidden
                )  # [B, N, C]

        # Return CLS token (global average pooling)
        if self.config["return_cls_token"]:
            cls_token = self.extract_cls_token(last_hidden)  # [B, C]
            result["cls_token"] = cls_token

        # Return register tokens (corner sampling)
        if self.config["return_register_tokens"]:
            register_tokens = self.extract_register_tokens(
                last_hidden
            )  # [B, 4, C]
            result["register_tokens"] = register_tokens

        # Return all hidden states
        if self.config["return_all_hidden_states"]:
            all_hidden = outputs.hidden_states  # Tuple of [B, C, H', W'] or [B, N, C]
            if all_hidden is None:
                raise ValueError(
                    "Model did not return hidden_states. Make sure output_hidden_states=True "
                    "is passed to the model forward call."
                )
            # Convert tuple to list if needed
            if isinstance(all_hidden, tuple):
                all_hidden = list(all_hidden)
            if self.config["return_as_feature_maps"]:
                # Convert tokens to spatial maps if needed
                converted_hidden = []
                for h in all_hidden:
                    if len(h.shape) == 3:
                        # It's tokens [B, N, C], convert to spatial
                        B, N, C = h.shape
                        # Infer spatial dimensions assuming square patches
                        spatial_size = int(N ** 0.5)
                        if spatial_size * spatial_size == N:
                            # Perfect square - reshape to spatial
                            h_spatial = h.transpose(1, 2).view(B, C, spatial_size, spatial_size)
                            converted_hidden.append(h_spatial)
                        else:
                            # Not a perfect square, use patch dimensions
                            h_spatial = h.transpose(1, 2).view(B, C, patch_h, patch_w)
                            converted_hidden.append(h_spatial)
                    else:
                        # Already spatial [B, C, H, W]
                        converted_hidden.append(h)
                result["all_hidden_states"] = converted_hidden
            else:
                # Convert each to token-like format (or keep as tokens)
                result["all_hidden_states"] = [
                    self.feature_maps_to_tokens(h) for h in all_hidden
                ]

        # Return selected hidden states
        if self.config["return_selected_layers"] is not None:
            selected_layers = self.config["return_selected_layers"]
            if not hasattr(outputs, "hidden_states"):
                raise AttributeError(
                    f"Model outputs do not have 'hidden_states' attribute. "
                    f"Available attributes: {dir(outputs)}"
                )
            all_hidden = outputs.hidden_states
            if all_hidden is None:
                raise ValueError(
                    "Model did not return hidden_states. Make sure output_hidden_states=True "
                    "is passed to the model forward call."
                )
            # Convert tuple to list if needed
            if isinstance(all_hidden, tuple):
                all_hidden = list(all_hidden)
            elif not isinstance(all_hidden, (list, tuple)):
                raise TypeError(
                    f"Expected hidden_states to be a tuple or list, got {type(all_hidden)}"
                )
            # Check if we have any layers
            if len(all_hidden) == 0:
                raise ValueError(
                    "Model returned empty hidden_states. The model may not support "
                    "output_hidden_states or may have a different structure."
                )
            # Validate layer indices
            max_layer_idx = len(all_hidden) - 1
            invalid_layers = [i for i in selected_layers if i > max_layer_idx or i < 0]
            if invalid_layers:
                raise IndexError(
                    f"Selected layer indices {invalid_layers} are out of range. "
                    f"Model has {len(all_hidden)} layers (indices 0-{max_layer_idx})."
                )
            try:
                selected_hidden = [all_hidden[i] for i in selected_layers]
            except (IndexError, TypeError) as e:
                raise IndexError(
                    f"Failed to access hidden states at indices {selected_layers}. "
                    f"Model has {len(all_hidden)} layers. Error: {e}"
                ) from e
            if self.config["return_as_feature_maps"]:
                # Convert tokens to spatial maps if needed
                converted_hidden = []
                for h in selected_hidden:
                    if len(h.shape) == 3:
                        # It's tokens [B, N, C], convert to spatial
                        B, N, C = h.shape
                        # Infer spatial dimensions assuming square patches
                        spatial_size = int(N ** 0.5)
                        if spatial_size * spatial_size == N:
                            # Perfect square - reshape to spatial
                            h_spatial = h.transpose(1, 2).view(B, C, spatial_size, spatial_size)
                            converted_hidden.append(h_spatial)
                        else:
                            # Not a perfect square, use patch dimensions
                            h_spatial = h.transpose(1, 2).view(B, C, patch_h, patch_w)
                            converted_hidden.append(h_spatial)
                    else:
                        # Already spatial [B, C, H, W]
                        converted_hidden.append(h)
                result["selected_hidden_states"] = converted_hidden
            else:
                # Convert each to token-like format (or keep as tokens)
                result["selected_hidden_states"] = [
                    self.feature_maps_to_tokens(h) for h in selected_hidden
                ]

        return result

    def preprocess_image(self, image_tensor):
        """
        Preprocess image for DINOv3 ConvNeXt using the proper processor.

        Args:
            image_tensor: [B, 3, H, W] - Raw image tensor with values in [0, 1]

        Returns:
            [B, 3, H, W] - Preprocessed tensor ready for DINOv3 ConvNeXt
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


class DPTReassembleLayer_ConvNext(nn.Module):
    """
    Reassemble layer for ConvNeXt feature maps (already spatial).
    Handles projection and upsampling/downsampling without token reshaping.
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
                nn.Conv2d(2 * in_channels, in_channels, kernel_size=1), nn.GELU()
            )

        # Channel projection
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Spatial resampling based on scale factor
        if scale_factor == 4.0:
            # 4x upsampling
            self.resample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=8,
                stride=4,
                padding=2,
                bias=True,
            )
        elif scale_factor == 2.0:
            # 2x upsampling
            self.resample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            )
        elif scale_factor == 1.0:
            # No resampling
            self.resample = nn.Identity()
        elif scale_factor == 0.5:
            # 0.5x downsampling
            self.resample = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W] - Spatial feature maps from ConvNeXt (already spatial)
        Returns:
            [B, out_channels, H', W'] - Spatial feature map after projection and resampling
        """
        # Project channels
        feature_map = self.proj(feature_map)  # [B, out_channels, H, W]

        # Resample spatial resolution
        output = self.resample(feature_map)  # [B, out_channels, H', W']

        return output


class DPT_Decoder_ConvNext(nn.Module):
    """
    DPT decoder adapted for ConvNeXt feature maps.
    Works with spatial feature maps directly (no token reshaping needed).
    Implements multi-scale reassembly, progressive fusion, and RGB prediction head.

    Dropout:
        Controlled by config["dropout"] (float in [0,1]). Applied as 2D dropout
        after each fusion addition and between stages in the head to regularize
        training. Shapes are preserved; tensor dims are:
        - inputs hidden_states: List[4] of [B, C, H, W] (spatial feature maps)
        - output: [B, C_out, H, W]
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        # Default configuration
        default_config = {
            "feature_dim": 768,  # ConvNeXt feature dimension (can be int or list)
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
        self.config["feature_dim"] = [192, 384, 768, 1536]
        self.out_image_size = self.config["output_image_size"]
        
        # Handle feature_dim: can be int (same for all) or list (per-stage)
        feature_dim = self.config["feature_dim"]
        if isinstance(feature_dim, int):
            # Use same feature_dim for all stages
            feature_dims = [feature_dim] * len(self.config["reassemble_out_channels"])
        elif isinstance(feature_dim, (list, tuple)):
            # Use per-stage feature dimensions
            if len(feature_dim) != len(self.config["reassemble_out_channels"]):
                raise ValueError(
                    f"feature_dim list length ({len(feature_dim)}) must match "
                    f"reassemble_out_channels length ({len(self.config['reassemble_out_channels'])})"
                )
            feature_dims = list(feature_dim)
        else:
            raise TypeError(
                f"feature_dim must be int or list/tuple, got {type(feature_dim)}"
            )
        
        # Create reassemble layers for multi-scale feature extraction
        self.reassemble_layers = nn.ModuleList(
            [
                DPTReassembleLayer_ConvNext(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    scale_factor=scale,
                    readout_type=self.config["readout_type"],
                )
                for in_ch, out_ch, scale in zip(
                    feature_dims,
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

        # RGB prediction head - split into stages to avoid large intermediate tensors
        # This allows progressive upsampling to handle very large output sizes
        self.rgb_head_stage1 = nn.Sequential(
            # First stage: 256 -> 128 channels with spatial refinement
            nn.Conv2d(self.config["fusion_hidden_size"], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
            self.drop2d,
        )
        self.rgb_head_stage2 = nn.Sequential(
            # Second stage: 128 -> 64 channels with feature refinement
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
            self.drop2d,
        )
        self.rgb_head_stage3 = nn.Sequential(
            # Third stage: 64 -> 32 channels
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
            self.drop2d,
        )
        self.rgb_head_final = nn.Sequential(
            # Final RGB projection
            nn.Conv2d(32, self.config["output_channels"], kernel_size=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(
        self,
        hidden_states: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through DPT decoder for ConvNeXt.

        Args:
            hidden_states: List of 4 tensors from DINOv3_ConvNext selected layers
                          Each tensor: [B, C, H, W] (spatial feature maps)
                          Can have different channel dimensions per stage

        Returns:
            rgb_output: [B, 3, H, W] - RGB image in [0, 1] range
        """
        input_height, input_width = self.out_image_size
        
        # Validate input
        if len(hidden_states) != len(self.reassemble_layers):
            raise ValueError(
                f"Expected {len(self.reassemble_layers)} hidden states, "
                f"got {len(hidden_states)}"
            )
        
        # Apply reassemble layers to create multi-scale feature maps
        reassembled_features = []
        for i, (hidden_state, reassemble) in enumerate(zip(hidden_states, self.reassemble_layers)):
            # Verify channel dimension matches
            actual_channels = hidden_state.shape[1]
            expected_channels = reassemble.in_channels
            if actual_channels != expected_channels:
                actual_dims = [h.shape[1] for h in hidden_states]
                raise ValueError(
                    f"Stage {i}: Expected {expected_channels} channels, "
                    f"got {actual_channels} channels. "
                    f"Hidden state shape: {hidden_state.shape}.\n\n"
                    f"To fix this, set 'feature_dim' in your decoder config to:\n"
                    f"  feature_dim: {actual_dims}\n\n"
                    f"Or in YAML format:\n"
                    f"  FEATURE_DIM: {actual_dims}\n\n"
                    f"Current decoder config feature_dim: {self.config.get('feature_dim', 'not set')}"
                )
            feature_map = reassemble(hidden_state)  # [B, out_ch, H', W']
            reassembled_features.append(feature_map)

        # Progressive fusion from smallest to largest scale
        # Use actual spatial dimensions from reassembled features to ensure matching
        
        # Start with smallest scale (stage 3)
        fused = self.fusion_blocks[3](reassembled_features[3])  # [B, 256, H3, W3]
        h3, w3 = fused.shape[2], fused.shape[3]
        
        # Interpolate to match stage 2 size
        h2, w2 = reassembled_features[2].shape[2], reassembled_features[2].shape[3]
        fused = F.interpolate(
            fused, size=(h2, w2), mode="bilinear", align_corners=True
        )  # [B, 256, H2, W2]

        # Add stage 2 features
        stage2_fused = self.fusion_blocks[2](reassembled_features[2])  # [B, 256, H2, W2]
        # Ensure spatial dimensions match (should already match, but be safe)
        if fused.shape[2:] != stage2_fused.shape[2:]:
            stage2_fused = F.interpolate(
                stage2_fused, size=(fused.shape[2], fused.shape[3]), 
                mode="bilinear", align_corners=True
            )
        fused = fused + stage2_fused  # [B, 256, H2, W2]
        fused = self.drop2d(fused)
        
        # Interpolate to match stage 1 size
        h1, w1 = reassembled_features[1].shape[2], reassembled_features[1].shape[3]
        fused = F.interpolate(
            fused, size=(h1, w1), mode="bilinear", align_corners=True
        )  # [B, 256, H1, W1]

        # Add stage 1 features
        stage1_fused = self.fusion_blocks[1](reassembled_features[1])  # [B, 256, H1, W1]
        # Ensure spatial dimensions match
        if fused.shape[2:] != stage1_fused.shape[2:]:
            stage1_fused = F.interpolate(
                stage1_fused, size=(fused.shape[2], fused.shape[3]), 
                mode="bilinear", align_corners=True
            )
        fused = fused + stage1_fused  # [B, 256, H1, W1]
        fused = self.drop2d(fused)
        
        # Interpolate to match stage 0 size
        h0, w0 = reassembled_features[0].shape[2], reassembled_features[0].shape[3]
        fused = F.interpolate(
            fused, size=(h0, w0), mode="bilinear", align_corners=True
        )  # [B, 256, H0, W0]

        # Add stage 0 features
        stage0_fused = self.fusion_blocks[0](reassembled_features[0])  # [B, 256, H0, W0]
        # Ensure spatial dimensions match
        if fused.shape[2:] != stage0_fused.shape[2:]:
            stage0_fused = F.interpolate(
                stage0_fused, size=(fused.shape[2], fused.shape[3]), 
                mode="bilinear", align_corners=True
            )
        fused = fused + stage0_fused  # [B, 256, H0, W0]
        fused = self.drop2d(fused)
        
        # Apply RGB head with progressive upsampling to avoid large intermediate tensors
        # Calculate target size early to do progressive upsampling
        target_size = self.config["output_image_size"] or (input_height, input_width)
        target_h, target_w = target_size
        
        # Stage 1: Process at current resolution
        x = self.rgb_head_stage1(fused)  # [B, 128, H, W]
        
        # Progressive upsampling: process at lower resolution first, then upsample
        # This avoids creating huge intermediate tensors
        # Calculate safe intermediate size (max 896x896 to avoid INT_MAX issues with batch_size=32)
        max_safe_size = 896
        current_h, current_w = x.shape[2], x.shape[3]
        
        # If target is very large, process at intermediate size first
        if target_h > max_safe_size or target_w > max_safe_size:
            # Process stage 2 at current or intermediate size
            intermediate_h = min(max_safe_size, max(current_h * 2, target_h // 2))
            intermediate_w = min(max_safe_size, max(current_w * 2, target_w // 2))
            
            # Upsample to intermediate size if needed
            if x.shape[2:] != (intermediate_h, intermediate_w):
                x = F.interpolate(
                    x, size=(intermediate_h, intermediate_w), mode="bilinear", align_corners=True
                )
            
            # Stage 2: Process at intermediate size
            x = self.rgb_head_stage2(x)  # [B, 64, H_intermediate, W_intermediate]
            
            # Stage 3: Process at intermediate size
            x = self.rgb_head_stage3(x)  # [B, 32, H_intermediate, W_intermediate]
            
            # Final RGB projection at intermediate size
            x = self.rgb_head_final(x)  # [B, 3, H_intermediate, W_intermediate]
            
            # Final upsample to target size
            rgb_output = F.interpolate(
                x, size=(target_h, target_w), mode="bilinear", align_corners=True
            )
        else:
            # Target size is manageable, process normally
            # Upsample to target size after stage 1
            if x.shape[2:] != (target_h, target_w):
                x = F.interpolate(
                    x, size=(target_h, target_w), mode="bilinear", align_corners=True
                )
            
            # Stage 2: Process
            x = self.rgb_head_stage2(x)  # [B, 64, H_target, W_target]
            
            # Stage 3: Process
            x = self.rgb_head_stage3(x)  # [B, 32, H_target, W_target]
            
            # Final RGB projection
            rgb_output = self.rgb_head_final(x)  # [B, 3, H_target, W_target]

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

        # ---- Detect ConvNeXt from encoder config ----
        # Check if encoder name contains "convnext" (case-insensitive)
        is_convnext = False
        if isinstance(dinov3, dict):
            model_name = dinov3.get("model_name", "").lower()
            is_convnext = "convnext" in model_name
        elif isinstance(dinov3, (DINOv3, DINOv3_ConvNext)):
            # If it's already an instance, check its type
            is_convnext = isinstance(dinov3, DINOv3_ConvNext)
            if isinstance(dinov3, DINOv3):
                model_name = dinov3.config.get("model_name", "").lower()
                is_convnext = "convnext" in model_name

        # ---- RGB (DINOv3 or DINOv3_ConvNext) ----
        # Accept either an instance or a DINOv3(**cfg) / DINOv3_ConvNext(**cfg) dict
        if is_convnext:
            self.dinov3 = _build(dinov3, DINOv3_ConvNext)
            self.use_convnext = True
        else:
            self.dinov3 = _build(dinov3, DINOv3)
            self.use_convnext = False

        self.image_size = self.dinov3.config["image_size"]
        self.patch_size = patch_size
        self.embed_dim = self.dinov3.feature_dim

        # For ConvNeXt, try to detect per-stage feature dimensions from model config
        convnext_feature_dims = None
        if self.use_convnext:
            # Try to get feature dimensions from ConvNeXt model config
            if hasattr(self.dinov3.dinov3, "config"):
                model_config = self.dinov3.dinov3.config
                # ConvNeXt models typically have hidden_sizes in config
                if hasattr(model_config, "hidden_sizes"):
                    all_feature_dims = list(model_config.hidden_sizes)
                    # Get selected layers to match feature dimensions
                    selected_layers = self.dinov3.config.get("return_selected_layers", None)
                    if selected_layers is not None and len(selected_layers) > 0:
                        # Map selected layer indices to feature dimensions
                        # Note: hidden_sizes might include initial embedding, so we need to map correctly
                        # For ConvNeXt, hidden_sizes typically corresponds to stage outputs
                        # If we have more stages than selected layers, we need to map correctly
                        if len(all_feature_dims) >= len(selected_layers):
                            # Try to map selected layers to feature dimensions
                            # This is a heuristic - may need adjustment based on actual model structure
                            max_layer = max(selected_layers)
                            if max_layer < len(all_feature_dims):
                                # Use feature dimensions corresponding to selected layers
                                convnext_feature_dims = [all_feature_dims[i] for i in selected_layers]
                            else:
                                # Fallback: use last N dimensions
                                convnext_feature_dims = all_feature_dims[-len(selected_layers):]
                        else:
                            # Not enough stages, use what we have
                            convnext_feature_dims = all_feature_dims
                    else:
                        # No selected layers specified, use last 4 stages
                        if len(all_feature_dims) >= 4:
                            convnext_feature_dims = all_feature_dims[-4:]
                        else:
                            convnext_feature_dims = all_feature_dims
                    # if convnext_feature_dims:
                    #     logger.info(
                    #         f"ConvNeXt feature dimensions: {convnext_feature_dims}"
                    #     )

        # ---- Decoders (DPT_Decoder or DPT_Decoder_ConvNext) ----
        # Handle flexible decoder configuration with legacy support
        def build_dpt(dec, decoder_name=None):
            """Build a decoder from config or instance (standard, FiLM-conditioned, or ConvNext).

            Accepts either an instantiated `DPT_Decoder`/`DPT_Decoder_ConvNext`/`FiLMConditionedDPT` or a
            config dict. When a config dict is provided, if it contains
            `use_film` (or `USE_FILM`) set to True, a `FiLMConditionedDPT` is
            created; otherwise a standard `DPT_Decoder` or `DPT_Decoder_ConvNext` is created
            based on whether we're using ConvNeXt encoder. The config
            dict is passed as a single dictionary argument to the decoder's
            constructor, augmented with the resolved `feature_dim`.

            If `from_pretrained` (or `FROM_PRETRAINED`) is set and not empty,
            the decoder weights are loaded from that path and the decoder is frozen.

            Args:
                dec: Decoder instance or config dict
                decoder_name: Optional decoder name for prefix stripping when loading weights
            """
            if isinstance(dec, (DPT_Decoder, DPT_Decoder_ConvNext, FiLMConditionedDPT)):
                return dec
            if isinstance(dec, dict):
                # Extract pretrained path before building decoder
                pretrained_path = dec.get(
                    "from_pretrained", dec.get("FROM_PRETRAINED", "")
                )
                # Extract decoder learning rate
                decoder_lr = dec.get("decoder_lr", dec.get("DECODER_LR", None))
                # Determine whether to build FiLM-conditioned or standard decoder
                use_film = bool(dec.get("use_film", dec.get("USE_FILM", False)))
                # Build config dict for the decoder class
                # For ConvNeXt, use detected feature dimensions if available, otherwise use embed_dim
                # But allow explicit feature_dim in decoder config to override auto-detection
                if "feature_dim" in dec or "FEATURE_DIM" in dec:
                    # User explicitly set feature_dim, use it
                    feature_dim = dec.get("feature_dim", dec.get("FEATURE_DIM", self.embed_dim))
                elif is_convnext and convnext_feature_dims is not None:
                    # Use auto-detected dimensions
                    feature_dim = convnext_feature_dims
                else:
                    # Fallback to embed_dim
                    feature_dim = self.embed_dim
                config = {
                    "feature_dim": feature_dim,
                    **dec,
                }
                # Remove the control flags from the config passed into the module
                config.pop("use_film", None)
                config.pop("USE_FILM", None)
                config.pop("from_pretrained", None)
                config.pop("FROM_PRETRAINED", None)
                config.pop("DECODER_LR", None)

                # Create decoder instance - use ConvNext decoder if using ConvNext encoder
                if use_film:
                    decoder = FiLMConditionedDPT(config)
                elif is_convnext:
                    decoder = DPT_Decoder_ConvNext(config)
                else:
                    decoder = DPT_Decoder(config)

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
                    # Filter out keys with shape mismatches before loading
                    filtered_state_dict = {}
                    incompatible_keys = []
                    model_state_dict = decoder.state_dict()
                    
                    for key, value in state_dict.items():
                        if key in model_state_dict:
                            model_shape = model_state_dict[key].shape
                            checkpoint_shape = value.shape
                            if model_shape == checkpoint_shape:
                                filtered_state_dict[key] = value
                            else:
                                incompatible_keys.append(
                                    f"{key}: checkpoint shape {checkpoint_shape} != model shape {model_shape}"
                                )
                        else:
                            # Key not in model, skip it
                            pass
                    
                    # Load filtered state dict
                    missing_keys, unexpected_keys = decoder.load_state_dict(
                        filtered_state_dict, strict=False
                    )
                    
                    # Log warnings about incompatible keys
                    if incompatible_keys:
                        import warnings
                        warnings.warn(
                            f"Some weights from {pretrained_path} had incompatible shapes and were skipped:\n"
                            + "\n".join(incompatible_keys[:10])  # Show first 10
                            + (f"\n... and {len(incompatible_keys) - 10} more" if len(incompatible_keys) > 10 else "")
                            + "\nThis is likely because the pretrained model used different feature_dim values."
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

                # Freeze decoder if decoder_lr is 0 (or None and from_pretrained was set)
                # If decoder_lr is explicitly 0, freeze regardless of pretrained status
                if decoder_lr is not None and decoder_lr == 0.0:
                    for param in decoder.parameters():
                        param.requires_grad = False
                    decoder.eval()  
                    logger.info(
                        f"Decoder '{decoder_name}' frozen due to DECODER_LR=0.0"
                    )
                else:
                    for param in decoder.parameters():
                        param.requires_grad = True
                    decoder.train()  
                    logger.info(
                        f"Decoder '{decoder_name}' un-frozen due to DECODER_LR={decoder_lr}"
                    )
                # Store decoder_lr for optimizer setup (will be None if not specified)
                decoder.decoder_lr = decoder_lr
                return decoder
            raise TypeError(
                "Decoder must be DPT_Decoder/DPT_Decoder_ConvNext/FiLMConditionedDPT instance or dict."
            )

        self.decoder_names = list(decoders.keys())
        self.decoders = nn.ModuleDict()
        # Store decoder learning rates for optimizer setup
        self.decoder_lrs = {}
        for decoder_name, decoder_config in decoders.items():
            decoder = build_dpt(decoder_config, decoder_name=decoder_name)
            self.decoders[decoder_name] = decoder
            # Store the learning rate for this decoder (None means use default/base LR)
            self.decoder_lrs[decoder_name] = getattr(decoder, "decoder_lr", None)

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
        # 1) RGB → DINO tokens/features

        rgb_in = self.dinov3.preprocess_image(model_input_dict["rgb"])
        dinov3_output = self.dinov3(rgb_in)
        rgb_features = dinov3_output["selected_hidden_states"]

        # For ConvNeXt, ensure features are spatial feature maps [B, C, H, W]
        # For regular DINOv3, they are tokens [B, N, C]
        if self.use_convnext:
            # ConvNeXt outputs might be tokens, convert to spatial if needed
            # But if return_as_feature_maps=True, they should already be spatial
            # Check first feature to determine format
            if len(rgb_features[0].shape) == 3:
                # It's tokens [B, N, C], but ConvNext decoder expects spatial
                # This shouldn't happen if config is correct, but handle gracefully
                logger.warning(
                    "ConvNeXt encoder returned tokens but decoder expects spatial maps. "
                    "Set return_as_feature_maps=True in encoder config."
                )

        # 6) Decode with flexible decoder heads
        outputs = {}
        for decoder_name in self.decoder_names:
            decoder_output = self.decoders[decoder_name](rgb_features)
            outputs[decoder_name] = decoder_output

        # Optional: Add tokens for debugging/analysis
        # outputs.update({
        #     "rgb_tokens": rgb_features,
        # })

        return outputs

    def extract_tokens(self, image):
        rgb_in = self.dinov3.preprocess_image(image)
        tokens_list = self.dinov3(rgb_in)["selected_hidden_states"]
        return tokens_list

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

    def forward(self, hidden_states, return_mask: bool = False):
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
    
    def extract_tokens(self, image):
        rgb_in = self.dinov3.preprocess_image(image)
        tokens_list = self.dinov3(rgb_in)["selected_hidden_states"]
        return tokens_list


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

        # Extract TokenInpainter class and module from config
        if token_inpainter_cfg is None:
            token_inpainter_cfg = {}

        # Get class name and module (with defaults for backward compatibility)
        self.token_inpainter_class_name = token_inpainter_cfg.pop(
            "token_inpainter_class", "TokenInpainter"
        )
        self.token_inpainter_module_name = token_inpainter_cfg.pop(
            "token_inpainter_module", "models"
        )

        # Dynamically import the module
        token_inpainter_module = importlib.import_module(self.token_inpainter_module_name)

        # Get the TokenInpainter class from the module
        TokenInpainterClass = getattr(
            token_inpainter_module, self.token_inpainter_class_name
        )

        # Filter kwargs to only include parameters that the constructor accepts
        # This prevents errors when passing unexpected parameters
        sig = inspect.signature(TokenInpainterClass.__init__)
        accepted_params = set(sig.parameters.keys()) - {"self"}  # Exclude 'self'
        filtered_cfg = {
            k: v for k, v in token_inpainter_cfg.items() if k in accepted_params
        }

        # Instantiate the TokenInpainter with filtered config
        self.token_inpaint = TokenInpainterClass(dim=dim, **filtered_cfg)

    def extract_tokens(self, image):
        rgb_in = self.dinov3.preprocess_image(image)
        tokens_list = self.dinov3(rgb_in)["selected_hidden_states"]
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
        if "patch_mask_override" in model_input_dict:
            patchmask = pixel_mask_to_patch_mask(
                model_input_dict["patch_mask_override"],
                patch_size=self.patch_size,
                threshold=0.1,
                invert=False,
                soft=True if "soft" in self.token_inpainter_class_name.lower() else False
            )
        else:
            patchmask = pixel_mask_to_patch_mask(
                outputs["highlight"],
                patch_size=self.patch_size,
                threshold=0.1,
                invert=False,
                soft=True if "soft" in self.token_inpainter_class_name.lower() else False
            )

        # patchmask = 1 : MUST IMPAINT THE TOKEN
        # patchmask = 0 : IS TEACHER TOKEN
        outputs["patch_mask"] = patchmask

        ### THIRD: Inpaint the tokens in the mask
        # Detect if using soft masks (float) or boolean masks
        is_soft_mask = patchmask.dtype.is_floating_point
        
        completed_tokens = []
        for n, T in enumerate(tokens_list):  # (B,N,C)
            # Prepare visibility mask for token inpainter
            # TokenInpainter expects: True/1.0 = visible/teacher, False/0.0 = masked/inpaint
            if is_soft_mask:
                visibility_mask = 1.0 - patchmask  # [B, N] float in [0,1]
            else:
                visibility_mask = torch.logical_not(patchmask)  # [B, N] bool
            
            T_inpainted = self.token_inpaint(T, visibility_mask)  # refined all tokens
            
            # Blend: keep teacher tokens on context; use predicted tokens on masked patches
            if is_soft_mask:
                # Soft blending: patchmask=1.0 → use T_inpainted, patchmask=0.0 → use T
                patchmask_expanded = patchmask.unsqueeze(-1)  # [B, N, 1]
                T_comp = patchmask_expanded * T_inpainted + (1.0 - patchmask_expanded) * T  # (B,N,C)
            else:
                # Boolean selection: use torch.where
                T_comp = torch.where(
                    patchmask.unsqueeze(-1), T_inpainted, T
                )  # (B,N,C)
            completed_tokens.append(T_comp)

        outputs["tokens_inpainted"] = T_inpainted
        outputs["tokens_completed"] = completed_tokens
        # 4) Decode with completed tokens (do NOT pass mask to decoder)
        for name, dec in self.decoders.items():
            if name == "highlight":
                continue
            outputs[name] = dec(completed_tokens)

        return outputs
