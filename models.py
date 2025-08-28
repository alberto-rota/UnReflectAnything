import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from typing import List, Dict, Optional


class DINOv3(nn.Module):
    """
    Configurable DINOv3 model with flexible return options.
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
        assert (
            input_h % self.patch_size == 0
        ), f"Height {input_h} must be divisible by patch size {self.patch_size}"
        assert (
            input_w % self.patch_size == 0
        ), f"Width {input_w} must be divisible by patch size {self.patch_size}"

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
            "output_image_size": (448, 448),  # If None, maintains input size
            "output_channels": 3, # Set to 4 for RGBA output
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

        # RGB prediction head
        self.rgb_head = nn.Sequential(
            # First stage: 256 -> 128 channels with spatial refinement
            nn.Conv2d(self.config["fusion_hidden_size"], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            ),  # 192x192 -> 384x384
            # Second stage: 128 -> 64 channels with feature refinement
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
            # Third stage: 64 -> 32 channels
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if self.config["use_bn"] else nn.Identity(),
            nn.ReLU(inplace=True),
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
        batch_size = hidden_states[0].shape[0]
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
            fused, scale_factor=2, mode="bilinear", align_corners=False
        )  # [B, 256, 24, 24]

        # Add stage 2 features
        fused = fused + self.fusion_blocks[2](
            reassembled_features[2]
        )  # [B, 256, 24, 24]
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=False
        )  # [B, 256, 48, 48]

        # Add stage 1 features
        fused = fused + self.fusion_blocks[1](
            reassembled_features[1]
        )  # [B, 256, 48, 48]
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=False
        )  # [B, 256, 96, 96]

        # Add stage 0 features
        fused = fused + self.fusion_blocks[0](
            reassembled_features[0]
        )  # [B, 256, 96, 96]
        fused = F.interpolate(
            fused, scale_factor=2, mode="bilinear", align_corners=False
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
                rgb_output, size=target_size, mode="bilinear", align_corners=False
            )

        return rgb_output


# ---------- 1) POL preprocessing ----------
class PolarizationPreprocess(nn.Module):
    """
    Input:
      aolp: (B,1,H,W) in radians, typically [0, pi)
      dolp: (B,1,H,W) in [0,1]
    Output:
      (B,3,H,W) = [cos(2*AoLP), sin(2*AoLP), DoLP]
    """

    def __init__(self, dinov3_model_name: str, height: int, width: int):
        super().__init__()
        self.prep_fn = AutoImageProcessor.from_pretrained(dinov3_model_name)
        self.prep_fn.do_normalize = False
        self.prep_fn.do_rescale = False
        self.prep_fn.size = {"height": height, "width": width}

    def forward(self, aolp: torch.Tensor, dolp: torch.Tensor) -> torch.Tensor:
        # aolp = self.prep_fn(images=aolp, return_tensors="pt")["pixel_values"]
        # dolp = self.prep_fn(images=dolp, return_tensors="pt")["pixel_values"]
        cos2 = torch.cos(2.0 * aolp)
        sin2 = torch.sin(2.0 * aolp)
        cos2sin2dolp = torch.cat([cos2, sin2, dolp.clamp(0, 1)], dim=1)
        cropped_pol = self.prep_fn(images=cos2sin2dolp, return_tensors="pt")[
            "pixel_values"
        ]
        return cropped_pol


# ---------- 2) Tiny ViT-style POL encoder ----------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):  # x: (B,C,H,W)
        x = self.proj(x)  # (B,embed_dim,H/P,W/P)
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C) with N = Hp*Wp
        return x, (Hp, Wp)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc2(self.drop(self.act(self.fc1(x))))
        return self.drop(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=12, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=4.0, drop=drop)

    def forward(self, x):
        # Self-attention
        x = (
            x
            + self.attn(
                self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False
            )[0]
        )
        x = x + self.mlp(self.norm2(x))
        return x


class POLViTEncoder(nn.Module):
    """
    ViT-like encoder for POL features; match DINOv3 hidden dim and patch size.

    Args:
        config: Dict containing configuration parameters:
            - in_ch: int, input channels (default: 3 for cos2θ, sin2θ, DoLP)
            - embed_dim: int, embedding dimension (default: 768)
            - depth: int, number of transformer blocks (default: 4)
            - n_heads: int, number of attention heads (default: 12)
            - patch_size: int, patch size for embedding (default: 16)
            - drop: float, dropout rate (default: 0.0)

    Returns:
        tokens: (B, N, embed_dim)  # N = (H/P)*(W/P)
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        # Default configuration
        default_config = {
            "in_ch": 3,
            "embed_dim": 768,
            "depth": 4,
            "n_heads": 12,
            "patch_size": 16,
            "drop": 0.0,
        }

        self.config = {**default_config, **(config or {})}

        self.patch = PatchEmbed(
            self.config["in_ch"], self.config["embed_dim"], self.config["patch_size"]
        )
        self.pos = None  # initialized at first forward pass
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.config["embed_dim"],
                    self.config["n_heads"],
                    self.config["drop"],
                )
                for _ in range(self.config["depth"])
            ]
        )
        self.norm = nn.LayerNorm(self.config["embed_dim"])

    def forward(self, x):  # x: (B,3,H,W)
        x, (Hp, Wp) = self.patch(x)  # (B,N,C)
        # learnable 2D pos embedding (initialized on first run to match N)
        N = x.shape[1]
        if (self.pos is None) or (self.pos.shape[1] != N):
            self.pos = nn.Parameter(torch.zeros(1, N, x.shape[2], device=x.device))
            nn.init.trunc_normal_(self.pos, std=0.02)
        x = x + self.pos
        pol_hidden_states = []
        for blk in self.blocks:
            x = blk(x)
            pol_hidden_states.append(self.norm(x))
        return pol_hidden_states  # (B,N,C)


# ---------- 3) Cross-attention block ----------
class CrossAttentionBlock(nn.Module):
    """
    One cross-attention layer with residual + MLP.
    Q from x_q (e.g., RGB tokens), K/V from x_kv (e.g., POL tokens).
    Shapes:
      x_q:  (B, Nq, C)
      x_kv: (B, Nk, C)
    Returns:
      (B, Nq, C) fused features.
    """

    def __init__(self, dim=768, n_heads=12, drop=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=drop, batch_first=True
        )
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=4.0, drop=drop)

    def forward(self, x_q, x_kv, attn_mask=None):
        q = self.norm_q(x_q)
        kv = self.norm_kv(x_kv)
        x, _ = self.cross_attn(q, kv, kv, attn_mask=attn_mask, need_weights=False)
        x = x_q + x  # residual after attention
        x = x + self.mlp(self.norm_out(x))  # residual after MLP
        return x


# ---------- 4) Simple fusion wrapper ----------
class RGBPOLCrossFuse(nn.Module):
    """
    One-way (RGB<-POL) or two-way (bi-directional) fusion.
    If bi_directional=True, returns concat([RGB_fused, POL_fused]) projected back to dim.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        default_config = {
            "embed_dim": 768,
            "n_heads": 12,
            "dropout": 0.0,
            "bi_directional": False,
        }
        self.config = {**default_config, **(config or {})}
        self.rgb_from_pol = CrossAttentionBlock(self.config["embed_dim"], self.config["n_heads"], self.config["dropout"])
        self.bi = self.config["bi_directional"]
        if self.config["bi_directional"]:
            self.pol_from_rgb = CrossAttentionBlock(self.config["embed_dim"], self.config["n_heads"], self.config["dropout"])
            self.proj = nn.Linear(2 * self.config["embed_dim"], self.config["embed_dim"])

    def forward(self, rgb_tokens, pol_tokens, attn_mask=None):
        rgb_fused = self.rgb_from_pol(
            rgb_tokens, pol_tokens, attn_mask
        )  # Q=RGB, K/V=POL
        if not self.bi:
            return rgb_fused
        pol_fused = self.pol_from_rgb(
            pol_tokens, rgb_tokens, attn_mask
        )  # Q=POL, K/V=RGB
        fused = torch.cat([rgb_fused, pol_fused], dim=-1)
        return self.proj(fused)  # (B, N_rgb, dim) if N_rgb == N_pol; else up to you


# ------------------ small helpers ------------------
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


# ------------------ top-level model ------------------


class RGBPOLDecomposer(nn.Module):
    """
    RGB + POL decomposition with cross-attention and three DPT decoders.

    Inputs (forward):
      batch["rgb"] : (B,3,H,W) in [0,1]
      batch["AoP"] : (B,1,H,W) radians
      batch["DoP"] : (B,1,H,W) in [0,1]

    Returns:
      {
        "specular":  (B,3,H,W),
        "diffuse":   (B,3,H,W),
        "highlight": (B,1 or 3,H,W)  # depends on decoder config
        "recon":     (B,3,H,W),      # typically specular + diffuse
        "tokens": {
           "rgb": (B,N,C),
           "pol": (B,N,C),
           "cross": (B,N,C)
        }
      }
    """

    def __init__(
        self,
        # 1) RGB encoder (DINOv3) — instance or config dict
        dinov3,
        # 2) POL encoder — instance or configs (preprocess is created inside if not passed)
        pol_encoder=None,  # POLViTEncoder instance or dict
        pol_preprocess=None,  # PolarizationPreprocess instance or dict
        pol_cross_attn=None,  # RGBPOLCrossFuse instance or dict
        # 3) Decoders — three DPT_Decoder instances or dict configs
        spec_decoder=None,  # DPT_Decoder instance or dict
        diffuse_decoder=None,  # DPT_Decoder instance or dict
        highlight_decoder=None,  # DPT_Decoder instance or dict
        # Optional: if your DINO wrapper needs these hints
        patch_size: int = 16,
    ):
        super().__init__()

        # ---- RGB (DINOv3) ----
        # Accept either an instance or a DINOv3(**cfg) dict
        self.dinov3 = _build(dinov3, DINOv3)

        self.image_size = self.dinov3.config["image_size"]
        self.patch_size = patch_size
        self.embed_dim = self.dinov3.feature_dim

        # ---- POL branch ----
        # Preprocess (AoLP, DoLP) → [cos2θ, sin2θ, DoLP]
        if pol_preprocess is None:
            self.pol_pre = PolarizationPreprocess(
                self.dinov3.config["model_name"], self.image_size, self.image_size
            )
        else:
            self.pol_pre = _build(pol_preprocess, PolarizationPreprocess)

        # Encoder (ViT-like), align dim/patch with DINO
        if pol_encoder is None:
            self.pol_enc = POLViTEncoder(
                in_ch=3,
                embed_dim=self.embed_dim,
                depth=4,
                n_heads=12,
                patch_size=patch_size,
            )
        else:
            self.pol_enc = _build(pol_encoder, POLViTEncoder)

        # Cross-attention: Q=RGB, K/V=POL
        if pol_cross_attn is None:
            self.cross = nn.ModuleList(
                [
                    RGBPOLCrossFuse(
                        embed_dim=self.embed_dim, n_heads=12, dropout=0.1, bi_directional=False
                    )
                    for _ in range(4)
                ]
            )
        else:
            self.cross = nn.ModuleList(
                [
                    _build(pol_cross_attn, RGBPOLCrossFuse)
                    for _ in range(4)
                ]
            )

        # ---- Decoders (DPT_Decoder) ----
        # Each can control its own out_channels inside the config (e.g., 3 for S/D, 1 for H)
        if spec_decoder is None:
            # Minimal default: your DPT_Decoder likely needs at least decoder_config / dim / image_size
            spec_decoder = {
                "decoder_config": {"use_bn": True, "readout_type": "project"},
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
            }
        if diffuse_decoder is None:
            diffuse_decoder = {
                "decoder_config": {"use_bn": True, "readout_type": "project"},
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
            }
        if highlight_decoder is None:
            # Often a 1-channel mask is useful; set out_channels=1 if your DPT_Decoder supports it.
            highlight_decoder = {
                "decoder_config": {
                    "use_bn": True,
                    "readout_type": "project",
                    "out_channels": 1,
                },
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
            }

        # Normalize configs → instances
        def build_dpt(dec):
            if isinstance(dec, DPT_Decoder):
                return dec
            if isinstance(dec, dict):
                # DPT_Decoder takes a single config dict
                config = {
                    "feature_dim": self.embed_dim,
                    **dec,
                }
                return DPT_Decoder(config)
            raise TypeError("Decoder must be DPT_Decoder instance or dict.")
        self.decS = build_dpt(spec_decoder)
        self.decD = build_dpt(diffuse_decoder)
        self.decH = build_dpt(highlight_decoder)

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

    def forward(self, batch):
        # 1) RGB → DINO tokens
        rgb_in = self.dinov3.preprocess_image(batch["rgb"])
        rgb_tokens = self.dinov3(rgb_in)["selected_hidden_states"]

        # 2) POL → preprocess → POL tokens
        pol_in = self.pol_pre(batch["AoP"], batch["DoP"])  # (B,3,H,W)
        pol_tokens = self.pol_enc(pol_in)  # (B,N,C)
        # 3) CROSS (Q=RGB, K/V=POL)
        cross_tokens = []
        for i in range(4):
            cross_tokens.append(self.cross[i](rgb_tokens[i], pol_tokens[i]))

        # 6) Decode with three DPT_Decoder heads
        S = self.decS(cross_tokens)  # Specular  (B,3,H,W)
        D = self.decD(cross_tokens)  # Diffuse   (B,3,H,W)
        H = self.decH(cross_tokens)  # Highlight (B,3,H,W)

        return {
            "specular": S,
            "diffuse": D,
            "highlight": H,
            "rgb_tokens": rgb_tokens, 
            "pol_tokens": pol_tokens, 
            "cross_tokens": cross_tokens,
        }

class RGBDistillDecomposer(nn.Module):
    """
    RGB with cross-attention and three DPT decoders.

    Inputs (forward):
      batch["rgb"] : (B,3,H,W) in [0,1]

    Returns:
      {
        "specular":  (B,3,H,W),
        "diffuse":   (B,3,H,W),
        "highlight": (B,1 or 3,H,W)  # depends on decoder config
        "recon":     (B,3,H,W),      # typically specular + diffuse
        "tokens": {
           "rgb": (B,N,C),
        }
      }
    """

    def __init__(
        self,
        # 1) RGB encoder (DINOv3) — instance or config dict
        dinov3,
        # 3) Decoders — three DPT_Decoder instances or dict configs
        spec_decoder=None,  # DPT_Decoder instance or dict
        diffuse_decoder=None,  # DPT_Decoder instance or dict
        highlight_decoder=None,  # DPT_Decoder instance or dict
        # Optional: if your DINO wrapper needs these hints
        patch_size: int = 16,
    ):
        super().__init__()

        # ---- RGB (DINOv3) ----
        # Accept either an instance or a DINOv3(**cfg) dict
        self.dinov3 = _build(dinov3, DINOv3)

        self.image_size = self.dinov3.config["image_size"]
        self.patch_size = patch_size
        self.embed_dim = self.dinov3.feature_dim

        # ---- Decoders (DPT_Decoder) ----
        # Each can control its own out_channels inside the config (e.g., 3 for S/D, 1 for H)
        if spec_decoder is None:
            # Minimal default: your DPT_Decoder likely needs at least decoder_config / dim / image_size
            spec_decoder = {
                "decoder_config": {"use_bn": True, "readout_type": "project"},
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
            }
        if diffuse_decoder is None:
            diffuse_decoder = {
                "decoder_config": {"use_bn": True, "readout_type": "project"},
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
            }
        if highlight_decoder is None:
            # Often a 1-channel mask is useful; set out_channels=1 if your DPT_Decoder supports it.
            highlight_decoder = {
                "decoder_config": {
                    "use_bn": True,
                    "readout_type": "project",
                    "out_channels": 1,
                },
                "embed_dim": self.embed_dim,
                "image_size": self.image_size,
            }

        # Normalize configs → instances
        def build_dpt(dec):
            if isinstance(dec, DPT_Decoder):
                return dec
            if isinstance(dec, dict):
                # DPT_Decoder takes a single config dict
                config = {
                    "feature_dim": self.embed_dim,
                    **dec,
                }
                return DPT_Decoder(config)
            raise TypeError("Decoder must be DPT_Decoder instance or dict.")
        self.decS = build_dpt(spec_decoder)
        self.decD = build_dpt(diffuse_decoder)
        self.decH = build_dpt(highlight_decoder)

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

    def forward(self, batch):
        # 1) RGB → DINO tokens
        rgb_in = self.dinov3.preprocess_image(batch["rgb"])
        rgb_tokens = self.dinov3(rgb_in)["selected_hidden_states"]

        # 6) Decode with three DPT_Decoder heads
        S = self.decS(rgb_tokens)  # Specular  (B,3,H,W)
        D = self.decD(rgb_tokens)  # Diffuse   (B,3,H,W)
        H = self.decH(rgb_tokens)  # Highlight (B,3,H,W)

        return {
            "specular": S,
            "diffuse": D,
            "highlight": H,
            "rgb_tokens": rgb_tokens, 
        }


def get_model_parameter_summary(model):
    """
    Generate a comprehensive parameter summary for RGBPOLDecomposer or RGBDistillDecomposer models.
    
    Args:
        model: RGBPOLDecomposer or RGBDistillDecomposer instance
        
    Returns:
        dict: Detailed parameter summary with counts and breakdowns
    """
    if not isinstance(model, (RGBPOLDecomposer, RGBDistillDecomposer)):
        raise ValueError("Model must be RGBPOLDecomposer or RGBDistillDecomposer")
    
    def count_parameters(module, trainable_only=False):
        """Count parameters in a module."""
        if trainable_only:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in module.parameters())
    
    def count_parameters_by_name(module, name_patterns):
        """Count parameters for modules matching name patterns."""
        total_params = 0
        trainable_params = 0
        
        for name, child in module.named_modules():
            if any(pattern in name for pattern in name_patterns):
                total_params += sum(p.numel() for p in child.parameters())
                trainable_params += sum(p.numel() for p in child.parameters() if p.requires_grad)
        
        return total_params, trainable_params
    
    # Initialize summary
    summary = {
        "model_type": model.__class__.__name__,
        "total_parameters": 0,
        "trainable_parameters": 0,
        "frozen_parameters": 0,
        "components": {}
    }
    
    # RGB Encoder (DINOv3)
    dinov3_total, dinov3_trainable = count_parameters(model.dinov3, trainable_only=False), count_parameters(model.dinov3, trainable_only=True)
    summary["components"]["rgb_encoder"] = {
        "total": dinov3_total,
        "trainable": dinov3_trainable,
        "frozen": dinov3_total - dinov3_trainable,
        "description": "DINOv3 backbone for RGB feature extraction"
    }
    
    # POL components (only for RGBPOLDecomposer)
    if isinstance(model, RGBPOLDecomposer):
        # POL Preprocessing
        pol_pre_total, pol_pre_trainable = count_parameters(model.pol_pre, trainable_only=False), count_parameters(model.pol_pre, trainable_only=True)
        summary["components"]["pol_preprocessing"] = {
            "total": pol_pre_total,
            "trainable": pol_pre_trainable,
            "frozen": pol_pre_total - pol_pre_trainable,
            "description": "Polarization preprocessing (AoLP, DoLP → [cos2θ, sin2θ, DoLP])"
        }
        
        # POL Encoder
        pol_enc_total, pol_enc_trainable = count_parameters(model.pol_enc, trainable_only=False), count_parameters(model.pol_enc, trainable_only=True)
        summary["components"]["pol_encoder"] = {
            "total": pol_enc_total,
            "trainable": pol_enc_trainable,
            "frozen": pol_enc_total - pol_enc_trainable,
            "description": "POLViT encoder for polarization feature extraction"
        }
        
        # Cross-attention modules
        cross_total, cross_trainable = count_parameters(model.cross, trainable_only=False), count_parameters(model.cross, trainable_only=True)
        summary["components"]["cross_attention"] = {
            "total": cross_total,
            "trainable": cross_trainable,
            "frozen": cross_total - cross_trainable,
            "description": "RGB-POL cross-attention fusion modules"
        }
    
    # Decoders
    decoders = {
        "specular_decoder": model.decS,
        "diffuse_decoder": model.decD,
        "highlight_decoder": model.decH
    }
    
    for name, decoder in decoders.items():
        dec_total, dec_trainable = count_parameters(decoder, trainable_only=False), count_parameters(decoder, trainable_only=True)
        summary["components"][name] = {
            "total": dec_total,
            "trainable": dec_trainable,
            "frozen": dec_total - dec_trainable,
            "description": f"DPT decoder for {name.replace('_decoder', '')} component"
        }
    
    # Calculate totals
    summary["total_parameters"] = sum(comp["total"] for comp in summary["components"].values())
    summary["trainable_parameters"] = sum(comp["trainable"] for comp in summary["components"].values())
    summary["frozen_parameters"] = summary["total_parameters"] - summary["trainable_parameters"]
    
    return summary


def print_model_parameter_summary(model, detailed=True):
    """
    Print a formatted parameter summary for RGBPOLDecomposer or RGBDistillDecomposer models.
    
    Args:
        model: RGBPOLDecomposer or RGBDistillDecomposer instance
        detailed: Whether to print detailed breakdown by component
    """
    summary = get_model_parameter_summary(model)
    
    print(f"\n{'='*60}")
    print(f"MODEL PARAMETER SUMMARY: {summary['model_type']}")
    print(f"{'='*60}")
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total Parameters:     {summary['total_parameters']:,}")
    print(f"   Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"   Frozen Parameters:    {summary['frozen_parameters']:,}")
    print(f"   Trainable Ratio:      {summary['trainable_parameters']/summary['total_parameters']*100:.1f}%")
    
    if detailed:
        print(f"\n🔍 DETAILED BREAKDOWN:")
        print(f"{'Component':<25} {'Total':<12} {'Trainable':<12} {'Frozen':<12} {'Ratio':<8}")
        print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
        
        for comp_name, comp_data in summary["components"].items():
            ratio = comp_data["trainable"] / comp_data["total"] * 100 if comp_data["total"] > 0 else 0
            print(f"{comp_name:<25} {comp_data['total']:<12,} {comp_data['trainable']:<12,} {comp_data['frozen']:<12,} {ratio:<7.1f}%")
        
        print(f"\n📝 COMPONENT DESCRIPTIONS:")
        for comp_name, comp_data in summary["components"].items():
            print(f"   • {comp_name}: {comp_data['description']}")
    
    print(f"\n{'='*60}")


def get_model_size_mb(model):
    """
    Calculate model size in MB (approximate).
    
    Args:
        model: RGBPOLDecomposer or RGBDistillDecomposer instance
        
    Returns:
        float: Model size in MB
    """
    summary = get_model_parameter_summary(model)
    # Assuming float32 (4 bytes per parameter)
    size_mb = summary["total_parameters"] * 4 / (1024 * 1024)
    return size_mb
