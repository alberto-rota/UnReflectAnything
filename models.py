import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor


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
                - image_size: int, input image size (default: 896) 
                - freeze_backbone: bool, whether to freeze DINOv3 parameters (default: True)
                - return_last_hidden_state: bool, return last hidden state (default: True)
                - return_all_hidden_states: bool, return all hidden states (default: False)
                - return_selected_layers: list[int], specific layer indices to return (default: None)
                - return_as_feature_maps: bool, reshape patch tokens to spatial format (default: False)
        """
        super().__init__()
        
        # Configuration with defaults
        self.config = {
            'model_name': "facebook/dinov3-vitb16-pretrain-lvd1689m",
            'image_size': 896,
            'freeze_backbone': True,
            'return_last_hidden_state': True,
            'return_all_hidden_states': False,
            'return_selected_layers': None,
            'return_as_feature_maps': False,
            **config  # Override defaults with user config
        }
        
        # DINOv3 backbone
        self.dinov3 = AutoModel.from_pretrained(self.config['model_name'])
        self.processor = AutoImageProcessor.from_pretrained(self.config['model_name'])
        self.processor.size = {'height': self.config['image_size'], 'width': self.config['image_size']}
        
        # Freeze parameters if requested
        if self.config['freeze_backbone']:
            for param in self.dinov3.parameters():
                param.requires_grad = False
        
        # Model properties
        self.feature_dim = self.dinov3.config.hidden_size  # 768 for ViT-B/16
        self.patch_size = self.dinov3.config.patch_size    # 16 for DINOv3
        self.dinov3.config.image_size = self.config['image_size']
    
    def get_patch_spatial_dims(self, input_height, input_width):
        """Calculate spatial dimensions of patch features based on input size."""
        patch_h = input_height // self.patch_size
        patch_w = input_width // self.patch_size
        return patch_h, patch_w
    
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
                - 'all_hidden_states': List of [B, N_tokens, feature_dim] or [B, feature_dim, patch_h, patch_w]
                - 'selected_hidden_states': List of selected layer outputs
        """
        batch_size, _, input_h, input_w = rgb_image.shape
        
        # Ensure input dimensions are compatible with patch size
        assert input_h % self.patch_size == 0, f"Height {input_h} must be divisible by patch size {self.patch_size}"
        assert input_w % self.patch_size == 0, f"Width {input_w} must be divisible by patch size {self.patch_size}"
        
        # Calculate patch spatial dimensions
        patch_h, patch_w = self.get_patch_spatial_dims(input_h, input_w)
        
        # Get DINOv3 outputs
        need_all_hidden_states = (self.config['return_all_hidden_states'] or 
                                 self.config['return_selected_layers'] is not None)
        
        outputs = self.dinov3(rgb_image, output_hidden_states=need_all_hidden_states)
        
        # Prepare return dictionary
        result = {}
        
        # Return last hidden state
        if self.config['return_last_hidden_state']:
            last_hidden = outputs.last_hidden_state  # [B, N_tokens, feature_dim]
            if self.config['return_as_feature_maps']:
                last_hidden = self.tokens_to_feature_maps(last_hidden, batch_size, patch_h, patch_w)
            result['last_hidden_state'] = last_hidden
        
        # Return all hidden states
        if self.config['return_all_hidden_states']:
            all_hidden = outputs.hidden_states  # Tuple of [B, N_tokens, feature_dim]
            if self.config['return_as_feature_maps']:
                all_hidden = [self.tokens_to_feature_maps(h, batch_size, patch_h, patch_w) 
                             for h in all_hidden]
            result['all_hidden_states'] = all_hidden
        
        # Return selected hidden states
        if self.config['return_selected_layers'] is not None:
            selected_layers = self.config['return_selected_layers']
            all_hidden = outputs.hidden_states
            selected_hidden = [all_hidden[i] for i in selected_layers]
            if self.config['return_as_feature_maps']:
                selected_hidden = [self.tokens_to_feature_maps(h, batch_size, patch_h, patch_w) 
                                  for h in selected_hidden]
            result['selected_hidden_states'] = selected_hidden
        
        return result
    
    def preprocess_image(self, image_tensor):
        """
        Preprocess image for DINOv3 using the proper processor.
        
        Args:
            image_tensor: [B, 3, H, W] - Raw image tensor with values in [0, 1]
        
        Returns:
            [B, 3, H, W] - Preprocessed tensor ready for DINOv3
        """
        # Convert to PIL images for proper preprocessing
        batch_size = image_tensor.shape[0]
        processed_images = []
        
        for i in range(batch_size):
            # Convert tensor to PIL Image (expecting [0, 1] range)
            img_tensor = image_tensor[i]  # [3, H, W]
            img_tensor = (img_tensor * 255).byte()  # Convert to [0, 255] range
            img_tensor = img_tensor.permute(1, 2, 0)  # [H, W, 3]
            
            # Convert to PIL Image
            import PIL.Image
            img_pil = PIL.Image.fromarray(img_tensor.cpu().numpy(), mode='RGB')
            
            # Process with DINOv3 processor
            processed = self.processor(images=img_pil, return_tensors="pt")
            processed_images.append(processed["pixel_values"])
        
        # Stack all processed images
        processed_batch = torch.cat(processed_images, dim=0)  # [B, 3, H, W]
        
        return processed_batch.to(image_tensor.device)


# class TransformerInpaintingDecoder(nn.Module):
#     """
#     Transformer-based decoder for reconstructing RGB images from DINOv3 features.
#     Supports inpainting by combining global and local attention mechanisms.
#     """
    
#     def __init__(self, config):
#         """
#         Initialize the decoder.
        
#         Args:
#             config: dict containing:
#                 - feature_dim: int, input feature dimension from DINOv3 (default: 768)
#                 - hidden_dim: int, hidden dimension for transformer layers (default: 512)
#                 - num_heads: int, number of attention heads (default: 8)
#                 - num_global_layers: int, number of global transformer layers (default: 4)
#                 - num_local_layers: int, number of local transformer layers (default: 2)
#                 - patch_size: int, patch size used by DINOv3 (default: 16)
#                 - dropout: float, dropout rate (default: 0.1)
#         """
#         super().__init__()
        
#         # Configuration with defaults
#         self.config = {
#             'feature_dim': 768,
#             'hidden_dim': 512, 
#             'num_heads': 8,
#             'num_global_layers': 4,
#             'num_local_layers': 2,
#             'patch_size': 16,
#             'dropout': 0.1,
#             **config
#         }
        
#         feature_dim = self.config['feature_dim']
#         hidden_dim = self.config['hidden_dim']
#         num_heads = self.config['num_heads']
        
#         # Feature projection
#         self.feature_proj = nn.Sequential(
#             nn.Linear(feature_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(self.config['dropout'])
#         )
        
#         # Global transformer - processes all patches to understand overall context
#         global_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 4,
#             dropout=self.config['dropout'],
#             activation='gelu',
#             batch_first=True
#         )
#         self.global_transformer = nn.TransformerEncoder(
#             global_layer, 
#             num_layers=self.config['num_global_layers']
#         )
        
#         # Local transformer - focuses on spatial neighborhoods  
#         local_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 2,
#             dropout=self.config['dropout'],
#             activation='gelu', 
#             batch_first=True
#         )
#         self.local_transformer = nn.TransformerEncoder(
#             local_layer,
#             num_layers=self.config['num_local_layers']
#         )
        
#         # Feature fusion
#         self.fusion = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(self.config['dropout'])
#         )
        
#         # Progressive upsampling decoder
#         self.decoder = nn.Sequential(
#             # First upsampling: patch_res -> patch_res * 2
#             nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
#             nn.BatchNorm2d(hidden_dim // 2),
#             nn.ReLU(inplace=True),
            
#             # Second upsampling: patch_res * 2 -> patch_res * 4
#             nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1),
#             nn.BatchNorm2d(hidden_dim // 4),
#             nn.ReLU(inplace=True),
            
#             # Third upsampling: patch_res * 4 -> patch_res * 8
#             nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, stride=2, padding=1),
#             nn.BatchNorm2d(hidden_dim // 8),
#             nn.ReLU(inplace=True),
            
#             # Fourth upsampling: patch_res * 8 -> patch_res * 16 (full resolution)
#             nn.ConvTranspose2d(hidden_dim // 8, 64, 4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
            
#             # Final RGB output
#             nn.Conv2d(64, 3, 3, padding=1),
#             nn.Sigmoid()  # RGB values [0, 1]
#         )
    
#     def apply_local_attention_windows(self, x, patch_h, patch_w, window_size=7):
#         """
#         Apply local attention in sliding windows for spatial locality.
        
#         Args:
#             x: [B, N_patches, hidden_dim] - Feature tokens
#             patch_h, patch_w: int - Spatial dimensions of patches
#             window_size: int - Size of attention windows
            
#         Returns:
#             [B, N_patches, hidden_dim] - Locally attended features
#         """
#         B, N, D = x.shape
        
#         # Reshape to spatial format
#         x_spatial = x.view(B, patch_h, patch_w, D)  # [B, H, W, D]
        
#         # Pad for windowing if needed
#         pad_h = (window_size - patch_h % window_size) % window_size
#         pad_w = (window_size - patch_w % window_size) % window_size
        
#         if pad_h > 0 or pad_w > 0:
#             x_spatial = F.pad(x_spatial, (0, 0, 0, pad_w, 0, pad_h))
        
#         H_pad, W_pad = x_spatial.shape[1], x_spatial.shape[2]
        
#         # Create windows
#         num_windows_h = H_pad // window_size
#         num_windows_w = W_pad // window_size
        
#         # Reshape into windows: [B, num_windows, window_size^2, D]
#         x_windows = x_spatial.view(
#             B, num_windows_h, window_size, num_windows_w, window_size, D
#         ).permute(0, 1, 3, 2, 4, 5).contiguous()
        
#         x_windows = x_windows.view(
#             B * num_windows_h * num_windows_w, window_size * window_size, D
#         )
        
#         # Apply local transformer to each window
#         x_windows = self.local_transformer(x_windows)
        
#         # Reshape back to spatial format
#         x_windows = x_windows.view(
#             B, num_windows_h, num_windows_w, window_size, window_size, D
#         ).permute(0, 1, 3, 2, 4, 5).contiguous()
        
#         x_local = x_windows.view(B, H_pad, W_pad, D)
        
#         # Remove padding if it was added
#         if pad_h > 0 or pad_w > 0:
#             x_local = x_local[:, :patch_h, :patch_w, :]
        
#         # Reshape back to token format
#         x_local = x_local.view(B, N, D)
        
#         return x_local
    
#     def forward(self, dinov3_features, patch_h, patch_w):
#         """
#         Forward pass for image reconstruction.
        
#         Args:
#             dinov3_features: [B, N_patches, feature_dim] or [B, feature_dim, patch_h, patch_w]
#                            DINOv3 features (patch tokens without CLS/register tokens)
#             patch_h, patch_w: int - Spatial dimensions of the patch grid
            
#         Returns:
#             [B, 3, H, W] - Reconstructed RGB image at full resolution
#         """
#         B = dinov3_features.shape[0]
        
#         # Handle both token and feature map formats
#         if dinov3_features.dim() == 4:  # [B, feature_dim, patch_h, patch_w]
#             # Convert feature maps to tokens
#             x = dinov3_features.flatten(2).transpose(1, 2)  # [B, N_patches, feature_dim]
#         else:  # [B, N_patches, feature_dim] 
#             x = dinov3_features
        
#         # Project features to hidden dimension
#         x = self.feature_proj(x)  # [B, N_patches, hidden_dim]
        
#         # Global attention - understand overall context
#         x_global = self.global_transformer(x)  # [B, N_patches, hidden_dim]
        
#         # Local attention - capture spatial details
#         x_local = self.apply_local_attention_windows(x, patch_h, patch_w)  # [B, N_patches, hidden_dim]
        
#         # Fuse global and local features
#         x_fused = self.fusion(torch.cat([x_global, x_local], dim=-1))  # [B, N_patches, hidden_dim]
        
#         # Reshape to spatial format for convolution
#         x_spatial = x_fused.transpose(1, 2).view(
#             B, self.config['hidden_dim'], patch_h, patch_w
#         )  # [B, hidden_dim, patch_h, patch_w]
        
#         # Progressive upsampling to reconstruct full resolution image
#         reconstructed_image = self.decoder(x_spatial)  # [B, 3, H, W]
        
#         return reconstructed_image


# class DINOv3Inpainter(nn.Module):
#     """
#     Complete inpainting model combining DINOv3 encoder and Transformer decoder.
#     """
    
#     def __init__(self, encoder_config, decoder_config):
#         """
#         Initialize the inpainting model.
        
#         Args:
#             encoder_config: dict - Configuration for DINOv3 encoder
#             decoder_config: dict - Configuration for Transformer decoder
#         """
#         super().__init__()
        
#         # DINOv3 encoder with feature maps output
#         encoder_config = {
#             'return_as_feature_maps': True,
#             'return_last_hidden_state': True,
#             **encoder_config
#         }
#         self.encoder = DINOv3(encoder_config)
        
#         # Transformer decoder 
#         decoder_config = {
#             'feature_dim': self.encoder.feature_dim,
#             'patch_size': self.encoder.patch_size,
#             **decoder_config
#         }
#         self.decoder = TransformerInpaintingDecoder(decoder_config)
    
#     def forward(self, rgb_image):
#         """
#         Forward pass for inpainting.
        
#         Args:
#             rgb_image: [B, 3, H, W] - Input RGB image (preprocessed for DINOv3)
        
#         Returns:
#             [B, 3, H, W] - Inpainted RGB image
#         """
#         B, _, H, W = rgb_image.shape
        
#         # Extract features using DINOv3
#         encoder_outputs = self.encoder(rgb_image)
#         features = encoder_outputs['last_hidden_state']  # [B, feature_dim, patch_h, patch_w]
        
#         # Get patch dimensions
#         patch_h, patch_w = self.encoder.get_patch_spatial_dims(H, W)
        
#         # Reconstruct image using transformer decoder
#         reconstructed = self.decoder(features, patch_h, patch_w)  # [B, 3, H_out, W_out]
        
#         # Resize to match input resolution if needed
#         if reconstructed.shape[-2:] != (H, W):
#             reconstructed = F.interpolate(
#                 reconstructed, size=(H, W), mode='bilinear', align_corners=False
#             )
        
#         return reconstructed


def create_inpainting_model(encoder_config=None, decoder_config=None, device='cuda'):
    """
    Create and initialize the complete inpainting model.
    
    Args:
        encoder_config: dict - DINOv3 encoder configuration
        decoder_config: dict - Transformer decoder configuration  
        device: str - Device to place model on
        
    Returns:
        DINOv3Inpainter model
    """
    if encoder_config is None:
        encoder_config = {}
    if decoder_config is None:
        decoder_config = {}
        
    model = DINOv3Inpainter(encoder_config, decoder_config)
    model = model.to(device)
    return model


def pad_to_patch_size(image_tensor, patch_size=16):
    """
    Pad image tensor to make dimensions divisible by patch_size.
    
    Args:
        image_tensor: [B, 3, H, W] - Input image tensor
        patch_size: int - Patch size (default 16 for DINOv3)
    
    Returns:
        padded_tensor: [B, 3, H_new, W_new] - Padded tensor
        original_size: tuple - Original (H, W) for cropping back
    """
    _, _, h, w = image_tensor.shape
    
    # Calculate padding needed
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    if pad_h > 0 or pad_w > 0:
        # Pad with reflection to avoid border artifacts
        padded = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        return padded, (h, w)
    else:
        return image_tensor, (h, w)
    
    import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


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
        readout_type: str = "ignore"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.readout_type = readout_type
        
        # Readout projection if using "project" method
        if readout_type == "project":
            self.readout_project = nn.Sequential(
                nn.Linear(2 * in_channels, in_channels),
                nn.GELU()
            )
        
        # Channel projection
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Spatial resampling based on scale factor
        if scale_factor == 4.0:
            # 4x upsampling: 24x24 -> 96x96
            self.resample = nn.ConvTranspose2d(
                out_channels, out_channels,
                kernel_size=8, stride=4, padding=2, bias=True
            )
        elif scale_factor == 2.0:
            # 2x upsampling: 24x24 -> 48x48
            self.resample = nn.ConvTranspose2d(
                out_channels, out_channels,
                kernel_size=4, stride=2, padding=1, bias=True
            )
        elif scale_factor == 1.0:
            # No resampling: 24x24 -> 24x24
            self.resample = nn.Identity()
        elif scale_factor == 0.5:
            # 0.5x downsampling: 24x24 -> 12x12
            self.resample = nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, stride=2, padding=1, bias=True
            )
    
    def forward(self, hidden_state: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        """
        Args:
            hidden_state: [B, N_tokens, feature_dim] - includes CLS + register + patch tokens
            patch_h, patch_w: Spatial dimensions of patches
        Returns:
            [B, out_channels, H', W'] - Spatial feature map
        """
        batch_size = hidden_state.shape[0]
        
        # For DINOv3: Remove CLS token (index 0) and register tokens (indices 1-4)
        # Keep only patch tokens starting from index 5
        patch_tokens = hidden_state[:, 5:, :]  # [B, patch_h*patch_w, feature_dim]
        
        # Handle readout token integration
        if self.readout_type == "project":
            # Get CLS token and expand to all spatial positions
            readout = hidden_state[:, 0:1, :].expand_as(patch_tokens)  # [B, patch_h*patch_w, feature_dim]
            # Concatenate and project
            patch_tokens = torch.cat([patch_tokens, readout], dim=-1)  # [B, patch_h*patch_w, 2*feature_dim]
            patch_tokens = self.readout_project(patch_tokens)  # [B, patch_h*patch_w, feature_dim]
        elif self.readout_type == "add":
            # Add CLS token to all patch tokens
            readout = hidden_state[:, 0:1, :]  # [B, 1, feature_dim]
            patch_tokens = patch_tokens + readout
        
        # Reshape to spatial format
        patch_tokens = patch_tokens.transpose(1, 2)  # [B, feature_dim, patch_h*patch_w]
        patch_tokens = patch_tokens.reshape(batch_size, self.in_channels, patch_h, patch_w)  # [B, feature_dim, patch_h, patch_w]
        
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
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        use_bn: bool = False
    ):
        super().__init__()
        
        self.residual_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not use_bn)
        self.residual_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
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


class DPTRGBDecoder(nn.Module):
    """
    DPT decoder adapted for RGB output from DINOv3 features.
    Implements multi-scale reassembly, progressive fusion, and RGB prediction head.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        super().__init__()
        
        # Default configuration
        default_config = {
            'feature_dim': 768,  # DINOv3 ViT-B feature dimension
            'reassemble_out_channels': [96, 192, 384, 768],  # Neck hidden sizes
            'reassemble_factors': [4.0, 2.0, 1.0, 0.5],  # Spatial scale factors
            'fusion_hidden_size': 256,
            'readout_type': 'ignore',  # 'ignore', 'add', or 'project'
            'use_bn': False,
            'output_image_size': (448,448),  # If None, maintains input size
        }
        
        self.config = {**default_config, **(config or {})}
        self.out_image_size = self.config['output_image_size']
        # Create reassemble layers for multi-scale feature extraction
        self.reassemble_layers = nn.ModuleList([
            DPTReassembleLayer(
                in_channels=self.config['feature_dim'],
                out_channels=out_ch,
                scale_factor=scale,
                readout_type=self.config['readout_type']
            )
            for out_ch, scale in zip(
                self.config['reassemble_out_channels'],
                self.config['reassemble_factors']
            )
        ])
        
        # Create fusion blocks for progressive feature combination
        fusion_in_channels = self.config['reassemble_out_channels']
        self.fusion_blocks = nn.ModuleList([
            DPTFeatureFusionBlock(
                in_channels=ch,
                out_channels=self.config['fusion_hidden_size'],
                use_bn=self.config['use_bn']
            )
            for ch in fusion_in_channels
        ])
        
        # RGB prediction head
        self.rgb_head = nn.Sequential(
            # First stage: 256 -> 128 channels with spatial refinement
            nn.Conv2d(self.config['fusion_hidden_size'], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if self.config['use_bn'] else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 192x192 -> 384x384
            
            # Second stage: 128 -> 64 channels with feature refinement
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if self.config['use_bn'] else nn.Identity(),
            nn.ReLU(inplace=True),
            
            # Third stage: 64 -> 32 channels
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if self.config['use_bn'] else nn.Identity(),
            nn.ReLU(inplace=True),
            
            # Final RGB projection
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1] range
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
        for i, (hidden_state, reassemble) in enumerate(zip(hidden_states, self.reassemble_layers)):
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
        fused = F.interpolate(fused, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 256, 24, 24]
        
        # Add stage 2 features
        fused = fused + self.fusion_blocks[2](reassembled_features[2])  # [B, 256, 24, 24]
        fused = F.interpolate(fused, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 256, 48, 48]
        
        # Add stage 1 features
        fused = fused + self.fusion_blocks[1](reassembled_features[1])  # [B, 256, 48, 48]
        fused = F.interpolate(fused, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 256, 96, 96]
        
        # Add stage 0 features
        fused = fused + self.fusion_blocks[0](reassembled_features[0])  # [B, 256, 96, 96]
        fused = F.interpolate(fused, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 256, 192, 192]
        
        # Apply RGB head
        rgb_output = self.rgb_head(fused)  # [B, 3, 384, 384]
        
        # Resize to original input size if needed
        if self.config['output_image_size'] or (rgb_output.shape[-2:] != (input_height, input_width)):
            target_size = self.config['output_image_size'] or (input_height, input_width)
            rgb_output = F.interpolate(
                rgb_output,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        
        return rgb_output


class DINOv3toDPTRGB(nn.Module):
    """
    Complete model combining DINOv3 encoder with DPT RGB decoder.
    Compatible with the provided DINOv3 class.
    """
    
    def __init__(
        self,
        dinov3_model,
        decoder_config: Optional[Dict] = None,
        selected_layers: List[int] = [2, 5, 8, 11]
    ):
        super().__init__()
        
        self.dinov3 = dinov3_model
        self.selected_layers = selected_layers
        
        # Configure DINOv3 to return selected hidden states
        self.dinov3.config['return_selected_layers'] = selected_layers
        self.dinov3.config['return_as_feature_maps'] = False  # We need tokens for reassembly
        
        # Initialize decoder with DINOv3's feature dimension
        decoder_config = decoder_config or {}
        decoder_config['feature_dim'] = self.dinov3.feature_dim
        
        self.decoder = DPTRGBDecoder(decoder_config)
    
    def forward(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_image: [B, 3, H, W] - Input RGB image (preprocessed for DINOv3)
        
        Returns:
            rgb_output: [B, 3, H, W] - Predicted RGB image in [0, 1] range
        """
        batch_size, _, input_h, input_w = rgb_image.shape
        
        # Get DINOv3 features from selected layers
        dinov3_output = self.dinov3(rgb_image)
        hidden_states = dinov3_output['selected_hidden_states']  # List of [B, N_tokens, feature_dim]
        
        # Pass through DPT decoder
        rgb_output = self.decoder(hidden_states, input_h, input_w)
        
        return rgb_output
    
    def freeze_encoder(self):
        """Freeze DINOv3 encoder parameters for efficient fine-tuning."""
        for param in self.dinov3.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze DINOv3 encoder parameters for full fine-tuning."""
        for param in self.dinov3.parameters():
            param.requires_grad = True
    
    def get_parameter_groups(self, base_lr: float = 1e-4):
        """
        Get parameter groups with differential learning rates.
        
        Args:
            base_lr: Base learning rate for the RGB head
        
        Returns:
            List of parameter groups for optimizer
        """
        return [
            {'params': self.dinov3.parameters(), 'lr': base_lr * 0.1},  # Encoder: 10% of base LR
            {'params': self.decoder.reassemble_layers.parameters(), 'lr': base_lr * 0.5},  # Reassemble: 50%
            {'params': self.decoder.fusion_blocks.parameters(), 'lr': base_lr * 0.5},  # Fusion: 50%
            {'params': self.decoder.rgb_head.parameters(), 'lr': base_lr}  # RGB head: 100%
        ]
        
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 1) POL preprocessing ----------
class PolarizationPreprocess(nn.Module):
    """
    Input:
      aolp: (B,1,H,W) in radians, typically [0, pi)
      dolp: (B,1,H,W) in [0,1]
    Output:
      (B,3,H,W) = [cos(2*AoLP), sin(2*AoLP), DoLP]
    """
    def forward(self, aolp: torch.Tensor, dolp: torch.Tensor) -> torch.Tensor:
        cos2 = torch.cos(2.0 * aolp)
        sin2 = torch.sin(2.0 * aolp)
        return torch.cat([cos2, sin2, dolp.clamp(0,1)], dim=1)

# ---------- 2) Tiny ViT-style POL encoder ----------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):           # x: (B,C,H,W)
        x = self.proj(x)            # (B,embed_dim,H/P,W/P)
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
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
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
            'in_ch': 3,
            'embed_dim': 768,
            'depth': 4,
            'n_heads': 12,
            'patch_size': 16,
            'drop': 0.0
        }
        
        self.config = {**default_config, **(config or {})}
        
        self.patch = PatchEmbed(
            self.config['in_ch'], 
            self.config['embed_dim'], 
            self.config['patch_size']
        )
        self.pos = None  # initialized at first forward pass
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.config['embed_dim'], 
                self.config['n_heads'], 
                self.config['drop']
            ) for _ in range(self.config['depth'])
        ])
        self.norm = nn.LayerNorm(self.config['embed_dim'])

    def forward(self, x):  # x: (B,3,H,W)
        x, (Hp, Wp) = self.patch(x)  # (B,N,C)
        # learnable 2D pos embedding (initialized on first run to match N)
        N = x.shape[1]
        if (self.pos is None) or (self.pos.shape[1] != N):
            self.pos = nn.Parameter(torch.zeros(1, N, x.shape[2], device=x.device))
            nn.init.trunc_normal_(self.pos, std=0.02)
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)  # (B,N,C)

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
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=4.0, drop=drop)

    def forward(self, x_q, x_kv, attn_mask=None):
        q = self.norm_q(x_q)
        kv = self.norm_kv(x_kv)
        x, _ = self.cross_attn(q, kv, kv, attn_mask=attn_mask, need_weights=False)
        x = x_q + x                      # residual after attention
        x = x + self.mlp(self.norm_out(x))  # residual after MLP
        return x

# ---------- 4) Simple fusion wrapper ----------
class RGBPOLCrossFuse(nn.Module):
    """
    One-way (RGB<-POL) or two-way (bi-directional) fusion.
    If bi_directional=True, returns concat([RGB_fused, POL_fused]) projected back to dim.
    """
    def __init__(self, dim=768, n_heads=12, drop=0.0, bi_directional=False):
        super().__init__()
        self.rgb_from_pol = CrossAttentionBlock(dim, n_heads, drop)
        self.bi = bi_directional
        if bi_directional:
            self.pol_from_rgb = CrossAttentionBlock(dim, n_heads, drop)
            self.proj = nn.Linear(2*dim, dim)

    def forward(self, rgb_tokens, pol_tokens, attn_mask=None):
        rgb_fused = self.rgb_from_pol(rgb_tokens, pol_tokens, attn_mask)   # Q=RGB, K/V=POL
        if not self.bi:
            return rgb_fused
        pol_fused = self.pol_from_rgb(pol_tokens, rgb_tokens, attn_mask)   # Q=POL, K/V=RGB
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
        pol_encoder=None,                     # POLViTEncoder instance or dict
        pol_preprocess=None,                  # PolarizationPreprocess instance or dict
        pol_cross_attn=None,                  # RGBPOLCrossFuse instance or dict

        # 3) Decoders — three DPTRGBDecoder instances or dict configs
        spec_decoder=None,                    # DPTRGBDecoder instance or dict
        diffuse_decoder=None,                 # DPTRGBDecoder instance or dict
        highlight_decoder=None,               # DPTRGBDecoder instance or dict

        # Optional: if your DINO wrapper needs these hints
        image_size: int = 896,
        patch_size: int = 16,
    ):
        super().__init__()

        # ---- RGB (DINOv3) ----
        # Accept either an instance or a DINOv3(**cfg) dict
        self.dinov3 = _build(dinov3, DINOv3)

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = self.dinov3.feature_dim

        # ---- POL branch ----
        # Preprocess (AoLP, DoLP) → [cos2θ, sin2θ, DoLP]
        if pol_preprocess is None:
            self.pol_pre = PolarizationPreprocess()
        else:
            self.pol_pre = _build(pol_preprocess, PolarizationPreprocess)

        # Encoder (ViT-like), align dim/patch with DINO
        if pol_encoder is None:
            self.pol_enc = POLViTEncoder(in_ch=3, embed_dim=self.embed_dim, depth=4, n_heads=12, patch_size=patch_size)
        else:
            self.pol_enc = _build(pol_encoder, POLViTEncoder)

        # Cross-attention: Q=RGB, K/V=POL
        if pol_cross_attn is None:
            self.cross = RGBPOLCrossFuse(dim=self.embed_dim, n_heads=12, drop=0.0, bi_directional=False)
        else:
            self.cross = _build(pol_cross_attn, RGBPOLCrossFuse)

        # ---- Decoders (DPTRGBDecoder) ----
        # Each can control its own out_channels inside the config (e.g., 3 for S/D, 1 for H)
        if spec_decoder is None:
            # Minimal default: your DPTRGBDecoder likely needs at least decoder_config / dim / image_size
            spec_decoder = {"decoder_config": {"use_bn": True, "readout_type": "project"},
                            "embed_dim": self.embed_dim, "image_size": image_size}
        if diffuse_decoder is None:
            diffuse_decoder = {"decoder_config": {"use_bn": True, "readout_type": "project"},
                               "embed_dim": self.embed_dim, "image_size": image_size}
        if highlight_decoder is None:
            # Often a 1-channel mask is useful; set out_channels=1 if your DPTRGBDecoder supports it.
            highlight_decoder = {"decoder_config": {"use_bn": True, "readout_type": "project", "out_channels": 1},
                                 "embed_dim": self.embed_dim, "image_size": image_size}

        # Normalize configs → instances
        def build_dpt(dec):
            if isinstance(dec, DPTRGBDecoder):
                return dec
            if isinstance(dec, dict):
                # DPTRGBDecoder takes a single config dict
                config = {
                    'feature_dim': self.embed_dim,
                    **dec.get("decoder_config", {})
                }
                return DPTRGBDecoder(config)
            raise TypeError("Decoder must be DPTRGBDecoder instance or dict.")

        self.decS = build_dpt(spec_decoder)
        self.decD = build_dpt(diffuse_decoder)
        self.decH = build_dpt(highlight_decoder)

    def _rgb_tokens(self, rgb_preproc):
        """Extract DINOv3 tokens and infer (Hp, Wp) if wrapper doesn’t return them."""
        out = self.dinov3(rgb_preproc)
        tokens = out.get("last_hidden_state", out.get("tokens"))
        if tokens is None:
            raise KeyError("DINOv3 wrapper must return 'last_hidden_state' or 'tokens'.")
        Hp = self.image_size // self.patch_size
        Wp = self.image_size // self.patch_size
        return tokens, (Hp, Wp)

    def forward(self, batch):
        # 1) RGB → DINO tokens
        rgb_in = self.dinov3.preprocess_image(batch["rgb"])
        rgb_tokens, grid_hw = self._rgb_tokens(rgb_in)         # (B,N,C), (Hp,Wp)

        # 2) POL → preprocess → POL tokens
        pol_in = self.pol_pre(batch["AoP"], batch["DoP"])      # (B,3,H,W)
        pol_tokens = self.pol_enc(pol_in)                      # (B,N,C)

        # 3) CROSS (Q=RGB, K/V=POL)
        cross_tokens = self.cross(rgb_tokens, pol_tokens)      # (B,N,C)
        # 5) DPT decoders expect multi-scale hidden states
        # For now, use the same fused tokens for all scales (this can be improved)
        hidden_states = [cross_tokens] * 4  # DPT expects 4 scale levels
        
        # 6) Decode with three DPTRGBDecoder heads
        S = self.decS(hidden_states)      # Specular  (B,3,H,W)
        D = self.decD(hidden_states)      # Diffuse   (B,3,H,W)
        H = self.decH(hidden_states)      # Highlight (B,3,H,W)

        return {
            "specular": S,
            "diffuse": D,
            "highlight": H,
            "tokens": {"rgb": rgb_tokens, "pol": pol_tokens, "cross": cross_tokens}
        }
