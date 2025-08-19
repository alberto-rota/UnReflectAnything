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


class TransformerInpaintingDecoder(nn.Module):
    """
    Transformer-based decoder for reconstructing RGB images from DINOv3 features.
    Supports inpainting by combining global and local attention mechanisms.
    """
    
    def __init__(self, config):
        """
        Initialize the decoder.
        
        Args:
            config: dict containing:
                - feature_dim: int, input feature dimension from DINOv3 (default: 768)
                - hidden_dim: int, hidden dimension for transformer layers (default: 512)
                - num_heads: int, number of attention heads (default: 8)
                - num_global_layers: int, number of global transformer layers (default: 4)
                - num_local_layers: int, number of local transformer layers (default: 2)
                - patch_size: int, patch size used by DINOv3 (default: 16)
                - dropout: float, dropout rate (default: 0.1)
        """
        super().__init__()
        
        # Configuration with defaults
        self.config = {
            'feature_dim': 768,
            'hidden_dim': 512, 
            'num_heads': 8,
            'num_global_layers': 4,
            'num_local_layers': 2,
            'patch_size': 16,
            'dropout': 0.1,
            **config
        }
        
        feature_dim = self.config['feature_dim']
        hidden_dim = self.config['hidden_dim']
        num_heads = self.config['num_heads']
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config['dropout'])
        )
        
        # Global transformer - processes all patches to understand overall context
        global_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=self.config['dropout'],
            activation='gelu',
            batch_first=True
        )
        self.global_transformer = nn.TransformerEncoder(
            global_layer, 
            num_layers=self.config['num_global_layers']
        )
        
        # Local transformer - focuses on spatial neighborhoods  
        local_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=self.config['dropout'],
            activation='gelu', 
            batch_first=True
        )
        self.local_transformer = nn.TransformerEncoder(
            local_layer,
            num_layers=self.config['num_local_layers']
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config['dropout'])
        )
        
        # Progressive upsampling decoder
        self.decoder = nn.Sequential(
            # First upsampling: patch_res -> patch_res * 2
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            # Second upsampling: patch_res * 2 -> patch_res * 4
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            
            # Third upsampling: patch_res * 4 -> patch_res * 8
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True),
            
            # Fourth upsampling: patch_res * 8 -> patch_res * 16 (full resolution)
            nn.ConvTranspose2d(hidden_dim // 8, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final RGB output
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()  # RGB values [0, 1]
        )
    
    def apply_local_attention_windows(self, x, patch_h, patch_w, window_size=7):
        """
        Apply local attention in sliding windows for spatial locality.
        
        Args:
            x: [B, N_patches, hidden_dim] - Feature tokens
            patch_h, patch_w: int - Spatial dimensions of patches
            window_size: int - Size of attention windows
            
        Returns:
            [B, N_patches, hidden_dim] - Locally attended features
        """
        B, N, D = x.shape
        
        # Reshape to spatial format
        x_spatial = x.view(B, patch_h, patch_w, D)  # [B, H, W, D]
        
        # Pad for windowing if needed
        pad_h = (window_size - patch_h % window_size) % window_size
        pad_w = (window_size - patch_w % window_size) % window_size
        
        if pad_h > 0 or pad_w > 0:
            x_spatial = F.pad(x_spatial, (0, 0, 0, pad_w, 0, pad_h))
        
        H_pad, W_pad = x_spatial.shape[1], x_spatial.shape[2]
        
        # Create windows
        num_windows_h = H_pad // window_size
        num_windows_w = W_pad // window_size
        
        # Reshape into windows: [B, num_windows, window_size^2, D]
        x_windows = x_spatial.view(
            B, num_windows_h, window_size, num_windows_w, window_size, D
        ).permute(0, 1, 3, 2, 4, 5).contiguous()
        
        x_windows = x_windows.view(
            B * num_windows_h * num_windows_w, window_size * window_size, D
        )
        
        # Apply local transformer to each window
        x_windows = self.local_transformer(x_windows)
        
        # Reshape back to spatial format
        x_windows = x_windows.view(
            B, num_windows_h, num_windows_w, window_size, window_size, D
        ).permute(0, 1, 3, 2, 4, 5).contiguous()
        
        x_local = x_windows.view(B, H_pad, W_pad, D)
        
        # Remove padding if it was added
        if pad_h > 0 or pad_w > 0:
            x_local = x_local[:, :patch_h, :patch_w, :]
        
        # Reshape back to token format
        x_local = x_local.view(B, N, D)
        
        return x_local
    
    def forward(self, dinov3_features, patch_h, patch_w):
        """
        Forward pass for image reconstruction.
        
        Args:
            dinov3_features: [B, N_patches, feature_dim] or [B, feature_dim, patch_h, patch_w]
                           DINOv3 features (patch tokens without CLS/register tokens)
            patch_h, patch_w: int - Spatial dimensions of the patch grid
            
        Returns:
            [B, 3, H, W] - Reconstructed RGB image at full resolution
        """
        B = dinov3_features.shape[0]
        
        # Handle both token and feature map formats
        if dinov3_features.dim() == 4:  # [B, feature_dim, patch_h, patch_w]
            # Convert feature maps to tokens
            x = dinov3_features.flatten(2).transpose(1, 2)  # [B, N_patches, feature_dim]
        else:  # [B, N_patches, feature_dim] 
            x = dinov3_features
        
        # Project features to hidden dimension
        x = self.feature_proj(x)  # [B, N_patches, hidden_dim]
        
        # Global attention - understand overall context
        x_global = self.global_transformer(x)  # [B, N_patches, hidden_dim]
        
        # Local attention - capture spatial details
        x_local = self.apply_local_attention_windows(x, patch_h, patch_w)  # [B, N_patches, hidden_dim]
        
        # Fuse global and local features
        x_fused = self.fusion(torch.cat([x_global, x_local], dim=-1))  # [B, N_patches, hidden_dim]
        
        # Reshape to spatial format for convolution
        x_spatial = x_fused.transpose(1, 2).view(
            B, self.config['hidden_dim'], patch_h, patch_w
        )  # [B, hidden_dim, patch_h, patch_w]
        
        # Progressive upsampling to reconstruct full resolution image
        reconstructed_image = self.decoder(x_spatial)  # [B, 3, H, W]
        
        return reconstructed_image


class DINOv3Inpainter(nn.Module):
    """
    Complete inpainting model combining DINOv3 encoder and Transformer decoder.
    """
    
    def __init__(self, encoder_config, decoder_config):
        """
        Initialize the inpainting model.
        
        Args:
            encoder_config: dict - Configuration for DINOv3 encoder
            decoder_config: dict - Configuration for Transformer decoder
        """
        super().__init__()
        
        # DINOv3 encoder with feature maps output
        encoder_config = {
            'return_as_feature_maps': True,
            'return_last_hidden_state': True,
            **encoder_config
        }
        self.encoder = DINOv3(encoder_config)
        
        # Transformer decoder 
        decoder_config = {
            'feature_dim': self.encoder.feature_dim,
            'patch_size': self.encoder.patch_size,
            **decoder_config
        }
        self.decoder = TransformerInpaintingDecoder(decoder_config)
    
    def forward(self, rgb_image):
        """
        Forward pass for inpainting.
        
        Args:
            rgb_image: [B, 3, H, W] - Input RGB image (preprocessed for DINOv3)
        
        Returns:
            [B, 3, H, W] - Inpainted RGB image
        """
        B, _, H, W = rgb_image.shape
        
        # Extract features using DINOv3
        encoder_outputs = self.encoder(rgb_image)
        features = encoder_outputs['last_hidden_state']  # [B, feature_dim, patch_h, patch_w]
        
        # Get patch dimensions
        patch_h, patch_w = self.encoder.get_patch_spatial_dims(H, W)
        
        # Reconstruct image using transformer decoder
        reconstructed = self.decoder(features, patch_h, patch_w)  # [B, 3, H_out, W_out]
        
        # Resize to match input resolution if needed
        if reconstructed.shape[-2:] != (H, W):
            reconstructed = F.interpolate(
                reconstructed, size=(H, W), mode='bilinear', align_corners=False
            )
        
        return reconstructed


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