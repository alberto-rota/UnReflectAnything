"""
Unit tests for RGBPOLDecomposer model with shape verification.
Tests loading from YAML config and verifying intermediate tensor shapes.
"""

import pytest
import torch
import yaml
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import RGBPOLDecomposer, DINOv3, POLViTEncoder, DPTRGBDecoder
from main import create_model_from_config, load_and_process_config


class TestRGBPOLDecomposer:
    """Test suite for RGBPOLDecomposer model."""
    
    @pytest.fixture(scope="class")
    def config_path(self):
        """Path to test configuration file."""
        return "config_train.yaml"
    
    @pytest.fixture(scope="class")
    def config(self, config_path):
        """Load and process configuration from YAML."""
        config = load_and_process_config(config_path=config_path)
        return config
    
    @pytest.fixture(scope="class")
    def device(self):
        """Get device for model creation."""
        return torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture(scope="class")
    def model(self, config, device):
        """Create model from configuration."""
        model = create_model_from_config(config, device)
        model.eval()
        return model
    
    @pytest.fixture
    def batch_size(self):
        """Batch size for testing."""
        return 2
    
    @pytest.fixture
    def image_size(self, config):
        """Get image size from config."""
        model_config = config.get("MODEL", {})
        return model_config.get("RGB_ENCODER", {}).get("IMAGE_SIZE", 224)
    
    @pytest.fixture
    def patch_size(self, config):
        """Get patch size from config."""
        model_config = config.get("MODEL", {})
        return model_config.get("POL_ENCODER", {}).get("PATCH_SIZE", 16)
    
    @pytest.fixture
    def embed_dim(self, config):
        """Get embedding dimension from config."""
        model_config = config.get("MODEL", {})
        return model_config.get("POL_ENCODER", {}).get("EMBED_DIM", 384)
    
    @pytest.fixture
    def selected_layers(self, config):
        """Get selected layers from config."""
        model_config = config.get("MODEL", {})
        return model_config.get("RGB_ENCODER", {}).get("RETURN_SELECTED_LAYERS", [3, 6, 9, 12])
    
    @pytest.fixture
    def sample_batch(self, batch_size, image_size, device):
        """Create a sample batch for testing."""
        batch = {
            "rgb": torch.randn(batch_size, 3, image_size, image_size, device=device),
            "AoP": torch.rand(batch_size, 1, image_size, image_size, device=device) * 3.14159,  # [0, π)
            "DoP": torch.rand(batch_size, 1, image_size, image_size, device=device),  # [0, 1]
        }
        return batch
    
    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert isinstance(model, RGBPOLDecomposer)
        assert hasattr(model, 'dinov3')
        assert hasattr(model, 'pol_enc')
        assert hasattr(model, 'pol_pre')
        assert hasattr(model, 'cross')
        assert hasattr(model, 'decS')
        assert hasattr(model, 'decD')
        assert hasattr(model, 'decH')
    
    def test_dinov3_config(self, model, image_size, selected_layers):
        """Test DINOv3 configuration matches expected values."""
        assert model.dinov3.config["image_size"] == image_size
        assert model.dinov3.config["return_selected_layers"] == selected_layers
        assert model.dinov3.config["return_as_feature_maps"] == False
        # DINOv3 ViT-S/16 has 384 hidden dim
        assert model.dinov3.feature_dim == 384
        assert model.dinov3.patch_size == 16
    
    def test_pol_encoder_config(self, model, embed_dim, patch_size):
        """Test POL encoder configuration."""
        assert model.pol_enc.config["embed_dim"] == embed_dim
        assert model.pol_enc.config["patch_size"] == patch_size
        assert model.pol_enc.config["in_ch"] == 3  # cos(2θ), sin(2θ), DoLP
        assert model.pol_enc.config["depth"] == 4
        assert model.pol_enc.config["n_heads"] == 12
    
    def test_cross_attention_modules(self, model, selected_layers):
        """Test cross-attention modules match selected layers."""
        assert isinstance(model.cross, torch.nn.ModuleList)
        assert len(model.cross) == len(selected_layers)
        for cross_module in model.cross:
            assert hasattr(cross_module, 'rgb_from_pol')
            assert cross_module.config["embed_dim"] == 384
            assert cross_module.config["n_heads"] == 12
    
    def test_decoder_configuration(self, model):
        """Test decoder configurations."""
        for decoder_name in ['decS', 'decD', 'decH']:
            decoder = getattr(model, decoder_name)
            assert isinstance(decoder, DPTRGBDecoder)
            assert decoder.config["feature_dim"] == 384
            assert len(decoder.reassemble_layers) == 4
            assert len(decoder.fusion_blocks) == 4
    
    def test_forward_pass_output_keys(self, model, sample_batch):
        """Test that forward pass returns expected dictionary keys."""
        with torch.no_grad():
            output = model(sample_batch)
        
        expected_keys = {"specular", "diffuse", "highlight", "rgb_tokens", "pol_tokens", "cross_tokens"}
        assert set(output.keys()) == expected_keys
    
    def test_rgb_token_shapes(self, model, sample_batch, batch_size, image_size, patch_size, embed_dim, selected_layers):
        """Test RGB token shapes from DINOv3."""
        with torch.no_grad():
            output = model(sample_batch)
        
        rgb_tokens = output["rgb_tokens"]
        assert isinstance(rgb_tokens, list)
        assert len(rgb_tokens) == len(selected_layers)
        
        # Calculate expected number of patches
        n_patches_h = n_patches_w = image_size // patch_size  # 224 / 16 = 14
        n_patches = n_patches_h * n_patches_w  # 14 * 14 = 196
        
        # DINOv3 tokens include: 1 CLS + 4 register + n_patches patch tokens
        expected_n_tokens = 1 + 4 + n_patches  # 1 + 4 + 196 = 201
        
        for i, tokens in enumerate(rgb_tokens):
            assert tokens.shape == (batch_size, expected_n_tokens -5, embed_dim), \
                f"RGB tokens at layer {i} have shape {tokens.shape}, expected {(batch_size, expected_n_tokens, embed_dim)}"
    
    def test_pol_token_shapes(self, model, sample_batch, batch_size, image_size, patch_size, embed_dim, selected_layers):
        """Test POL token shapes from POL encoder."""
        with torch.no_grad():
            output = model(sample_batch)
        
        pol_tokens = output["pol_tokens"]
        assert isinstance(pol_tokens, list)
        assert len(pol_tokens) == 4  # POL encoder has 4 layers (depth=4)
        
        # POL encoder outputs patch tokens only (no CLS/register tokens)
        n_patches_h = n_patches_w = image_size // patch_size  # 224 / 16 = 14
        n_patches = n_patches_h * n_patches_w  # 14 * 14 = 196
        
        for i, tokens in enumerate(pol_tokens):
            assert tokens.shape == (batch_size, n_patches, embed_dim), \
                f"POL tokens at layer {i} have shape {tokens.shape}, expected {(batch_size, n_patches, embed_dim)}"
    
    def test_cross_token_shapes(self, model, sample_batch, batch_size, image_size, patch_size, embed_dim, selected_layers):
        """Test cross-attention token shapes."""
        with torch.no_grad():
            output = model(sample_batch)
        
        cross_tokens = output["cross_tokens"]
        assert isinstance(cross_tokens, list)
        assert len(cross_tokens) == len(selected_layers)
        
        # Cross tokens should have same shape as RGB tokens (output of cross-attention)
        n_patches_h = n_patches_w = image_size // patch_size
        n_patches = n_patches_h * n_patches_w
        expected_n_tokens = 1 + 4 + n_patches  # Same as RGB tokens
        
        for i, tokens in enumerate(cross_tokens):
            assert tokens.shape == (batch_size, expected_n_tokens -5, embed_dim), \
                f"Cross tokens at layer {i} have shape {tokens.shape}, expected {(batch_size, expected_n_tokens, embed_dim)}"
    
    def test_output_image_shapes(self, model, sample_batch, batch_size, config):
        """Test output image tensor shapes."""
        with torch.no_grad():
            output = model(sample_batch)
        
        # Get output image size from decoder config
        model_config = config.get("MODEL", {})
        output_size = model_config.get("DECODER", {}).get("OUTPUT_IMAGE_SIZE", 224)
        if isinstance(output_size, list):
            output_h, output_w = output_size
        else:
            output_h = output_w = output_size
        
        # Test specular output
        assert output["specular"].shape == (batch_size, 3, output_h, output_w), \
            f"Specular shape {output['specular'].shape}, expected {(batch_size, 3, output_h, output_w)}"
        
        # Test diffuse output  
        assert output["diffuse"].shape == (batch_size, 3, output_h, output_w), \
            f"Diffuse shape {output['diffuse'].shape}, expected {(batch_size, 3, output_h, output_w)}"
        
        # Test highlight output (can be 1 or 3 channels depending on config)
        h_channels = output["highlight"].shape[1]
        assert h_channels in [1, 3], f"Highlight channels {h_channels} not in [1, 3]"
        assert output["highlight"].shape[0] == batch_size
        assert output["highlight"].shape[2:] == (output_h, output_w)
    
    def test_output_value_ranges(self, model, sample_batch):
        """Test that output values are in expected ranges."""
        with torch.no_grad():
            output = model(sample_batch)
        
        # RGB outputs should be in [0, 1] due to sigmoid activation
        for key in ["specular", "diffuse", "highlight"]:
            tensor = output[key]
            assert tensor.min() >= 0.0, f"{key} has values below 0: min={tensor.min()}"
            assert tensor.max() <= 1.0, f"{key} has values above 1: max={tensor.max()}"
    
    def test_pol_preprocessing(self, model, batch_size, image_size, device):
        """Test polarization preprocessing."""
        aolp = torch.rand(batch_size, 1, image_size, image_size, device=device) * 3.14159
        dolp = torch.rand(batch_size, 1, image_size, image_size, device=device)
        
        with torch.no_grad():
            pol_preprocessed = model.pol_pre(aolp, dolp)
        
        # Should output 3 channels: cos(2θ), sin(2θ), DoLP
        assert pol_preprocessed.shape == (batch_size, 3, image_size, image_size)
        
        # Check that cos and sin are in [-1, 1]
        assert pol_preprocessed[:, 0, :, :].min() >= -1.0  # cos(2θ)
        assert pol_preprocessed[:, 0, :, :].max() <= 1.0
        assert pol_preprocessed[:, 1, :, :].min() >= -1.0  # sin(2θ)
        assert pol_preprocessed[:, 1, :, :].max() <= 1.0
        
        # DoLP should be in [0, 1]
        assert pol_preprocessed[:, 2, :, :].min() >= 0.0
        assert pol_preprocessed[:, 2, :, :].max() <= 1.0
    
    def test_gradient_flow(self, model, sample_batch):
        """Test that gradients can flow through the model."""
        model.train()
        
        # Ensure at least decoder parameters require gradients
        for param in model.decS.parameters():
            param.requires_grad = True
        
        output = model(sample_batch)
        loss = output["specular"].mean() + output["diffuse"].mean() + output["highlight"].mean()
        loss.backward()
        
        # Check that at least some decoder parameters have gradients
        has_gradient = False
        for param in model.decS.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradient = True
                break
        
        assert has_gradient, "No gradients found in decoder parameters"
        
        # Clean up gradients
        model.zero_grad()
        model.eval()
    
    def test_reassemble_layer_scales(self, model):
        """Test reassemble layer configurations match expected scales."""
        expected_scales = [4.0, 2.0, 1.0, 0.5]
        expected_out_channels = [48, 96, 192, 384]
        
        for decoder_name in ['decS', 'decD', 'decH']:
            decoder = getattr(model, decoder_name)
            assert len(decoder.reassemble_layers) == len(expected_scales)
            
            for i, layer in enumerate(decoder.reassemble_layers):
                assert layer.scale_factor == expected_scales[i]
                assert layer.out_channels == expected_out_channels[i]
    
    def test_memory_efficiency(self, model, sample_batch):
        """Test model memory efficiency with no_grad context."""
        import gc
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # Get initial memory if CUDA available
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        # Run inference
        with torch.no_grad():
            output = model(sample_batch)
        
        # Check memory usage didn't explode
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_mb = (peak_memory - initial_memory) / (1024 * 1024)
            assert memory_mb < 4000, f"Model used {memory_mb:.2f} MB, which seems excessive"
    
    def test_deterministic_output(self, model, sample_batch):
        """Test that model produces deterministic outputs in eval mode."""
        model.eval()
        
        with torch.no_grad():
            output1 = model(sample_batch)
            output2 = model(sample_batch)
        
        for key in ["specular", "diffuse", "highlight"]:
            assert torch.allclose(output1[key], output2[key], atol=1e-6), \
                f"Non-deterministic output for {key}"


# Additional integration tests
class TestConfigIntegration:
    """Test configuration loading and model creation integration."""
    
    def test_config_override(self):
        """Test that command-line arguments can override config values."""
        from main import load_and_process_config
        
        config = load_and_process_config(
            config_path="config_train.yaml",
            unknown_args=["--batch_size=8", "--epochs=50"]
        )
        
        assert config.BATCH_SIZE == 8
        assert config.EPOCHS == 50
    
    def test_boot_mode(self):
        """Test boot mode configuration."""
        from main import load_and_process_config
        
        config = load_and_process_config(
            config_path="config_train.yaml",
            boot_mode=True
        )
        
        assert config.BATCH_SIZE == 1
        assert config.EPOCHS == 1
        assert config.NO_WANDB == True
    
    def test_model_parameter_count(self):
        """Test that model parameter count is reasonable."""
        from main import create_model_from_config, load_and_process_config
        
        config = load_and_process_config(config_path="config_train.yaml")
        device = torch.device("cpu")  # Use CPU for this test
        model = create_model_from_config(config, device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model should have reasonable number of parameters
        assert total_params > 1_000_000, f"Model seems too small: {total_params:,} params"
        assert total_params < 500_000_000, f"Model seems too large: {total_params:,} params"
        
        # With frozen backbone, trainable params should be much less
        if config.MODEL.RGB_ENCODER.FREEZE_BACKBONE:
            assert trainable_params < total_params * 0.5, \
                f"Too many trainable params with frozen backbone: {trainable_params:,} / {total_params:,}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])