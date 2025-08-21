"""
Example usage of the new flexible RGBPOLDecomposer initialization.
Shows different ways to initialize the model with instances or configurations.
"""

import torch
from models import (
    DINOv3, POLViTEncoder, DPTRGBDecoder,
    RGBPOLDecomposer, create_rgb_pol_decomposer, create_rgb_pol_decomposer_from_instances
)

def example_1_all_config_dicts():
    """Example 1: Initialize everything from configuration dictionaries."""
    
    # Create model with all default configurations  
    model = create_rgb_pol_decomposer(device='cuda')
    print("Example 1a: Created model with all defaults")
    
    # Create model with custom configurations
    model = create_rgb_pol_decomposer(
        dinov3_config={
            'model_name': 'facebook/dinov3-vits16-pretrain-lvd1689m',
            'image_size': 896,
            'freeze_backbone': True
        },
        pol_encoder_config={
            'embed_dim': 768,
            'depth': 6,  # Deeper POL encoder
            'n_heads': 12
        },
        decoder_configs={
            'use_bn': True,
            'readout_type': 'project',
            'fusion_hidden_size': 512
        },
        cross_attention_config={
            'bi_directional': True  # Enable bidirectional fusion
        },
        device='cuda'
    )
    print("Example 1b: Created model with custom configs")
    
    return model


def example_2_different_decoder_configs():
    """Example 2: Different configurations for each decoder."""
    
    model = create_rgb_pol_decomposer(
        dinov3_config={
            'model_name': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'image_size': 896
        },
        decoder_configs={
            'specular': {
                'use_bn': True,
                'fusion_hidden_size': 512,
                'readout_type': 'project'
            },
            'diffuse': {
                'use_bn': False, 
                'fusion_hidden_size': 256,
                'readout_type': 'add'
            },
            'reconstruction': {
                'use_bn': True,
                'fusion_hidden_size': 384,
                'readout_type': 'ignore'
            }
        },
        device='cuda'
    )
    print("Example 2: Created model with different decoder configs")
    
    return model


def example_3_mixed_instances_and_configs():
    """Example 3: Mix of pre-initialized instances and configurations."""
    
    # Pre-initialize some components
    dinov3_model = DINOv3({
        'model_name': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'image_size': 896,
        'freeze_backbone': True,
        'return_last_hidden_state': True,
        'return_as_feature_maps': False
    })
    
    pol_encoder = POLViTEncoder({
        'in_ch': 3,
        'embed_dim': 768,
        'depth': 4,
        'n_heads': 12,
        'patch_size': 16
    })
    
    # Use direct initialization with mix of instances and configs
    model = RGBPOLDecomposer(
        dinov3_model_or_config=dinov3_model,  # Pre-initialized instance
        pol_encoder_or_config=pol_encoder,    # Pre-initialized instance
        specular_decoder_or_config={'use_bn': True, 'fusion_hidden_size': 512},  # Config dict
        diffuse_decoder_or_config={'use_bn': False, 'fusion_hidden_size': 256},  # Config dict
        reconstruction_decoder_or_config=None,  # Use defaults
        cross_attention_config={'bi_directional': False}
    ).cuda()
    
    print("Example 3: Created model with mixed instances and configs")
    
    return model


def example_4_all_instances():
    """Example 4: Pre-initialize all components as instances."""
    
    # Initialize all components separately
    dinov3_model = DINOv3({
        'model_name': 'facebook/dinov3-vitb16-pretrain-lvd1689m', 
        'image_size': 896,
        'freeze_backbone': True,
        'return_last_hidden_state': True,
        'return_as_feature_maps': False
    })
    
    pol_encoder = POLViTEncoder({
        'in_ch': 3,
        'embed_dim': 768,
        'depth': 4,
        'n_heads': 12,
        'patch_size': 16,
        'drop': 0.1
    })
    
    # Three different decoder configurations
    specular_decoder = DPTRGBDecoder({
        'feature_dim': 768,
        'use_bn': True,
        'fusion_hidden_size': 512,
        'readout_type': 'project'
    })
    
    diffuse_decoder = DPTRGBDecoder({
        'feature_dim': 768,
        'use_bn': False,
        'fusion_hidden_size': 256, 
        'readout_type': 'add'
    })
    
    reconstruction_decoder = DPTRGBDecoder({
        'feature_dim': 768,
        'use_bn': True,
        'fusion_hidden_size': 384,
        'readout_type': 'ignore'
    })
    
    # Create model from instances
    model = create_rgb_pol_decomposer_from_instances(
        dinov3_model=dinov3_model,
        pol_encoder=pol_encoder,
        specular_decoder=specular_decoder,
        diffuse_decoder=diffuse_decoder,
        reconstruction_decoder=reconstruction_decoder,
        cross_attention_config={'bi_directional': True},
        device='cuda'
    )
    
    print("Example 4: Created model from all pre-initialized instances")
    
    return model


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    
    batch_size = 2
    height = 896  
    width = 896
    
    # Create dummy batch
    batch = {
        'rgb': torch.randn(batch_size, 3, height, width, device='cuda'),
        'AoP': torch.randn(batch_size, 1, height, width, device='cuda') * 3.14159,  # [0, π) 
        'DoP': torch.rand(batch_size, 1, height, width, device='cuda')  # [0, 1]
    }
    
    print(f"Input shapes: RGB {batch['rgb'].shape}, AoP {batch['AoP'].shape}, DoP {batch['DoP'].shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(batch)
    
    print(f"Output shapes:")
    print(f"  Specular: {outputs['specular'].shape}")
    print(f"  Diffuse: {outputs['diffuse'].shape}")
    print(f"  Reconstruction: {outputs['reconstruction'].shape}")
    print(f"  Token shapes: RGB {outputs['tokens']['rgb'].shape}, POL {outputs['tokens']['pol'].shape}, Fused {outputs['tokens']['fused'].shape}")
    
    return outputs


if __name__ == "__main__":
    print("=== RGBPOLDecomposer Flexible Initialization Examples ===\n")
    
    # Example 1: All config dictionaries
    print("--- Example 1: Config Dictionaries ---")
    model1 = example_1_all_config_dicts()
    test_forward_pass(model1)
    print()
    
    # Example 2: Different decoder configs
    print("--- Example 2: Different Decoder Configs ---") 
    model2 = example_2_different_decoder_configs()
    test_forward_pass(model2)
    print()
    
    # Example 3: Mixed instances and configs
    print("--- Example 3: Mixed Instances and Configs ---")
    model3 = example_3_mixed_instances_and_configs()
    test_forward_pass(model3)
    print()
    
    # Example 4: All instances
    print("--- Example 4: All Pre-initialized Instances ---")
    model4 = example_4_all_instances()
    test_forward_pass(model4)
    print()
    
    print("=== All examples completed successfully! ===")
