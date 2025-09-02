#!/usr/bin/env python3
"""
Test script for model parameter summary functions.
Demonstrates usage with both RGBPOLDecomposer and RGBDistillDecomposer models.
"""

import torch
from models import RGBPOLDecomposer, RGBDistillDecomposer, get_model_parameter_summary, print_model_parameter_summary, get_model_size_mb


def test_rgb_distill_decomposer():
    """Test parameter summary for RGBDistillDecomposer."""
    print("\n" + "="*80)
    print("TESTING RGBDistillDecomposer")
    print("="*80)
    
    # Create model
    dinov3_config = {
        "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "image_size": 896,
        "freeze_backbone": True,
        "return_selected_layers": [2, 5, 8, 11]
    }
    
    model = RGBDistillDecomposer(
        dinov3=dinov3_config,
        patch_size=16
    )
    
    # Get and print summary
    print_model_parameter_summary(model, detailed=True)
    
    # Get size in MB
    size_mb = get_model_size_mb(model)
    print(f"\n💾 Model Size: {size_mb:.1f} MB")
    
    return model


def test_rgb_pol_decomposer():
    """Test parameter summary for RGBPOLDecomposer."""
    print("\n" + "="*80)
    print("TESTING RGBPOLDecomposer")
    print("="*80)
    
    # Create model
    dinov3_config = {
        "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "image_size": 896,
        "freeze_backbone": True,
        "return_selected_layers": [2, 5, 8, 11]
    }
    
    model = RGBPOLDecomposer(
        dinov3=dinov3_config,
        patch_size=16
    )
    
    # Get and print summary
    print_model_parameter_summary(model, detailed=True)
    
    # Get size in MB
    size_mb = get_model_size_mb(model)
    print(f"\n💾 Model Size: {size_mb:.1f} MB")
    
    return model


def compare_models():
    """Compare both models side by side."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Create both models
    dinov3_config = {
        "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "image_size": 896,
        "freeze_backbone": True,
        "return_selected_layers": [2, 5, 8, 11]
    }
    
    rgb_distill = RGBDistillDecomposer(dinov3=dinov3_config, patch_size=16)
    rgb_pol = RGBPOLDecomposer(dinov3=dinov3_config, patch_size=16)
    
    # Get summaries
    distill_summary = get_model_parameter_summary(rgb_distill)
    pol_summary = get_model_parameter_summary(rgb_pol)
    
    print(f"\n{'Metric':<25} {'RGBDistill':<15} {'RGBPOL':<15} {'Difference':<15}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")
    
    print(f"{'Total Parameters':<25} {distill_summary['total_parameters']:<15,} {pol_summary['total_parameters']:<15,} {pol_summary['total_parameters'] - distill_summary['total_parameters']:<15,}")
    print(f"{'Trainable Parameters':<25} {distill_summary['trainable_parameters']:<15,} {pol_summary['trainable_parameters']:<15,} {pol_summary['trainable_parameters'] - distill_summary['trainable_parameters']:<15,}")
    print(f"{'Frozen Parameters':<25} {distill_summary['frozen_parameters']:<15,} {pol_summary['frozen_parameters']:<15,} {pol_summary['frozen_parameters'] - distill_summary['frozen_parameters']:<15,}")
    
    distill_size = get_model_size_mb(rgb_distill)
    pol_size = get_model_size_mb(rgb_pol)
    print(f"{'Model Size (MB)':<25} {distill_size:<15.1f} {pol_size:<15.1f} {pol_size - distill_size:<15.1f}")
    
    print(f"\n📈 RGBPOL has {pol_summary['total_parameters'] - distill_summary['total_parameters']:,} additional parameters")
    print(f"   This represents a {(pol_summary['total_parameters'] - distill_summary['total_parameters']) / distill_summary['total_parameters'] * 100:.1f}% increase")


if __name__ == "__main__":
    print("🧪 Testing Model Parameter Summary Functions")
    print("="*80)
    
    try:
        # Test individual models
        rgb_distill = test_rgb_distill_decomposer()
        rgb_pol = test_rgb_pol_decomposer()
        
        # Compare models
        compare_models()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
