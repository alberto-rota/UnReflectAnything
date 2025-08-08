# HighLightRenderer

## Overview

The `HighLightRenderer` class is a PyTorch module that renders realistic reflection artifacts on images based on 3D point clouds, light sources, and surface geometry. It simulates specular reflections using the law of reflection and the Phong lighting model, making it useful for computer vision tasks that require realistic lighting effects.

## Key Features

- **Surface Normal Estimation**: Automatically estimates surface normals from 3D point clouds using gradient-based methods
- **Phong Reflection Model**: Implements realistic specular reflections with configurable surface roughness
- **Light Attenuation**: Supports distance-based light attenuation for realistic lighting
- **Multi-format Support**: Works with both RGB images and feature maps
- **GPU Optimized**: Fully vectorized operations for high-performance rendering

## Class Definition

```python
class HighLightRenderer(nn.Module):
    def __init__(self, height, width, patch_size=16):
        """
        Initialize the HighLightRenderer.
        
        Args:
            height (int): Image height in pixels
            width (int): Image width in pixels  
            patch_size (int): Patch size for feature resolution (default: 16)
        """
```

## Input Data Format

### Required Inputs

1. **cloud** (`torch.Tensor`): 3D point cloud in homogeneous coordinates
   - Shape: `[B, 4, N]` where B=batch size, N=number of points
   - Format: `[x, y, z, 1]` homogeneous coordinates

2. **rgb_vec** (`torch.Tensor`): RGB values or features for each point
   - Shape: `[B, 3, N]` for RGB or `[B, E, N]` for features
   - Values: RGB in range [0, 1] or feature vectors

3. **camera_K** (`torch.Tensor`): Camera intrinsic matrix
   - Shape: `[B, 3, 3]`
   - Format: Standard camera intrinsics matrix

4. **camera_T** (`torch.Tensor`): Camera pose transformation
   - Shape: `[B, 4, 4]` (homogeneous matrix) or `[B, 6]` (Euler angles)
   - Format: Camera-to-world transformation

5. **light_position** (`torch.Tensor`): Light source position
   - Shape: `[B, 3]` or `[3]` (broadcasted to batch)
   - Format: World coordinates `[x, y, z]`

### Optional Parameters

- **light_intensity** (`float`): Light intensity scalar (default: 1.0)
- **light_color** (`torch.Tensor`): Light color `[3]` or `[B, 3]` (default: white `[1,1,1]`)
- **surface_roughness** (`float`): Surface roughness 0-1 (0=mirror, 1=diffuse, default: 0.1)
- **light_attenuation** (`tuple`): Distance attenuation coefficients `(constant, linear, quadratic)` (default: `(1.0, 0.1, 0.01)`)
- **reflection_strength** (`float`): Overall reflection strength multiplier 0-1 (default: 0.5)

## Output Format

The `forward()` method returns a dictionary containing:

```python
{
    'warped': torch.Tensor,           # [B, C, H, W] - Final image with reflections
    'reflection_intensity': torch.Tensor,  # [B, 3, N] - Reflection intensity per point
    'surface_normals': torch.Tensor,   # [B, 3, N] - Estimated surface normals
    'reflection_only': torch.Tensor,   # [B, C, H, W] - Reflection-only visualization
    'light_position': torch.Tensor,    # [B, 3, 1] - Light position used
    'camera_position': torch.Tensor,   # [B, 3, 1] - Camera position extracted
    # ... additional outputs from Project class
}
```

## Usage Examples

### Basic Usage

```python
import torch
from projections import HighLightRenderer

# Initialize renderer
renderer = HighLightRenderer(height=480, width=640)

# Prepare input data
B, N = 2, 1000  # batch size, number of points
cloud = torch.randn(B, 4, N)  # 3D points in homogeneous coordinates
rgb_vec = torch.rand(B, 3, N)  # RGB values [0,1]
camera_K = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)  # Camera intrinsics
camera_T = torch.randn(B, 6)  # Camera pose (Euler angles)
light_position = torch.tensor([10.0, 5.0, 15.0])  # Light position

# Render with reflections
result = renderer(
    cloud=cloud,
    rgb_vec=rgb_vec, 
    camera_K=camera_K,
    camera_T=camera_T,
    light_position=light_position,
    reflection_strength=0.3
)

# Access results
image_with_reflections = result['warped']  # [B, 3, H, W]
reflection_only = result['reflection_only']  # [B, 3, H, W]
```

### Advanced Usage with Custom Lighting

```python
# Custom light setup
light_color = torch.tensor([1.0, 0.8, 0.6])  # Warm light
light_intensity = 2.0
surface_roughness = 0.05  # Very smooth surface (near mirror)

result = renderer(
    cloud=cloud,
    rgb_vec=rgb_vec,
    camera_K=camera_K, 
    camera_T=camera_T,
    light_position=light_position,
    light_intensity=light_intensity,
    light_color=light_color,
    surface_roughness=surface_roughness,
    light_attenuation=(1.0, 0.05, 0.001),  # Custom attenuation
    reflection_strength=0.7
)
```

### Feature Map Rendering

```python
# For feature maps (e.g., 64-dimensional features)
feature_vec = torch.randn(B, 64, N)  # Feature vectors

result = renderer(
    cloud=cloud,
    rgb_vec=feature_vec,  # Can handle features too
    camera_K=camera_K,
    camera_T=camera_T, 
    light_position=light_position
)

# Features with reflections applied to first 3 channels
enhanced_features = result['warped']  # [B, 64, H//16, W//16]
```

## Technical Details

### Surface Normal Estimation

The renderer estimates surface normals using a gradient-based approach:

1. Reshapes 3D point cloud to spatial grid format
2. Calculates gradients in X and Y directions
3. Computes normal vectors using cross product
4. Normalizes and ensures normals point toward camera

### Phong Reflection Model

The reflection calculation follows the Phong model:

1. **Light Direction**: `L = (light_pos - surface_pos) / ||light_pos - surface_pos||`
2. **View Direction**: `V = (camera_pos - surface_pos) / ||camera_pos - surface_pos||`
3. **Reflection Direction**: `R = 2(N·L)N - L`
4. **Specular Component**: `specular = (R·V)^shininess`
5. **Final Intensity**: `reflection = light_intensity × attenuation × (N·L) × specular × light_color`

### Light Attenuation

Distance-based attenuation follows the formula:
```
attenuation = 1 / (constant + linear×distance + quadratic×distance²)
```

### Surface Roughness

The `surface_roughness` parameter controls the shininess:
- `0.0`: Perfect mirror (very sharp reflections)
- `0.1`: Smooth surface (default)
- `1.0`: Diffuse surface (no specular reflections)

## Dependencies

- **PyTorch**: Core tensor operations and neural network modules
- **geometry.py**: `euler2mat()` function for pose conversion
- **Project class**: For 3D-to-2D projection

## Performance Considerations

- **GPU Memory**: Large point clouds may require significant GPU memory
- **Batch Processing**: Supports batched operations for efficiency
- **Vectorization**: All operations are vectorized for optimal performance
- **Feature Resolution**: Feature maps are automatically scaled by `patch_size`

## Common Use Cases

1. **Computer Vision Training**: Adding realistic reflections to synthetic data
2. **Data Augmentation**: Enhancing training datasets with lighting variations
3. **Visualization**: Creating realistic renderings of 3D scenes
4. **Research**: Studying the effects of lighting on computer vision models

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or number of points
2. **NaN Values**: Check for invalid camera poses or light positions
3. **Poor Reflections**: Adjust `surface_roughness` and `reflection_strength`
4. **Wrong Scale**: Ensure camera intrinsics match image resolution

### Parameter Tuning

- **Weak Reflections**: Increase `reflection_strength` (0.5 → 0.8)
- **Sharp Reflections**: Decrease `surface_roughness` (0.1 → 0.05)
- **Distant Light**: Adjust `light_attenuation` for realistic falloff
- **Color Cast**: Modify `light_color` for different lighting conditions
