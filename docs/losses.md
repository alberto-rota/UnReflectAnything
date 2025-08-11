# Losses

This module contains various loss functions used in the UnReflectAnything framework.

## Overview

The losses module provides:
- Reflection removal losses
- Depth estimation losses
- Geometric consistency losses
- Multi-scale losses

## Loss Functions

### Reflection Removal Losses
- **Photometric Loss**: Ensures color consistency
- **Perceptual Loss**: Maintains visual quality
- **Adversarial Loss**: Improves realism

### Depth Losses
- **L1/L2 Depth Loss**: Direct depth supervision
- **Smoothness Loss**: Encourages smooth depth maps
- **Edge-Aware Loss**: Preserves depth discontinuities

### Geometric Losses
- **Reprojection Loss**: Geometric consistency
- **Scale-Aware Loss**: Handles scale ambiguity
- **Pose Consistency Loss**: Multi-view consistency

## Usage

```python
# Example loss usage
from losses import ReflectionLoss, DepthLoss

reflection_loss = ReflectionLoss()
depth_loss = DepthLoss()
```
