# Dataset Overview

This section provides an overview of the dataset handling capabilities in UnReflectAnything.

## Supported Datasets

The framework supports various datasets for reflection removal and depth estimation:

### SCARED Dataset
- Surgical scene dataset with reflections
- Multiple camera viewpoints
- Depth ground truth available

### Custom Datasets
- Support for custom dataset formats
- Flexible data loading pipeline
- Multi-modal data support

## Dataset Structure

```
dataset/
├── images/          # RGB images
├── depth/           # Depth maps
├── masks/           # Reflection masks
└── metadata/        # Camera parameters, poses, etc.
```

## Data Loading

The dataset module provides:
- Efficient data loading with caching
- Multi-threaded data loading
- GPU-optimized data transfer
- Automatic data augmentation
