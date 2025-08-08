# MONO3D Dataset Module

This module provides classes and functions for handling monocular 3D camera pose estimation datasets.

## Structure

- `__init__.py` - Exports all public module components
- `base.py` - Contains the base `Mono3D_Dataset` class
- `multi_dataset.py` - Contains the `MultiDataset` class for combining multiple datasets
- `specialized.py` - Contains specialized dataset classes (SCARED, CHOLEC80, GRASP)
- `utils.py` - Utility functions for handling datasets
- `loader.py` - Dataset loading and configuration functions

## Main Classes

### Mono3D_Dataset

Base dataset class that handles loading videos, frames, and camera poses, and provides methods for curriculum learning, augmentation, and various output formats.

### MultiDataset

Extends `torch.utils.data.ConcatDataset` to combine multiple `Mono3D_Dataset` instances with unified sampling, curriculum learning, and inspection capabilities.

### Specialized Datasets

- `SCARED` - Dataset class for SCARED dataset
- `CHOLEC80` - Dataset class for CHOLEC80 dataset  
- `GRASP` - Dataset class for GRASP dataset with custom dimensions

## Key Functions

- `initialize_from_config()` - Initialize datasets and dataloaders from configuration
- `adapt_intrinsics_two_step()` - Adapt intrinsic camera matrix for resizing and cropping
- `split_videos()` - Split a list of videos into training and validation sets

## Usage

```python
from mono3d.dataset import Mono3D_Dataset, MultiDataset, initialize_from_config

# Initialize from config
result = initialize_from_config(config)
training_ds = result["dataset"]["Training"] 
training_dl = result["dataset"]["training_dl"]

# Or create dataset instances directly
dataset = Mono3D_Dataset(
    path="path/to/dataset",
    frameskip=[1, 2, 4],
    # ... other options
) 