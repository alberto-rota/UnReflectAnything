"""
Dataset module for monocular 3D camera pose estimation and polarization-based reflection removal.

This module provides various dataset implementations for computer vision tasks:

Core Datasets:
- Mono3D_Dataset: Base class for monocular 3D pose estimation
- HighlightDataset: Wrapper for highlight/specular reflection detection
- MultiDataset: Combines multiple datasets for training

Polarization Datasets:
- RGBP_Dataset: Base class for RGB + Polarization data processing
- SCRREAM_Dataset: SCRREAM dataset for reflection removal
- HOUSECAT6D_Dataset: HOUSECAT6D dataset for 6D pose estimation
- POLARGB_Dataset: PolaRGB dataset for polarization-guided processing

Specialized Datasets:
- SCARED: Stereo Correspondence and Reconstruction of Endoscopic Data
- CHOLEC80: Cholecystectomy dataset for laparoscopic surgery
- GRASP: Robotic grasping dataset

Utilities:
- Camera intrinsics manipulation functions
- Video splitting and data loading utilities
"""

# from .highlight import HighlightDataset
from .rgbp import (
    HOUSECAT6D_Dataset,
    POLARGB_Dataset,
    RGBP_Dataset,
    SCRREAM_Dataset,
    from_config,
)
from .utils import (
    adapt_intrinsics_two_step,
    center_crop_intrinsics,
    resize_intrinsics,
    split_videos,
)

__all__ = [
    # Core datasets
    # "HighlightDataset",
    # Specialized datasets
    # RGBP/Polarization datasets
    "RGBP_Dataset",
    "SCRREAM_Dataset",
    "HOUSECAT6D_Dataset",
    "POLARGB_Dataset",
    # Dataset creation functions
    "initialize_from_config",
    "from_config",
    # Utility functions
    "adapt_intrinsics_two_step",
    "split_videos",
    "resize_intrinsics",
    "center_crop_intrinsics",
]
