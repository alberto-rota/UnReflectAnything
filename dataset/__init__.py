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

from .base import Mono3D_Dataset
from .highlight import HighlightDataset
from .loader import initialize_from_config
from .multi_dataset import MultiDataset
from .rgbp import (
    HOUSECAT6D_Dataset,
    POLARGB_Dataset,
    RGBP_Dataset,
    SCRREAM_Dataset,
    create_datasets_from_config,
    create_optimized_dataloader,
    load_config_and_create_datasets,
)
from .specialized import CHOLEC80, GRASP, SCARED
from .utils import (
    adapt_intrinsics_two_step,
    center_crop_intrinsics,
    resize_intrinsics,
    split_videos,
)

__all__ = [
    # Core datasets
    "Mono3D_Dataset",
    "MultiDataset",
    "HighlightDataset",
    # Specialized datasets
    "SCARED",
    "CHOLEC80",
    "GRASP",
    # RGBP/Polarization datasets
    "RGBP_Dataset",
    "SCRREAM_Dataset",
    "HOUSECAT6D_Dataset",
    "POLARGB_Dataset",
    # Dataset creation functions
    "initialize_from_config",
    "create_datasets_from_config",
    "create_optimized_dataloader",
    "load_config_and_create_datasets",
    # Utility functions
    "adapt_intrinsics_two_step",
    "split_videos",
    "resize_intrinsics",
    "center_crop_intrinsics",
]
