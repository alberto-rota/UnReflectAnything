"""
Utility functions for the dataset module.
"""

from typing import List, Optional, Tuple
import torch
import random


def split_videos(
    videos: List[str], tr_perc: float = 0.8, test_vids: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Split a list of videos into training and validation sets.

    Args:
        videos: List of video names.
        tr_perc: Percentage of videos to include in the training set. Defaults to 0.8.
        test_vids: List of videos to exclude from training/validation. Defaults to None.

    Returns:
        A tuple containing the training and validation sets.
    """
    # Remove test videos from the list
    if test_vids is not None:
        videos = [vid for vid in videos if vid not in test_vids]
    random.shuffle(videos)
    tr_len = int(len(videos) * tr_perc)
    return videos[:tr_len], videos[tr_len:]


def resize_intrinsics(K: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    """
    Resize intrinsic camera matrix based on scaling factors.

    Args:
        K: The intrinsic camera matrix of shape (3, 3)
        sx: Scaling factor for width
        sy: Scaling factor for height

    Returns:
        Resized intrinsic matrix of shape (3, 3)
    """
    K_new = K.clone()
    # Focal lengths
    K_new[0, 0] *= sx
    K_new[1, 1] *= sy
    # Principal point
    K_new[0, 2] *= sx
    K_new[1, 2] *= sy
    return K_new


def center_crop_intrinsics(
    K: torch.Tensor,
    final_width: int,
    final_height: int,
    backbone_width: int,
    backbone_height: int,
) -> torch.Tensor:
    """
    Adjust intrinsic camera matrix for center cropping.

    Args:
        K: The intrinsic camera matrix of shape (3, 3)
        final_width: Final image width
        final_height: Final image height
        backbone_width: Backbone width before cropping
        backbone_height: Backbone height before cropping

    Returns:
        Adjusted intrinsic matrix of shape (3, 3)
    """
    K_new = K.clone()

    # Crop offsets
    offset_x = (backbone_width - final_width) / 2.0
    offset_y = (backbone_height - final_height) / 2.0

    # Shift the principal point by the top/left offset
    K_new[0, 2] -= offset_x
    K_new[1, 2] -= offset_y

    return K_new


def adapt_intrinsics_two_step(
    K: torch.Tensor,
    orig_width: int,
    orig_height: int,
    backbone_width: int,
    backbone_height: int,
    final_width: int,
    final_height: int,
) -> torch.Tensor:
    """
    Adapt intrinsic camera matrix for resizing and cropping in two steps.

    Args:
        K: The intrinsic camera matrix of shape (3, 3)
        orig_width: Original image width
        orig_height: Original image height
        backbone_width: Backbone width after resizing
        backbone_height: Backbone height after resizing
        final_width: Final image width after cropping
        final_height: Final image height after cropping

    Returns:
        Adapted intrinsic matrix of shape (3, 3)
    """
    # 1) Resize step
    sx = backbone_width / orig_width
    sy = backbone_height / orig_height
    K_resize = resize_intrinsics(K, sx, sy)

    # 2) Center-crop step
    K_crop = center_crop_intrinsics(
        K_resize, final_width, final_height, backbone_width, backbone_height
    )
    return K_crop
