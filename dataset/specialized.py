"""
Specialized dataset classes for specific data sources.
"""

from typing import List, Any, Dict
from .base import Mono3D_Dataset


class SCARED(Mono3D_Dataset):
    """
    SCARED dataset class for monocular 3D camera pose estimation.
    Extends the base Mono3D_Dataset.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the SCARED dataset.
        
        Args:
            **kwargs: Additional arguments passed to the parent class
        """
        params = {
            "original_width": 1280,
            "original_height": 1024,
            "with_depth": True,
        }
        params.update(kwargs)
        super().__init__(**params)

    @staticmethod
    def videonames() -> List[str]:
        """
        Get the list of video names for the SCARED dataset.
        
        Returns:
            List of video names
        """
        return [f"v{i}" for i in range(1, 35)]


class CHOLEC80(Mono3D_Dataset):
    """
    CHOLEC80 dataset class for monocular 3D camera pose estimation.
    Extends the base Mono3D_Dataset.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the CHOLEC80 dataset.
        
        Args:
            **kwargs: Additional arguments passed to the parent class
        """
        params = {
            "original_width": 1280,
            "original_height": 1024,
            "with_depth": False,
        }
        params.update(kwargs)
        super().__init__(**params)

    @staticmethod
    def videonames() -> List[str]:
        """
        Get the list of video names for the CHOLEC80 dataset.
        
        Returns:
            List of video names
        """
        return []  # [f"v{i}" for i in range(1, 12)]


class GRASP(Mono3D_Dataset):
    """
    GRASP dataset class for monocular 3D camera pose estimation.
    Extends the base Mono3D_Dataset with custom dimensions.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the GRASP dataset.
        
        Args:
            **kwargs: Additional arguments passed to the parent class
        """
        params = {
            "original_width": 640,
            "original_height": 400,
            "with_depth": False,
        }
        params.update(kwargs)
        super().__init__(**params)

    @staticmethod
    def videonames() -> List[str]:
        """
        Get the list of video names for the StereomMIS dataset.
        
        Returns:
            List of video names
        """
        return [f"v{i}" for i in range(1, 14)]
