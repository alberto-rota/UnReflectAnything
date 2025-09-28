"""
Highlight detection dataset wrapper for monocular 3D camera pose estimation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


from .mono3d_dataset import Mono3D_Dataset

class HighlightDataset(Mono3D_Dataset):
    """
    Dataset wrapper that adds highlight detection functionality to any specialized dataset.

    This class can wrap SCARED, CHOLEC80, GRASP, or any other dataset that extends Mono3D_Dataset
    and provides additional functionality to detect bright highlights/specular reflections in images.
    """

    def __init__(
        self,
        base_dataset: Optional[Union[Mono3D_Dataset, type]] = None,
        brightness_threshold: float = 0.93,
        return_mask: bool = False,
        rect_size: Optional[Tuple[int, int]] = None,
        return_rect: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the HighlightDataset wrapper.

        Args:
            base_dataset: Either an instantiated dataset object
                or a dataset class to wrap (e.g., SCARED, CHOLEC80, GRASP)
            brightness_threshold: Threshold for highlight detection (0.0-1.0). Default: 0.93
            return_mask: Whether to return highlight masks in the output. Default: False
            rect_size: Target rectangle size as (height, width).
                If provided, returns coordinates of rectangle of this size with fewest highlighted pixels.
            return_rect: Whether to return cropped rectangles and their masks. Default: False
            **kwargs: Additional arguments passed to the base dataset class (only used if base_dataset is a class)
        """
        self.brightness_threshold = brightness_threshold
        self.return_mask = return_mask
        self.rect_size = rect_size
        self.return_rect = return_rect

        # Check if base_dataset is an instance or a class
        if base_dataset is not None and hasattr(base_dataset, "__getitem__"):
            # It's an instantiated dataset - wrap it
            self._wrap_existing_dataset(base_dataset)
        elif base_dataset is not None:
            # It's a dataset class - instantiate it
            self._create_from_class(base_dataset, kwargs)
        else:
            # No base dataset provided - initialize as standard Mono3D_Dataset
            super().__init__(**kwargs)

    def _wrap_existing_dataset(self, dataset_instance: Mono3D_Dataset) -> None:
        """
        Wrap an existing dataset instance.

        Args:
            dataset_instance: An already instantiated Mono3D_Dataset or subclass
        """
        self.base_dataset = dataset_instance

        # Copy all relevant attributes from the base dataset
        attributes_to_copy = [
            # Core dataset attributes
            "rgbpathlist",
            "pathlist",
            "depthpathlist",
            "poseslist",
            "intrinsicslist",
            "distortionslist",
            "Tlist",
            "Tinvlist",
            "sampler",
            # Configuration attributes
            "numframes",
            "numvideos",
            "name",
            "fps",
            "frameskip",
            "frameskip_set",
            "frameskip_set_curriculum",
            "frameskip_curriculum_step",
            "manual_frameskip",
            # Dimension and transformation attributes
            "height",
            "width",
            "original_height",
            "original_width",
            "aspect_ratio",
            "backbone_height",
            "backbone_width",
            "resize_transform",
            # Feature flags
            "as_euler",
            "as_embedding",
            "unit_translation",
            "as_quat",
            "with_paths",
            "with_frameskip",
            "with_intrinsics",
            "with_distortions",
            "with_fundamental",
            "with_global_poses",
            "with_depth",
            "transforms_only",
            "target_pose_only",
            "random_pose",
            "random_pose_ranges",
            # Augmentation settings
            "color_augmentation_prob",
            "geometric_augmentation_prob",
            "reverse_augmentation_prob",
            "standstill_augmentation_prob",
            # Learning parameters
            "standardize",
            "target_length",
            "curriculum_factor",
            # Storage and caching
            "DEVICE",
            "is_gcs",
            "preload_in_memory",
            "preload_transforms",
            "frame_cache",
            "depth_cache",
            "embedding_cache",
            # Other attributes
            "excluded",
            "order_check",
        ]

        for attr in attributes_to_copy:
            if hasattr(dataset_instance, attr):
                setattr(self, attr, getattr(dataset_instance, attr))

        # Copy storage backend attributes if they exist (for GCS support)
        if hasattr(dataset_instance, "storage_client"):
            self.storage_client = dataset_instance.storage_client
        if hasattr(dataset_instance, "bucket"):
            self.bucket = dataset_instance.bucket
        if hasattr(dataset_instance, "gcs_prefix"):
            self.gcs_prefix = dataset_instance.gcs_prefix

    def _create_from_class(
        self, base_dataset_class: type, kwargs: Dict[str, Any]
    ) -> None:
        """
        Create a new dataset instance from a dataset class.
        
        This method instantiates the base dataset class with the provided parameters
        and copies relevant attributes to enable highlight detection functionality.
        
        Args:
            base_dataset_class: Dataset class to instantiate (e.g., SCARED, CHOLEC80, GRASP)
            kwargs: Keyword arguments to pass to the dataset class constructor
        """
        self.base_dataset_class = base_dataset_class
        self.base_params = kwargs.copy()

        # Create a temporary instance to get default parameters
        temp_instance = base_dataset_class(**kwargs)

        # Copy relevant attributes from the base dataset
        for attr in ["original_width", "original_height", "with_depth"]:
            if hasattr(temp_instance, attr):
                setattr(self, attr, getattr(temp_instance, attr))

        # Initialize with the same parameters as the base class
        super().__init__(**kwargs)

    @classmethod
    def from_dataset_class(
        cls,
        base_dataset_class: type,
        brightness_threshold: float = 0.93,
        return_mask: bool = False,
        rect_size: Optional[Tuple[int, int]] = None,
        return_rect: bool = False,
        **kwargs,
    ) -> "HighlightDataset":
        """
        Create a HighlightDataset by wrapping a specific dataset class.
        
        This is a convenient factory method for creating highlight datasets
        from dataset classes like SCARED, CHOLEC80, or GRASP.
        
        Args:
            base_dataset_class: Dataset class to wrap (e.g., SCARED, CHOLEC80, GRASP)
            brightness_threshold: Threshold for highlight detection (0.0-1.0). Default: 0.93
            return_mask: Whether to return highlight masks in output. Default: False
            rect_size: Target rectangle size as (height, width) for finding regions
                      with least highlights. Default: None
            return_rect: Whether to return cropped rectangles and their masks. Default: False
            **kwargs: Additional arguments passed to the dataset class constructor
            
        Returns:
            HighlightDataset instance wrapping the specified dataset class
            
        Example:
            >>> highlight_scared = HighlightDataset.from_dataset_class(
            ...     SCARED, brightness_threshold=0.9, return_mask=True,
            ...     root='/path/to/scared', frameskip=2
            ... )
        """
        """
        Factory method to create a HighlightDataset from a base dataset class.

        Args:
            base_dataset_class: The base dataset class to wrap
            brightness_threshold: Threshold for highlight detection
            return_mask: Whether to return highlight masks
            rect_size: Target rectangle size as (height, width)
            return_rect: Whether to return cropped rectangles and their masks
            **kwargs: Additional arguments for the base dataset

        Returns:
            Configured highlight detection dataset
        """
        return cls(
            base_dataset=base_dataset_class,
            brightness_threshold=brightness_threshold,
            return_mask=return_mask,
            rect_size=rect_size,
            return_rect=return_rect,
            **kwargs,
        )

    def _compute_highlight_mask(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Compute binary mask for highlights in a single frame.

        Args:
            frame: Input frame tensor of shape (C, H, W)

        Returns:
            Binary mask of shape (1, H, W) where 1 indicates highlight pixels
        """
        if frame.dim() == 3:
            # Convert RGB to grayscale using luminance formula: 0.299*R + 0.587*G + 0.114*B
            if frame.shape[0] == 3:
                # RGB image
                grayscale = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
            else:
                # Assume single channel or take mean across channels
                grayscale = frame.mean(dim=0)
        elif frame.dim() == 2:
            # Already grayscale
            grayscale = frame
        else:
            raise ValueError(f"Unexpected frame dimensions: {frame.shape}")

        # Create binary mask for pixels above threshold
        highlight_mask = (grayscale > self.brightness_threshold).float().unsqueeze(0)

        return highlight_mask

    def _find_rectangle_with_least_highlights(
        self, binary_mask: torch.Tensor, rect_size: Optional[Tuple[int, int]]
    ) -> Tuple[int, int, int, int]:
        """
        Find rectangle of specified size with the fewest highlighted pixels.
        Fully vectorized implementation with O(H*W) complexity and no for loops.

        Args:
            binary_mask: Binary mask of shape (H, W) where 1 indicates highlights
            rect_size: Target rectangle size as (height, width)

        Returns:
            (top, left, bottom, right) pixel coordinates of the rectangle with fewest highlights,
            or (0, 0, 0, 0) if target size doesn't fit in the image
        """
        if rect_size is None:
            return torch.tensor([0, 0, 0, 0]).float()

        target_height, target_width = rect_size
        img_height, img_width = binary_mask.shape

        # Check if target size fits in the image
        if target_height > img_height or target_width > img_width:
            return torch.tensor([0, 0, 0, 0]).float()

        # Calculate all possible positions where the rectangle can fit
        max_top = img_height - target_height + 1
        max_left = img_width - target_width + 1

        # Create cumulative sum array (integral image)
        cumsum = torch.cumsum(torch.cumsum(binary_mask, dim=0), dim=1)

        # Add zero padding for easier boundary handling
        padded_cumsum = F.pad(cumsum, (1, 0, 1, 0), value=0)

        # Create coordinate grids for all possible top-left positions
        tops = torch.arange(max_top, device=binary_mask.device)[
            :, None
        ]  # Shape: (max_top, 1)
        lefts = torch.arange(max_left, device=binary_mask.device)[
            None, :
        ]  # Shape: (1, max_left)

        # Calculate bottom-right coordinates for all rectangles
        bottoms = tops + target_height - 1  # Shape: (max_top, 1)
        rights = lefts + target_width - 1  # Shape: (1, max_left)

        # Vectorized calculation of all rectangle sums using integral image
        # This broadcasts to shape (max_top, max_left)
        highlights_counts = (
            padded_cumsum[bottoms + 1, rights + 1]  # Bottom-right corner
            - padded_cumsum[tops, rights + 1]  # Subtract top strip
            - padded_cumsum[bottoms + 1, lefts]  # Subtract left strip
            + padded_cumsum[tops, lefts]  # Add back overlap
        )

        # Find the position with minimum highlights
        min_highlights = torch.min(highlights_counts)
        min_pos = torch.argmin(highlights_counts)

        # Convert flat index back to 2D coordinates
        best_top = min_pos // max_left
        best_left = min_pos % max_left
        best_bottom = best_top + target_height - 1
        best_right = best_left + target_width - 1

        return torch.tensor([best_top, best_left, best_bottom, best_right]).int()

    def _compute_rectangles_for_framestack(self, framestack):
        """
        Compute rectangles with least highlights for a stack of frames.

        Args:
            framestack (torch.Tensor): Frame stack of shape (N, C, H, W)

        Returns:
            list: List of rectangle coordinates (top, left, bottom, right) for each frame
        """
        rectangles = []

        for i in range(framestack.shape[0]):
            # Get highlight mask for this frame
            highlight_mask = self._compute_highlight_mask(framestack[i])
            highlight_mask = highlight_mask.squeeze(
                0
            )  # Remove channel dimension: (H, W)

            # Find rectangle with least highlights
            rect_coords = self._find_rectangle_with_least_highlights(
                highlight_mask, self.rect_size
            )
            rectangles.append(rect_coords)

        return torch.stack(rectangles)

    def _compute_framestack_masks(self, framestack):
        """
        Compute highlight masks for a stack of frames.

        Args:
            framestack (torch.Tensor): Frame stack of shape (N, C, H, W)

        Returns:
            torch.Tensor: Mask stack of shape (N, 1, H, W)
        """
        masks = []
        for i in range(framestack.shape[0]):
            mask = self._compute_highlight_mask(framestack[i])
            masks.append(mask)

        return torch.stack(masks, dim=0)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset with optional highlight masks and rectangle crops.

        Args:
            idx (int): Index of the frame to retrieve

        Returns:
            dict: Dictionary containing the requested data, including:
                - highlight_masks: if return_mask=True
                - rect_crop: if return_rect=True, list of cropped rectangle frames
                - rect_mask: if return_rect=True, list of highlight masks for cropped rectangles
                - rectangles: if rect_size is specified, list of rectangle coordinates
        """
        # Get the base dataset item
        output = super().__getitem__(idx)

        # If the base __getitem__ returned None (error case), pass it through
        if output is None:
            return None

        # Add highlight masks if requested
        if self.return_mask and "framestack" in output:
            framestack = output["framestack"]
            highlight_masks = self._compute_framestack_masks(framestack)
            output["highlight_masks"] = highlight_masks

            # Also add some statistics about highlight coverage
            total_pixels = framestack.shape[-2] * framestack.shape[-1]  # H * W
            highlight_coverage = []

            for i in range(highlight_masks.shape[0]):
                mask = highlight_masks[i, 0]  # Remove channel dimension for counting
                highlight_pixels = mask.sum().item()
                coverage_percent = (highlight_pixels / total_pixels) * 100
                highlight_coverage.append(coverage_percent)

            output["highlight_coverage"] = torch.tensor(
                highlight_coverage, dtype=torch.float32
            )

        # Add rectangle detection if rect_size is specified
        if self.rect_size is not None and "framestack" in output:
            framestack = output["framestack"]
            rectangles = self._compute_rectangles_for_framestack(framestack)
            output["rect_coords"] = rectangles

            # Add cropped rectangles and their masks if return_rect is True
            if self.return_rect:
                cropped_frames, cropped_masks = self._crop_rectangles_from_framestack(
                    framestack, rectangles
                )
                output["rect_crop"] = torch.stack(cropped_frames)
                output["rect_mask"] = torch.stack(cropped_masks)

        return output

    @staticmethod
    def videonames():
        """
        Get video names. This should be overridden by the specific implementation
        or delegate to the base dataset class.
        """
        return []

    def get_highlight_statistics(self, idx):
        """
        Get highlight statistics for a specific frame without full data loading.

        Args:
            idx (int): Frame index

        Returns:
            dict: Statistics about highlights in the frame(s)
        """
        # Load just the framestack for analysis
        framestack, _, _ = self._load_frame_pair(idx)
        highlight_masks = self._compute_framestack_masks(framestack)

        stats = {
            "brightness_threshold": self.brightness_threshold,
            "frame_shapes": framestack.shape,
            "highlight_coverage": [],
        }

        total_pixels = framestack.shape[-2] * framestack.shape[-1]

        for i in range(highlight_masks.shape[0]):
            mask = highlight_masks[i, 0]
            highlight_pixels = mask.sum().item()
            coverage_percent = (highlight_pixels / total_pixels) * 100

            frame_stats = {
                "frame_idx": i,
                "highlight_pixels": int(highlight_pixels),
                "total_pixels": int(total_pixels),
                "coverage_percent": coverage_percent,
            }
            stats["highlight_coverage"].append(frame_stats)

        return stats

    def _crop_rectangle_from_frame(self, frame, rect_coords):
        """
        Crop a rectangle from a frame and create its corresponding mask.

        Args:
            frame (torch.Tensor): Input frame tensor of shape (C, H, W)
            rect_coords (tuple): Rectangle coordinates (top, left, bottom, right)

        Returns:
            tuple: (cropped_frame, cropped_mask) where:
                - cropped_frame: Tensor of shape (C, rect_height, rect_width)
                - cropped_mask: Tensor of shape (1, rect_height, rect_width)
        """
        top, left, bottom, right = rect_coords

        # Check if rectangle is valid (non-zero coordinates)
        if top == 0 and left == 0 and bottom == 0 and right == 0:
            # Invalid rectangle, return empty tensors
            if self.rect_size is not None:
                target_height, target_width = self.rect_size
                empty_frame = torch.zeros(
                    frame.shape[0],
                    target_height,
                    target_width,
                    device=frame.device,
                    dtype=frame.dtype,
                )
                empty_mask = torch.zeros(
                    1,
                    target_height,
                    target_width,
                    device=frame.device,
                    dtype=frame.dtype,
                )
                return empty_frame, empty_mask
            else:
                return frame, torch.zeros(
                    1,
                    frame.shape[1],
                    frame.shape[2],
                    device=frame.device,
                    dtype=frame.dtype,
                )

        # Crop the frame
        cropped_frame = frame[:, top : bottom + 1, left : right + 1]

        # Create highlight mask for the cropped region
        cropped_mask = self._compute_highlight_mask(cropped_frame)

        return cropped_frame, cropped_mask

    def _crop_rectangles_from_framestack(self, framestack, rectangles):
        """
        Crop rectangles from a framestack and create their corresponding masks.

        Args:
            framestack (torch.Tensor): Frame stack of shape (N, C, H, W)
            rectangles (list): List of rectangle coordinates for each frame

        Returns:
            tuple: (cropped_frames, cropped_masks) where:
                - cropped_frames: List of tensors, each of shape (C, rect_height, rect_width)
                - cropped_masks: List of tensors, each of shape (1, rect_height, rect_width)
        """
        cropped_frames = []
        cropped_masks = []

        for i in range(framestack.shape[0]):
            frame = framestack[i]
            rect_coords = rectangles[i]

            cropped_frame, cropped_mask = self._crop_rectangle_from_frame(
                frame, rect_coords
            )
            cropped_frames.append(cropped_frame)
            cropped_masks.append(cropped_mask)

        return cropped_frames, cropped_masks


# Convenience factory functions for common dataset types
class HighlightSCARED(HighlightDataset):
    """
    SCARED dataset with highlight detection capabilities.
    
    Wraps the SCARED (Stereo Correspondence and Reconstruction of Endoscopic Data)
    dataset with automatic highlight detection for surgical scene analysis.
    Highlights often correspond to specular reflections from surgical instruments
    and tissue surfaces under surgical lighting.
    """

    def __init__(
        self,
        brightness_threshold: float = 0.93,
        return_mask: bool = False,
        rect_size: Optional[Tuple[int, int]] = None,
        return_rect: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize SCARED dataset with highlight detection.
        
        Args:
            brightness_threshold: Threshold for highlight detection (0.0-1.0). Default: 0.93
            return_mask: Whether to return highlight masks. Default: False
            rect_size: Target rectangle size (height, width) for highlight-free regions. Default: None
            return_rect: Whether to return cropped rectangles. Default: False
            **kwargs: Additional arguments passed to SCARED dataset constructor
        """
        from .mono3d_dataset import SCARED  # Import here to avoid circular imports

        params = {
            "original_width": 1280,
            "original_height": 1024,
            "with_depth": True,
        }
        params.update(kwargs)

        super().__init__(
            base_dataset=SCARED,
            brightness_threshold=brightness_threshold,
            return_mask=return_mask,
            rect_size=rect_size,
            return_rect=return_rect,
            **params,
        )

    @staticmethod
    def videonames() -> List[str]:
        """Get the list of video names for the SCARED dataset."""
        return [f"v{i}" for i in range(1, 35)]


class HighlightCHOLEC80(HighlightDataset):
    """
    CHOLEC80 dataset with highlight detection capabilities.
    
    Wraps the CHOLEC80 (Cholecystectomy) dataset with automatic highlight detection
    for laparoscopic surgery analysis. Highlights typically occur on surgical
    instruments, gallbladder surface, and other reflective anatomical structures.
    """

    def __init__(
        self,
        brightness_threshold: float = 0.93,
        return_mask: bool = False,
        rect_size: Optional[Tuple[int, int]] = None,
        return_rect: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize CHOLEC80 dataset with highlight detection.
        
        Args:
            brightness_threshold: Threshold for highlight detection (0.0-1.0). Default: 0.93
            return_mask: Whether to return highlight masks. Default: False
            rect_size: Target rectangle size (height, width) for highlight-free regions. Default: None
            return_rect: Whether to return cropped rectangles. Default: False
            **kwargs: Additional arguments passed to CHOLEC80 dataset constructor
        """
        from . import CHOLEC80  # Import here to avoid circular imports

        params = {
            "original_width": 1280,
            "original_height": 1024,
            "with_depth": False,
        }
        params.update(kwargs)

        super().__init__(
            base_dataset=CHOLEC80,
            brightness_threshold=brightness_threshold,
            return_mask=return_mask,
            rect_size=rect_size,
            return_rect=return_rect,
            **params,
        )

    @staticmethod
    def videonames() -> List[str]:
        """Get the list of video names for the CHOLEC80 dataset."""
        return []  # Same as CHOLEC80


class HighlightGRASP(HighlightDataset):
    """
    GRASP dataset with highlight detection capabilities.
    
    Wraps the GRASP dataset with automatic highlight detection for robotic
    manipulation and grasping analysis. Highlights often appear on robotic
    end-effectors, grasped objects, and reflective surfaces in the workspace.
    """

    def __init__(
        self,
        brightness_threshold: float = 0.93,
        return_mask: bool = False,
        rect_size: Optional[Tuple[int, int]] = None,
        return_rect: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize GRASP dataset with highlight detection.
        
        Args:
            brightness_threshold: Threshold for highlight detection (0.0-1.0). Default: 0.93
            return_mask: Whether to return highlight masks. Default: False
            rect_size: Target rectangle size (height, width) for highlight-free regions. Default: None
            return_rect: Whether to return cropped rectangles. Default: False
            **kwargs: Additional arguments passed to GRASP dataset constructor
        """
        from . import GRASP  # Import here to avoid circular imports

        params = {
            "original_width": 640,
            "original_height": 400,
            "with_depth": False,
        }
        params.update(kwargs)

        super().__init__(
            base_dataset=GRASP,
            brightness_threshold=brightness_threshold,
            return_mask=return_mask,
            rect_size=rect_size,
            return_rect=return_rect,
            **params,
        )

    @staticmethod
    def videonames() -> List[str]:
        """Get the list of video names for the GRASP dataset."""
        return [f"v{i}" for i in range(1, 14)]


def draw_rect(
    image_tensor: torch.Tensor,
    color: Union[str, Tuple[float, ...], List[float]],
    thickness: int,
    rect_coords: Tuple[int, int, int, int],
) -> torch.Tensor:
    """
    Draw a rectangle on a torch tensor image.

    Args:
        image_tensor: Input image tensor of shape (C, H, W) or (H, W)
        color: Color specification:
            - Hex string (e.g., "#FF0000", "#ff0000")
            - RGB tuple/list (e.g., (255, 0, 0) or (1.0, 0.0, 0.0))
            - Color name string (e.g., "red", "green", "blue")
        thickness: Thickness of the rectangle border in pixels
        rect_coords: Rectangle coordinates as (top, left, bottom, right)

    Returns:
        torch.Tensor: Image tensor with rectangle drawn, same shape as input
    """
    # Handle empty or invalid rectangle coordinates
    if rect_coords is None or len(rect_coords) != 4:
        return image_tensor.clone()

    top, left, bottom, right = rect_coords

    # Handle case where no valid rectangle was found
    if top == bottom == left == right == 0:
        return image_tensor.clone()

    # Clone the input to avoid modifying the original
    output = image_tensor.clone()

    # Handle different input shapes
    if output.dim() == 2:
        # Grayscale image (H, W) -> convert to (1, H, W)
        output = output.unsqueeze(0)
        was_grayscale = True
    elif output.dim() == 3:
        # RGB image (C, H, W)
        was_grayscale = False
    else:
        raise ValueError(f"Unsupported image tensor shape: {output.shape}")

    C, H, W = output.shape

    # Parse color
    color_values = _parse_color(color, C)

    # Ensure coordinates are within image bounds
    top = max(0, min(H - 1, int(top)))
    bottom = max(0, min(H - 1, int(bottom)))
    left = max(0, min(W - 1, int(left)))
    right = max(0, min(W - 1, int(right)))

    # Ensure valid rectangle (top <= bottom, left <= right)
    if top > bottom or left > right:
        return image_tensor.clone()

    # Calculate thickness bounds
    thickness = max(1, int(thickness))

    # Draw rectangle borders
    for t in range(thickness):
        # Calculate border positions with thickness
        top_border = max(0, top - t)
        bottom_border = min(H - 1, bottom + t)
        left_border = max(0, left - t)
        right_border = min(W - 1, right + t)

        # Draw horizontal borders (top and bottom)
        if top_border < H and left_border <= right_border:
            for c in range(C):
                output[c, top_border, left_border : right_border + 1] = color_values[c]

        if (
            bottom_border < H
            and bottom_border != top_border
            and left_border <= right_border
        ):
            for c in range(C):
                output[c, bottom_border, left_border : right_border + 1] = color_values[
                    c
                ]

        # Draw vertical borders (left and right)
        if left_border < W and top_border <= bottom_border:
            for c in range(C):
                output[c, top_border : bottom_border + 1, left_border] = color_values[c]

        if (
            right_border < W
            and right_border != left_border
            and top_border <= bottom_border
        ):
            for c in range(C):
                output[c, top_border : bottom_border + 1, right_border] = color_values[
                    c
                ]

    # Return original shape
    if was_grayscale:
        output = output.squeeze(0)

    return output


def _parse_color(
    color: Union[str, Tuple[float, ...], List[float]], num_channels: int
) -> torch.Tensor:
    """
    Parse color specification into tensor values.

    Args:
        color: Color specification
        num_channels: Number of channels in the target image

    Returns:
        Color values for each channel
    """
    # Color name mapping
    color_names = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
        "white": (1.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0),
        "orange": (1.0, 0.5, 0.0),
        "purple": (0.5, 0.0, 0.5),
    }

    if isinstance(color, str):
        if color.startswith("#"):
            # Hex color
            hex_color = color[1:]
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                rgb = (r, g, b)
            elif len(hex_color) == 3:
                r = int(hex_color[0], 16) / 15.0
                g = int(hex_color[1], 16) / 15.0
                b = int(hex_color[2], 16) / 15.0
                rgb = (r, g, b)
            else:
                raise ValueError(f"Invalid hex color format: {color}")
        else:
            # Color name
            color_lower = color.lower()
            if color_lower in color_names:
                rgb = color_names[color_lower]
            else:
                raise ValueError(f"Unknown color name: {color}")
    elif isinstance(color, (tuple, list)):
        if len(color) >= 3:
            rgb = tuple(color[:3])
            # Check if values are in 0-255 range (convert to 0-1)
            if all(isinstance(x, int) and 0 <= x <= 255 for x in rgb):
                rgb = tuple(x / 255.0 for x in rgb)
            elif not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in rgb):
                raise ValueError(
                    f"RGB values must be in range [0, 255] or [0, 1]: {rgb}"
                )
        else:
            raise ValueError(f"RGB color must have at least 3 values: {color}")
    else:
        raise ValueError(f"Unsupported color type: {type(color)}")

    # Handle different number of channels
    if num_channels == 1:
        # Convert to grayscale using luminance formula
        gray_value = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return torch.tensor([gray_value], dtype=torch.float32)
    elif num_channels == 3:
        return torch.tensor(list(rgb), dtype=torch.float32)
    elif num_channels == 4:
        # Add alpha channel (fully opaque)
        return torch.tensor(list(rgb) + [1.0], dtype=torch.float32)
    else:
        # For other channel numbers, repeat the first channel value
        return torch.tensor([rgb[0]] * num_channels, dtype=torch.float32)
