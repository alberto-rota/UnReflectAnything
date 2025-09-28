import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from logger import get_logger

logger = get_logger(__name__).set_context("DATASET")

try:
    import polanalyser as pa

    POLANALYSER_AVAILABLE = True
except ImportError:
    POLANALYSER_AVAILABLE = False
    logger.warning("Warning: polanalyser not available. Using manual implementation.")


class RGBP_Dataset(Dataset):
    """
    Optimized version of RGBP_Dataset dataset with performance improvements:
    1. Reduced tensor/numpy conversions
    2. Optional caching
    3. Simplified processing pipeline
    4. Batch-friendly operations
    5. Consistent image sizing for batching
    6. Support for both single file clockwise and separate polarization files
    7. Flexible scene filtering with include/exclude patterns

    Polarization formats supported:
    - "single_file_clock": Single file with 4 polarization images arranged clockwise
    - "separate_files": Four separate files ending with _000, _045, _090, _135

    Scene filtering:
    - include: str or List[str] - Include scenes that match (substring or exact)
    - exclude: str or List[str] - Exclude scenes that match (substring or exact)
    """

    def __init__(
        self,
        root_dir: str,
        rho_s: float = 0.6,
        eps: float = 1e-8,
        rgb_ext: str = ".png",
        pol_ext: str = ".png",
        transform=None,
        # Polarization data format
        polarization_format: str = "single_file_clock",  # "single_file_clock", "separate_files" or "mosaic"
        # Image sizing parameters
        target_size: Optional[Tuple[int, int]] = (874, 1132),  # Default size (H, W)
        resize_mode: str = "crop",  # "crop", "resize", or "pad"
        # Performance optimizations
        use_cache: bool = False,
        cache_size: int = 100,
        simplify_upsampling: bool = True,  # Use simple bicubic instead of edge-aware
        precompute_stokes: bool = False,  # Precompute Stokes parameters
        # Reduced smoothing options
        smooth_specular: bool = True,
        gaussian_sigma: float = 0.0,
        dolp_min_intensity: float = 0.05,
        dolp_min_value: float = 0.04,
        # Scene filtering
        include: Optional[Union[str, List[str]]] = None,
        exclude: Optional[Union[str, List[str]]] = None,
        # Few images mode for quick testing
        few_images: bool = False,
        overfit_test: bool = False,
        # Deprecated parameters (for backward compatibility)
        # Highlight detection (optional)
        highlight_enable: bool = False,
        highlight_brightness_threshold: float = 0.93,
        highlight_return_mask: bool = False,
        highlight_rect_size: Optional[Tuple[int, int]] = None,
        highlight_return_rect: bool = False,
        highlight_return_rect_as_rgb: bool = False,
    ):
        self.root_dir = os.path.expandvars(root_dir)
        self.rho_s = rho_s
        self.eps = eps
        self.rgb_ext = rgb_ext
        self.pol_ext = pol_ext
        self.transform = transform

        # Polarization format validation
        self.polarization_format = polarization_format.lower()
        if self.polarization_format not in [
            "single_file_clock",
            "separate_files",
            "mosaic",
        ]:
            raise ValueError(
                f"polarization_format must be one of ['single_file_clock', 'separate_files', 'mosaic'], got {polarization_format}"
            )

        # Image sizing parameters
        self.target_size = target_size
        self.resize_mode = resize_mode.lower()
        if self.resize_mode not in ["crop", "resize", "pad"]:
            raise ValueError(
                f"resize_mode must be one of ['crop', 'resize', 'pad'], got {resize_mode}"
            )

        self.use_cache = use_cache
        self.simplify_upsampling = simplify_upsampling
        self.precompute_stokes = precompute_stokes

        # Simplified smoothing config
        self.smooth_specular = smooth_specular
        self.gaussian_sigma = gaussian_sigma
        self.dolp_min_intensity = dolp_min_intensity
        self.dolp_min_value = dolp_min_value

        # Store filtering parameters
        self.include = include
        self.exclude = exclude or []
        self.few_images = few_images

        self.scene_pairs = self._find_scene_pairs()

        # Limit to 32 samples if few_images is True
        if self.few_images and len(self.scene_pairs) > 8:
            original_count = len(self.scene_pairs)
            self.scene_pairs = self.scene_pairs[:8]
            logger.info(
                f"Few images mode: Limited dataset from {original_count} to 8 samples"
            )

        # Setup caching if enabled
        if self.use_cache:
            self._cache_intrinsics = {}
            # Make cached methods
            self._load_intrinsics_cached = lru_cache(maxsize=cache_size)(
                self._load_intrinsics_impl
            )
            if self.precompute_stokes:
                self._load_pol_cached = lru_cache(maxsize=cache_size)(
                    self._load_and_process_polarization_impl
                )
        self.overfit_test = overfit_test
        if self.overfit_test:
            self.scene_pairs = len(self.scene_pairs) * [self.scene_pairs[0]]
            logger.info(
                f"Overfit test mode: Limited dataset from {len(self.scene_pairs)} to 1 sample"
            )

        # Highlight detection configuration
        self.highlight_enabled = highlight_enable
        self.highlight_brightness_threshold = highlight_brightness_threshold
        self.highlight_return_mask = highlight_return_mask
        self.highlight_rect_size = highlight_rect_size
        self.highlight_return_rect = highlight_return_rect
        self.highlight_return_rect_as_rgb = highlight_return_rect_as_rgb

    def _resize_tensor(
        self, tensor: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Resize tensor to target size using specified mode.

        Args:
            tensor: Input tensor of shape [C, H, W] or [H, W]
            target_size: Target size as (H, W)

        Returns:
            Resized tensor of shape [C, target_H, target_W] or [target_H, target_W]
        """
        if tensor.dim() == 2:
            # Single channel tensor [H, W]
            tensor = tensor.unsqueeze(0)  # [1, H, W]
            was_2d = True
        else:
            was_2d = False

        current_size = tensor.shape[-2:]  # [H, W]

        if self.resize_mode == "crop":
            # Center crop to target size
            if current_size[0] > target_size[0]:
                start_h = (current_size[0] - target_size[0]) // 2
                end_h = start_h + target_size[0]
                tensor = tensor[..., start_h:end_h, :]
            if current_size[1] > target_size[1]:
                start_w = (current_size[1] - target_size[1]) // 2
                end_w = start_w + target_size[1]
                tensor = tensor[..., :, start_w:end_w]

        elif self.resize_mode == "resize":
            # Resize to target size using bilinear interpolation
            tensor = F.interpolate(
                tensor.unsqueeze(0),  # Add batch dimension
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # Remove batch dimension

        elif self.resize_mode == "pad":
            # Pad to target size with zeros
            pad_h = max(0, target_size[0] - current_size[0])
            pad_w = max(0, target_size[1] - current_size[1])

            if pad_h > 0 or pad_w > 0:
                # Pad format: (pad_left, pad_right, pad_top, pad_bottom)
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top

                tensor = F.pad(
                    tensor,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=0,
                )

        if was_2d:
            tensor = tensor.squeeze(0)  # Remove channel dimension

        return tensor

    def _resize_rgb_tensor(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Resize RGB tensor to target size.

        Args:
            rgb: RGB tensor of shape [3, H, W] in [0, 1]

        Returns:
            Resized RGB tensor of shape [3, target_H, target_W] in [0, 1]
        """
        if self.target_size is None:
            return rgb

        return self._resize_tensor(rgb, self.target_size)

    def _resize_polarization_data(
        self, pol_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Resize all polarization data tensors to target size.

        Args:
            pol_data: Dictionary containing polarization tensors with shapes [C, H, W]
                     Keys include 'I0', 'I45', 'I90', 'I135', 'S0', 'S1', 'S2', 'DoLP', etc.

        Returns:
            Dictionary with same keys but tensors resized to target size [C, target_H, target_W]
        """
        """
        Resize all polarization data tensors to target size.

        Args:
            pol_data: Dictionary containing polarization data tensors

        Returns:
            Dictionary with all tensors resized to target size
        """
        if self.target_size is None:
            return pol_data

        resized_data = {}
        for key, tensor in pol_data.items():
            resized_data[key] = self._resize_tensor(tensor, self.target_size)
        return resized_data

    def _should_include_scene(self, scene_name: str) -> bool:
        """
        Check if a scene should be included based on include/exclude filters.

        Args:
            scene_name: Name of the scene to check

        Returns:
            True if scene should be included, False otherwise
        """
        """
        Check if a scene should be included based on include/exclude filters.

        Args:
            scene_name: Name of the scene to check

        Returns:
            True if scene should be included, False otherwise
        """
        # Check exclude filter first
        if self.exclude:
            if isinstance(self.exclude, str):
                # Single string - check if scene name contains the exclude string
                if self.exclude in scene_name:
                    return False
            elif isinstance(self.exclude, list):
                # List of strings - check exact matches and substring matches
                for exclude_pattern in self.exclude:
                    if exclude_pattern == scene_name or exclude_pattern in scene_name:
                        return False

        # Check include filter
        if self.include is not None:
            if isinstance(self.include, str):
                # Single string - check if scene name contains the include string
                return self.include in scene_name
            elif isinstance(self.include, list):
                # List of strings - check exact matches and substring matches
                for include_pattern in self.include:
                    if include_pattern == scene_name or include_pattern in scene_name:
                        return True
                # If include list is provided but no match found, exclude the scene
                return False

        # If no include filter specified and not excluded, include the scene
        return True

    def _find_scene_pairs(self) -> List[Tuple[str, str, str, bool]]:
        """
        Find matching RGB, polarization, and intrinsics file triplets.

        Scans the root directory structure to find corresponding files:
        - RGB images in rgb/ subdirectory
        - Polarization data in pol/ subdirectory (optional)
        - Camera intrinsics in intrinsics/ subdirectory (optional)

        Returns:
            List of tuples (rgb_path, pol_path, intrinsics_path, has_pol_data) for valid scene pairs
        """
        """Find valid RGB-polarization pairs based on polarization format"""
        scene_pairs = []
        for scene_name in os.listdir(self.root_dir):
            # Use new filtering logic
            if not self._should_include_scene(scene_name):
                continue

            scene_path = os.path.join(self.root_dir, scene_name)
            if not os.path.isdir(scene_path):
                continue

            rgb_dir = os.path.join(scene_path, "rgb")
            pol_dir = os.path.join(scene_path, "pol")
            intrinsics_path = os.path.join(scene_path, "intrinsics.txt")

            # Check if RGB directory exists (required)
            if not os.path.exists(rgb_dir):
                continue

            # Check if polarization directory exists (optional)
            has_pol_data = os.path.exists(pol_dir)

            # Set intrinsics_path to None if file doesn't exist
            if not os.path.exists(intrinsics_path):
                intrinsics_path = None

            rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(self.rgb_ext)]

            # Get polarization files if pol directory exists
            pol_files = []
            if has_pol_data:
                pol_files = [f for f in os.listdir(pol_dir) if f.endswith(self.pol_ext)]

            for rgb_file in rgb_files:
                if not has_pol_data:
                    # No polarization data available - include RGB-only sample
                    scene_pairs.append(
                        (
                            os.path.join(rgb_dir, rgb_file),
                            None,  # No polarization path
                            intrinsics_path,
                            False,  # No polarization data
                        )
                    )
                elif self.polarization_format == "single_file_clock":
                    # Original behavior: single polarization file
                    pol_file = rgb_file.replace(self.rgb_ext, self.pol_ext)
                    if pol_file in pol_files:
                        scene_pairs.append(
                            (
                                os.path.join(rgb_dir, rgb_file),
                                os.path.join(pol_dir, pol_file),
                                intrinsics_path,
                                True,  # Has polarization data
                            )
                        )

                elif self.polarization_format == "separate_files":
                    # New behavior: separate polarization files (_000, _045, _090, _135)
                    base_name = rgb_file.replace(self.rgb_ext, "")
                    pol_files_needed = [
                        f"{base_name}_000{self.pol_ext}",
                        f"{base_name}_045{self.pol_ext}",
                        f"{base_name}_090{self.pol_ext}",
                        f"{base_name}_135{self.pol_ext}",
                    ]

                    # Check if all 4 polarization files exist
                    if all(pol_file in pol_files for pol_file in pol_files_needed):
                        # Store the base path for separate files (we'll construct individual paths later)
                        pol_base_path = os.path.join(pol_dir, base_name)
                        scene_pairs.append(
                            (
                                os.path.join(rgb_dir, rgb_file),
                                pol_base_path,  # Base path for separate files
                                intrinsics_path,
                                True,  # Has polarization data
                            )
                        )

        return scene_pairs

    def __len__(self) -> int:
        return len(self.scene_pairs)

    def get_loaded_scenes(self) -> List[str]:
        """
        Get list of scene names that were loaded into the dataset.

        Returns:
            List of scene names (without file extensions) that passed filtering
        """
        """Get list of scene names that are actually loaded in the dataset."""
        loaded_scenes = set()
        for rgb_path, _, _ in self.scene_pairs:
            scene_name = os.path.basename(os.path.dirname(os.path.dirname(rgb_path)))
            loaded_scenes.add(scene_name)
        return sorted(list(loaded_scenes))

    @staticmethod
    def _to_luminance_torch(rgb: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB tensor to luminance using standard weights.

        Args:
            rgb: RGB tensor of shape [H, W, 3] in range [0, 1]

        Returns:
            Luminance tensor of shape [H, W] in range [0, 1]
        """
        """Convert RGB to luminance staying in torch. Input: [...,3] in [0,1]."""
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    def _load_intrinsics_impl(self, intrinsics_path: str) -> torch.Tensor:
        """
        Load camera intrinsics from file (implementation without caching).

        Args:
            intrinsics_path: Path to intrinsics file (typically .txt or .npy)

        Returns:
            Camera intrinsics matrix of shape [3, 3]
        """
        """Implementation for loading intrinsics (cacheable)"""
        try:
            K = np.loadtxt(intrinsics_path).reshape(3, 3).astype(np.float32)
            return torch.from_numpy(K)
        except Exception:
            # warnings.warn(f"Could not load intrinsics from {intrinsics_path}: {e}")
            return torch.eye(3, dtype=torch.float32)

    def _load_intrinsics(self, intrinsics_path: str) -> torch.Tensor:
        """
        Load camera intrinsics with optional caching.

        Args:
            intrinsics_path: Path to intrinsics file

        Returns:
            Camera intrinsics matrix of shape [3, 3]
        """
        """Load intrinsics with optional caching"""
        if self.use_cache:
            return self._load_intrinsics_cached(intrinsics_path)
        else:
            return self._load_intrinsics_impl(intrinsics_path)

    def _simple_upsample(
        self, f_spec_half: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Simple bilinear upsampling - much faster than edge-aware methods.

        Args:
            f_spec_half: Specular fraction tensor of shape [H, W] at half resolution
            target_size: Target size as (H, W) for upsampling

        Returns:
            Upsampled tensor of shape [target_H, target_W] clamped to [0, 1]
        """
        # Use torch interpolation instead of OpenCV
        f_batch = f_spec_half.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        f_up = torch.nn.functional.interpolate(
            f_batch, size=target_size, mode="bilinear", align_corners=False
        )
        return f_up.squeeze(0).squeeze(0).clamp(0, 1)

    def _load_and_process_polarization_impl(
        self, pol_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Load and process polarization data based on format.

        Args:
            pol_path: For single_file_clock format, this is the full file path.
                     For separate_files format, this is the base path without extension.
        """
        if self.polarization_format == "single_file_clock":
            return self._load_single_file_polarization(pol_path)
        elif self.polarization_format == "separate_files":
            return self._load_separate_polarization_files(pol_path)
        else:
            raise ValueError(f"Unknown polarization format: {self.polarization_format}")

    def _load_single_file_polarization(self, pol_path: str) -> Dict[str, torch.Tensor]:
        """Load polarization data from single file with clockwise arrangement"""

        # Load image directly as torch tensor
        pol_img = Image.open(pol_path).convert("RGB")
        pol_rgb = torch.from_numpy(np.asarray(pol_img, dtype=np.float32)) / 255.0

        H, W, _ = pol_rgb.shape
        hh, hw = H // 2, W // 2

        # Split quadrants (clockwise arrangement)
        I0_rgb = pol_rgb[:hh, :hw, :]
        I45_rgb = pol_rgb[:hh, hw:, :]
        I90_rgb = pol_rgb[hh:, hw:, :]
        I135_rgb = pol_rgb[hh:, :hw, :]

        # Convert to luminance (staying in torch)
        I0 = self._to_luminance_torch(I0_rgb)
        I45 = self._to_luminance_torch(I45_rgb)
        I90 = self._to_luminance_torch(I90_rgb)
        I135 = self._to_luminance_torch(I135_rgb)

        # Compute Stokes parameters
        S0 = I0 + I90
        S1 = I0 - I90
        S2 = I45 - I135

        # Compute DoLP and AoP
        DoLP = torch.clamp(
            torch.sqrt(S1**2 + S2**2) / torch.clamp(S0, min=self.eps), 0.0, 1.0
        )
        AoP = 0.5 * torch.atan2(S2, S1)

        # Apply DoLP masking
        valid_mask = (S0 >= self.dolp_min_intensity).float()
        DoLP = DoLP * valid_mask
        DoLP = torch.where(DoLP < self.dolp_min_value, torch.zeros_like(DoLP), DoLP)

        # Compute specular fraction
        f_spec = DoLP * S0
        # Normalize to be max 1
        f_spec = torch.clamp(f_spec / torch.max(f_spec), 0.0, 1.0)
        # f_spec = torch.clamp(DoLP / max(self.rho_s, 1e-6), 0.0, 1.0)

        # Additional parameters (simplified)
        S3 = torch.zeros_like(S0)

        # Create polarization data dictionary
        pol_data = {
            "I0": I0_rgb.permute(2, 0, 1),
            "I45": I45_rgb.permute(2, 0, 1),
            "I90": I90_rgb.permute(2, 0, 1),
            "I135": I135_rgb.permute(2, 0, 1),
            "S0": S0.unsqueeze(0),
            "S1": S1.unsqueeze(0),
            "S2": S2.unsqueeze(0),
            "S3": S3.unsqueeze(0),
            "stokes": torch.cat([S0, S1, S2], dim=0).unsqueeze(0),
            "intensity": S0.unsqueeze(0),
            "DoLP": DoLP.unsqueeze(0),
            "AoP": AoP.unsqueeze(0),
            "AoLP": AoP.unsqueeze(0),
            "DoP": DoLP.unsqueeze(0),
            "DoCP": torch.zeros_like(DoLP).unsqueeze(0),
            "ellipticity_angle": torch.zeros_like(DoLP).unsqueeze(0),
            "f_spec": f_spec.unsqueeze(0),
        }

        # Resize all polarization data to target size if specified
        if self.target_size is not None:
            pol_data = self._resize_polarization_data(pol_data)

        return pol_data

    def _load_and_process_polarization(self, pol_path: str) -> Dict[str, torch.Tensor]:
        """Load polarization with optional caching"""
        if self.use_cache and self.precompute_stokes:
            return self._load_pol_cached(pol_path)
        else:
            return self._load_and_process_polarization_impl(pol_path)

    def _load_rgb_and_separate(
        self, rgb_path: str, f_spec_half: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Optimized RGB loading and separation"""

        # Load RGB directly as torch tensor
        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb = torch.from_numpy(np.asarray(rgb_img, dtype=np.float32)) / 255.0
        H, W, _ = rgb.shape

        # Upsample specular fraction
        if self.simplify_upsampling:
            f_full = self._simple_upsample(f_spec_half, (H, W))
        else:
            # Fall back to original method if needed
            f_full = self._edge_aware_upsample_original(f_spec_half, rgb)

        # Compute specular/diffuse separation
        rgb_chw = rgb.permute(2, 0, 1)  # 3xHxW
        f_expanded = f_full.unsqueeze(0)  # 1xHxW

        I_spec = (f_expanded * rgb_chw).clamp(0, 1)
        I_diff = (rgb_chw - I_spec).clamp(0, 1)

        # Resize RGB data to target size if specified
        if self.target_size is not None:
            rgb_chw = self._resize_rgb_tensor(rgb_chw)
            I_spec = self._resize_tensor(I_spec, self.target_size)
            I_diff = self._resize_tensor(I_diff, self.target_size)

        return {
            "rgb": rgb_chw,
            "specular": I_spec,
            "diffuse": I_diff,
        }

    def _load_rgb_only(self, rgb_path: str) -> Dict[str, torch.Tensor]:
        """
        Load RGB data only (when polarization data is not available).

        Args:
            rgb_path: Path to RGB image file

        Returns:
            Dictionary containing RGB data with dummy specular/diffuse components
        """
        # Load RGB directly as torch tensor
        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb = torch.from_numpy(np.asarray(rgb_img, dtype=np.float32)) / 255.0

        # Convert to CHW format
        rgb_chw = rgb.permute(2, 0, 1)  # [3, H, W]

        # Resize RGB data to target size if specified
        if self.target_size is not None:
            rgb_chw = self._resize_rgb_tensor(rgb_chw)

        # Create dummy specular and diffuse components (all zeros for specular, RGB for diffuse)
        I_spec = torch.zeros_like(rgb_chw)  # [3, H, W] - no specular component
        I_diff = rgb_chw.clone()  # [3, H, W] - all RGB is diffuse

        return {
            "rgb": rgb_chw,
            "specular": I_spec,
            "diffuse": I_diff,
        }

    def _compute_highlight_mask(self, frame_chw: torch.Tensor) -> torch.Tensor:
        """
        Compute binary highlight mask for a single RGB frame.

        Args:
            frame_chw: Tensor of shape [C, H, W]

        Returns:
            Tensor of shape [1, H, W] with 1 at highlight pixels
        """
        if frame_chw.dim() == 3:
            if frame_chw.shape[0] == 3:
                # Use luminance weights consistent with HighlightDataset
                grayscale = (
                    0.299 * frame_chw[0] + 0.587 * frame_chw[1] + 0.114 * frame_chw[2]
                )
            else:
                grayscale = frame_chw.mean(dim=0)
        elif frame_chw.dim() == 2:
            grayscale = frame_chw
        else:
            raise ValueError(
                f"Unexpected frame dimensions for highlight mask: {frame_chw.shape}"
            )

        mask = (grayscale > self.highlight_brightness_threshold).float().unsqueeze(0)
        return mask

    def _find_rectangle_with_least_highlights(
        self, binary_mask_hw: torch.Tensor, rect_size: Optional[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Find rect (top, left, bottom, right) with fewest highlights in a binary mask.

        Args:
            binary_mask_hw: Tensor [H, W] of 0/1
            rect_size: (height, width) or None

        Returns:
            Int tensor [4] = (top, left, bottom, right)
        """
        if rect_size is None:
            return torch.tensor([0, 0, 0, 0]).int()

        target_height, target_width = rect_size
        img_height, img_width = binary_mask_hw.shape

        if target_height > img_height or target_width > img_width:
            return torch.tensor([0, 0, 0, 0]).int()

        max_top = img_height - target_height + 1
        max_left = img_width - target_width + 1

        cumsum = torch.cumsum(torch.cumsum(binary_mask_hw, dim=0), dim=1)
        padded_cumsum = F.pad(cumsum, (1, 0, 1, 0), value=0)

        tops = torch.arange(max_top, device=binary_mask_hw.device)[:, None]
        lefts = torch.arange(max_left, device=binary_mask_hw.device)[None, :]
        bottoms = tops + target_height - 1
        rights = lefts + target_width - 1

        highlight_counts = (
            padded_cumsum[bottoms + 1, rights + 1]
            - padded_cumsum[tops, rights + 1]
            - padded_cumsum[bottoms + 1, lefts]
            + padded_cumsum[tops, lefts]
        )

        min_pos = torch.argmin(highlight_counts)
        best_top = (min_pos // max_left).item()
        best_left = (min_pos % max_left).item()
        best_bottom = best_top + target_height - 1
        best_right = best_left + target_width - 1

        return torch.tensor([best_top, best_left, best_bottom, best_right]).int()

    def _crop_rectangle_from_rgb(
        self, rgb_chw: torch.Tensor, rect_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Crop rectangle from RGB and compute its highlight mask.

        Returns:
            (cropped_rgb [C,h,w], cropped_mask [1,h,w])
        """
        top, left, bottom, right = [int(v) for v in rect_coords]

        if top == 0 and left == 0 and bottom == 0 and right == 0:
            if self.highlight_rect_size is not None:
                h, w = self.highlight_rect_size
                empty_frame = torch.zeros(
                    rgb_chw.shape[0], h, w, device=rgb_chw.device, dtype=rgb_chw.dtype
                )
                empty_mask = torch.zeros(
                    1, h, w, device=rgb_chw.device, dtype=rgb_chw.dtype
                )
                return empty_frame, empty_mask
            else:
                return rgb_chw, torch.zeros(
                    1,
                    rgb_chw.shape[1],
                    rgb_chw.shape[2],
                    device=rgb_chw.device,
                    dtype=rgb_chw.dtype,
                )

        cropped_rgb = rgb_chw[:, top : bottom + 1, left : right + 1]
        cropped_mask = self._compute_highlight_mask(cropped_rgb)
        return cropped_rgb, cropped_mask

    def _edge_aware_upsample_original(
        self, f_half: torch.Tensor, rgb_full: torch.Tensor
    ) -> torch.Tensor:
        """Original edge-aware upsampling (fallback)"""
        # Convert to numpy for OpenCV operations
        f_np = f_half.cpu().numpy()
        rgb_np = rgb_full.cpu().numpy()

        H, W, _ = rgb_np.shape
        f_up = cv2.resize(f_np, (W, H), interpolation=cv2.INTER_CUBIC)

        # Simple bilateral filter instead of guided filter for speed
        f_up = cv2.bilateralFilter(
            f_up.astype(np.float32), d=5, sigmaColor=0.1 * 255, sigmaSpace=5
        )

        return torch.from_numpy(np.clip(f_up, 0.0, 1.0))

    def _load_separate_polarization_files(
        self, pol_base_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Load polarization data from separate files (_000, _045, _090, _135).

        Args:
            pol_base_path: Base path without extension (e.g., "/path/to/pol/000034")

        Returns:
            Dictionary containing polarization data tensors
        """
        # Construct individual file paths
        pol_paths = {
            "000": f"{pol_base_path}_000{self.pol_ext}",
            "045": f"{pol_base_path}_045{self.pol_ext}",
            "090": f"{pol_base_path}_090{self.pol_ext}",
            "135": f"{pol_base_path}_135{self.pol_ext}",
        }

        # Load individual polarization images
        pol_images = {}
        for angle, path in pol_paths.items():
            img = Image.open(path).convert("RGB")
            pol_images[angle] = (
                torch.from_numpy(np.asarray(img, dtype=np.float32)) / 255.0
            )

        # Extract RGB data for each polarization angle
        I0_rgb = pol_images["000"]  # 0 degrees
        I45_rgb = pol_images["045"]  # 45 degrees
        I90_rgb = pol_images["090"]  # 90 degrees
        I135_rgb = pol_images["135"]  # 135 degrees

        # Convert to luminance (staying in torch)
        I0 = self._to_luminance_torch(I0_rgb)
        I45 = self._to_luminance_torch(I45_rgb)
        I90 = self._to_luminance_torch(I90_rgb)
        I135 = self._to_luminance_torch(I135_rgb)

        # Compute Stokes parameters
        S0 = I0 + I90
        S1 = I0 - I90
        S2 = I45 - I135

        # Compute DoLP and AoP
        R = torch.sqrt(S1**2 + S2**2)
        DoLP = torch.clamp(R / torch.clamp(S0, min=self.eps), 0.0, 1.0)
        AoP = 0.5 * torch.atan2(S2, S1)

        # Apply DoLP masking
        valid_mask = (S0 >= self.dolp_min_intensity).float()
        DoLP = DoLP * valid_mask
        DoLP = torch.where(DoLP < self.dolp_min_value, torch.zeros_like(DoLP), DoLP)

        # Compute specular fraction
        f_spec = torch.clamp(DoLP / max(self.rho_s, 1e-6), 0.0, 1.0)

        # Additional parameters (simplified)
        S3 = torch.zeros_like(S0)

        # Create polarization data dictionary
        pol_data = {
            "I0": I0_rgb.permute(2, 0, 1),
            "I45": I45_rgb.permute(2, 0, 1),
            "I90": I90_rgb.permute(2, 0, 1),
            "I135": I135_rgb.permute(2, 0, 1),
            "S0": S0.unsqueeze(0),
            "S1": S1.unsqueeze(0),
            "S2": S2.unsqueeze(0),
            "S3": S3.unsqueeze(0),
            "stokes": torch.cat([S0, S1, S2], dim=0).unsqueeze(0),
            "intensity": S0.unsqueeze(0),
            "DoLP": DoLP.unsqueeze(0),
            "AoP": AoP.unsqueeze(0),
            "AoLP": AoP.unsqueeze(0),
            "DoP": DoLP.unsqueeze(0),
            "DoCP": torch.zeros_like(DoLP).unsqueeze(0),
            "ellipticity_angle": torch.zeros_like(DoLP).unsqueeze(0),
            "f_spec": f_spec.unsqueeze(0),
        }

        # Resize all polarization data to target size if specified
        if self.target_size is not None:
            pol_data = self._resize_polarization_data(pol_data)

        return pol_data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a single dataset sample with optimized processing.

        Args:
            idx: Index of the sample to load (0 <= idx < len(dataset))

        Returns:
            Dictionary containing:
            - Polarization data: 'I0', 'I45', 'I90', 'I135', 'S0', 'S1', 'S2', 'DoLP', 'AoP', 'f_spec' (if available)
            - RGB data: 'rgb', 'specular', 'diffuse'
            - Camera data: 'intrinsics' [3, 3]
            All image tensors have shape [C, H, W] where H, W match target_size if specified
        """
        rgb_path, pol_path, intrinsics_path, has_pol_data = self.scene_pairs[idx]

        # Load data (potentially from cache)
        intrinsics = self._load_intrinsics(intrinsics_path)

        if has_pol_data:
            # Load polarization data
            pol_data = self._load_and_process_polarization(pol_path)
            # Get specular fraction for RGB processing
            f_spec = pol_data["f_spec"].squeeze(0)
            rgb_data = self._load_rgb_and_separate(rgb_path, f_spec)
            # Combine results
            sample = {**pol_data, **rgb_data, "intrinsics": intrinsics}
        else:
            # No polarization data available - load RGB only
            rgb_data = self._load_rgb_only(rgb_path)
            sample = {**rgb_data, "intrinsics": intrinsics}

        # Optional highlight outputs (computed on the resized rgb)
        if self.highlight_enabled and "rgb" in sample:
            rgb_chw = sample["rgb"]
            mask = self._compute_highlight_mask(rgb_chw)

            if self.highlight_return_mask:
                sample["highlight_masks"] = mask
                total_pixels = rgb_chw.shape[-2] * rgb_chw.shape[-1]
                coverage_percent = (mask.sum() / max(total_pixels, 1)) * 100.0
                sample["highlight_coverage"] = coverage_percent.to(torch.float32)

            if self.highlight_rect_size is not None:
                rect_coords = self._find_rectangle_with_least_highlights(
                    mask.squeeze(0), self.highlight_rect_size
                )
                sample["rect_coords"] = rect_coords

                if self.highlight_return_rect:
                    rect_rgb, rect_mask = self._crop_rectangle_from_rgb(
                        rgb_chw, rect_coords
                    )

                    # If both target_size and rect_size are set, resize the rect to target size
                    # Keep native rect size when it's used as the main RGB (cropped training)
                    if (
                        self.target_size is not None
                        and self.highlight_rect_size is not None
                        and not self.highlight_return_rect_as_rgb
                    ):
                        rect_rgb = self._resize_rgb_tensor(rect_rgb)
                        rect_mask = F.interpolate(
                            rect_mask.unsqueeze(0), size=self.target_size, mode="nearest", align_corners=None
                        ).squeeze(0)

                    sample["rect_crop"] = rect_rgb
                    sample["rect_mask"] = rect_mask

                    if self.highlight_return_rect_as_rgb:
                        sample["uncropped_rgb"] = sample["rgb"]
                        sample["rgb"] = rect_rgb

        if self.transform:
            sample = self.transform(sample)

        return sample


# Dataset-specific classes inheriting from base RGBP_Dataset class
class SCRREAM_Dataset(RGBP_Dataset):
    """
    SCRREAM dataset implementation for polarization-based reflection removal.

    Inherits all functionality from the base RGBP_Dataset class.
    This class can be extended with SCRREAM-specific preprocessing,
    data augmentation, or validation logic as needed.

    The SCRREAM dataset contains RGB images with corresponding polarization
    data for training reflection removal models.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize SCRREAM dataset.

        Args:
            **kwargs: All arguments passed to parent RGBP_Dataset class
        """
        super().__init__(**kwargs)
        # Add any SCRREAM-specific initialization here


class HOUSECAT6D_Dataset(RGBP_Dataset):
    """
    HOUSECAT6D dataset implementation for 6D pose estimation with polarization.

    Inherits all functionality from the base RGBP_Dataset class.
    This class can be extended with HOUSECAT6D-specific preprocessing,
    pose annotation loading, or 6D pose-specific data augmentation.

    The HOUSECAT6D dataset provides RGB and polarization data along with
    6D object pose annotations for training pose estimation models.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize HOUSECAT6D dataset.

        Args:
            **kwargs: All arguments passed to parent RGBP_Dataset class
        """
        super().__init__(**kwargs)
        # Add any HOUSECAT6D-specific initialization here


class POLARGB_Dataset(RGBP_Dataset):
    """
    PolaRGB dataset implementation for polarization-guided RGB processing.

    Inherits all functionality from the base RGBP_Dataset class.
    This class can be extended with PolaRGB-specific preprocessing,
    polarization analysis, or RGB enhancement techniques.

    The PolaRGB dataset combines RGB imagery with polarization measurements
    for improved scene understanding and image enhancement tasks.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize PolaRGB dataset.

        Args:
            **kwargs: All arguments passed to parent RGBP_Dataset class
        """
        super().__init__(**kwargs)
        # Add any PolaRGB-specific initialization here


class SCARED_Dataset(RGBP_Dataset):
    """
    SCARED dataset implementation for polarization-guided RGB processing.

    Inherits all functionality from the base RGBP_Dataset class.
    This class can be extended with SCARED-specific preprocessing,
    polarization analysis, or RGB enhancement techniques.

    The SCARED dataset combines RGB imagery with polarization measurements
    for improved scene understanding and image enhancement tasks.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize SCARED dataset.

        Args:
            **kwargs: All arguments passed to parent RGBP_Dataset class
        """
        super().__init__(**kwargs)
        # Add any SCARED-specific initialization here


def from_config(
    config: Dict, dataset_names: Optional[List[str]] = None
) -> Dict[str, Union[Dataset, None]]:
    """
    Create datasets from configuration file.

    This function reads the configuration file and creates training and validation datasets
    based on the VAL_SCENES parameter. The logic is:
    - VAL_SCENES: defines which scenes to use for validation
    - TRAIN_SCENES: if provided and not None/[], overrides the default training scenes
    - If TRAIN_SCENES is None/[], training uses all scenes except those in VAL_SCENES

    Args:
        config: Configuration dictionary loaded from config file
        dataset_names: Optional list of dataset names to load. If None, loads all available datasets.

    Returns:
        Dictionary with 'training', 'validation', and 'test' keys containing ConcatDataset objects
    """
    if dataset_names is None:
        # Get all available dataset names from config
        dataset_names = []

        datasets_config = config.DATASETS

        if isinstance(datasets_config, dict):
            datasets_value = datasets_config

            if datasets_value is not None:
                dataset_names = [
                    name
                    for name in datasets_value.keys()
                    if isinstance(datasets_value[name], dict)
                ]

    if not dataset_names:
        raise ValueError(
            "No datasets found in configuration. Check DATASETS section in config file."
        )

    # Map dataset names to classes - this is where you add new dataset classes
    dataset_classes = {
        "SCRREAM": SCRREAM_Dataset,
        "HOUSECAT6D": HOUSECAT6D_Dataset,
        "POLARGB": POLARGB_Dataset,
        "SCARED": SCARED_Dataset,
        # Future datasets will be added here by the user
    }

    train_datasets = []
    val_datasets = []

    # Global config parameters
    global_config = config

    # Get global scene configuration
    global_train_scenes = global_config.get("TRAIN_SCENES", {}).get("value")
    global_val_scenes = global_config.get("VAL_SCENES", {}).get("value")

    logger.info(f"Processing {len(dataset_names)} datasets: {dataset_names}")

    for dataset_name in dataset_names:
        if dataset_name not in dataset_classes:
            logger.warning(
                f"Warning: Dataset class for '{dataset_name}' not found. Skipping."
            )
            logger.info(f"Available classes: {list(dataset_classes.keys())}")
            continue

        # Get dataset-specific config
        datasets_value = global_config.DATASETS
        if datasets_value is None:
            raise ValueError("DATASETS['value'] is None in config")
        dataset_config = datasets_value[dataset_name]
        # Get root directory
        root_dir = os.path.expandvars(dataset_config.ROOT_DIR)
        if not os.path.exists(root_dir):
            logger.warning(
                f"Warning: Root directory '{root_dir}' for dataset '{dataset_name}' not found. Skipping."
            )
            continue

        # Extract configuration parameters with fallbacks to global config
        def get_config_value(param_name, default_value):
            """Helper to get parameter from dataset config or global config"""
            dataset_value = dataset_config.get(param_name)
            if dataset_value is not None:
                return dataset_value
            global_param = global_config.get(param_name, {})
            if isinstance(global_param, dict) and "value" in global_param:
                return global_param["value"]
            return default_value

        dataset_params = {
            "root_dir": root_dir,
            "rho_s": get_config_value("RHO_S", 0.6),
            "eps": get_config_value("EPS", 1e-8),
            "target_size": tuple(get_config_value("TARGET_SIZE", [224, 224])),
            "resize_mode": get_config_value("RESIZE_MODE", "crop"),
            "use_cache": get_config_value("USE_CACHE", True),
            "simplify_upsampling": get_config_value("SIMPLIFY_UPSAMPLING", True),
            "few_images": get_config_value("FEW_IMAGES", False),
            "polarization_format": get_config_value(
                "POLARIZATION_FORMAT", "single_file_clock"
            ),
            # Highlight options
            "highlight_enable": get_config_value("HIGHLIGHT_ENABLE", False),
            "highlight_brightness_threshold": get_config_value(
                "HIGHLIGHT_BRIGHTNESS_THRESHOLD", 0.93
            ),
            "highlight_return_mask": get_config_value("HIGHLIGHT_RETURN_MASK", False),
            "highlight_return_rect": get_config_value("HIGHLIGHT_RETURN_RECT", False),
            "highlight_return_rect_as_rgb": get_config_value("HIGHLIGHT_RETURN_RECT_AS_RGB", False),
        }

        # Handle optional tuple conversion for rect size if provided
        rect_size_val = get_config_value("HIGHLIGHT_RECT_SIZE", None)
        if rect_size_val is not None:
            try:
                dataset_params["highlight_rect_size"] = tuple(rect_size_val)
            except Exception:
                dataset_params["highlight_rect_size"] = None

        # Get scenes configuration with priority: global > dataset-specific
        dataset_train_scenes = dataset_config.get("TRAIN_SCENES", [])
        dataset_val_scenes = dataset_config.get("VAL_SCENES", [])

        # Final scene determination with clear precedence
        val_scenes = (
            global_val_scenes if global_val_scenes is not None else dataset_val_scenes
        )

        # TRAIN_SCENES override logic: if global TRAIN_SCENES is provided and not empty, use it
        if global_train_scenes is not None and len(global_train_scenes) > 0:
            train_scenes = global_train_scenes
            logger.info(f"Using global TRAIN_SCENES for {dataset_name}: {train_scenes}")
        elif dataset_train_scenes and len(dataset_train_scenes) > 0:
            train_scenes = dataset_train_scenes
            logger.info(
                f"Using dataset-specific TRAIN_SCENES for {dataset_name}: {train_scenes}"
            )
        else:
            # Use all scenes except validation scenes
            train_scenes = None
            logger.info(
                f"Using all scenes except VAL_SCENES for {dataset_name} training"
            )

        # Get dataset class
        dataset_class = dataset_classes[dataset_name]

        # Create training dataset
        if train_scenes is not None and len(train_scenes) > 0:
            dataset_params.update({"highlight_enable": True})
            
            # Use specific training scenes
            train_dataset = dataset_class(
                include=train_scenes,
                # highlight_enable=dataset_config.get("HIGHLIGHT_ENABLE"),
                # highlight_brightness_threshold=dataset_config.get("HIGHLIGHT_BRIGHTNESS_THRESHOLD"),
                # highlight_return_mask=dataset_config.get("HIGHLIGHT_RETURN_MASK"),
                # highlight_rect_size=dataset_config.get("HIGHLIGHT_RECT_SIZE"),
                # highlight_return_rect=dataset_config.get("HIGHLIGHT_RETURN_RECT"),
                # highlight_return_rect_as_rgb=dataset_config.get("HIGHLIGHT_RETURN_RECT_AS_RGB"),
                **dataset_params,
            )
            if len(train_dataset) > 0:
                train_datasets.append(train_dataset)
                logger.info(
                    f"  ✓ Created training dataset for {dataset_name}: {len(train_dataset)} samples from specific scenes"
                )
            else:
                logger.warning(f"  ✗ Training dataset for {dataset_name} is empty")
        else:
            # Use all scenes except validation scenes
            exclude_scenes = val_scenes if val_scenes and len(val_scenes) > 0 else []
            train_dataset = dataset_class(
                exclude=exclude_scenes,
                # highlight_enable=dataset_config.get("HIGHLIGHT_ENABLE"),
                # highlight_brightness_threshold=dataset_config.get("HIGHLIGHT_BRIGHTNESS_THRESHOLD"),
                # highlight_return_mask=dataset_config.get("HIGHLIGHT_RETURN_MASK"),
                # highlight_rect_size=dataset_config.get("HIGHLIGHT_RECT_SIZE"),
                # highlight_return_rect=dataset_config.get("HIGHLIGHT_RETURN_RECT"),
                # highlight_return_rect_as_rgb=dataset_config.get("HIGHLIGHT_RETURN_RECT_AS_RGB"),
                **dataset_params,
            )
            if len(train_dataset) > 0:
                train_datasets.append(train_dataset)
                excluded_text = (
                    f" (excluding {len(exclude_scenes)} val scenes)"
                    if exclude_scenes
                    else ""
                )
                logger.info(
                    f"  ✓ Created training dataset for {dataset_name}: {len(train_dataset)} samples{excluded_text}"
                )
            else:
                logger.warning(f"  ✗ Training dataset for {dataset_name} is empty")

        # Create validation dataset
        if val_scenes and len(val_scenes) > 0:
            # Overrides global highlight_enable. Validation dataset should have original images
            dataset_params.update({"highlight_enable": False})
            val_dataset = dataset_class(
                include=val_scenes,
                **dataset_params,
            )
            if len(val_dataset) > 0:
                val_datasets.append(val_dataset)
                logger.info(
                    f"  ✓ Created validation dataset for {dataset_name}: {len(val_dataset)} samples from {len(val_scenes)} scenes"
                )
            else:
                logger.warning(f"  ✗ Validation dataset for {dataset_name} is empty")
        else:
            logger.warning(f"  ! No validation scenes specified for {dataset_name}")

    # Create ConcatDatasets
    result = {
        "training": ConcatDataset(train_datasets) if train_datasets else None,
        "validation": ConcatDataset(val_datasets) if val_datasets else None,
        "test": ConcatDataset(val_datasets) if val_datasets else None,
    }

    # Print summary
    logger.info("=== Dataset Creation Summary ===")
    logger.info(
        f"Training:   {len(result['training']) if result['training'] else 0} total samples"
    )
    logger.info(
        f"Validation: {len(result['validation']) if result['validation'] else 0} total samples"
    )
    logger.info(
        f"Test:       {len(result['test']) if result['test'] else 0} total samples"
    )

    # Print detailed breakdown if multiple datasets
    # if len(dataset_names) > 1:
    #     logger.info(f"Dataset breakdown:")
    #     for i, dataset_name in enumerate(dataset_names):
    #         if i < len(train_datasets):
    #             logger.info(f"  {dataset_name} - Train: {len(train_datasets[i])}, Val: {len(val_datasets[i]) if i < len(val_datasets) else 0}")

    return result
