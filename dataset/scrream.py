import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, List
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache
import warnings
import torch.nn.functional as F

try:
    import polanalyser as pa
    POLANALYSER_AVAILABLE = True
except ImportError:
    POLANALYSER_AVAILABLE = False
    print("Warning: polanalyser not available. Using manual implementation.")


class SCRREAM(Dataset):
    """
    Optimized version of SCRREAM dataset with performance improvements:
    1. Reduced tensor/numpy conversions
    2. Optional caching
    3. Simplified processing pipeline  
    4. Batch-friendly operations
    5. Consistent image sizing for batching
    """

    def __init__(
        self,
        root_dir: str,
        rho_s: float = 0.6,
        eps: float = 1e-8,
        rgb_ext: str = ".png",
        pol_ext: str = ".png",
        transform=None,
        # Image sizing parameters
        target_size: Optional[Tuple[int, int]] = (874, 1132),  # Default size (H, W)
        resize_mode: str = "crop",  # "crop", "resize", or "pad"
        # Performance optimizations
        use_cache: bool = False,
        cache_size: int = 100,
        simplify_upsampling: bool = True,  # Use simple bicubic instead of edge-aware
        precompute_stokes: bool = False,   # Precompute Stokes parameters
        # Reduced smoothing options
        smooth_specular: bool = True,
        gaussian_sigma: float = 0.0,
        dolp_min_intensity: float = 0.05,
        dolp_min_value: float = 0.04,
        # Scene filtering
        scene_names: Optional[List[str]] = None,
        ignore_scenes: Optional[List[str]] = None,
        # Few images mode for quick testing
        few_images: bool = False,
    ):
        self.root_dir = root_dir
        self.rho_s = rho_s
        self.eps = eps
        self.rgb_ext = rgb_ext
        self.pol_ext = pol_ext
        self.transform = transform
        
        # Image sizing parameters
        self.target_size = target_size
        self.resize_mode = resize_mode.lower()
        if self.resize_mode not in ["crop", "resize", "pad"]:
            raise ValueError(f"resize_mode must be one of ['crop', 'resize', 'pad'], got {resize_mode}")
        
        self.use_cache = use_cache
        self.simplify_upsampling = simplify_upsampling
        self.precompute_stokes = precompute_stokes
        
        # Simplified smoothing config
        self.smooth_specular = smooth_specular
        self.gaussian_sigma = gaussian_sigma
        self.dolp_min_intensity = dolp_min_intensity
        self.dolp_min_value = dolp_min_value
        
        self.scene_names = scene_names
        self.ignore_scenes = ignore_scenes or []
        self.few_images = few_images

        self.scene_pairs = self._find_scene_pairs()
        
        # Limit to 100 samples if few_images is True
        if self.few_images and len(self.scene_pairs) > 100:
            original_count = len(self.scene_pairs)
            self.scene_pairs = self.scene_pairs[:100]
            print(f"Few images mode: Limited dataset from {original_count} to 100 samples")
        
        # Setup caching if enabled
        if self.use_cache:
            self._cache_intrinsics = {}
            # Make cached methods
            self._load_intrinsics_cached = lru_cache(maxsize=cache_size)(self._load_intrinsics_impl)
            if self.precompute_stokes:
                self._load_pol_cached = lru_cache(maxsize=cache_size)(self._load_and_process_polarization_impl)

    def _resize_tensor(self, tensor: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
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
                mode='bilinear',
                align_corners=False
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
                
                tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
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

    def _resize_polarization_data(self, pol_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

    def _find_scene_pairs(self) -> List[Tuple[str, str, str]]:
        """Same as original but could be optimized with os.scandir for large datasets"""
        scene_pairs = []
        for scene_name in os.listdir(self.root_dir):
            if scene_name in self.ignore_scenes:
                continue
                
            if self.scene_names is not None and len(self.scene_names) > 0:
                if scene_name not in self.scene_names:
                    continue
            
            scene_path = os.path.join(self.root_dir, scene_name)
            if not os.path.isdir(scene_path):
                continue

            rgb_dir = os.path.join(scene_path, "rgb")
            pol_dir = os.path.join(scene_path, "pol")
            intrinsics_path = os.path.join(scene_path, "intrinsics.txt")
            
            if not (os.path.exists(rgb_dir) and os.path.exists(pol_dir) and os.path.exists(intrinsics_path)):
                continue

            rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(self.rgb_ext)]
            pol_files = [f for f in os.listdir(pol_dir) if f.endswith(self.pol_ext)]
            
            for rgb_file in rgb_files:
                pol_file = rgb_file.replace(self.rgb_ext, self.pol_ext)
                if pol_file in pol_files:
                    scene_pairs.append((
                        os.path.join(rgb_dir, rgb_file),
                        os.path.join(pol_dir, pol_file),
                        intrinsics_path
                    ))
        return scene_pairs

    def __len__(self) -> int:
        return len(self.scene_pairs)
    
    def get_loaded_scenes(self) -> List[str]:
        """Get list of scene names that are actually loaded in the dataset."""
        loaded_scenes = set()
        for rgb_path, _, _ in self.scene_pairs:
            scene_name = os.path.basename(os.path.dirname(os.path.dirname(rgb_path)))
            loaded_scenes.add(scene_name)
        return sorted(list(loaded_scenes))

    @staticmethod
    def _to_luminance_torch(rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to luminance staying in torch. Input: [...,3] in [0,1]."""
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    def _load_intrinsics_impl(self, intrinsics_path: str) -> torch.Tensor:
        """Implementation for loading intrinsics (cacheable)"""
        try:
            K = np.loadtxt(intrinsics_path).reshape(3, 3).astype(np.float32)
            return torch.from_numpy(K)
        except Exception as e:
            warnings.warn(f"Could not load intrinsics from {intrinsics_path}: {e}")
            return torch.eye(3, dtype=torch.float32)

    def _load_intrinsics(self, intrinsics_path: str) -> torch.Tensor:
        """Load intrinsics with optional caching"""
        if self.use_cache:
            return self._load_intrinsics_cached(intrinsics_path)
        else:
            return self._load_intrinsics_impl(intrinsics_path)

    def _simple_upsample(self, f_spec_half: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Simple bicubic upsampling - much faster than edge-aware"""
        # Use torch interpolation instead of OpenCV
        f_batch = f_spec_half.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        f_up = torch.nn.functional.interpolate(
            f_batch, size=target_size, mode='bilinear', align_corners=False
        )
        return f_up.squeeze(0).squeeze(0).clamp(0, 1)

    def _load_and_process_polarization_impl(self, pol_path: str) -> Dict[str, torch.Tensor]:
        """Optimized polarization processing staying mostly in torch"""
        
        # Load image directly as torch tensor
        pol_img = Image.open(pol_path).convert("RGB")
        pol_rgb = torch.from_numpy(np.asarray(pol_img, dtype=np.float32)) / 255.0
        
        H, W, _ = pol_rgb.shape
        hh, hw = H // 2, W // 2
        
        # Split quadrants
        I0_rgb   = pol_rgb[:hh, :hw, :]
        I45_rgb  = pol_rgb[:hh, hw:, :]
        I90_rgb  = pol_rgb[hh:, hw:, :]
        I135_rgb = pol_rgb[hh:, :hw, :]

        # Convert to luminance (staying in torch)
        I0   = self._to_luminance_torch(I0_rgb)
        I45  = self._to_luminance_torch(I45_rgb)
        I90  = self._to_luminance_torch(I90_rgb)
        I135 = self._to_luminance_torch(I135_rgb)

        # Compute Stokes parameters
        S0 = I0 + I90
        S1 = I0 - I90
        S2 = I45 - I135

        # Optional Gaussian smoothing (convert to numpy only if needed)
        # if self.gaussian_sigma > 0:
        #     # Only convert to numpy for this operation
        #     S0_np = cv2.GaussianBlur(S0.numpy(), (0, 0), self.gaussian_sigma)
        #     S1_np = cv2.GaussianBlur(S1.numpy(), (0, 0), self.gaussian_sigma)
        #     S2_np = cv2.GaussianBlur(S2.numpy(), (0, 0), self.gaussian_sigma)
        #     S0, S1, S2 = torch.from_numpy(S0_np), torch.from_numpy(S1_np), torch.from_numpy(S2_np)

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
            'I0': I0_rgb.permute(2, 0, 1),
            'I45': I45_rgb.permute(2, 0, 1),
            'I90': I90_rgb.permute(2, 0, 1),
            'I135': I135_rgb.permute(2, 0, 1),
            'S0': S0.unsqueeze(0),
            'S1': S1.unsqueeze(0),
            'S2': S2.unsqueeze(0),
            'S3': S3.unsqueeze(0),
            'intensity': S0.unsqueeze(0),
            'DoLP': DoLP.unsqueeze(0),
            'AoP': AoP.unsqueeze(0),
            'AoLP': AoP.unsqueeze(0),
            'DoP': DoLP.unsqueeze(0),
            'DoCP': torch.zeros_like(DoLP).unsqueeze(0),
            'ellipticity_angle': torch.zeros_like(DoLP).unsqueeze(0),
            'f_spec': f_spec.unsqueeze(0),
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

    def _load_rgb_and_separate(self, rgb_path: str, f_spec_half: torch.Tensor) -> Dict[str, torch.Tensor]:
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
            'rgb': rgb_chw,
            'specular': I_spec,
            'diffuse': I_diff,
        }

    def _edge_aware_upsample_original(self, f_half: torch.Tensor, rgb_full: torch.Tensor) -> torch.Tensor:
        """Original edge-aware upsampling (fallback)"""
        # Convert to numpy for OpenCV operations
        f_np = f_half.cpu().numpy()
        rgb_np = rgb_full.cpu().numpy()
        
        H, W, _ = rgb_np.shape
        f_up = cv2.resize(f_np, (W, H), interpolation=cv2.INTER_CUBIC)
        
        # Simple bilateral filter instead of guided filter for speed
        f_up = cv2.bilateralFilter(f_up.astype(np.float32), d=5, sigmaColor=0.1*255, sigmaSpace=5)
        
        return torch.from_numpy(np.clip(f_up, 0.0, 1.0))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Optimized data loading"""
        rgb_path, pol_path, intrinsics_path = self.scene_pairs[idx]
        
        # Load data (potentially from cache)
        intrinsics = self._load_intrinsics(intrinsics_path)
        pol_data = self._load_and_process_polarization(pol_path)
        
        # Get specular fraction for RGB processing
        f_spec = pol_data['f_spec'].squeeze(0)
        rgb_data = self._load_rgb_and_separate(rgb_path, f_spec)

        # Combine results
        sample = {**pol_data, **rgb_data, 'intrinsics': intrinsics}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


# Optimized dataloader function
def create_optimized_dataloader(
    root_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    use_cache: bool = True,
    simplify_upsampling: bool = True,
    target_size: Optional[Tuple[int, int]] = (874, 1132),
    resize_mode: str = "crop",
    few_images: bool = False,
    **dataset_kwargs
) -> DataLoader:
    """Create optimized dataloader with performance improvements"""
    
    dataset = SCRREAM(
        root_dir,
        use_cache=use_cache,
        simplify_upsampling=simplify_upsampling,
        target_size=target_size,
        resize_mode=resize_mode,
        few_images=few_images,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
        prefetch_factor=2 if num_workers > 0 else 2,  # Prefetch batches
    )

