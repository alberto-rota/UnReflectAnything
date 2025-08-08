import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from sklearn.decomposition import PCA
from skimage import measure

def vox2world(
    vox_coords: torch.Tensor,
    volume_origin: torch.Tensor,
    voxel_size: float,
    device: torch.device
) -> torch.Tensor:
    """
    Convert voxel grid indices to world coordinates.

    Transforms discrete voxel coordinates to continuous world coordinates
    by scaling by voxel size and offsetting by volume origin.

    Args:
        vox_coords: [N, 3] tensor of voxel indices (x, y, z)
        volume_origin: [3] tensor of volume origin in world coordinates
        voxel_size: Size of each voxel in mm
        device: PyTorch device for computations

    Returns:
        world_pts: [N, 3] tensor of world coordinates in mm
    """
    world_pts = torch.zeros_like(vox_coords, dtype=torch.float32, device=device)
    for i in range(3):
        world_pts[:, i] = volume_origin[i] + vox_coords[:, i].float() * voxel_size
    return world_pts


def cam2pix(cam_pts: torch.Tensor, intr: torch.Tensor) -> torch.Tensor:
    """
    Project 3D camera coordinates to 2D pixel coordinates using camera intrinsics.

    Uses the standard pinhole camera model: u = fx * X/Z + cx, v = fy * Y/Z + cy

    Args:
        cam_pts: [N, 3] tensor of 3D points in camera coordinate system
        intr: [3, 3] camera intrinsic matrix

    Returns:
        pix: [N, 2] tensor of pixel coordinates (u, v) as integers
    """
    # Extract focal lengths and principal point from intrinsic matrix
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]

    # Project using pinhole camera model and round to nearest pixel
    pix = torch.empty((cam_pts.shape[0], 2), dtype=torch.int64, device=cam_pts.device)
    pix[:, 0] = ((cam_pts[:, 0] * fx / cam_pts[:, 2]) + cx).round().to(torch.int64)
    pix[:, 1] = ((cam_pts[:, 1] * fy / cam_pts[:, 2]) + cy).round().to(torch.int64)
    return pix


def world_to_grid(
    pts: torch.Tensor,
    volume_origin: torch.Tensor,
    volume_dim: torch.Tensor,
    voxel_size: float
) -> torch.Tensor:
    """
    Convert world-space points to normalized grid coords in [-1,1] for grid_sample
    (with align_corners=True).
    """
    # 1) shift into local volume frame
    rel = (pts - volume_origin.to(pts))
    
    # 2) divide by the physical size of the volume *minus one voxel*  
    #    (so that corners map exactly to ±1 under align_corners=True)
    physical_size = (volume_dim.to(pts).float() - 1.0) * voxel_size
    rel /= physical_size
    
    # 3) map [0,1] → [−1,1]
    norm = rel * 2.0 - 1.0
    
    # 4) return in (x,y,z) order
    return norm



def interpolate_features(
    world_pts: np.ndarray,
    feature_volume: torch.Tensor,
    volume_origin: torch.Tensor,
    feature_dim: torch.Tensor,
    featvoxel_size_x: float,
    featvoxel_size_y: float,
    featvoxel_size_z: float,
    device: torch.device
) -> np.ndarray:
    """
    Interpolate features from the feature volume at given world coordinates.

    Args:
        world_pts: [N, 3] numpy array of world coordinates
        feature_volume: [X_f, Y_f, Z_f, E] tensor of feature values
        volume_origin: [3] tensor of volume origin in world coordinates
        feature_dim: [3] tensor of feature volume dimensions
        featvoxel_size_x/y/z: Size of feature voxels in each dimension
        device: PyTorch device for computations

    Returns:
        features: [N, E] numpy array of interpolated feature vectors
    """
    wp = torch.as_tensor(world_pts, dtype=torch.float32, device=device)

    # --- world → voxel indices ------------------------------------------------
    vx = (wp[:, 0] - volume_origin[0]) / featvoxel_size_x
    vy = (wp[:, 1] - volume_origin[1]) / featvoxel_size_y
    vz = (wp[:, 2] - volume_origin[2]) / featvoxel_size_z
    feat_coords = torch.stack([vx, vy, vz], dim=-1)  # [N,3]

    # --- normalise for grid_sample (align_corners=True) -----------------------
    dims = feature_dim.float() - 1  # [3]
    feat_coords_norm = 2.0 * feat_coords / dims - 1.0  # [N,3]

    # --- prepare tensors ------------------------------------------------------
    feat_vol = feature_volume.permute(3, 2, 1, 0).contiguous()  # C,Z,Y,X
    feat_vol = feat_vol.unsqueeze(0)  # 1,C,D,H,W
    grid = feat_coords_norm.contiguous().view(1, 1, 1, -1, 3)  # 1,1,1,N,3

    # --- trilinear sample -----------------------------------------------------
    samp = F.grid_sample(
        feat_vol,
        grid,
        mode="bilinear",  # trilinear
        padding_mode="border",
        align_corners=True,
    )  # 1,C,1,1,N
    samp = samp.squeeze(0).squeeze(1).squeeze(1).permute(1, 0)  # N,E
    return samp.cpu().numpy()


def generate_ray_samples(
    K: torch.Tensor,
    T: torch.Tensor,
    im_h: int,
    im_w: int,
    near: float,
    far: float,
    n_samples: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate ray samples for raycasting operations.

    Args:
        K: [3, 3] camera intrinsic matrix
        T: [4, 4] camera-to-world transformation matrix
        im_h: Image height in pixels
        im_w: Image width in pixels
        near: Near clipping plane distance
        far: Far clipping plane distance
        n_samples: Number of samples along each ray
        device: PyTorch device for computations

    Returns:
        sample_pts: [im_h, im_w, n_samples, 3] tensor of sample points
        origins: [im_h, im_w, 3] tensor of ray origins
        dirs_world: [im_h, im_w, 3] tensor of ray directions
    """
    # Generate pixel coordinates - shape: [im_h, im_w]
    i, j = torch.meshgrid(
        torch.arange(im_h, device=device, dtype=torch.float32),
        torch.arange(im_w, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Extract camera intrinsics
    fx, fy = K[0, 0], K[1, 1]  # focal lengths
    cx, cy = K[0, 2], K[1, 2]  # principal point
    
    # Convert pixels to normalized camera coordinates - shape: [im_h, im_w, 3]
    dirs_cam = torch.stack([
        (j - cx) / fx,  # x direction
        (i - cy) / fy,  # y direction  
        torch.ones_like(i)  # z direction (forward)
    ], dim=-1)
    
    # Transform ray directions to world coordinates
    R = T[:3, :3]  # rotation matrix
    t = T[:3, 3]   # translation vector
    dirs_world = (dirs_cam @ R.t())  # shape: [im_h, im_w, 3]
    
    # Normalize ray directions
    dirs_world = F.normalize(dirs_world, dim=-1)
    
    # Ray origins (camera position) - shape: [im_h, im_w, 3]
    origins = t.view(1, 1, 3).expand(im_h, im_w, 3)
    
    # Sample depths along rays - shape: [n_samples]
    depths = torch.linspace(near, far, n_samples, device=device)
    
    # Generate sample points along all rays - shape: [im_h, im_w, n_samples, 3]
    sample_pts = origins.unsqueeze(2) + dirs_world.unsqueeze(2) * depths.view(1, 1, -1, 1)
    
    return sample_pts, origins, dirs_world


def find_surface_intersections(
    tsdf_samples: torch.Tensor,
    depths: torch.Tensor,
    surface_threshold: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find surface intersections from TSDF samples along rays.

    Args:
        tsdf_samples: [im_h, im_w, n_samples] tensor of TSDF values
        depths: [n_samples] tensor of depth values
        surface_threshold: Threshold for surface detection

    Returns:
        depth_map: [im_h, im_w] tensor of intersection depths
        valid_hit: [im_h, im_w] tensor of boolean mask for valid hits
    """
    im_h, im_w, n_samples = tsdf_samples.shape
    device = tsdf_samples.device
    # Find first point along each ray where TSDF <= surface_threshold
    # Find closest point to 0 along each ray - Shape: [H, W]
    abs_tsdf = torch.abs(tsdf_samples)  # [H, W, N]
    hit_indices = torch.argmin(abs_tsdf, dim=2)  # [H, W]
    min_tsdf_vals = torch.min(abs_tsdf, dim=2)[0]  # [H, W] 
    has_hit = min_tsdf_vals <= surface_threshold
    
    # Check if any hit occurred (if no hit, argmax returns 0, but first sample might not be a hit)
    # has_hit = hit_mask.any(dim=2)  # shape: [im_h, im_w]
    valid_hit = has_hit & (hit_indices > 0)  # Exclude hits at first sample (likely invalid)
    
    # Calculate depths at hit points
    hit_depths = depths[hit_indices]  # shape: [im_h, im_w]
    
    # For more accurate depth, interpolate between the sample before and after the hit
    # Get TSDF values at hit points
    i_coords = torch.arange(im_h, device=device).view(-1, 1).expand(-1, im_w)
    j_coords = torch.arange(im_w, device=device).view(1, -1).expand(im_h, -1)
    
    # Clamp hit_indices to valid range for interpolation
    hit_indices_clamped = torch.clamp(hit_indices, 1, n_samples - 1)
    
    # Get TSDF values before and at hit point
    tsdf_before = tsdf_samples[i_coords, j_coords, hit_indices_clamped - 1]
    tsdf_at = tsdf_samples[i_coords, j_coords, hit_indices_clamped]
    
    # Get corresponding depths
    depth_before = depths[hit_indices_clamped - 1]
    depth_at = depths[hit_indices_clamped]
    
    # Linear interpolation to find more precise intersection point
    # Avoid division by zero
    tsdf_diff = tsdf_before - tsdf_at
    tsdf_diff = torch.where(torch.abs(tsdf_diff) < 1e-6, torch.ones_like(tsdf_diff) * 1e-6, tsdf_diff)
    
    # Interpolation factor
    t = (tsdf_before - surface_threshold) / tsdf_diff
    t = torch.clamp(t, 0, 1)
    
    # Interpolated depth
    depth_interp = depth_before + t * (depth_at - depth_before)
    
    # Final depth map: use interpolated depth where valid hit, otherwise 0
    depth_map = torch.where(valid_hit, depth_interp, torch.zeros_like(depth_interp))
    
    return depth_map, valid_hit


def extract_point_cloud_from_features(
    feature_volume: torch.Tensor,
    feature_weights: torch.Tensor,
    volume_origin: torch.Tensor,
    featvoxel_size_x: float,
    featvoxel_size_y: float,
    featvoxel_size_z: float,
    feature_vector_length: int,
    pca: Optional[PCA] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract point cloud directly from feature volume for efficiency.
    
    Args:
        feature_volume: [X_f, Y_f, Z_f, E] feature volume tensor
        feature_weights: [X_f, Y_f, Z_f] feature weights tensor
        volume_origin: [3] volume origin tensor
        featvoxel_size_x/y/z: Feature voxel sizes
        feature_vector_length: Dimension of feature vectors
        pca: Optional pre-fitted PCA transformer
        
    Returns:
        points: [N, 3] point coordinates
        colors: [N, 3] RGB colors
    """
    feature_vol = feature_volume.detach().cpu().numpy()  # [X_f, Y_f, Z_f, E]
    feature_weights_vol = feature_weights.detach().cpu().numpy()  # [X_f, Y_f, Z_f]
    
    # Find voxels with meaningful features (non-zero weights indicating fusion has occurred)
    valid_mask = feature_weights_vol > 0
    valid_indices = np.where(valid_mask)  # tuple of (x_indices, y_indices, z_indices)
    
    if len(valid_indices[0]) == 0:
        # No valid feature voxels found
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # Convert feature voxel indices to world coordinates
    feat_vox_coords = np.stack(valid_indices, axis=1).astype(np.float32)  # [N, 3]
    
    # Transform to world coordinates using feature voxel sizes
    points = np.zeros_like(feat_vox_coords)
    volume_origin_np = volume_origin.cpu().numpy()
    points[:, 0] = volume_origin_np[0] + feat_vox_coords[:, 0] * featvoxel_size_x
    points[:, 1] = volume_origin_np[1] + feat_vox_coords[:, 1] * featvoxel_size_y  
    points[:, 2] = volume_origin_np[2] + feat_vox_coords[:, 2] * featvoxel_size_z
    
    # Extract features directly (no interpolation needed)
    feats = feature_vol[valid_indices]  # [N, E]
    
    # Apply PCA to features for colors
    if pca is not None:
        comps = pca.transform(feats)
    else:
        # Fit PCA on-the-fly
        comps = PCA(n_components=3, svd_solver="randomized").fit_transform(feats)
    
    # Normalize to 0-255 range
    comps -= comps.min(0)
    comps /= np.maximum(np.ptp(comps, 0), 1e-8)
    colors = (comps * 255).astype(np.uint8)
    
    return points, colors


def extract_point_cloud_from_tsdf(
    tsdf_volume: torch.Tensor,
    color_volume: torch.Tensor,
    volume_origin: torch.Tensor,
    voxel_size: float,
    threshold: float = 0.05,
    feature_volume: Optional[torch.Tensor] = None,
    feature_as_colors: bool = False,
    pca: Optional[PCA] = None,
    interpolate_features_func=None,
    **interpolate_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract point cloud using marching cubes on TSDF volume.
    
    Args:
        tsdf_volume: [X, Y, Z] TSDF volume tensor
        color_volume: [X, Y, Z, 3] color volume tensor
        volume_origin: [3] volume origin tensor
        voxel_size: Voxel size in mm
        threshold: TSDF threshold for surface extraction
        feature_volume: Optional feature volume for feature-based colors
        feature_as_colors: Whether to use features as colors
        pca: Optional PCA transformer
        interpolate_features_func: Function to interpolate features
        **interpolate_kwargs: Additional arguments for feature interpolation
        
    Returns:
        points: [N, 3] point coordinates
        colors: [N, 3] RGB colors
    """
    # Move data to CPU for processing with scikit-image
    tsdf_vol = tsdf_volume.detach().cpu().numpy()
    color_vol = color_volume.detach().cpu().numpy()

    # Extract surface vertices using marching cubes
    verts = measure.marching_cubes(tsdf_vol, level=threshold)[0]
    verts_ind = np.round(verts).astype(int)

    # Transform voxel coordinates to world coordinates
    points = verts * voxel_size + volume_origin.cpu().numpy()

    if feature_as_colors and feature_volume is not None and interpolate_features_func is not None:
        # Use PCA-transformed features as colors
        feats = interpolate_features_func(points, **interpolate_kwargs)  # [N, E]
        if pca is not None:
            comps = pca.transform(feats)
        else:
            # Fit PCA on-the-fly
            comps = PCA(n_components=3, svd_solver="randomized").fit_transform(feats)
        # Normalize to 0-255 range
        comps -= comps.min(0)
        comps /= np.maximum(np.ptp(comps, 0), 1e-8)
        colors = (comps * 255).astype(np.uint8)
    else:
        # Use RGB colors from color volume
        colors = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]].astype(
            np.uint8
        )

    return points, colors


def extract_mesh_from_tsdf(
    tsdf_volume: torch.Tensor,
    color_volume: torch.Tensor,
    volume_origin: torch.Tensor,
    voxel_size: float,
    threshold: float = 0.05,
    feature_volume: Optional[torch.Tensor] = None,
    feature_as_colors: bool = False,
    pca: Optional[PCA] = None,
    interpolate_features_func=None,
    **interpolate_kwargs
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Extract mesh using marching cubes on TSDF volume.
    
    Args:
        tsdf_volume: [X, Y, Z] TSDF volume tensor
        color_volume: [X, Y, Z, 3] color volume tensor
        volume_origin: [3] volume origin tensor
        voxel_size: Voxel size in mm
        threshold: TSDF threshold for surface extraction
        feature_volume: Optional feature volume for feature-based colors
        feature_as_colors: Whether to use features as colors
        pca: Optional PCA transformer
        interpolate_features_func: Function to interpolate features
        **interpolate_kwargs: Additional arguments for feature interpolation
        
    Returns:
        verts_world: [N, 3] vertex positions in world coordinates
        faces: [M, 3] triangle face indices
        norms: [N, 3] vertex normal vectors
        colors: [N, 3] vertex colors as RGB values (0-255)
        features: [N, E] vertex feature vectors (if feature_as_colors=True)
    """
    # Move data to CPU for processing with scikit-image
    tsdf_vol = tsdf_volume.detach().cpu().numpy()
    color_vol = color_volume.detach().cpu().numpy()

    # Extract mesh using marching cubes algorithm
    verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=threshold)
    verts_ind = np.round(verts).astype(int)

    # Transform voxel coordinates to world coordinates
    verts_world = verts * voxel_size + volume_origin.cpu().numpy()

    if feature_as_colors and feature_volume is not None and interpolate_features_func is not None:
        feats = interpolate_features_func(verts_world, **interpolate_kwargs)     # [N,E]
        if pca is not None:
            comps = pca.transform(feats)
        else:
            # fit PCA on-the-fly
            comps = PCA(n_components=3, svd_solver="randomized").fit_transform(feats)
        comps -= comps.min(0)
        comps /= np.maximum(np.ptp(comps,0), 1e-8)
        colors = (comps * 255).astype(np.uint8)
    else:
        colors = color_vol[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]].astype(np.uint8)

    return verts_world, faces, norms, colors


def print_volume_summary(
    device: torch.device,
    voxel_size: float,
    truncation_margin: float,
    volume_bounds: torch.Tensor,
    volume_dim: torch.Tensor,
    tsdf: torch.Tensor,
    color: torch.Tensor,
    feature: Optional[torch.Tensor] = None,
    feature_dim: Optional[torch.Tensor] = None,
    featvoxel_size_x: Optional[float] = None,
    featvoxel_size_y: Optional[float] = None,
    featvoxel_size_z: Optional[float] = None,
    feature_vector_length: Optional[int] = None,
    volumes_to_show: str = "all"  # "all", "tsdf", "color", "features", or combinations like "tsdf,color"
) -> None:
    """
    Print a comprehensive summary of the TSDF volume properties.
    
    Args:
        device: PyTorch device
        voxel_size: Main volume voxel size in mm
        truncation_margin: Truncation margin in mm
        volume_bounds: [3, 2] volume bounds tensor for main volume
        volume_dim: [3] volume dimensions tensor for main volume
        tsdf: TSDF volume tensor
        color: Color volume tensor
        feature: Optional feature volume tensor
        feature_dim: Optional feature volume dimensions tensor
        featvoxel_size_x/y/z: Feature voxel sizes
        feature_vector_length: Feature vector dimension
        volumes_to_show: Which volumes to display - "all", "tsdf", "color", "features", or comma-separated combinations
    """
    # Parse which volumes to show
    if volumes_to_show.lower() == "all":
        show_volumes = ["tsdf", "color", "features"]
    else:
        show_volumes = [v.strip().lower() for v in volumes_to_show.split(",")]
    
    # Helper function to calculate occupied bounds for a volume
    def get_occupied_bounds(volume: torch.Tensor, volume_origin: torch.Tensor, voxel_sizes: Union[float, Tuple[float, float, float]]) -> dict:
        """Calculate bounding box of non-zero/occupied voxels."""
        if volume.dim() == 4:  # Feature volume [X, Y, Z, E] or color volume [X, Y, Z, 3]
            # For multi-channel volumes, check if any channel is non-zero
            occupied_mask = (volume != 0).any(dim=-1)  # [X, Y, Z]
        else:  # TSDF volume [X, Y, Z] 
            # For TSDF, consider values != 1.0 as occupied (since 1.0 = unobserved)
            occupied_mask = (volume != 1.0)  # [X, Y, Z]
        
        if not occupied_mask.any():
            return {
                'min_coords': [0, 0, 0],
                'max_coords': [0, 0, 0],
                'world_min': [0.0, 0.0, 0.0],
                'world_max': [0.0, 0.0, 0.0],
                'world_size': [0.0, 0.0, 0.0],
                'occupied_voxels': 0,
                'total_voxels': volume.shape[0] * volume.shape[1] * volume.shape[2]
            }
        
        # Find min/max occupied voxel coordinates
        occupied_indices = torch.nonzero(occupied_mask)  # [N, 3]
        min_coords = occupied_indices.min(dim=0)[0].cpu().numpy()  # [3]
        max_coords = occupied_indices.max(dim=0)[0].cpu().numpy()  # [3]
        
        # Convert to world coordinates
        if isinstance(voxel_sizes, float):
            voxel_sizes = (voxel_sizes, voxel_sizes, voxel_sizes)
        
        world_min = volume_origin.cpu().numpy() + min_coords * np.array(voxel_sizes)
        world_max = volume_origin.cpu().numpy() + max_coords * np.array(voxel_sizes)
        world_size = world_max - world_min
        
        return {
            'min_coords': min_coords.tolist(),
            'max_coords': max_coords.tolist(),
            'world_min': world_min.tolist(),
            'world_max': world_max.tolist(),
            'world_size': world_size.tolist(),
            'occupied_voxels': int(occupied_mask.sum().item()),
            'total_voxels': volume.shape[0] * volume.shape[1] * volume.shape[2]
        }
    
    print(f"\n{'='*80}")
    print(f"TSDF VOLUME SUMMARY")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Main Volume Voxel Size: {voxel_size:.1f} mm")
    print(f"Truncation Margin: {truncation_margin:.1f} mm")
    
    # Collect volume information
    volume_info = []
    
    # TSDF Volume
    if "tsdf" in show_volumes:
        occupied_bounds = get_occupied_bounds(tsdf, volume_bounds[:, 0], voxel_size)
        tsdf_stats = {
            'min': tsdf.min().item(),
            'max': tsdf.max().item(), 
            'mean': tsdf.mean().item(),
            'std': tsdf.std().item()
        }
        memory_mb = tsdf.element_size() * tsdf.nelement() / (1024 * 1024)
        
        volume_info.append({
            'name': 'TSDF',
            'shape': list(tsdf.shape),
            'voxel_size': [voxel_size, voxel_size, voxel_size],
            'world_bounds_min': volume_bounds[:, 0].cpu().numpy().tolist(),
            'world_bounds_max': volume_bounds[:, 1].cpu().numpy().tolist(),
            'world_size': (volume_bounds[:, 1] - volume_bounds[:, 0]).cpu().numpy().tolist(),
            'occupied_bounds': occupied_bounds,
            'stats': tsdf_stats,
            'stats_desc': 'TSDF values (0=surface, 1=unobserved)',
            'memory_mb': memory_mb
        })
    
    # Color Volume  
    if "color" in show_volumes:
        occupied_bounds = get_occupied_bounds(color, volume_bounds[:, 0], voxel_size)
        # Calculate statistics per channel and overall
        color_stats = {}
        for i, channel in enumerate(['R', 'G', 'B']):
            color_stats[channel] = {
                'min': color[..., i].min().item(),
                'max': color[..., i].max().item(),
                'mean': color[..., i].mean().item(),
                'std': color[..., i].std().item()
            }
        
        # Overall color magnitude statistics
        color_magnitude = torch.norm(color, dim=-1)  # [X, Y, Z] - L2 norm across RGB
        color_stats['magnitude'] = {
            'min': color_magnitude.min().item(),
            'max': color_magnitude.max().item(),
            'mean': color_magnitude.mean().item(),
            'std': color_magnitude.std().item()
        }
        
        memory_mb = color.element_size() * color.nelement() / (1024 * 1024)
        
        volume_info.append({
            'name': 'Color',
            'shape': list(color.shape),
            'voxel_size': [voxel_size, voxel_size, voxel_size],
            'world_bounds_min': volume_bounds[:, 0].cpu().numpy().tolist(),
            'world_bounds_max': volume_bounds[:, 1].cpu().numpy().tolist(),
            'world_size': (volume_bounds[:, 1] - volume_bounds[:, 0]).cpu().numpy().tolist(),
            'occupied_bounds': occupied_bounds,
            'stats': color_stats,
            'stats_desc': 'RGB values (0-255) + magnitude',
            'memory_mb': memory_mb
        })
    
    # Feature Volume
    if "features" in show_volumes and feature is not None:
        # Calculate feature volume world bounds
        feat_world_bounds_min = volume_bounds[:, 0].cpu().numpy()  # Same origin as main volume
        feat_voxel_sizes = (featvoxel_size_x, featvoxel_size_y, featvoxel_size_z)
        feat_world_bounds_max = feat_world_bounds_min + feature_dim.cpu().numpy() * np.array(feat_voxel_sizes)
        feat_world_size = feat_world_bounds_max - feat_world_bounds_min
        
        occupied_bounds = get_occupied_bounds(feature, torch.tensor(feat_world_bounds_min), feat_voxel_sizes)
        
        # Feature statistics - magnitude across feature dimension
        feature_magnitude = torch.norm(feature, dim=-1)  # [X_f, Y_f, Z_f] - L2 norm across feature dimension
        feature_stats = {
            'magnitude': {
                'min': feature_magnitude.min().item(),
                'max': feature_magnitude.max().item(),
                'mean': feature_magnitude.mean().item(),
                'std': feature_magnitude.std().item()
            }
        }
        
        # Also show per-dimension statistics (first few dimensions)
        for i in range(min(3, feature_vector_length)):
            feature_stats[f'dim_{i}'] = {
                'min': feature[..., i].min().item(),
                'max': feature[..., i].max().item(),
                'mean': feature[..., i].mean().item(),
                'std': feature[..., i].std().item()
            }
        
        memory_mb = feature.element_size() * feature.nelement() / (1024 * 1024)
        
        volume_info.append({
            'name': 'Features',
            'shape': list(feature.shape),
            'voxel_size': list(feat_voxel_sizes),
            'world_bounds_min': feat_world_bounds_min.tolist(),
            'world_bounds_max': feat_world_bounds_max.tolist(),
            'world_size': feat_world_size.tolist(),
            'occupied_bounds': occupied_bounds,
            'stats': feature_stats,
            'stats_desc': f'{feature_vector_length}D features + magnitude',
            'memory_mb': memory_mb
        })
    
    # Print comprehensive information for each volume
    total_memory = 0
    for i, vol_info in enumerate(volume_info):
        if i > 0:
            print(f"\n{'-'*80}")
        
        print(f"\n{vol_info['name'].upper()} VOLUME:")
        print(f"  Shape: {vol_info['shape']}")
        print(f"  Voxel Size (X,Y,Z): {vol_info['voxel_size'][0]:.1f}, {vol_info['voxel_size'][1]:.1f}, {vol_info['voxel_size'][2]:.1f} mm")
        print(f"  Memory: {vol_info['memory_mb']:.1f} MB")
        
        print(f"\n  World Bounds:")
        print(f"    Min: ({vol_info['world_bounds_min'][0]:.1f}, {vol_info['world_bounds_min'][1]:.1f}, {vol_info['world_bounds_min'][2]:.1f})")
        print(f"    Max: ({vol_info['world_bounds_max'][0]:.1f}, {vol_info['world_bounds_max'][1]:.1f}, {vol_info['world_bounds_max'][2]:.1f})")
        print(f"    Size: ({vol_info['world_size'][0]:.1f}, {vol_info['world_size'][1]:.1f}, {vol_info['world_size'][2]:.1f})")
        
        occ = vol_info['occupied_bounds']
        print(f"\n  Occupied Voxels: {occ['occupied_voxels']:,} / {occ['total_voxels']:,} ({100*occ['occupied_voxels']/occ['total_voxels']:.1f}%)")
        if occ['occupied_voxels'] > 0:
            print(f"  Occupied Voxel Bounds:")
            print(f"    Min Index: ({occ['min_coords'][0]}, {occ['min_coords'][1]}, {occ['min_coords'][2]})")
            print(f"    Max Index: ({occ['max_coords'][0]}, {occ['max_coords'][1]}, {occ['max_coords'][2]})")
            print(f"    World Min: ({occ['world_min'][0]:.1f}, {occ['world_min'][1]:.1f}, {occ['world_min'][2]:.1f})")
            print(f"    World Max: ({occ['world_max'][0]:.1f}, {occ['world_max'][1]:.1f}, {occ['world_max'][2]:.1f})")
            print(f"    World Size: ({occ['world_size'][0]:.1f}, {occ['world_size'][1]:.1f}, {occ['world_size'][2]:.1f})")
        
        print(f"\n  Statistics ({vol_info['stats_desc']}):")
        for stat_name, stat_values in vol_info['stats'].items():
            if isinstance(stat_values, dict):
                print(f"    {stat_name.upper()}: min={stat_values['min']:.1f}, max={stat_values['max']:.1f}, mean={stat_values['mean']:.1f}, std={stat_values['std']:.1f}")
        
        total_memory += vol_info['memory_mb']
    
    print(f"\n{'-'*80}")
    print(f"TOTAL MEMORY: {total_memory:.1f} MB")
    print(f"{'='*80}") 