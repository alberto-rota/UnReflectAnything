import torch
import numpy as np
from typing import Tuple, Optional, Union
from sklearn.decomposition import PCA
from .tsdf_utils import (
    vox2world, cam2pix, world_to_grid, interpolate_features,
    generate_ray_samples, find_surface_intersections,
    extract_point_cloud_from_features, extract_point_cloud_from_tsdf,
    extract_mesh_from_tsdf, print_volume_summary
)


class TSDistanceFeatureColorVolume:
    """
    Volumetric TSDF (Truncated Signed Distance Function) Fusion of RGB-D Images using PyTorch.

    This implementation uses a one-sided TSDF approach (no negative values) for efficient
    3D reconstruction from RGB-D camera data. The volume stores distance values, weights,
    color information, and optionally feature information in a voxelized 3D grid.

    Attributes:
        device (torch.device): The device (CPU/GPU) where computations are performed
        volume_bounds (torch.Tensor): Volume bounds in world coordinates [3, 2] (min/max per axis)
        voxel_size (float): Size of each voxel in meters
        truncation_margin (float): Truncation margin for TSDF values
        volume_dim (torch.Tensor): Volume dimensions in voxels [X, Y, Z]
        volume_origin (torch.Tensor): Origin point of the volume in world coordinates
        tsdf (torch.Tensor): TSDF values for each voxel
        weights (torch.Tensor): Fusion weights for each voxel
        color (torch.Tensor): RGB color values for each voxel
        feature (torch.Tensor, optional): Feature vectors for each voxel (if feature_dim specified)
        feature_vector_length (int, optional): Dimension of feature vectors
        feature_dim (torch.Tensor, optional): Feature volume dimensions [X_f, Y_f, Z] (lower resolution)
        feature_weights (torch.Tensor, optional): Fusion weights for feature volume
        vox_coords (torch.Tensor): Precomputed voxel coordinates [N, 3]
    """

    def __init__(
        self,
        vol_bnds: Union[np.ndarray, torch.Tensor],
        voxel_size: float,
        margin: Optional[float] = None,
        device: Optional[torch.device] = None,
        feature_vector_length: Optional[int] = None,
    ) -> None:
        """
        Initialize the TSDF volume with specified bounds and resolution.

        Args:
            vol_bnds: (3, 2) array of xyz bounds (min/max) in meters
            voxel_size: Volume discretization in meters - smaller values give higher resolution
            margin: Truncation margin for TSDF. If None, defaults to 2.0 * voxel_size
            device: PyTorch device (CPU or GPU). If None, automatically selects GPU if available
            feature_vector_length: Dimension of feature vectors to store per voxel. If None, no feature volume is created

        Raises:
            AssertionError: If vol_bnds is not of shape (3, 2)
        """
        # Convert volume bounds to tensor if needed
        if not isinstance(vol_bnds, torch.Tensor):
            vol_bnds = torch.tensor(vol_bnds, dtype=torch.float32)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Initialize device and basic parameters
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.volume_bounds = vol_bnds.to(self.device)
        self.voxel_size = float(voxel_size)
        self.truncation_margin = margin if margin is not None else 2.0 * self.voxel_size
        self.feature_vector_length = feature_vector_length

        # Compute volume dimensions and adjust bounds to fit exact voxel grid
        self.volume_dim = torch.ceil(
            (self.volume_bounds[:, 1] - self.volume_bounds[:, 0]) / self.voxel_size
        ).to(torch.int64)
        self.volume_bounds[:, 1] = self.volume_bounds[:, 0] + self.volume_dim * self.voxel_size
        self.volume_origin = self.volume_bounds[:, 0].clone()

        # Initialize voxel volumes
        # TSDF volume: stores truncated signed distance values (starts with 1.0 = unobserved)
        self.tsdf = torch.ones(
            self.volume_dim.tolist(), dtype=torch.float32, device=self.device
        )
        # Weight volume: tracks confidence/number of observations per voxel
        self.weights = torch.zeros(
            self.volume_dim.tolist(), dtype=torch.float32, device=self.device
        )
        # Color volume: stores RGB values (0-255 range)
        self.color = torch.zeros(
            (*self.volume_dim.tolist(), 3), dtype=torch.float32, device=self.device
        )

        # Feature volume: stores feature vectors at lower resolution (initialized on first use)
        if self.feature_vector_length is not None:
            self.feature = (
                None  # Will be initialized when first feature map is provided
            )
            self.feature_dim = None  # Will store feature volume dimensions
            self.feature_weights = None  # Will store feature volume weights
            self._featvoxel_size_x = None  # Will store feature voxel sizes
            self._featvoxel_size_y = None
            self._featvoxel_size_z = None
        else:
            self.feature = None
            self.feature_dim = None
            self.feature_weights = None
            self._featvoxel_size_x = None
            self._featvoxel_size_y = None
            self._featvoxel_size_z = None

        # Precompute all voxel coordinates for efficient processing
        # This creates a flattened list of all (x, y, z) voxel indices
        xx, yy, zz = torch.meshgrid(
            torch.arange(self.volume_dim[0], device=self.device),
            torch.arange(self.volume_dim[1], device=self.device),
            torch.arange(self.volume_dim[2], device=self.device),
            indexing="ij",  # Use matrix indexing convention
        )
        self.vox_coords = torch.stack(
            [xx.flatten(), yy.flatten(), zz.flatten()], dim=1
        ).to(torch.int64)

    def vox2world(self, vox_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert voxel grid indices to world coordinates.

        Transforms discrete voxel coordinates to continuous world coordinates
        by scaling by voxel size and offsetting by volume origin.

        Args:
            vox_coords: [N, 3] tensor of voxel indices (x, y, z)

        Returns:
            world_pts: [N, 3] tensor of world coordinates in meters
        """
        return vox2world(vox_coords, self.volume_origin, self.voxel_size, self.device)

    def cam2pix(self, cam_pts: torch.Tensor, intr: torch.Tensor) -> torch.Tensor:
        """
        Project 3D camera coordinates to 2D pixel coordinates using camera intrinsics.

        Uses the standard pinhole camera model: u = fx * X/Z + cx, v = fy * Y/Z + cy

        Args:
            cam_pts: [N, 3] tensor of 3D points in camera coordinate system
            intr: [3, 3] camera intrinsic matrix

        Returns:
            pix: [N, 2] tensor of pixel coordinates (u, v) as integers
        """
        return cam2pix(cam_pts, intr)

    def integrate(
        self,
        frame: Union[np.ndarray, torch.Tensor],  # [3, H, W] or [H, W, 3] RGB image (values 0-255 or 0-1)
        depthmap: Union[np.ndarray, torch.Tensor],  # [1, H, W] or [H, W] depth image in meters
        K: Union[np.ndarray, torch.Tensor],  # [3, 3] camera intrinsic matrix
        T: Union[np.ndarray, torch.Tensor],  # [4, 4] camera-to-world transformation matrix
        featuremap: Optional[Union[np.ndarray, torch.Tensor]] = None,  # [E, H_f, W_f] or [H_f, W_f, E] feature map
        weights: float = 1.0,
    ) -> None:
        """
        Integrate an RGB-D frame into the one-sided TSDF volume.

        This is the core fusion algorithm that updates the TSDF, color, and optionally
        feature volumes with new observations. Uses a one-sided approach where only
        voxels in front of the observed surface are updated.

        Args:
            frame: [3, H, W] or [H, W, 3] RGB image (values 0-255 or 0-1)
            depthmap: [1, H, W] or [H, W] depth image in meters
            K: [3, 3] camera intrinsic matrix
            T: [4, 4] camera-to-world transformation matrix
            weights: Weight for this observation in the fusion process
            featuremap: [E, H_f, W_f] or [H_f, W_f, E] feature map (optional). Will be upsampled to match frame size.
                     Only used if feature volume was initialized (feature_dim is not None)
        """
        # Convert all inputs to tensors on the correct device
        if not isinstance(frame, torch.Tensor):
            frame = torch.tensor(frame, dtype=torch.float32, device=self.device)
        if not isinstance(depthmap, torch.Tensor):
            depthmap = torch.tensor(depthmap, dtype=torch.float32, device=self.device)
        if not isinstance(K, torch.Tensor):
            K = torch.tensor(K, dtype=torch.float32, device=self.device)
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=torch.float32, device=self.device)

        # Convert frame to HWC format if needed
        if frame.shape[0] == 3:  # CHW format
            frame = frame.permute(1, 2, 0)  # Convert to HWC
        elif frame.shape[-1] != 3:
            raise ValueError(f"Frame must be [3, H, W] or [H, W, 3], got shape {frame.shape}")

        # Convert depthmap to HW format if needed
        if depthmap.shape[0] == 1:  # CHW format
            depthmap = depthmap.squeeze(0)  # Convert to HW
        elif len(depthmap.shape) != 2:
            raise ValueError(f"Depthmap must be [1, H, W] or [H, W], got shape {depthmap.shape}")

        # Normalize color to 0-255 range if needed
        if frame.max() <= 1.0:
            frame = frame * 255.0

        # Ensure all tensors are on the correct device
        frame = frame.to(self.device)
        depthmap = depthmap.to(self.device)
        K = K.to(self.device)
        T = T.to(self.device)
        im_h, im_w = frame.shape[:2]

        # Process featuremap if provided and feature volume exists
        if featuremap is not None and self.feature_vector_length is not None:
            # Convert featuremap to tensor if needed
            if not isinstance(featuremap, torch.Tensor):
                featuremap = torch.tensor(
                    featuremap, dtype=torch.float32, device=self.device
                )
            
            # Convert featuremap to HWC format if needed
            if featuremap.shape[0] == self.feature_vector_length:  # CHW format
                featuremap = featuremap.permute(2,1, 0)  # Convert to HWC
            elif featuremap.shape[0] != self.feature_vector_length:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.feature_vector_length}, got {featuremap.shape[0]}"
                )
            featuremap = featuremap.permute(1,0,2).to(self.device)
            feat_h, feat_w = featuremap.shape[:2]

            # Initialize feature volume on first use
            if self.feature is None:
                # Calculate feature voxel size based on resolution difference
                # Feature voxel size = mainvoxel_size * (image_res / feature_res)
                # Z-axis voxel size should match main volume since depth doesn't scale with image resolution
                featvoxel_size_x = 1#self.voxel_size * (im_w / feat_w)
                featvoxel_size_y = 1#self.voxel_size * (im_h / feat_h)
                featvoxel_size_z = 1#self.voxel_size * (im_h / feat_h) # Same as main volume - depth dimension unchanged

                # Compute feature volume dimensions to cover same world bounds
                feat_vol_x = int(
                    torch.ceil(
                        (self.volume_bounds[0, 1] - self.volume_bounds[0, 0])
                        / featvoxel_size_x
                    ).item()
                )
                feat_vol_y = int(
                    torch.ceil(
                        (self.volume_bounds[1, 1] - self.volume_bounds[1, 0])
                        / featvoxel_size_y
                    ).item()
                )
                feat_vol_z = int(
                    torch.ceil(
                        (self.volume_bounds[2, 1] - self.volume_bounds[2, 0])
                        / featvoxel_size_z
                    ).item()
                )

                self.feature_dim = torch.tensor(
                    [feat_vol_x, feat_vol_y, feat_vol_z],
                    dtype=torch.int64,
                    device=self.device,
                )
                self._featvoxel_size_x = featvoxel_size_x
                self._featvoxel_size_y = featvoxel_size_y
                self._featvoxel_size_z = featvoxel_size_z

                # Initialize feature volume and weights
                self.feature = torch.zeros(
                    (feat_vol_x, feat_vol_y, feat_vol_z, self.feature_vector_length),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.feature_weights = torch.zeros(
                    (feat_vol_x, feat_vol_y, feat_vol_z),
                    dtype=torch.float32,
                    device=self.device,
                )

            # Get stored voxel sizes
            featvoxel_size_x = self._featvoxel_size_x
            featvoxel_size_y = self._featvoxel_size_y
            featvoxel_size_z = self._featvoxel_size_z

        # Transform voxel coordinates to camera frame
        # Step 1: Convert voxel indices to world coordinates
        world_pts = vox2world(self.vox_coords, self.volume_origin, self.voxel_size, self.device)
        # Step 2: Convert to homogeneous coordinates
        world_pts_h = torch.cat(
            [world_pts, torch.ones((world_pts.shape[0], 1), device=self.device)], dim=1
        )
        # Step 3: Transform world points to camera frame using inverse pose
        cam_pts = (torch.inverse(T) @ world_pts_h.t()).t()[:, :3]

        # Project camera points to image pixels
        pix_z = cam_pts[:, 2]  # Depth values in camera frame
        pix_xy = cam2pix(cam_pts, K)
        pix_x, pix_y = pix_xy[:, 0], pix_xy[:, 1]

        # Create mask for valid projections (within image bounds and positive depth)
        valid_pix = (
            (pix_x >= 0)
            & (pix_x < im_w)
            & (pix_y >= 0)
            & (pix_y < im_h)
            & (pix_z > 0)  # Points must be in front of camera
        )

        # Sample depth values from the depth image at projected pixel locations
        depth_val = torch.zeros_like(pix_z)
        idxs = torch.nonzero(valid_pix, as_tuple=True)[0]
        depth_val[idxs] = depthmap[pix_y[idxs], pix_x[idxs]]

        # Compute signed distance: positive means voxel is in front of surface
        depth_diff = depth_val - pix_z

        # One-sided TSDF: only update voxels in front of surface within truncation margin
        # This means: observed depth > 0 AND 0 ≤ (depth_val - pix_z) ≤ trunc_margin
        in_front = (
            (depth_val > 0) & (depth_diff >= 0) & (depth_diff <= self.truncation_margin)
        )
        valid_inds = torch.nonzero(in_front, as_tuple=True)[0]
        if valid_inds.numel() == 0:
            return  # No valid voxels to update

        # Normalize distances to [0, 1] range for TSDF storage
        dist = torch.clamp(depth_diff[valid_inds] / self.truncation_margin, max=1.0)

        # Get voxel indices for updating
        vx = self.vox_coords[valid_inds, 0]
        vy = self.vox_coords[valid_inds, 1]
        vz = self.vox_coords[valid_inds, 2]

        # Retrieve current TSDF and weight values
        w_old = self.weights[vx, vy, vz]
        tsdf_old = self.tsdf[vx, vy, vz]

        # Weighted fusion of TSDF values
        w_new = w_old + weights
        tsdf_new = (w_old * tsdf_old + weights * dist) / w_new

        # Update TSDF and weight volumes
        self.weights[vx, vy, vz] = w_new
        self.tsdf[vx, vy, vz] = tsdf_new

        # Fuse colors (vectorized across all 3 channels)
        old_colors = self.color[vx, vy, vz, :]  # [N, 3]
        new_colors = frame[pix_y[valid_inds], pix_x[valid_inds], :]  # [N, 3]
        fused_colors = (
            w_old.unsqueeze(-1) * old_colors + weights * new_colors
        ) / w_new.unsqueeze(-1)
        # Clamp colors to valid range [0, 255]
        self.color[vx, vy, vz, :] = torch.clamp(fused_colors, 0, 255)

        # Fuse featuremap if provided and feature volume exists (vectorized across all feature channels)
        if featuremap is not None and self.feature is not None:
            # Map image pixel coordinates to feature map coordinates
            feat_pix_x = torch.floor(pix_x[valid_inds].float() * feat_w / im_w).long()
            feat_pix_y = torch.floor(pix_y[valid_inds].float() * feat_h / im_h).long()

            # Clamp to feature map bounds
            feat_pix_x = torch.clamp(feat_pix_x, 0, feat_w - 1)
            feat_pix_y = torch.clamp(feat_pix_y, 0, feat_h - 1)

            # Map world coordinates to feature volume voxel coordinates
            world_pts_valid = world_pts[valid_inds]
            feat_vx = (
                ((world_pts_valid[:, 0] - self.volume_origin[0]) / featvoxel_size_x)
                .round()
                .long()
            )
            feat_vy = (
                ((world_pts_valid[:, 1] - self.volume_origin[1]) / featvoxel_size_y)
                .round()
                .long()
            )
            feat_vz = (
                ((world_pts_valid[:, 2] - self.volume_origin[2]) / featvoxel_size_z)
                .round()
                .long()
            )
            # Clamp to feature volume bounds
            feat_vx = torch.clamp(feat_vx, 0, self.feature_dim[0] - 1).int()
            feat_vy = torch.clamp(feat_vy, 0, self.feature_dim[1] - 1).int()
            feat_vz = torch.clamp(feat_vz, 0, self.feature_dim[2] - 1).int()

            # Get old feature weights for this feature volume location
            feat_w_old = self.feature_weights[feat_vx, feat_vy, feat_vz]
            feat_w_new = feat_w_old + weights
            old_featuremap = self.feature[feat_vx, feat_vy, feat_vz, :]
            new_featuremap = featuremap[feat_pix_y, feat_pix_x, :]
            fused_featuremap = (feat_w_old.unsqueeze(-1) * old_featuremap + weights * new_featuremap) / feat_w_new.unsqueeze(-1)

            self.feature[feat_vx, feat_vy, feat_vz, :] = fused_featuremap
            self.feature_weights[feat_vx, feat_vy, feat_vz] = feat_w_new

    def world_to_grid(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Convert world-space points to normalized grid coordinates for grid_sample.

        Transforms world coordinates to the [-1, 1] coordinate system expected
        by PyTorch's grid_sample function.

        Args:
            pts: [..., 3] tensor of points in world coordinates

        Returns:
            norm: [..., 3] tensor of normalized grid coordinates in [-1, 1] range
        """
        return world_to_grid(pts, self.volume_origin, self.volume_dim, self.voxel_size)

    def _interpolate_features(self, world_pts: np.ndarray) -> np.ndarray:
        if self.feature is None:
            raise ValueError("No feature volume available for interpolation")
        
        return interpolate_features(
            world_pts,
            self.feature,
            self.volume_origin,
            self.feature_dim,
            self._featvoxel_size_x,
            self._featvoxel_size_y,
            self._featvoxel_size_z,
            self.device
        )

    def raycast(
        self,
        K: Union[np.ndarray, torch.Tensor],
        T: Union[np.ndarray, torch.Tensor],
        im_h: int,
        im_w: int,
        target: str = "depthmap",
        near: float = 0.1,
        far: float = 10.0,
        n_samples: int = 1000,
        surface_threshold: float = 0.05,  # Added surface threshold parameter
    ) -> torch.Tensor:
        """
        Ray-cast the TSDF volume to produce synthetic depth maps or feature maps.

        Generates a depth image or feature map by casting rays through the TSDF volume and finding
        zero-crossings (surface intersections) using the one-sided TSDF values.

        Args:
            K: [3, 3] camera intrinsic matrix
            T: [4, 4] camera-to-world transformation matrix  
            im_h: Output map height in pixels
            im_w: Output map width in pixels
            target: Target output type - "depthmap" or "features"
            near: Near clipping plane distance in meters
            far: Far clipping plane distance in meters
            n_samples: Number of samples along each ray
            surface_threshold: TSDF threshold value for detecting surface intersections (default: 0.05)

        Returns:
            If target="depthmap": [1, im_h, im_w] tensor of depths in meters (0 where no surface hit)
            If target="features": [im_h, im_w, E] tensor of feature vectors (0 where no surface hit)
        
        Raises:
            ValueError: If target is not "depthmap" or "features", or if no feature volume is available for features
        """
        if target not in ["depthmap", "features"]:
            raise ValueError(f"Target must be 'depthmap' or 'features', got {target}")
        
        if target == "features" and self.feature is None:
            raise ValueError("No feature volume available for feature raycasting")
    
        # Convert inputs to tensors on correct device
        if not isinstance(K, torch.Tensor):
            K = torch.tensor(K, dtype=torch.float32, device=self.device)
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=torch.float32, device=self.device)
    
        K = K.to(self.device)
        T = T.to(self.device)

        # Generate ray samples using helper function
        sample_pts, origins, dirs_world = generate_ray_samples(
            K, T, im_h, im_w, near, far, n_samples, self.device
        )
        depths = torch.linspace(near, far, n_samples, device=self.device)
    
        # Flatten for efficient TSDF sampling - shape: [im_h*im_w*n_samples, 3]
        sample_pts_flat = sample_pts.reshape(-1, 3)
    
        # Convert world points to normalized grid coordinates for grid_sample
        grid_coords = world_to_grid(sample_pts_flat, self.volume_origin, self.volume_dim, self.voxel_size)  # shape: [im_h*im_w*n_samples, 3]
    
        # Reshape for grid_sample: [1, im_h*im_w, n_samples, 1, 3]  
        grid_coords = grid_coords.view(1, im_h * im_w, n_samples, 1, 3)
    
        # Prepare TSDF volume for sampling: [1, 1, Z, Y, X]
        tsdf_for_sample = self.tsdf.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
    
        # Sample TSDF values using trilinear interpolation
        tsdf_samples = torch.nn.functional.grid_sample(
            tsdf_for_sample,
            grid_coords,
            mode='nearest',  # trilinear for 3D
            padding_mode='border',
            align_corners=True
        )  # shape: [1, 1, im_h*im_w, n_samples, 1]
    
        # Reshape to [im_h, im_w, n_samples]
        tsdf_samples = tsdf_samples.squeeze().view(im_h, im_w, n_samples)
    
        if target == "depthmap":
            # Find surface intersections using helper function
            depth_map, valid_hit = find_surface_intersections(tsdf_samples, depths)
            return depth_map.unsqueeze(0)
        
        elif target == "features":
            # Find surface intersections by looking for zero crossings
            # For one-sided TSDF: surface is where TSDF transitions from positive to ~0
            # Look for points where TSDF drops significantly (surface hit)
        
            # Find first point along each ray where TSDF <= surface_threshold
            hit_mask = tsdf_samples <= surface_threshold  # shape: [im_h, im_w, n_samples]
        
            # Get index of first hit along each ray
            hit_indices = hit_mask.float().argmax(dim=2)  # shape: [im_h, im_w]
        
            # Check if any hit occurred (if no hit, argmax returns 0, but first sample might not be a hit)
            has_hit = hit_mask.any(dim=2)  # shape: [im_h, im_w]
            valid_hit = has_hit & (hit_indices > 0)  # Exclude hits at first sample (likely invalid)
        
            # For more accurate intersection point, interpolate between the sample before and after the hit
            # Get coordinate arrays for indexing
            i_coords = torch.arange(im_h, device=self.device).view(-1, 1).expand(-1, im_w)
            j_coords = torch.arange(im_w, device=self.device).view(1, -1).expand(im_h, -1)
        
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
            t_interp = (tsdf_before - surface_threshold) / tsdf_diff
            t_interp = torch.clamp(t_interp, 0, 1)
        
            # Interpolated depth
            depth_interp = depth_before + t_interp * (depth_at - depth_before)
        
            # Calculate world coordinates of surface intersection points
            surface_points = origins + dirs_world * depth_interp.unsqueeze(-1)  # shape: [im_h, im_w, 3]
        
            # Sample features at surface intersection points using the same logic as _interpolate_features
            # Flatten surface points for efficient feature sampling
            surface_points_flat = surface_points.reshape(-1, 3)  # shape: [im_h*im_w, 3]
        
            # Convert world coordinates to feature voxel indices
            vx = (surface_points_flat[:, 0] - self.volume_origin[0]) / self._featvoxel_size_x
            vy = (surface_points_flat[:, 1] - self.volume_origin[1]) / self._featvoxel_size_y
            vz = (surface_points_flat[:, 2] - self.volume_origin[2]) / self._featvoxel_size_z
            feat_coords = torch.stack([vx, vy, vz], dim=-1)  # shape: [im_h*im_w, 3]
        
            # Normalize for grid_sample (align_corners=True)
            dims = self.feature_dim.float() - 1  # [3]
            feat_coords_norm = 2.0 * feat_coords / dims - 1.0  # shape: [im_h*im_w, 3]
        
            # Prepare feature volume for sampling: [1, E, Z, Y, X]
            feat_vol = self.feature.permute(3, 2, 1, 0).contiguous().unsqueeze(0)  # [1, E, Z, Y, X]
        
            # Reshape coordinates for grid_sample: [1, 1, 1, im_h*im_w, 3]
            grid_feat = feat_coords_norm.contiguous().view(1, 1, 1, -1, 3)
        
            # Sample features using trilinear interpolation
            feature_samples = torch.nn.functional.grid_sample(
                feat_vol, 
                grid_feat,
                mode="nearest",       # trilinear for 3D
                padding_mode="border",
                align_corners=True,
            )  # shape: [1, E, 1, 1, im_h*im_w]
        
            # Reshape to [im_h, im_w, E]
            feature_samples = feature_samples.squeeze(0).squeeze(1).squeeze(1).permute(1, 0)  # [im_h*im_w, E]
            feature_map = feature_samples.view(im_h, im_w, self.feature_vector_length)
        
            # Zero out features where no valid surface hit occurred
            valid_hit_expanded = valid_hit.unsqueeze(-1).expand(-1, -1, self.feature_vector_length)
            feature_map = torch.where(valid_hit_expanded, feature_map, torch.zeros_like(feature_map))
        
            return feature_map

    def get_volume(
        self,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Get the current state of the TSDF, color, and optionally feature volumes.

        Note: Feature volume (if present) has lower spatial resolution than TSDF/color volumes.

        Useful for debugging, visualization, or saving/loading volume state.

        Returns:
            If no feature volume:
                Tuple containing:
                    - tsdf_vol: [X, Y, Z] TSDF values
                    - color_vol: [X, Y, Z, 3] RGB color values
            If feature volume exists:
                Tuple containing:
                    - tsdf_vol: [X, Y, Z] TSDF values
                    - color_vol: [X, Y, Z, 3] RGB color values
                    - feature_vol: [X_f, Y_f, Z, E] feature values (lower resolution)
        """
        if self.feature is not None:
            return self.tsdf, self.color, self.feature
        else:
            return self.tsdf, self.color

    def get_point_cloud(
        self,
        threshold: float = 0.05,
        feature_as_colors: bool = False,
        pca: Optional[PCA] = None,
        extract_from_features: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a point cloud from the voxel volume.

        Uses marching cubes to find surface vertices and extracts their
        world coordinates and colors separately. Optionally extracts points
        directly from the feature volume for higher efficiency.

        Args:
            threshold: TSDF level at which to extract the isosurface
                      (0.0 = exact zero-crossing, >0 = slightly in front of surface)
            feature_as_colors: Whether to use PCA-reduced features as colors instead of RGB colors.
                             Requires feature volume to exist. The first 3 PCA components are mapped to RGB.
            pca: Optional pre-fitted PCA transformer. If None and feature_as_colors=True, 
                PCA will be fitted on-the-fly.
            extract_from_features: If True and feature_as_colors=True, extract points directly
                                 from feature volume instead of using marching cubes on TSDF.
                                 This is more efficient but gives points at feature resolution.

        Returns:
            Tuple containing:
                - points: [N, 3] numpy array of 3D point coordinates in world space
                - colors: [N, 3] numpy array of RGB color values (0-255)
        """
        if extract_from_features and feature_as_colors and self.feature is not None:
            return extract_point_cloud_from_features(
                self.feature,
                self.feature_weights,
                self.volume_origin,
                self._featvoxel_size_x,
                self._featvoxel_size_y,
                self._featvoxel_size_z,
                self.feature_vector_length,
                pca
            )
        else:
            return extract_point_cloud_from_tsdf(
                self.tsdf,
                self.color,
                self.volume_origin,
                self.voxel_size,
                threshold,
                self.feature if feature_as_colors else None,
                feature_as_colors,
                pca,
                self._interpolate_features if feature_as_colors else None
            )

    def get_mesh(
        self,
        threshold: float = 0.05,
        feature_as_colors: bool = False,
        pca: Optional[PCA] = None,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Extract a colored mesh from the voxel volume using marching cubes.

        Generates a triangular mesh representation of the reconstructed surface
        with vertex colors and optionally features.

        Args:
            threshold: TSDF level at which to extract the isosurface
                      (0.0 = exact zero-crossing, >0 = slightly in front of surface)
            feature_as_colors: Whether to use PCA-reduced features as colors instead of RGB colors.
                             Requires feature volume to exist. The first 3 PCA components are mapped to RGB.

        Returns:
            If no feature volume and feature_as_colors=False:
                Tuple containing:
                    - verts_world: [N, 3] vertex positions in world coordinates
                    - faces: [M, 3] triangle face indices
                    - norms: [N, 3] vertex normal vectors
                    - colors: [N, 3] vertex colors as RGB values (0-255)
            If feature volume exists and feature_as_colors=True:
                Tuple containing:
                    - verts_world: [N, 3] vertex positions in world coordinates
                    - faces: [M, 3] triangle face indices
                    - norms: [N, 3] vertex normal vectors
                    - colors: [N, 3] vertex colors as RGB values (0-255), or PCA colors if feature_as_colors=True
                    - features: [N, E] vertex feature vectors
        """
        return extract_mesh_from_tsdf(
            self.tsdf,
            self.color,
            self.volume_origin,
            self.voxel_size,
            threshold,
            self.feature if feature_as_colors else None,
            feature_as_colors,
            pca,
            self._interpolate_features if feature_as_colors else None
        )

    def summary(self, volumes_to_show: str = "all") -> None:
        """
        Print a comprehensive summary of the TSDF volume properties.
        
        Args:
            volumes_to_show: Which volumes to display - "all", "tsdf", "color", "features", 
                           or comma-separated combinations like "tsdf,color"
        
        This method displays detailed information about:
        - Volume dimensions and bounds (world coordinates)
        - Voxel sizes and resolutions for each volume
        - Occupied voxel statistics and bounding boxes
        - TSDF, color, and feature volume statistics
        - Memory usage estimates
        """
        print_volume_summary(
            self.device,
            self.voxel_size,
            self.truncation_margin,
            self.volume_bounds,
            self.volume_dim,
            self.tsdf,
            self.color,
            self.feature,
            self.feature_dim,
            self._featvoxel_size_x,
            self._featvoxel_size_y,
            self._featvoxel_size_z,
            self.feature_vector_length,
            volumes_to_show
        )

    def trim(self) -> dict:
        """
        Trim all volumes to only contain occupied voxels, reducing memory usage.
        
        This method crops the TSDF, color, weights, and feature volumes (if present) to their
        minimum bounding boxes containing non-empty voxels. This significantly reduces memory
        usage while preserving all meaningful data.
        
        For TSDF volume: occupied voxels are those != 1.0 (since 1.0 = unobserved)
        For color/feature volumes: occupied voxels are those with any non-zero values
        
        Returns:
            dict: Summary of memory savings with keys:
                - 'original_memory_mb': Original total memory usage in MB
                - 'trimmed_memory_mb': New total memory usage in MB  
                - 'memory_saved_mb': Amount of memory saved in MB
                - 'memory_saved_percent': Percentage of memory saved
                - 'original_shape': Original volume shapes
                - 'trimmed_shape': New volume shapes
                - 'trim_bounds': Voxel indices used for trimming [min_coords, max_coords]
        """
        def calculate_memory_mb(tensor: torch.Tensor) -> float:
            """Calculate memory usage of a tensor in MB."""
            return tensor.element_size() * tensor.nelement() / (1024 * 1024)
        
        # Calculate original memory usage
        original_memory = calculate_memory_mb(self.tsdf) + calculate_memory_mb(self.color) + calculate_memory_mb(self.weights)
        if self.feature is not None:
            original_memory += calculate_memory_mb(self.feature) + calculate_memory_mb(self.feature_weights)
        
        # Find occupied bounds for main TSDF volume
        # For TSDF: consider values != 1.0 as occupied (since 1.0 = unobserved)
        occupied_mask = (self.tsdf != 1.0)  # Shape: [X, Y, Z]
        
        if not occupied_mask.any():
            # No occupied voxels - nothing to trim
            return {
                'original_memory_mb': original_memory,
                'trimmed_memory_mb': original_memory,
                'memory_saved_mb': 0.0,
                'memory_saved_percent': 0.0,
                'original_shape': {'tsdf': list(self.tsdf.shape), 'color': list(self.color.shape)},
                'trimmed_shape': {'tsdf': list(self.tsdf.shape), 'color': list(self.color.shape)},
                'trim_bounds': [[0, 0, 0], [0, 0, 0]]
            }
        
        # Find min/max occupied voxel coordinates
        occupied_indices = torch.nonzero(occupied_mask)  # Shape: [N, 3]
        min_coords = occupied_indices.min(dim=0)[0]  # Shape: [3]
        max_coords = occupied_indices.max(dim=0)[0]  # Shape: [3]
        
        # Add small padding to avoid edge effects (1 voxel on each side)
        min_coords = torch.clamp(min_coords - 1, min=0)
        max_coords = torch.clamp(max_coords + 1, max=self.volume_dim - 1)
        
        # Extract trimmed ranges
        x_range = slice(min_coords[0].item(), max_coords[0].item() + 1)
        y_range = slice(min_coords[1].item(), max_coords[1].item() + 1)
        z_range = slice(min_coords[2].item(), max_coords[2].item() + 1)
        
        # Store original shapes for reporting
        original_shapes = {
            'tsdf': list(self.tsdf.shape),
            'color': list(self.color.shape),
            'weights': list(self.weights.shape)
        }
        
        # Trim main volumes - Shape: [X, Y, Z] -> [X', Y', Z']
        self.tsdf = self.tsdf[x_range, y_range, z_range].contiguous()
        self.weights = self.weights[x_range, y_range, z_range].contiguous()
        # Color volume - Shape: [X, Y, Z, 3] -> [X', Y', Z', 3]
        self.color = self.color[x_range, y_range, z_range, :].contiguous()
        
        # Update volume dimensions and bounds
        new_volume_dim = torch.tensor([
            max_coords[0] - min_coords[0] + 1,
            max_coords[1] - min_coords[1] + 1,
            max_coords[2] - min_coords[2] + 1
        ], dtype=torch.int64, device=self.device)
        
        # Update volume origin (world coordinates of new min corner)
        min_coords_world = self.volume_origin + min_coords.to(self.device).float() * self.voxel_size
        
        # Update volume bounds
        self.volume_bounds[:, 0] = min_coords_world
        self.volume_bounds[:, 1] = min_coords_world + new_volume_dim.float() * self.voxel_size
        self.volume_origin = min_coords_world
        self.volume_dim = new_volume_dim
        
        # Trim feature volume if it exists
        if self.feature is not None:
            # Find occupied bounds for feature volume
            # For features: occupied voxels are those with any non-zero values
            feature_occupied_mask = (self.feature != 0).any(dim=-1)  # Shape: [X_f, Y_f, Z_f]
            
            if feature_occupied_mask.any():
                # Find min/max for feature volume
                feat_occupied_indices = torch.nonzero(feature_occupied_mask)  # Shape: [N, 3]
                feat_min_coords = feat_occupied_indices.min(dim=0)[0]  # Shape: [3]
                feat_max_coords = feat_occupied_indices.max(dim=0)[0]  # Shape: [3]
                
                # Add padding and clamp to feature volume bounds
                feat_min_coords = torch.clamp(feat_min_coords - 1, min=0)
                feat_max_coords = torch.clamp(feat_max_coords + 1, max=self.feature_dim - 1)
                
                # Extract feature trimmed ranges
                fx_range = slice(feat_min_coords[0].item(), feat_max_coords[0].item() + 1)
                fy_range = slice(feat_min_coords[1].item(), feat_max_coords[1].item() + 1)
                fz_range = slice(feat_min_coords[2].item(), feat_max_coords[2].item() + 1)
                
                # Store original feature shapes
                original_shapes['feature'] = list(self.feature.shape)
                original_shapes['feature_weights'] = list(self.feature_weights.shape)
                
                # Trim feature volumes
                self.feature = self.feature[fx_range, fy_range, fz_range, :].contiguous()
                self.feature_weights = self.feature_weights[fx_range, fy_range, fz_range].contiguous()
                
                # Update feature volume dimensions
                self.feature_dim = torch.tensor([
                    feat_max_coords[0] - feat_min_coords[0] + 1,
                    feat_max_coords[1] - feat_min_coords[1] + 1,
                    feat_max_coords[2] - feat_min_coords[2] + 1
                ], dtype=torch.int64, device=self.device)
        
        # Recompute voxel coordinates for new dimensions
        xx, yy, zz = torch.meshgrid(
            torch.arange(self.volume_dim[0], device=self.device),
            torch.arange(self.volume_dim[1], device=self.device),
            torch.arange(self.volume_dim[2], device=self.device),
            indexing="ij",
        )
        self.vox_coords = torch.stack(
            [xx.flatten(), yy.flatten(), zz.flatten()], dim=1
        ).to(torch.int64)
        
        # Calculate new memory usage
        trimmed_memory = calculate_memory_mb(self.tsdf) + calculate_memory_mb(self.color) + calculate_memory_mb(self.weights)
        if self.feature is not None:
            trimmed_memory += calculate_memory_mb(self.feature) + calculate_memory_mb(self.feature_weights)
        
        # Prepare trimmed shapes for reporting
        trimmed_shapes = {
            'tsdf': list(self.tsdf.shape),
            'color': list(self.color.shape),
            'weights': list(self.weights.shape)
        }
        if self.feature is not None:
            trimmed_shapes['feature'] = list(self.feature.shape)
            trimmed_shapes['feature_weights'] = list(self.feature_weights.shape)
        
        # Calculate savings
        memory_saved = original_memory - trimmed_memory
        memory_saved_percent = (memory_saved / original_memory * 100) if original_memory > 0 else 0.0
        
        return {
            'original_memory_mb': original_memory,
            'trimmed_memory_mb': trimmed_memory,
            'memory_saved_mb': memory_saved,
            'memory_saved_percent': memory_saved_percent,
            'original_shape': original_shapes,
            'trimmed_shape': trimmed_shapes,
            'trim_bounds': [min_coords.tolist(), max_coords.tolist()]
        }


class TSDistanceColorVolume:
    """
    Simplified RGB-only TSDF (Truncated Signed Distance Function) Fusion using PyTorch.
    
    This is a streamlined version focused purely on RGB-D reconstruction without feature volumes.
    Uses a one-sided TSDF approach for efficient 3D reconstruction from RGB-D camera data.
    
    Attributes:
        device (torch.device): The device (CPU/GPU) where computations are performed
        volume_bounds (torch.Tensor): Volume bounds in world coordinates [3, 2] (min/max per axis)  
        voxel_size (float): Size of each voxel in meters
        truncation_margin (float): Truncation margin for TSDF values
        volume_dim (torch.Tensor): Volume dimensions in voxels [X, Y, Z]
        volume_origin (torch.Tensor): Origin point of the volume in world coordinates
        tsdf (torch.Tensor): TSDF values for each voxel [X, Y, Z]
        weights (torch.Tensor): Fusion weights for each voxel [X, Y, Z]
        color (torch.Tensor): RGB color values for each voxel [X, Y, Z, 3]
        vox_coords (torch.Tensor): Precomputed voxel coordinates [N, 3]
    """
    
    def __init__(
        self,
        vol_bnds: Union[np.ndarray, torch.Tensor],  # [3, 2] volume bounds (min/max per axis)
        voxel_size: float,  # Voxel size in meters
        margin: Optional[float] = None,  # Truncation margin, defaults to 2.0 * voxel_size
        device: Optional[torch.device] = None,  # Device for computation
    ) -> None:
        """
        Initialize RGB-only TSDF volume.
        
        Args:
            vol_bnds: [3, 2] array of xyz bounds (min/max) in meters
            voxel_size: Volume discretization in meters - smaller = higher resolution
            margin: Truncation margin for TSDF. If None, defaults to 2.0 * voxel_size
            device: PyTorch device. If None, auto-selects GPU if available
        """
        # Convert volume bounds to tensor if needed - Shape: [3, 2]
        if not isinstance(vol_bnds, torch.Tensor):
            vol_bnds = torch.tensor(vol_bnds, dtype=torch.float32)
        assert vol_bnds.shape == (3, 2), f"vol_bnds should be [3, 2], got {vol_bnds.shape}"
        
        # Initialize device and parameters
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.volume_bounds = vol_bnds.to(self.device)  # Shape: [3, 2]
        self.voxel_size = float(voxel_size)
        self.truncation_margin = margin if margin is not None else 2.0 * self.voxel_size
        
        # Compute volume dimensions - Shape: [3]
        self.volume_dim = torch.ceil(
            (self.volume_bounds[:, 1] - self.volume_bounds[:, 0]) / self.voxel_size
        ).to(torch.int64)
        
        # Adjust bounds to fit exact voxel grid
        self.volume_bounds[:, 1] = self.volume_bounds[:, 0] + self.volume_dim * self.voxel_size
        self.volume_origin = self.volume_bounds[:, 0].clone()  # Shape: [3]
        
        # Initialize volumes - all GPU-optimized tensors
        # TSDF: 1.0 = unobserved, 0.0 = surface, values in [0, 1] - Shape: [X, Y, Z]
        self.tsdf = torch.ones(
            self.volume_dim.tolist(), dtype=torch.float32, device=self.device
        )
        
        # Weights: accumulates confidence per voxel - Shape: [X, Y, Z]  
        self.weights = torch.zeros(
            self.volume_dim.tolist(), dtype=torch.float32, device=self.device
        )
        
        # Colors: RGB values in [0, 255] range - Shape: [X, Y, Z, 3]
        self.color = torch.zeros(
            (*self.volume_dim.tolist(), 3), dtype=torch.float32, device=self.device
        )
        
        # Precompute voxel coordinates for vectorized operations - Shape: [N, 3]
        xx, yy, zz = torch.meshgrid(
            torch.arange(self.volume_dim[0], device=self.device),
            torch.arange(self.volume_dim[1], device=self.device), 
            torch.arange(self.volume_dim[2], device=self.device),
            indexing="ij"
        )
        self.vox_coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        
    def integrate(
        self,
        color_im: Union[np.ndarray, torch.Tensor],  # [H, W, 3] or [3, H, W] RGB image
        depth_im: Union[np.ndarray, torch.Tensor],  # [H, W] or [1, H, W] depth in meters
        cam_intr: Union[np.ndarray, torch.Tensor],  # [3, 3] camera intrinsics
        cam_pose: Union[np.ndarray, torch.Tensor],  # [4, 4] camera-to-world transform
        obs_weight: float = 1.0,  # Weight for this observation
    ) -> None:
        """
        Integrate RGB-D observation into TSDF volume using vectorized GPU operations.
        
        Args:
            color_im: RGB image - [H, W, 3] or [3, H, W], values 0-255 or 0-1
            depth_im: Depth image - [H, W] or [1, H, W], values in meters
            cam_intr: Camera intrinsics - [3, 3] matrix
            cam_pose: Camera-to-world pose - [4, 4] transformation matrix
            obs_weight: Weight for this observation in fusion
        """
        # Convert inputs to GPU tensors with proper shapes
        if not isinstance(color_im, torch.Tensor):
            color_im = torch.tensor(color_im, dtype=torch.float32, device=self.device)
        if not isinstance(depth_im, torch.Tensor):
            depth_im = torch.tensor(depth_im, dtype=torch.float32, device=self.device)
        if not isinstance(cam_intr, torch.Tensor):
            cam_intr = torch.tensor(cam_intr, dtype=torch.float32, device=self.device)
        if not isinstance(cam_pose, torch.Tensor):
            cam_pose = torch.tensor(cam_pose, dtype=torch.float32, device=self.device)
            
        # Ensure tensors are on correct device
        color_im = color_im.to(self.device)
        depth_im = depth_im.to(self.device)
        cam_intr = cam_intr.to(self.device)
        cam_pose = cam_pose.to(self.device)
        
        # Normalize color image to HWC format - Shape: [H, W, 3]
        if color_im.shape[0] == 3:  # CHW -> HWC
            color_im = color_im.permute(1, 2, 0)
        elif color_im.shape[-1] != 3:
            raise ValueError(f"Color image must be [H, W, 3] or [3, H, W], got {color_im.shape}")
            
        # Normalize depth image to HW format - Shape: [H, W]
        if len(depth_im.shape) == 3 and depth_im.shape[0] == 1:  # [1, H, W] -> [H, W]
            depth_im = depth_im.squeeze(0)
        elif len(depth_im.shape) != 2:
            raise ValueError(f"Depth image must be [H, W] or [1, H, W], got {depth_im.shape}")
            
        # Normalize color values to [0, 255] range
        if color_im.max() <= 1.0:
            color_im = color_im * 255.0
            
        im_h, im_w = color_im.shape[:2]
        
        # Vectorized coordinate transformation - Shape: [N, 3] -> [N, 3]
        world_pts = vox2world(self.vox_coords, self.volume_origin, self.voxel_size, self.device)
        
        # Convert to homogeneous coordinates - Shape: [N, 4]
        world_pts_h = torch.cat([world_pts, torch.ones(world_pts.shape[0], 1, device=self.device)], dim=1)
        
        # Transform to camera frame - Shape: [N, 3]
        cam_pts = (torch.inverse(cam_pose) @ world_pts_h.t()).t()[:, :3]
        
        # Project to image coordinates - Shape: [N, 2]
        pix_z = cam_pts[:, 2]  # Depth values
        pix_xy = cam2pix(cam_pts, cam_intr)
        pix_x, pix_y = pix_xy[:, 0], pix_xy[:, 1]
        
        # Create validity mask for projections - Shape: [N]
        valid_pix = (
            (pix_x >= 0) & (pix_x < im_w) & 
            (pix_y >= 0) & (pix_y < im_h) &
            (pix_z > 0)  # Points in front of camera
        )
        
        # Sample depth values at projected locations - Shape: [N]
        depth_val = torch.zeros_like(pix_z)
        valid_idxs = torch.nonzero(valid_pix, as_tuple=True)[0]
        if valid_idxs.numel() == 0:
            return  # No valid projections
            
        depth_val[valid_idxs] = depth_im[pix_y[valid_idxs], pix_x[valid_idxs]]
        
        # Compute signed distance (positive = in front of surface) - Shape: [N]
        depth_diff = depth_val - pix_z
        
        # One-sided TSDF: only update voxels in front of surface within truncation margin
        in_front = (
            (depth_val > 0) & 
            (depth_diff >= -self.truncation_margin) & 
            (depth_diff <= self.truncation_margin)
        )
        
        valid_inds = torch.nonzero(in_front, as_tuple=True)[0]
        if valid_inds.numel() == 0:
            return  # No valid voxels to update
            
        # Normalize distances to [0, 1] range - Shape: [M] where M = num valid voxels
        dist = torch.clamp(depth_diff[valid_inds] / self.truncation_margin, max=1.0)
        
        # Get voxel indices for updating - Shape: [M]
        vx = self.vox_coords[valid_inds, 0]
        vy = self.vox_coords[valid_inds, 1] 
        vz = self.vox_coords[valid_inds, 2]
        
        # Vectorized TSDF fusion - Shape: [M]
        w_old = self.weights[vx, vy, vz]
        tsdf_old = self.tsdf[vx, vy, vz]
        w_new = w_old + obs_weight
        tsdf_new = (w_old * tsdf_old + obs_weight * dist) / w_new
        
        # Update TSDF and weights
        self.weights[vx, vy, vz] = w_new
        self.tsdf[vx, vy, vz] = tsdf_new
        
        # Vectorized color fusion - Shape: [M, 3]
        old_colors = self.color[vx, vy, vz, :]  # [M, 3]
        new_colors = color_im[pix_y[valid_inds], pix_x[valid_inds], :]  # [M, 3]
        fused_colors = (w_old.unsqueeze(-1) * old_colors + obs_weight * new_colors) / w_new.unsqueeze(-1)
        
        # Update colors with clamping to valid range
        self.color[vx, vy, vz, :] = torch.clamp(fused_colors, 0, 255)
        
    def raycast(
        self,
        cam_intr: Union[np.ndarray, torch.Tensor],  # [3, 3] camera intrinsics
        cam_pose: Union[np.ndarray, torch.Tensor],  # [4, 4] camera-to-world transform
        im_h: int,  # Output image height
        im_w: int,  # Output image width
        near: float = 0.1,  # Near clipping plane
        far: float = 10.0,  # Far clipping plane
        n_samples: int = 500,  # Number of samples per ray
        surface_threshold: float = 0.05,  # TSDF threshold for surface detection
    ) -> torch.Tensor:
        """
        Raycast TSDF volume to generate synthetic depth map using GPU acceleration.
        
        Args:
            cam_intr: Camera intrinsics - [3, 3]
            cam_pose: Camera-to-world pose - [4, 4]
            im_h: Output depth map height
            im_w: Output depth map width
            near: Near clipping distance in meters
            far: Far clipping distance in meters
            n_samples: Samples per ray for surface intersection
            surface_threshold: TSDF value threshold for surface detection
            
        Returns:
            depth_map: [1, im_h, im_w] synthetic depth map in meters
        """
        # Convert inputs to GPU tensors
        if not isinstance(cam_intr, torch.Tensor):
            cam_intr = torch.tensor(cam_intr, dtype=torch.float32, device=self.device)
        if not isinstance(cam_pose, torch.Tensor):
            cam_pose = torch.tensor(cam_pose, dtype=torch.float32, device=self.device)
            
        cam_intr = cam_intr.to(self.device)
        cam_pose = cam_pose.to(self.device)
        
        # Generate ray samples - Shape: [im_h, im_w, n_samples, 3]
        sample_pts, _, _ = generate_ray_samples(
            cam_intr, cam_pose, im_h, im_w, near, far, n_samples, self.device
        )
        depths = torch.linspace(near, far, n_samples, device=self.device)
        
        # Flatten for efficient TSDF sampling - Shape: [im_h*im_w*n_samples, 3]
        sample_pts_flat = sample_pts.reshape(-1, 3)
        
        # Convert to normalized grid coordinates - Shape: [im_h*im_w*n_samples, 3]
        grid_coords = world_to_grid(sample_pts_flat, self.volume_origin, self.volume_dim, self.voxel_size)
        
        # Reshape for grid_sample - Shape: [1, im_h*im_w, n_samples, 1, 3]
        grid_coords = grid_coords.view(1, im_h * im_w, n_samples, 1, 3)
        
        # Prepare TSDF for sampling - Shape: [1, 1, Z, Y, X]
        tsdf_for_sample = self.tsdf.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
        
        # Sample TSDF values using trilinear interpolation - Shape: [1, 1, im_h*im_w, n_samples, 1]
        tsdf_samples = torch.nn.functional.grid_sample(
            tsdf_for_sample,
            grid_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # Reshape to [im_h, im_w, n_samples]
        tsdf_samples = tsdf_samples.squeeze().view(im_h, im_w, n_samples)
        
        # Find surface intersections - Shape: [im_h, im_w]
        depth_map, _ = find_surface_intersections(tsdf_samples, depths, surface_threshold)
        
        return depth_map.unsqueeze(0)  # Shape: [1, im_h, im_w]
        
    def get_mesh(
        self,
        threshold: float = 0.05,  # TSDF isosurface level
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract colored mesh from TSDF volume using marching cubes.
        
        Args:
            threshold: TSDF level for surface extraction (0.0 = exact surface)
            
        Returns:
            verts_world: [N, 3] vertex positions in world coordinates
            faces: [M, 3] triangle face indices
            norms: [N, 3] vertex normal vectors  
            colors: [N, 3] vertex RGB colors (0-255)
        """
        return extract_mesh_from_tsdf(
            self.tsdf,
            self.color,
            self.volume_origin,
            self.voxel_size,
            threshold,
            feature_volume=None,
            feature_as_colors=False,
            pca=None,
            interpolate_features_fn=None
        )
        
    def get_point_cloud(
        self,
        threshold: float = 0.05,  # TSDF isosurface level
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract colored point cloud from TSDF volume.
        
        Args:
            threshold: TSDF level for surface extraction
            
        Returns:
            points: [N, 3] point positions in world coordinates
            colors: [N, 3] point RGB colors (0-255)
        """
        return extract_point_cloud_from_tsdf(
            self.tsdf,
            self.color, 
            self.volume_origin,
            self.voxel_size,
            threshold,
            feature_volume=None,
            feature_as_colors=False,
            pca=None,
            interpolate_features_fn=None
        )
        
    def get_volume(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current TSDF and color volumes.
        
        Returns:
            tsdf_vol: [X, Y, Z] TSDF values
            color_vol: [X, Y, Z, 3] RGB color values
        """
        return self.tsdf, self.color
        
    def trim(self) -> dict:
        """
        Trim volume to occupied voxels only, reducing memory usage.
        
        Returns:
            dict: Memory usage statistics and trimming info
        """
        def calculate_memory_mb(tensor: torch.Tensor) -> float:
            return tensor.element_size() * tensor.nelement() / (1024 * 1024)
            
        # Calculate original memory
        original_memory = (
            calculate_memory_mb(self.tsdf) + 
            calculate_memory_mb(self.color) + 
            calculate_memory_mb(self.weights)
        )
        
        # Find occupied voxels (TSDF != 1.0 means observed)
        occupied_mask = (self.tsdf != 1.0)
        
        if not occupied_mask.any():
            return {
                'original_memory_mb': original_memory,
                'trimmed_memory_mb': original_memory,
                'memory_saved_mb': 0.0,
                'memory_saved_percent': 0.0
            }
            
        # Find bounding box of occupied voxels
        occupied_indices = torch.nonzero(occupied_mask)
        min_coords = occupied_indices.min(dim=0)[0]
        max_coords = occupied_indices.max(dim=0)[0]
        
        # Add padding to avoid edge effects
        min_coords = torch.clamp(min_coords - 1, min=0)
        max_coords = torch.clamp(max_coords + 1, max=self.volume_dim - 1)
        
        # Extract trimmed volumes
        x_range = slice(min_coords[0].item(), max_coords[0].item() + 1)
        y_range = slice(min_coords[1].item(), max_coords[1].item() + 1)
        z_range = slice(min_coords[2].item(), max_coords[2].item() + 1)
        
        self.tsdf = self.tsdf[x_range, y_range, z_range].contiguous()
        self.weights = self.weights[x_range, y_range, z_range].contiguous()
        self.color = self.color[x_range, y_range, z_range, :].contiguous()
        
        # Update volume parameters
        new_volume_dim = torch.tensor([
            max_coords[0] - min_coords[0] + 1,
            max_coords[1] - min_coords[1] + 1,
            max_coords[2] - min_coords[2] + 1
        ], dtype=torch.int64, device=self.device)
        
        min_coords_world = self.volume_origin + min_coords.to(self.device).float() * self.voxel_size
        
        self.volume_bounds[:, 0] = min_coords_world
        self.volume_bounds[:, 1] = min_coords_world + new_volume_dim.float() * self.voxel_size
        self.volume_origin = min_coords_world
        self.volume_dim = new_volume_dim
        
        # Recompute voxel coordinates
        xx, yy, zz = torch.meshgrid(
            torch.arange(self.volume_dim[0], device=self.device),
            torch.arange(self.volume_dim[1], device=self.device),
            torch.arange(self.volume_dim[2], device=self.device),
            indexing="ij"
        )
        self.vox_coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        
        # Calculate memory savings
        trimmed_memory = (
            calculate_memory_mb(self.tsdf) + 
            calculate_memory_mb(self.color) + 
            calculate_memory_mb(self.weights)
        )
        
        memory_saved = original_memory - trimmed_memory
        memory_saved_percent = (memory_saved / original_memory * 100) if original_memory > 0 else 0.0
        
        return {
            'original_memory_mb': original_memory,
            'trimmed_memory_mb': trimmed_memory,
            'memory_saved_mb': memory_saved,
            'memory_saved_percent': memory_saved_percent
        }
        
    def summary(self) -> None:
        """Print volume statistics and memory usage."""
        print(f"=== RGB TSDF Volume Summary ===")
        print(f"Device: {self.device}")
        print(f"Volume bounds: {self.volume_bounds.cpu().numpy()}")
        print(f"Volume dimensions: {self.volume_dim.cpu().numpy()}")
        print(f"Voxel size: {self.voxel_size:.4f} meters")
        print(f"Truncation margin: {self.truncation_margin:.4f} meters")
        
        # Calculate occupied voxels
        occupied = (self.tsdf != 1.0).sum().item()
        total = self.tsdf.numel()
        occupancy_percent = (occupied / total) * 100
        
        print(f"Occupied voxels: {occupied:,} / {total:,} ({occupancy_percent:.2f}%)")
        
        # Memory usage
        tsdf_mb = self.tsdf.element_size() * self.tsdf.nelement() / (1024 * 1024)
        color_mb = self.color.element_size() * self.color.nelement() / (1024 * 1024)
        weights_mb = self.weights.element_size() * self.weights.nelement() / (1024 * 1024)
        total_mb = tsdf_mb + color_mb + weights_mb
        
        print(f"Memory usage:")
        print(f"  TSDF: {tsdf_mb:.2f} MB")
        print(f"  Color: {color_mb:.2f} MB") 
        print(f"  Weights: {weights_mb:.2f} MB")
        print(f"  Total: {total_mb:.2f} MB")
