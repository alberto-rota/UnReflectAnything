import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import geometry

class BackProject(nn.Module):
    """Back-projects 2D points to 3D space using depth information.
    
    Supports both RGB images and feature maps with automatic resolution detection.

    Args:
        height (int): Original image height
        width (int): Original image width
        patch_size (int): Patch size for feature maps (default: 16)
    """

    def __init__(self, height: int, width: int, patch_size: int = 16):
        super().__init__()

        self.height = height
        self.width = width
        self.patch_size = patch_size
        
        # Feature map dimensions
        self.feat_height = height // patch_size
        self.feat_width = width // patch_size

        # Create meshgrid of pixel coordinates for original resolution
        x = torch.arange(0, width).float()
        y = torch.arange(0, height).float()
        xx, yy = torch.meshgrid(x, y, indexing="xy")

        # Register buffer for pixel coordinates [3, H*W]
        self.register_buffer(
            "pix_coords",
            torch.stack(
                [xx.reshape(-1), yy.reshape(-1), torch.ones_like(xx).reshape(-1)], dim=0
            ),
        )
        
        # Create meshgrid for feature resolution
        x_feat = torch.arange(0, self.feat_width).float()
        y_feat = torch.arange(0, self.feat_height).float()
        xx_feat, yy_feat = torch.meshgrid(x_feat, y_feat, indexing="xy")

        # Register buffer for feature pixel coordinates [3, Hf*Wf]
        self.register_buffer(
            "feat_pix_coords",
            torch.stack(
                [xx_feat.reshape(-1), yy_feat.reshape(-1), torch.ones_like(xx_feat).reshape(-1)], dim=0
            ),
        )

    def _scale_inverse_intrinsics_for_features(self, invK: torch.Tensor) -> torch.Tensor:
        """Scale inverse intrinsics matrix for feature resolution.
        
        For feature maps downsampled by patch_size, we need to adjust the inverse intrinsics:
        - Scale focal length terms by patch_size
        - Principal point terms remain unchanged in normalized coordinates
        """
        invK_scaled = invK.clone()
        invK_scaled[:, 0, 0] *= self.patch_size  # Scale 1/fx -> patch_size/fx
        invK_scaled[:, 1, 1] *= self.patch_size  # Scale 1/fy -> patch_size/fy
        # invK_scaled[:, 0, 2] *= self.patch_size  # Scale cx -> cx*patch_size
        # invK_scaled[:, 1, 2] *= self.patch_size  # Scale cy -> cy*patch_size
        return invK_scaled

    def forward(
        self,
        image: torch.Tensor,  # [B,3,H,W] or [B,E,Hf,Wf]
        depth: torch.Tensor,  # [B,1,H,W] or [B,H,W]
        invK: torch.Tensor,   # [B,3,3]
        points_match: torch.Tensor = None,
        batch_idx_match: torch.Tensor = None,
    ) -> dict:
        """
        Back-project image points to 3D space.

        Args:
            image: RGB image [B, 3, H, W] or feature map [B, E, Hf, Wf]
            depth: Depth map [B, 1, H, W] or [B, H, W]. If image is a feature map 
                and depth is at original resolution, it will be automatically downsampled.
            invK: Inverse camera intrinsics [B, 3, 3] (for original resolution)
            points_match: Specific pixel coordinates to project in either:
                - [B, N, 2] format (same number of points per batch)
                - [BN, 2] format (variable number of points per batch)
            batch_idx_match: If points_match is [BN, 2], indicates batch indices [BN, 1]

        Returns:
            dict containing:
                - xyz1: Homogeneous 3D points [B, 4, N] where N=H*W or Hf*Wf
                - depth: Flattened depth [B, 1, N]
                - rgb: Flattened RGB values [B, 3, N] (if RGB input)
                - features: Flattened feature values [B, E, N] (if feature input)
                - points_match_3d: 3D coordinates of matched points (format matches input)
                - batch_idx_match: Only returned if input used non-batched format
        """
        batch_size = depth.size(0)
        
        # Ensure depth has channel dimension [B,1,H,W]
        if len(depth.shape) == 3:
            depth = depth.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
        
        # Detect if input is RGB or feature map based on dimensions
        _, channels, img_h, img_w = image.shape
        is_feature_map = (img_h == self.feat_height and img_w == self.feat_width and 
                        channels != 3) or (channels > 3)
        
        if is_feature_map:
            # Working with feature maps - use feature resolution pixel coordinates and scaled intrinsics
            invK_scaled = self._scale_inverse_intrinsics_for_features(invK)
            pix_coords = self.feat_pix_coords.unsqueeze(0).expand(batch_size, -1, -1)
            current_height, current_width = self.feat_height, self.feat_width
            
            # Downsample depth map to feature resolution if needed
            if depth.shape[2] != self.feat_height or depth.shape[3] != self.feat_width:
                depth = F.interpolate(depth, size=(self.feat_height, self.feat_width), 
                                    mode='bilinear', align_corners=True)
        else:
            # Working with RGB images - use original resolution
            if img_h != self.height or img_w != self.width:
                raise ValueError(f"Expected RGB image size {self.height}x{self.width}, got {img_h}x{img_w}")
            invK_scaled = invK
            pix_coords = self.pix_coords.unsqueeze(0).expand(batch_size, -1, -1)
            current_height, current_width = self.height, self.width

        # Transform pixels to camera coordinates using batched matrix multiplication
        cam_points_plane = torch.bmm(invK_scaled, pix_coords)  # [B, 3, N]

        # Scale by depth
        depth_flat = depth.view(batch_size, 1, -1)  # [B, 1, N]
        cam_points = cam_points_plane * depth_flat  # [B, 3, N]

        # Create homogeneous coordinates
        ones = torch.ones_like(depth_flat)  # [B, 1, N]
        cam_points = torch.cat([cam_points, ones], dim=1)  # [B, 4, N]

        # Flatten image/features
        image_flat = image.reshape(batch_size, channels, -1)  # [B, C, N]

        result = {
            "xyz1": cam_points,
            "depth": depth_flat,
        }
        
        # Add appropriate data key based on input type
        if is_feature_map:
            result["features"] = image_flat
        else:
            result["rgb"] = image_flat

        # Handle points_match if provided
        if points_match is not None:
            # Check the format of points_match
            if len(points_match.shape) == 3:  # [B, N, 2] format
                # Convert points_match to homogeneous coordinates [B, N, 3]
                points_match_homo = torch.cat(
                    [points_match, torch.ones_like(points_match[..., :1])], dim=-1
                )

                # Transpose for batch matrix multiplication [B, 3, N]
                points_match_homo = points_match_homo.transpose(1, 2)

                # Transform to camera coordinates
                points_match_cam = torch.bmm(invK_scaled, points_match_homo)

                # Get depth values at the matched points
                # First get the pixel coordinates as integers
                points_match_px = points_match.int()
                batch_idx = (
                    torch.arange(batch_size, device=points_match.device)
                    .view(-1, 1)
                    .expand(-1, points_match.size(1))
                )

                # Sample depth values at these coordinates
                points_match_depth = (
                    depth[
                        batch_idx.reshape(-1),
                        torch.zeros_like(batch_idx.reshape(-1)),  # channel index
                        points_match_px[..., 1].reshape(-1).clamp(0, current_height - 1),
                        points_match_px[..., 0].reshape(-1).clamp(0, current_width - 1),
                    ]
                    .reshape(batch_size, -1)
                    .unsqueeze(1)
                )

                # Scale by depth
                points_match_3d = points_match_cam * points_match_depth  # [B, 3, N]

                # Add homogeneous coordinate
                ones_match = torch.ones_like(points_match_depth)  # [B, 1, N]
                points_match_3d = torch.cat(
                    [points_match_3d, ones_match], dim=1
                )  # [B, 4, N]

                result["points_match_3d"] = points_match_3d

            elif len(points_match.shape) == 2:  # [BN, 2] format
                assert (
                    batch_idx_match is not None
                ), "batch_idx_match must be provided for [BN, 2] format"
                assert (
                    batch_idx_match.shape[0] == points_match.shape[0]
                ), "batch_idx_match must have same length as points_match"

                # Get batch indices (flattened)s
                batch_indices = batch_idx_match.squeeze(-1).int()

                # Convert to homogeneous coordinates [BN, 3]
                points_match_homo = torch.cat(
                    [points_match, torch.ones_like(points_match[:, :1])], dim=1
                )  # [BN, 3]

                # Get invK for each point based on batch_indices
                point_invK = invK_scaled[batch_indices]  # [BN, 3, 3]

                # Transform to camera coordinates using batched matrix multiplication
                points_match_cam = torch.bmm(
                    point_invK, points_match_homo.unsqueeze(-1)
                ).squeeze(
                    -1
                )  # [BN, 3]

                # Clamp coordinates to valid image range
                point_y = points_match[:, 1].long().clamp(0, current_height - 1)
                point_x = points_match[:, 0].long().clamp(0, current_width - 1)

                # Sample depth values at these coordinates
                points_match_depth = depth[
                    batch_indices, 0, point_y, point_x  # channel index
                ].unsqueeze(
                    -1
                )  # [BN, 1]

                # Scale by depth
                points_match_3d = points_match_cam * points_match_depth  # [BN, 3]

                # Add homogeneous coordinate
                ones_match = torch.ones_like(points_match_depth)  # [BN, 1]
                points_match_3d = torch.cat(
                    [points_match_3d, ones_match], dim=1
                )  # [BN, 4]

                result["points_match_3d"] = points_match_3d
                result["batch_idx_match"] = batch_idx_match

        return result


class Project(nn.Module):
    """Projects 3D point clouds to 2D images.
    
    Supports both RGB and feature cloud projection with automatic detection.
    """
    
    def __init__(self, height, width, patch_size=16):
        super().__init__()
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.feat_height = height // patch_size
        self.feat_width = width // patch_size

    def _scale_intrinsics_for_features(self, K: torch.Tensor) -> torch.Tensor:
        """Scale intrinsics matrix for feature resolution."""
        K_scaled = K.clone()
        scale_factor = 1.0 / self.patch_size
        K_scaled[:, 0, 0] *= scale_factor  # fx
        K_scaled[:, 1, 1] *= scale_factor  # fy
        K_scaled[:, 0, 2] *= scale_factor  # cx
        K_scaled[:, 1, 2] *= scale_factor  # cy
        return K_scaled

    def forward(
        self,
        cloud: torch.Tensor,
        data_vec: torch.Tensor,  # Can be rgb_vec or feature_vec
        K: torch.Tensor,
        T: torch.Tensor,
        points_match_3d: torch.Tensor = None,
        batch_idx_match: torch.Tensor = None,
        missing_value: float = 0,
        median_kernel_size: int = 5,
        return_artifacts: bool = False,
        return_mask: bool = False,
        infilling_steps: int = 10,
        splat_fraction: float = 0.0,  # Controls bilinear splatting (0.0 = no splatting)
    ) -> dict:
        """
        Project 3D points to 2D image space.

        Args:
            cloud: 3D point cloud [B, 4, N]
            data_vec: RGB values [B, 3, N] or features [B, E, N]
            K: Camera intrinsics [B, 3, 3] (for original resolution)
            T: Camera pose [B, 4, 4] or [B, 6] (Euler)
            points_match_3d: 3D points to track in either:
                - [B, 4, N] format (same number of points per batch)
                - [BN, 4] format (variable number of points per batch)
            batch_idx_match: If points_match_3d is [BN, 4], indicates batch indices [BN, 1]
            missing_value: Value to fill in missing pixels
            median_kernel_size: Size of kernel for median filtering
            return_artifacts: Whether to return intermediate artifacts
            return_mask: Whether to return visibility mask
            infilling_steps: Number of infilling iterations
            splat_fraction: Fraction for bilinear splatting (0.0 = no splatting, 1.0 = full splatting)

        Returns:
            dict: Dictionary containing warped image/features, points, and other outputs
        """
        B, data_dim, N = data_vec.shape
        device = cloud.device

        # Detect if we're working with features or RGB
        is_feature_data = data_dim != 3
        
        if is_feature_data:
            # Use feature map resolution and scaled intrinsics
            target_height, target_width = self.feat_height, self.feat_width
            K_scaled = self._scale_intrinsics_for_features(K)
        else:
            # Use original RGB resolution
            target_height, target_width = self.height, self.width
            K_scaled = K

        if T.shape[1] == 6:
            T = geometry.euler2mat(T)
        T = torch.inverse(T)
        
        # Project point cloud to camera space
        cloud_cam = torch.bmm(T, cloud)  # B x 4 x N
        proj = torch.bmm(K_scaled, cloud_cam[:, :3, :])  # B x 3 x N
        uv = proj[:, :2, :] / proj[:, 2:3, :]  # B x 2 x N
        depth = cloud_cam[:, 2, :]  # B x N

        # Calculate fractional coordinates for bilinear splatting
        u_frac = (uv[:, 0, :] - uv[:, 0, :].int().float()) * splat_fraction  # B x N
        v_frac = (uv[:, 1, :] - uv[:, 1, :].int().float()) * splat_fraction  # B x N
        
        # Clamp projected coordinates to image boundaries
        v = uv[:, 1, :].int().clamp(0, target_height - 1)
        u = uv[:, 0, :].int().clamp(0, target_width - 1)

        # Compute linear indices for scatter operations
        batch_offset = (torch.arange(B, device=device) * target_height * target_width).view(B, 1)
        linear_idx = batch_offset + v * target_width + u  # B x N

        # Flatten for scatter operations
        flat_linear_idx = linear_idx.reshape(-1).long()  # (B*N,)
        flat_depth = depth.reshape(-1).long()  # (B*N,)
        flat_data = data_vec.permute(0, 2, 1).reshape(-1, data_dim)  # (B*N, data_dim)
        flat_u_frac = u_frac.reshape(-1)  # (B*N,)
        flat_v_frac = v_frac.reshape(-1)  # (B*N,)

        # Depth buffer initialization for scatter_reduce
        depth_buffer = torch.full(
            (B * target_height * target_width,), float("inf"), device=device
        ).long()
        # Use scatter_reduce to find the minimum depth per pixel
        depth_buffer = depth_buffer.scatter_reduce(
            0, flat_linear_idx, flat_depth, reduce="amin", include_self=True
        )

        gathered_depth = depth_buffer[flat_linear_idx]
        # Mask for selecting the closest point per pixel
        mask = torch.isclose(flat_depth, gathered_depth, atol=1e-6)

        # Filter data values using the mask
        flat_data_filtered = torch.zeros_like(flat_data)
        flat_data_filtered[mask] = flat_data[mask]

        # Create output image/feature map with bilinear splatting
        image_flat = -0.001 * torch.ones(
            B * target_height * target_width, data_dim, device=device, dtype=flat_data.dtype
        )
        
        if splat_fraction > 0.0:
            # Bilinear splatting: distribute each point to 4 nearest pixels
            # Get integer coordinates for the 4 neighbors
            u_int = uv[:, 0, :].int().reshape(-1)  # (B*N,)
            v_int = uv[:, 1, :].int().reshape(-1)  # (B*N,)
            
            # Calculate bilinear weights for the 4 neighbors
            # Each point contributes to 4 neighbors with distance-based weights
            weights = torch.stack([
                (1 - flat_u_frac) * (1 - flat_v_frac),  # top-left
                flat_u_frac * (1 - flat_v_frac),         # top-right  
                (1 - flat_u_frac) * flat_v_frac,         # bottom-left
                flat_u_frac * flat_v_frac                 # bottom-right
            ], dim=1)  # (B*N, 4)
            
            # Calculate coordinates for the 4 neighbors
            neighbor_coords = torch.stack([
                torch.stack([u_int, v_int], dim=1),                    # top-left
                torch.stack([u_int + 1, v_int], dim=1),               # top-right
                torch.stack([u_int, v_int + 1], dim=1),               # bottom-left
                torch.stack([u_int + 1, v_int + 1], dim=1)            # bottom-right
            ], dim=1)  # (B*N, 4, 2)
            
            # Clamp coordinates to image boundaries
            neighbor_coords = neighbor_coords.clamp(torch.tensor([0, 0], device=device), torch.tensor([target_width - 1, target_height - 1], device=device))
            
            # Calculate linear indices for all 4 neighbors
            batch_indices = torch.arange(B, device=device).repeat_interleave(N)  # (B*N,)
            batch_offset = batch_indices * target_height * target_width
            neighbor_linear_idx = batch_offset.unsqueeze(1) + neighbor_coords[:, :, 1] * target_width + neighbor_coords[:, :, 0]  # (B*N, 4)
            
            # Apply mask to only process valid points
            valid_mask = mask.unsqueeze(1).expand(-1, 4)  # (B*N, 4)
            
            # Flatten for scatter operations
            flat_neighbor_idx = neighbor_linear_idx[valid_mask]  # (valid_count * 4,)
            flat_weights = weights[valid_mask]  # (valid_count * 4,)
            
            # Get valid data and expand to 4 neighbors
            valid_data = flat_data_filtered[mask]  # (valid_count, data_dim)
            flat_data_expanded = valid_data.unsqueeze(1).expand(-1, 4, -1).reshape(-1, data_dim)  # (valid_count * 4, data_dim)
            
            # Weight the data by bilinear weights
            weighted_data = flat_data_expanded * flat_weights.unsqueeze(1)  # (valid_count * 4, data_dim)
            
            # Use scatter_add to accumulate weighted contributions
            # We need to handle each channel separately since scatter_add expects matching dimensions
            for i in range(data_dim):
                image_flat[:, i] = image_flat[:, i].scatter_add(0, flat_neighbor_idx, weighted_data[:, i])
        else:
            # No splatting: only use center pixel
            image_flat = image_flat.index_copy(
                0, flat_linear_idx[mask], flat_data_filtered[mask]
            )
        
        image = image_flat.view(B, target_height, target_width, data_dim).permute(0, 3, 1, 2)

        # Classification mask initialization:
        # Start with all pixels as holes (0)
        classification_mask = torch.zeros(
            B, target_height, target_width, device=device, dtype=torch.uint8
        )

        # Count the number of points projected to each pixel to identify occlusions
        count_buffer = torch.zeros(
            B * target_height * target_width, device=device, dtype=torch.int32
        )
        ones = torch.ones_like(flat_depth, dtype=torch.int32)
        count_buffer = count_buffer.scatter_reduce(
            0, flat_linear_idx, ones, reduce="sum"
        )
        count_buffer = count_buffer.view(B, target_height, target_width)

        # Populate the classification mask
        classification_mask[count_buffer == 1] = 1  # Valid pixels
        classification_mask[count_buffer > 1] = 2  # Overlapping pixels (occlusions)

        # Expand to match data channels for compatibility with output shape
        classification_mask = classification_mask.unsqueeze(1).expand(-1, data_dim, -1, -1)

        # Save warped image before hole filling and median filtering if artifacts are requested
        warped_image = image.clone() if return_artifacts else None

        # Inpainting with median filtering integrated
        base_mask = (image[:, :1, :, :] > missing_value).float()  # shape: (B, 1, H, W)
        mask_full = base_mask.expand_as(image)  # shape: (B, C, H, W)

        img_for_interp = image.clone()
        img_for_interp[mask_full == 0] = 0.0

        _, C, H, W = image.shape
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

        inpainted_img = image.clone()
        count_buffer = count_buffer.unsqueeze(1).expand(-1, data_dim, -1, -1)

        pad = median_kernel_size // 2
        for _ in range(infilling_steps):
            padded = F.pad(inpainted_img, (pad, pad, pad, pad), mode="reflect")
            patches = padded.unfold(2, median_kernel_size, 1).unfold(
                3, median_kernel_size, 1
            )
            patches = patches.contiguous().view(B, C, H, W, -1)
            median_img, _ = patches.median(dim=-1)
            final_img = inpainted_img.clone()
            final_img[mask_full == 0] = median_img[mask_full == 0]
            inpainted_img = final_img

        # Handle points_match_3d projection if provided
        uv_match = None
        if points_match_3d is not None:
            if len(points_match_3d.shape) == 3:  # [B, 4, N] format (batched)
                _, _, Np = points_match_3d.shape

                # Project points to camera space and then to image
                points_cam = torch.bmm(T, points_match_3d)  # [B, 4, N]
                proj_match = torch.bmm(K_scaled, points_cam[:, :3, :])  # [B, 3, N]
                uv_match = proj_match[:, :2, :] / proj_match[:, 2:3, :]  # [B, 2, N]
                uv_match = uv_match.permute(0, 2, 1)  # [B, N, 2]

            elif len(points_match_3d.shape) == 2:  # [BN, 4] format (non-batched)
                assert (
                    batch_idx_match is not None
                ), "batch_idx_match must be provided for [BN, 4] format"

                # Get batch indices
                batch_indices = batch_idx_match.squeeze(-1).int()

                # Get T and K for each point based on batch indices
                point_T = T[batch_indices]  # [BN, 4, 4]
                point_K = K_scaled[batch_indices]  # [BN, 3, 3]

                # Project each point to camera space
                points_cam = torch.bmm(point_T, points_match_3d.unsqueeze(-1)).squeeze(
                    -1
                )  # [BN, 4]

                # Project to image plane
                proj_match = torch.bmm(
                    point_K, points_cam[:, :3].unsqueeze(-1)
                ).squeeze(
                    -1
                )  # [BN, 3]

                # Calculate UV coordinates
                z = proj_match[:, 2].clamp(min=1e-10)
                uv_match = proj_match[:, :2] / z.unsqueeze(-1)  # [BN, 2]

        # Prepare the dictionary to return
        output = {}
        expandedmask = (
            (inpainted_img > 0).any(dim=1, keepdim=True).expand(-1, data_dim, -1, -1).int()
        )

        mask = expandedmask.bool()
        single_channel = mask[:, 0, :, :].float()
        kernel = torch.ones((1, 1, 3, 3), device=mask.device)
        neighbor_count = F.conv2d(single_channel.unsqueeze(1), kernel, padding=1)
        updated_channel = (neighbor_count >= 6).squeeze(1)
        holemask = updated_channel.unsqueeze(1).repeat(1, data_dim, 1, 1).int()

        if return_mask:
            output["mask"] = holemask
        else:
            output["mask"] = None

        output["warped"] = inpainted_img * holemask
        # Raw warped image before hole filling and median filtering if artifacts requested
        output["raw"] = warped_image if return_artifacts else None
        output["matches"] = uv_match

        # Include batch_idx_match in output if it was provided
        if batch_idx_match is not None and points_match_3d is not None:
            output["batch_idx_match"] = batch_idx_match

        return output


class Warp(nn.Module):
    """
    Warp module that combines back-projection and forward-projection operations.
    Transforms a source image/feature map to a target viewpoint based on depth and camera pose.
    Supports both RGB images and feature maps with automatic detection.
    """

    def __init__(self, height, width, patch_size=16):
        super().__init__()
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.backproject = BackProject(height, width, patch_size)
        self.forward_project = Project(height, width, patch_size)

    def forward(
        self,
        source_image: torch.Tensor,
        depth_map: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_pose: torch.Tensor,
        return_mask: bool = False,
        return_artifacts: bool = False,
        points_to_match: torch.Tensor = None,
        batch_idx_match: torch.Tensor = None,
        median_kernel_size: int = 5,
        infilling_steps: int = 10,
        splat_fraction: float = 0.0,
    ) -> dict:
        """
        Warp a source image/feature map to a target viewpoint based on depth and camera pose.

        Args:
            source_image (torch.Tensor): Source image [B, 3, H, W] or feature map [B, E, Hf, Wf]
            depth_map (torch.Tensor): Depth map [B, 1, H, W] or [B, 1, Hf, Wf]. If source_image is 
                                    a feature map and depth_map is at original resolution, it will be 
                                    automatically downsampled to match.
            camera_intrinsics (torch.Tensor): Camera intrinsics matrix [B, 3, 3] (for original resolution)
            camera_pose (torch.Tensor): Camera pose / transformation [B, 4, 4] or [B, 6] (Euler)
            return_mask (bool): Whether to return visibility mask
            return_artifacts (bool): Whether to return intermediate artifacts
            points_to_match (torch.Tensor, optional): Source points to track in either:
                - [B, N, 2] format (same number of points per batch)
                - [BN, 2] format (variable number of points per batch)
            batch_idx_match (torch.Tensor, optional): If points_to_match is [BN, 2], this tensor [BN, 1]
                indicates which batch each point belongs to
            median_kernel_size (int): Size of kernel for median filtering in forward projection

        Returns:
            dict: Dictionary containing warped image/features, points, and other outputs
        """
        # Validate input format
        batch_size = source_image.shape[0]

        if points_to_match is not None:
            if len(points_to_match.shape) == 2:  # [BN, 2] format
                assert (
                    points_to_match.shape[1] == 2
                ), "Last dimension of points_to_match must be 2"
                assert (
                    batch_idx_match is not None
                ), "batch_idx_match must be provided when points_to_match has shape [BN, 2]"
                assert (
                    batch_idx_match.shape[0] == points_to_match.shape[0]
                ), "batch_idx_match and points_to_match must have the same first dimension"
                assert (
                    len(batch_idx_match.shape) == 1
                ), "batch_idx_match must have shape [BN,]"
                assert torch.all(batch_idx_match >= 0) and torch.all(
                    batch_idx_match < batch_size
                ), f"batch_idx_match values must be in range [0, {batch_size-1}]"
            elif len(points_to_match.shape) == 3:  # [B, N, 2] format
                assert (
                    points_to_match.shape[0] == batch_size
                ), "Batch size of points_to_match must match source_image"
                assert (
                    points_to_match.shape[2] == 2
                ), "Last dimension of points_to_match must be 2"
                if batch_idx_match is not None:
                    print(
                        "Warning: batch_idx_match is ignored when points_to_match has shape [B, N, 2]"
                    )
                    batch_idx_match = None
            else:
                raise ValueError(
                    f"Invalid shape for points_to_match: {points_to_match.shape}"
                )

        # Back-project source image/features and/or points to 3D
        backproj_output = self.backproject(
            source_image,
            depth_map,
            torch.inverse(camera_intrinsics),
            points_match=points_to_match,
            batch_idx_match=batch_idx_match,
        )

        # Extract 3D points and data values (RGB or features)
        cloud = backproj_output["xyz1"]
        
        # Get the appropriate data vector - RGB for images, features for feature maps
        if "rgb" in backproj_output:
            data_vec = backproj_output["rgb"]
        elif "features" in backproj_output:
            data_vec = backproj_output["features"]
        else:
            raise ValueError("BackProject output must contain either 'rgb' or 'features' key")

        # Forward-project to create warped image/features and/or track points
        warp_output = self.forward_project(
            cloud,
            data_vec,
            camera_intrinsics,
            camera_pose,
            points_match_3d=backproj_output.get("points_match_3d", None),
            batch_idx_match=backproj_output.get("batch_idx_match", None),
            return_mask=return_mask,
            return_artifacts=return_artifacts,
            median_kernel_size=median_kernel_size,
            infilling_steps=infilling_steps,
            splat_fraction=splat_fraction,
        )
        
        # Merge outputs - warp_output takes precedence for overlapping keys
        warp_output.update(backproj_output)
        return warp_output
    
class Raycast(nn.Module):
    """
    Raycasts a point cloud to produce a synthetic depth map.

    Args:
        height (int): Image height in pixels
        width (int): Image width in pixels
        n_samples (int): Number of samples along each ray
    """

    def __init__(self, height: int, width: int, n_samples: int = 500):
        super().__init__()
        self.height = height
        self.width = width
        self.n_samples = n_samples

    def forward(
        self,
        point_cloud: torch.Tensor,
        cam_intr: torch.Tensor,
        cam_pose: torch.Tensor,
        near: float = 10,
        far: float = 1000,
        inpaint: bool = True,
        median_kernel_size: int = 5,
        num_iterations: int = 10,
        return_mask: bool = False,
    ) -> torch.Tensor:

        """
        Raycasts the point cloud to a depth map.

        Args:
            point_cloud: [B,N,4] or [N,4] Point cloud in world coordinates (homogeneous)
            cam_intr: [B,3,3] or [3,3] Camera intrinsic matrix  
            cam_pose: [B,4,4] or [4,4] Camera-to-world transform or [B,6] or [6] euler angles + translation
            near: Near clipping distance
            far: Far clipping distance

        Returns:
            depth_map: [B,H,W] or [H,W] Depth values per pixel (-1 = no hit)
        """
        device = point_cloud.device
        H, W = self.height, self.width

        # Add batch dimension if inputs are unbatched
        batched = len(point_cloud.shape) == 3
        if not batched:
            point_cloud = point_cloud.unsqueeze(0)  # [1,N,4]
            cam_intr = cam_intr.unsqueeze(0)       # [1,3,3]
            cam_pose = cam_pose.unsqueeze(0)       # [1,4,4] or [1,6]

        B, N = point_cloud.shape[:2]

        # Handle euler angles input
        if cam_pose.shape[-1] == 6:
            cam2world = geometry.euler2mat(cam_pose)  # [B,4,4]
        else:
            cam2world = cam_pose
        # Get world-to-camera transform
        world2cam = torch.inverse(cam2world)  # [B,4,4]

        # Transform point cloud to camera space (already in homogeneous coordinates)
        pc_cam_h = torch.bmm(world2cam, point_cloud.transpose(1,2))  # [B,4,N]
        pc_cam_h = pc_cam_h.transpose(1,2)  # [B,N,4]
        pc_cam = pc_cam_h[:,:,:3] / pc_cam_h[:,:,3:4]  # [B,N,3] Dehomogenize

        # Filter out points behind the camera and outside near/far planes
        valid = (pc_cam[:,:,2] > near) & (pc_cam[:,:,2] < far)  # [B,N]
        
        # Early exit if no valid points
        if not valid.any():
            depth_maps = torch.full((B, H, W), -1.0, device=device)
            return depth_maps.squeeze(0) if not batched else depth_maps
        
        # Flatten for vectorized processing
        pc_cam_flat = pc_cam.view(-1, 3)  # [B*N, 3]
        valid_flat = valid.view(-1)  # [B*N]
        
        # Create batch indices for each point
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N).reshape(-1)  # [B*N]
        
        # Apply validity mask
        pc_cam_valid = pc_cam_flat[valid_flat]  # [Valid, 3]
        batch_indices_valid = batch_indices[valid_flat]  # [Valid]
        
        # Vectorized projection using gather for intrinsics
        cam_intr_valid = cam_intr[batch_indices_valid]  # [Valid, 3, 3]
        
        # More efficient projection: use einsum for better performance
        # Project to image plane: K * [x, y, z]^T
        uv_h = torch.einsum('bij,bj->bi', cam_intr_valid, pc_cam_valid)  # [Valid, 3]
        uv = uv_h[:,:2] / pc_cam_valid[:,2:].clamp(min=1e-6)  # [Valid, 2]
        
        # Round and clamp pixel coordinates
        u = uv[:,0].round().long()
        v = uv[:,1].round().long()
        z = pc_cam_valid[:,2]
        
        # Filter points within image bounds
        mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u_valid = u[mask]
        v_valid = v[mask]
        z_valid = z[mask]
        batch_valid = batch_indices_valid[mask]
        
        # Initialize depth maps with -1
        depth_maps = torch.full((B, H, W), -1.0, device=device)
        
        if u_valid.numel() == 0:
            return depth_maps.squeeze(0) if not batched else depth_maps
        
        # Optimized z-buffering using advanced indexing
        # Create global pixel indices accounting for batch dimension
        global_pixel_indices = batch_valid * H * W + v_valid * W + u_valid  # [Valid]
        
        # Use scatter_reduce with amin for z-buffering (in-place for efficiency)
        depth_flat = torch.full((B * H * W,), float('inf'), device=device)
        depth_flat.scatter_reduce_(0, global_pixel_indices, z_valid, reduce='amin', include_self=False)
        
        # Reshape and replace inf with -1
        depth_maps = depth_flat.view(B, H, W)
        depth_maps[depth_maps == float('inf')] = -1.0
        
        # Remove batch dimension if input was unbatched
        if not batched:
            depth_maps = depth_maps.squeeze(0)
        
        if inpaint and depth_maps.numel() > 0:
            # Mask for holes (missing values are -1.0)
            missing_mask = (depth_maps == -1.0).float()  # [B,H,W]
            depth_maps_filled = depth_maps.clone()

            pad = median_kernel_size // 2
            for _ in range(num_iterations):  # Typical: 5–10 iterations
                padded = F.pad(depth_maps_filled.unsqueeze(1), (pad, pad, pad, pad), mode="reflect")  # [B,1,H+2p,W+2p]
                patches = padded.unfold(2, median_kernel_size, 1).unfold(3, median_kernel_size, 1)  # [B,1,H,W,k,k]
                patches = patches.contiguous().view(B, 1, H, W, -1)
                median = patches.median(dim=-1).values  # [B,1,H,W]

                # Only update missing values
                depth_maps_filled[missing_mask.bool()] = median.squeeze(1)[missing_mask.bool()]

            depth_maps = depth_maps_filled
            
        if return_mask:
            return depth_maps, missing_mask 
        else:
            return depth_maps
    
class DepthWarp(nn.Module):
    """
    Warps depth maps from identity pose (source) to target camera viewpoints by 
    converting to 3D points and raycasting from the target viewpoint.
    
    This class combines BackProject and Raycast to transform depth maps without 
    requiring RGB/feature data. Assumes source camera is at identity pose 
    (world coordinates) and uses the same intrinsics for source and target.
    
    Args:
        height (int): Original image height
        width (int): Original image width
        patch_size (int): Patch size (default: 16)
        n_samples (int): Number of samples for raycast (default: 500)
    """
    
    def __init__(self, height: int, width: int, patch_size: int = 16, n_samples: int = 500):
        super().__init__()
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.backproject = BackProject(height, width, patch_size)
        self.raycast = Raycast(height, width, n_samples)
    
    def forward(
        self,
        source_depth: torch.Tensor,  # [B, 1, H, W] or [B, H, W]
        intrinsics: torch.Tensor,  # [B, 3, 3] - camera intrinsics (same for source and target)
        target_pose: torch.Tensor,  # [B, 4, 4] or [B, 6] - target camera pose (cam2world)
        near: float = 10,
        far: float = 1000,
        inpaint: bool = True,
        median_kernel_size: int = 5,
        num_iterations: int = 10,
        return_mask: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Warp a depth map from source viewpoint (identity pose) to target viewpoint.
        
        Args:
            source_depth: Source depth map [B, 1, H, W] or [B, H, W]
            intrinsics: Camera intrinsics [B, 3, 3] (same for source and target)
            target_pose: Target camera pose [B, 4, 4] or [B, 6] (camera-to-world)
            near: Near clipping distance for raycast
            far: Far clipping distance for raycast
            inpaint: Whether to inpaint holes in the result
            median_kernel_size: Kernel size for median filtering during inpainting
            num_iterations: Number of inpainting iterations
            return_mask: Whether to return the hole mask
            
        Returns:
            If return_mask=False:
                warped_depth: Warped depth map [B, H, W]
            If return_mask=True:
                tuple: (warped_depth, hole_mask) where hole_mask indicates missing values [B, H, W]
        """
        device = source_depth.device
        batch_size = source_depth.shape[0]
        
        # Ensure depth has channel dimension [B, 1, H, W]
        if len(source_depth.shape) == 3:
            source_depth = source_depth.unsqueeze(1)
        
        # Create dummy RGB data since BackProject requires it but Raycast doesn't use it
        # We'll use zeros as placeholder since we only care about the 3D point cloud
        dummy_rgb = torch.zeros(batch_size, 3, self.height, self.width, 
                            device=device, dtype=source_depth.dtype)
        
        # Back-project source depth to 3D point cloud in source camera coordinates
        # Since source pose is identity, these are also world coordinates
        backproj_output = self.backproject(
            dummy_rgb,
            source_depth,
            torch.inverse(intrinsics)
        )
        
        # Get 3D points in world coordinates [B, 4, N]
        # (since source pose is identity, camera coords = world coords)
        points_world = backproj_output["xyz1"]
        
        # Transpose to [B, N, 4] format expected by Raycast
        points_world = points_world.transpose(1, 2)  # [B, N, 4]
        
        # Raycast from target viewpoint
        result = self.raycast(
            points_world,
            intrinsics,
            target_pose,
            near=near,
            far=far,
            inpaint=inpaint,
            median_kernel_size=median_kernel_size,
            num_iterations=num_iterations,
            return_mask=return_mask
        )
        
        return result    

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Union, Literal


class HighLightRenderer(nn.Module):
    """
    Renders reflection artifacts on images based on light sources and surface geometry.
    Simulates specular reflections using the law of reflection and Phong lighting model.
    """
    
    def __init__(self, height: int, width: int, patch_size: int = 16):
        super().__init__()
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.feat_height = height // patch_size
        self.feat_width = width // patch_size
        self.project = Project(height, width, patch_size)
        
        # Precompute neighborhood offsets for normal estimation
        self.register_buffer('_neighborhood_offsets', self._create_neighborhood_offsets())
        
    def _create_neighborhood_offsets(self, neighborhood_size: int = 3) -> torch.Tensor:
        """Create neighborhood offsets for normal estimation."""
        half_size = neighborhood_size // 2
        offsets = torch.meshgrid(
            torch.arange(-half_size, half_size + 1),
            torch.arange(-half_size, half_size + 1),
            indexing='ij'
        )
        # [2, neighborhood_size, neighborhood_size] -> [neighborhood_size^2, 2]
        offsets = torch.stack(offsets, dim=0).reshape(2, -1).T
        # Remove center point (0, 0)
        center_idx = len(offsets) // 2
        offsets = torch.cat([offsets[:center_idx], offsets[center_idx+1:]])
        return offsets  # [N_neighbors, 2]
    
    import torch
    import torch.nn.functional as F

    def estimate_surface_normals(self, cloud_xyz: torch.Tensor, neighborhood_size: int = 3) -> torch.Tensor:
        """
        Estimate surface normals using robust PCA-based method with local neighborhoods.
    
        Args:
            cloud_xyz: 3D points [B, 3, N]
            neighborhood_size: Size of neighborhood for normal estimation (should be odd)
        
        Returns:
            normals: Surface normals [B, 3, N]
        """
        B, _, N = cloud_xyz.shape  # [B, 3, N]
        device = cloud_xyz.device
    
        # Determine spatial resolution
        is_feature_res = N == (self.feat_height * self.feat_width)
        if is_feature_res:
            h, w = self.feat_height, self.feat_width
        else:
            h, w = self.height, self.width
        
        # Reshape to spatial format for neighborhood processing
        cloud_spatial = cloud_xyz.view(B, 3, h, w)  # [B, 3, H, W]
    
        # Use conv2d with unfold for efficient neighborhood extraction
        kernel_size = neighborhood_size
        pad_size = kernel_size // 2
    
        # Pad for boundary handling
        cloud_padded = F.pad(cloud_spatial, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
    
        # Extract neighborhoods using unfold (more efficient than advanced indexing)
        # unfold: [B, 3, H, W] -> [B, 3*k*k, H*W] where k=kernel_size
        neighborhoods = F.unfold(
            cloud_padded, 
            kernel_size=kernel_size, 
            stride=1
        )  # [B, 3*k*k, H*W]
    
        # Reshape to separate spatial dimensions and neighborhood
        neighborhoods = neighborhoods.view(B, 3, kernel_size*kernel_size, h*w)  # [B, 3, k*k, H*W]
        neighborhoods = neighborhoods.permute(0, 3, 1, 2)  # [B, H*W, 3, k*k]
    
        # Center point is the middle of the neighborhood
        center_idx = kernel_size*kernel_size // 2
        center_points = neighborhoods[:, :, :, center_idx:center_idx+1]  # [B, H*W, 3, 1]
    
        # Compute relative positions
        relative_pos = neighborhoods - center_points  # [B, H*W, 3, k*k]
    
        # Remove the center point from the neighborhood (it's all zeros anyway)
        mask = torch.ones(kernel_size*kernel_size, dtype=torch.bool, device=device)
        mask[center_idx] = False
        relative_pos = relative_pos[:, :, :, mask]  # [B, H*W, 3, k*k-1]
    
        # Compute covariance matrices: [B, H*W, 3, 3]
        cov_matrices = torch.matmul(relative_pos, relative_pos.transpose(-2, -1))
        cov_matrices = cov_matrices / (relative_pos.shape[-1] - 1)
    
        # Add regularization for numerical stability
        eps = 1e-6
        eye = torch.eye(3, device=device, dtype=cov_matrices.dtype)
        cov_matrices = cov_matrices + eps * eye.unsqueeze(0).unsqueeze(0)
    
        # Compute eigendecomposition - normal is eigenvector with smallest eigenvalue
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrices)  # [B, H*W, 3], [B, H*W, 3, 3]
            # Smallest eigenvalue corresponds to normal direction (index 0)
            normals = eigenvectors[:, :, :, 0]  # [B, H*W, 3]
        
            # Check for degenerate cases (very small eigenvalues)
            min_eigenvals = eigenvalues[:, :, 0]  # [B, H*W]
            degenerate_mask = min_eigenvals < eps * 10
        
            if degenerate_mask.any():
                # Fallback to gradient-based normals for degenerate cases
                grad_normals = self._compute_gradient_normals(cloud_spatial)  # [B, 3, H*W]
                grad_normals = grad_normals.permute(0, 2, 1)  # [B, H*W, 3]
                normals = torch.where(degenerate_mask.unsqueeze(-1), grad_normals, normals)
            
        except Exception as e:
            # Complete fallback to gradient-based method
            print(f"Eigendecomposition failed: {e}. Using gradient-based normals.")
            normals = self._compute_gradient_normals(cloud_spatial)  # [B, 3, H*W]
            normals = normals.permute(0, 2, 1)  # [B, H*W, 3]
    
        # Ensure consistent orientation (point towards camera)
        # Assume camera is at origin looking down -Z axis
        camera_dir = torch.tensor([0., 0., -1.], device=device, dtype=normals.dtype)
        dot_product = torch.sum(normals * camera_dir.view(1, 1, 3), dim=-1, keepdim=True)  # [B, H*W, 1]
        normals = torch.where(dot_product < 0, -normals, normals)
    
        # Normalize
        normals = F.normalize(normals, dim=-1, eps=1e-8)  # [B, H*W, 3]
    
        # Reshape back to point cloud format
        normals = normals.permute(0, 2, 1)  # [B, 3, H*W] -> [B, 3, N]
    
        return normals

    def _compute_gradient_normals(self, cloud_spatial: torch.Tensor) -> torch.Tensor:
        """
        Fallback method: compute normals using spatial gradients.
    
        Args:
            cloud_spatial: Points in spatial format [B, 3, H, W]
        
        Returns:
            normals: Normal vectors [B, 3, H*W]
        """
        B, _, H, W = cloud_spatial.shape
    
        # Compute gradients using Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=cloud_spatial.dtype, device=cloud_spatial.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=cloud_spatial.dtype, device=cloud_spatial.device).view(1, 1, 3, 3)
    
        # Apply to each coordinate
        grad_x = F.conv2d(cloud_spatial.view(B*3, 1, H, W), sobel_x, padding=1).view(B, 3, H, W)
        grad_y = F.conv2d(cloud_spatial.view(B*3, 1, H, W), sobel_y, padding=1).view(B, 3, H, W)
    
        # Cross product to get normals
        normals = torch.cross(grad_x.view(B, 3, -1), grad_y.view(B, 3, -1), dim=1)  # [B, 3, H*W]
    
        # Normalize
        normals = F.normalize(normals, dim=1, eps=1e-8)
    
        return normals
    
    def _gradient_based_normals(self, cloud_spatial: torch.Tensor) -> torch.Tensor:
        """Fallback gradient-based normal estimation."""
        # Calculate gradients using Sobel filters for better stability
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                              device=cloud_spatial.device, dtype=cloud_spatial.dtype) / 8.0
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                              device=cloud_spatial.device, dtype=cloud_spatial.dtype) / 8.0
        
        B, _, H, W = cloud_spatial.shape
        grad_x = torch.zeros_like(cloud_spatial)  # [B, 3, H, W]
        grad_y = torch.zeros_like(cloud_spatial)  # [B, 3, H, W]
        
        # Apply Sobel filters to each coordinate
        for i in range(3):
            grad_x[:, i:i+1] = F.conv2d(cloud_spatial[:, i:i+1], sobel_x, padding=1)
            grad_y[:, i:i+1] = F.conv2d(cloud_spatial[:, i:i+1], sobel_y, padding=1)
        
        # Cross product to get normals
        normals = torch.cross(grad_x, grad_y, dim=1)  # [B, 3, H, W]
        
        # Normalize
        normals = F.normalize(normals, dim=1, eps=1e-8)
        
        return normals.view(B, 3, -1)  # [B, 3, H*W]
    
    def _sample_planar_light(self, 
                                light_pose: torch.Tensor,     # [B, 4, 4]
                                light_width: float, 
                                light_height: float, 
                                num_samples: int,
                                projected_image: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample points on a rectangular planar light with optional image projection.
        
        Args:
            light_pose: Light pose matrix [B, 4, 4] (world to light transform)
            light_width: Width of light plane (in light's local X direction)
            light_height: Height of light plane (in light's local Y direction)
            num_samples: Number of sample points (will be made square: sqrt(num_samples)^2)
            projected_image: Optional RGB image to project [B, 3, H_img, W_img]
            
        Returns:
            sample_points: World coordinates of sample points [B, 3, num_samples_actual]
            sample_colors: Color at each sample point [B, 3, num_samples_actual]
            num_samples_actual: Actual number of samples used
        """
        B = light_pose.shape[0]
        device = light_pose.device
        dtype = light_pose.dtype
        
        # Make num_samples a perfect square for uniform grid
        samples_per_side = int(torch.sqrt(torch.tensor(num_samples, dtype=torch.float32)).item())
        actual_num_samples = samples_per_side ** 2
        
        # Create uniform grid in light's local coordinate system
        # Light plane spans [-width/2, width/2] x [-height/2, height/2] x [0]
        u = torch.linspace(-light_width/2, light_width/2, samples_per_side, device=device, dtype=dtype)
        v = torch.linspace(-light_height/2, light_height/2, samples_per_side, device=device, dtype=dtype)
        
        # Create grid: [samples_per_side, samples_per_side]
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        
        # Flatten to get sample points in light's local coordinates
        local_x = uu.flatten()  # [actual_num_samples]
        local_y = vv.flatten()  # [actual_num_samples]
        local_z = torch.zeros_like(local_x)  # Light plane at z=0 in local coords
        
        # Homogeneous coordinates in light space: [4, actual_num_samples]
        local_points = torch.stack([local_x, local_y, local_z, torch.ones_like(local_x)], dim=0)
        
        # Transform to world coordinates
        # light_pose is world-to-light, so we need light-to-world (inverse)
        light_to_world = torch.linalg.inv(light_pose)  # [B, 4, 4]
        
        # Broadcast and transform: [B, 4, 4] @ [4, actual_num_samples] -> [B, 4, actual_num_samples]
        world_points = torch.matmul(light_to_world, local_points.unsqueeze(0).expand(B, -1, -1))
        
        # Get colors for each sample point
        if projected_image is not None:
            # Sample image colors based on local coordinates
            sample_colors = self._sample_projected_image(
                local_x, local_y, light_width, light_height, projected_image
            )  # [B, 3, actual_num_samples]
        else:
            # Default to white light
            sample_colors = torch.ones(B, 3, actual_num_samples, device=device, dtype=dtype)
        
        # Return 3D coordinates: [B, 3, actual_num_samples], colors: [B, 3, actual_num_samples]
        return world_points[:, :3, :], sample_colors, actual_num_samples
    
    def _sample_projected_image(self, 
                               local_x: torch.Tensor,      # [num_samples]
                               local_y: torch.Tensor,      # [num_samples]
                               light_width: float,
                               light_height: float, 
                               projected_image: torch.Tensor) -> torch.Tensor:  # [B, 3, H, W]
        """
        Sample colors from projected image based on local light coordinates.
        
        Args:
            local_x, local_y: Local coordinates on light plane [num_samples]
            light_width, light_height: Physical dimensions of light plane
            projected_image: Image to project [B, 3, H_img, W_img]
            
        Returns:
            sampled_colors: Colors at sample points [B, 3, num_samples]
        """
        B, _, H_img, W_img = projected_image.shape
        num_samples = local_x.shape[0]
        device = projected_image.device
        
        # Convert local coordinates to UV coordinates [0, 1]
        # local_x spans [-light_width/2, light_width/2] -> map to [0, 1]
        # local_y spans [-light_height/2, light_height/2] -> map to [0, 1]
        u = (local_x + light_width/2) / light_width    # [num_samples]
        v = (local_y + light_height/2) / light_height  # [num_samples]
        
        # Clamp to valid range [0, 1]
        u = torch.clamp(u, 0.0, 1.0)
        v = torch.clamp(v, 0.0, 1.0)
        
        # Convert UV to image pixel coordinates
        # Note: We flip V coordinate because image (0,0) is top-left but light plane (0,0) is center
        u_img = u * (W_img - 1)  # [num_samples]
        v_img = (1.0 - v) * (H_img - 1)  # [num_samples] - flip V axis
        
        # Create grid for F.grid_sample (expects [-1, 1] range)
        grid_x = 2.0 * u - 1.0  # Convert [0, 1] -> [-1, 1]
        grid_y = 2.0 * v - 1.0  # Convert [0, 1] -> [-1, 1]
        
        # Create sampling grid: [B, 1, num_samples, 2]
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [num_samples, 2]
        grid = grid.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)  # [B, 1, num_samples, 2]
        
        # Sample from image using bilinear interpolation
        # F.grid_sample expects [B, C, H, W] and [B, H_out, W_out, 2]
        sampled = F.grid_sample(
            projected_image,  # [B, 3, H_img, W_img]
            grid,            # [B, 1, num_samples, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # [B, 3, 1, num_samples]
        
        # Reshape to [B, 3, num_samples]
        sampled_colors = sampled.squeeze(2)  # [B, 3, num_samples]
        
        return sampled_colors
    
    def calculate_reflection_intensity(self, 
                                     cloud_xyz: torch.Tensor,  # [B, 3, N]
                                     normals: torch.Tensor,    # [B, 3, N]
                                     camera_pos: torch.Tensor, # [B, 3, 1]
                                     light_pos: Optional[torch.Tensor] = None,  # [B, 3, 1] for point light
                                     light_pose: Optional[torch.Tensor] = None, # [B, 4, 4] for planar light
                                     light_width: Optional[float] = None,       # planar light width
                                     light_height: Optional[float] = None,      # planar light height
                                     light_samples: int = 16,                   # samples for planar light
                                     light_intensity: float = 1.0, 
                                     light_color: torch.Tensor = None, # [B, 3, 1] or [B, 3, N]
                                     surface_roughness: float = 0.1,
                                     light_attenuation: Tuple[float, float, float] = (1.0, 0.1, 0.01)) -> torch.Tensor:
        """
        Calculate reflection intensity using optimized Phong reflection model.
        Supports both point lights and planar (area) lights.
        
        Args:
            cloud_xyz: 3D surface points [B, 3, N]
            normals: Surface normals [B, 3, N]
            camera_pos: Camera position [B, 3, 1] 
            light_pos: Point light position [B, 3, 1] (for point light)
            light_pose: Planar light pose [B, 4, 4] (for planar light)
            light_width: Width of planar light (required for planar light)
            light_height: Height of planar light (required for planar light)
            light_samples: Number of samples for planar light integration
            light_intensity: Light intensity scalar
            light_color: Light color [B, 3, 1] or [B, 3, N]
            surface_roughness: Surface roughness parameter (0=mirror, 1=rough)
            light_attenuation: (constant, linear, quadratic) attenuation
            
        Returns:
            reflection_intensity: Reflection intensity [B, 3, N]
        """
        B, _, N = cloud_xyz.shape
        device = cloud_xyz.device
        
        # Determine light type and validate inputs
        is_point_light = light_pos is not None
        is_planar_light = light_pose is not None and light_width is not None and light_height is not None
        
        if not (is_point_light ^ is_planar_light):  # XOR - exactly one should be true
            raise ValueError("Specify exactly one light type: either light_pos for point light, or (light_pose, light_width, light_height) for planar light")
        
        # Ensure light_color broadcasting
        if light_color is not None and light_color.shape[-1] == 1:
            light_color = light_color.expand(-1, -1, N)  # [B, 3, N]
        elif light_color is None:
            light_color = torch.ones(B, 3, N, device=device, dtype=cloud_xyz.dtype)
        
        if is_point_light:
            # Point light calculation (original logic)
            light_positions = light_pos  # [B, 3, 1]
            
            # Vectorized direction calculations
            light_dir = light_positions - cloud_xyz  # [B, 3, N]
            light_distance = torch.norm(light_dir, dim=1, keepdim=True)  # [B, 1, N]
            light_dir = F.normalize(light_dir, dim=1, eps=1e-8)  # [B, 3, N]
            
            # Calculate reflections
            reflection_intensity = self._calculate_phong_reflection(
                cloud_xyz, normals, camera_pos, light_dir, light_distance,
                light_intensity, light_color, surface_roughness, light_attenuation
            )
            
        else:
            # Planar light calculation
            # Sample points on the planar light with image projection
            light_sample_points, light_sample_colors, actual_samples = self._sample_planar_light(
                light_pose, light_width, light_height, light_samples, 
                projected_image=getattr(self, '_current_projected_image', None)
            )  # [B, 3, actual_samples], [B, 3, actual_samples], int
            
            # Initialize accumulated reflection
            total_reflection = torch.zeros(B, 3, N, device=device, dtype=cloud_xyz.dtype)
            
            # Calculate reflection from each light sample
            for i in range(actual_samples):
                sample_pos = light_sample_points[:, :, i:i+1]  # [B, 3, 1]
                sample_color = light_sample_colors[:, :, i:i+1]  # [B, 3, 1]
                
                # Broadcast sample color to all surface points
                sample_color_full = sample_color.expand(-1, -1, N)  # [B, 3, N]
                
                # Direction from surface to this light sample
                light_dir = sample_pos - cloud_xyz  # [B, 3, N]
                light_distance = torch.norm(light_dir, dim=1, keepdim=True)  # [B, 1, N]
                light_dir = F.normalize(light_dir, dim=1, eps=1e-8)  # [B, 3, N]
                
                # Calculate reflection from this sample with its specific color
                sample_reflection = self._calculate_phong_reflection(
                    cloud_xyz, normals, camera_pos, light_dir, light_distance,
                    light_intensity, sample_color_full, surface_roughness, light_attenuation
                )  # [B, 3, N]
                
                total_reflection += sample_reflection
            
            # Average over all samples (Monte Carlo integration)
            reflection_intensity = total_reflection / actual_samples
            
            # Scale by light area (larger lights emit more total energy)
            light_area = light_width * light_height
            reflection_intensity *= light_area
        
        return reflection_intensity
    
    def _calculate_phong_reflection(self,
                                  cloud_xyz: torch.Tensor,      # [B, 3, N]
                                  normals: torch.Tensor,        # [B, 3, N]
                                  camera_pos: torch.Tensor,     # [B, 3, 1]
                                  light_dir: torch.Tensor,      # [B, 3, N]
                                  light_distance: torch.Tensor, # [B, 1, N]
                                  light_intensity: float,
                                  light_color: torch.Tensor,    # [B, 3, N]
                                  surface_roughness: float,
                                  light_attenuation: Tuple[float, float, float]) -> torch.Tensor:
        """
        Core Phong reflection calculation for a single light direction.
        
        Returns:
            reflection_intensity: [B, 3, N]
        """
        # View direction
        view_dir = camera_pos - cloud_xyz  # [B, 3, N]
        view_dir = F.normalize(view_dir, dim=1, eps=1e-8)  # [B, 3, N]
        
        # Optimized reflection calculation: R = 2(N·L)N - L
        dot_nl = torch.sum(normals * light_dir, dim=1, keepdim=True).clamp(0.0, 1.0)  # [B, 1, N]
        reflection_dir = 2.0 * dot_nl * normals - light_dir  # [B, 3, N]
        reflection_dir = F.normalize(reflection_dir, dim=1, eps=1e-8)  # [B, 3, N]
        
        # Specular calculation
        dot_rv = torch.sum(reflection_dir * view_dir, dim=1, keepdim=True).clamp(0.0, 1.0)  # [B, 1, N]
        
        # Optimized shininess calculation
        shininess = torch.clamp(torch.tensor(128.0 * (1.0 - surface_roughness)), min=1.0)
        specular = torch.pow(dot_rv, shininess)  # [B, 1, N]
        
        # Vectorized attenuation
        const_att, linear_att, quad_att = light_attenuation
        attenuation = 1.0 / (const_att + linear_att * light_distance + quad_att * light_distance.square())  # [B, 1, N]
        
        # Final intensity calculation - fully vectorized
        reflection_intensity = (light_intensity * attenuation * dot_nl * specular * light_color)  # [B, 3, N]
        
        return reflection_intensity
    
    def _prepare_common_inputs(self, 
                              cloud: torch.Tensor,      # [B, 4, N]
                              rgb_vec: torch.Tensor,    # [B, 3, N] or [B, E, N]
                              camera_T: torch.Tensor,   # [B, 4, 4] or [B, 6]
                              light_color: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare common inputs for both light types.
        
        Returns:
            cloud_xyz: [B, 3, N]
            camera_pos: [B, 3, 1] 
            light_color_processed: [B, 3, 1]
            normals: [B, 3, N]
        """
        B, _, N = cloud.shape
        device = cloud.device
        
        # Input validation
        assert cloud.shape[1] == 4, f"Expected cloud shape [B, 4, N], got {cloud.shape}"
        assert rgb_vec.shape[0] == B and rgb_vec.shape[2] == N, f"RGB shape mismatch: {rgb_vec.shape}"
        
        # Default light color setup
        if light_color is None:
            light_color = torch.ones(3, device=device, dtype=cloud.dtype)
        
        # Ensure correct shapes for light_color
        if light_color.dim() == 1:
            light_color = light_color.unsqueeze(0).expand(B, -1)  # [B, 3]
        light_color_processed = light_color.unsqueeze(-1)  # [B, 3, 1]
        
        # Extract camera position
        camera_T_processed = camera_T
        if camera_T.shape[1] == 6:
            camera_T_processed = geometry.euler2mat(camera_T)  # [B, 4, 4]
        camera_pos = camera_T_processed[:, :3, 3:4]  # [B, 3, 1]
        
        # Extract 3D coordinates
        cloud_xyz = cloud[:, :3, :]  # [B, 3, N]
        
        # Estimate surface normals
        normals = self.estimate_surface_normals(cloud_xyz)  # [B, 3, N]
        
        return cloud_xyz, camera_pos, light_color_processed, normals
    
    def _apply_reflections_and_project(self,
                                     cloud: torch.Tensor,           # [B, 4, N]
                                     rgb_vec: torch.Tensor,         # [B, 3, N] or [B, E, N]
                                     camera_K: torch.Tensor,        # [B, 3, 3]
                                     camera_T: torch.Tensor,        # [B, 4, 4] or [B, 6]
                                     reflection_intensity: torch.Tensor,  # [B, 3, N]
                                     normals: torch.Tensor,         # [B, 3, N]
                                     camera_pos: torch.Tensor,      # [B, 3, 1]
                                     light_info: Dict,              # Light-specific info
                                     reflection_strength: float) -> Dict[str, torch.Tensor]:
        """
        Apply reflections to RGB/features and project to image.
        
        Returns:
            Complete result dictionary
        """
        B, _, N = cloud.shape
        device = cloud.device
        
        # Apply reflection strength
        reflection_intensity = reflection_intensity * reflection_strength
        
        # Enhance RGB/features with reflections
        if rgb_vec.shape[1] == 3:
            # RGB case - add reflections directly
            enhanced_rgb = torch.clamp(rgb_vec + reflection_intensity, 0.0, 1.0)  # [B, 3, N]
        else:
            # Feature case - add reflections to first 3 channels if available
            enhanced_rgb = rgb_vec.clone()
            if rgb_vec.shape[1] >= 3:
                enhanced_rgb[:, :3, :] = torch.clamp(
                    enhanced_rgb[:, :3, :] + reflection_intensity, 0.0, 1.0
                )
        
        # Project to image
        projection_result = self.project(
            cloud, enhanced_rgb, camera_K, camera_T,
            return_artifacts=True, return_mask=True
        )
        
        # Create reflection-only visualization
        if rgb_vec.shape[1] > 3:
            reflection_features = torch.cat([
                reflection_intensity, 
                torch.zeros(B, rgb_vec.shape[1] - 3, N, device=device, dtype=reflection_intensity.dtype)
            ], dim=1)
        else:
            reflection_features = reflection_intensity
            
        reflection_only = self.project(
            cloud, reflection_features, camera_K, camera_T,
            return_artifacts=True, return_mask=True
        )
        
        # Build result dictionary
        result = {
            'reflection_intensity': reflection_intensity,      # [B, 3, N]
            'surface_normals': normals,                       # [B, 3, N]
            'reflection_only': reflection_only['warped'],     # [B, 3, H, W] or [B, E, H, W]
            'camera_position': camera_pos,                    # [B, 3, 1]
            'enhanced_features': enhanced_rgb,                # [B, 3, N] or [B, E, N]
        }
        
        # Add light-specific information
        result.update(light_info)
        
        # Add projection results
        result.update(projection_result)
        
        return result
    
    def forward_point_light(self,
                           cloud: torch.Tensor,                    # [B, 4, N]
                           rgb_vec: torch.Tensor,                  # [B, 3, N] or [B, E, N]
                           camera_K: torch.Tensor,                 # [B, 3, 3]
                           camera_T: torch.Tensor,                 # [B, 4, 4] or [B, 6]
                           light_position: torch.Tensor,           # [B, 3] or [3]
                           light_intensity: float = 1.0,
                           light_color: Optional[torch.Tensor] = None,     # [3] or [B, 3]
                           surface_roughness: float = 0.1,
                           light_attenuation: Tuple[float, float, float] = (1.0, 0.1, 0.01),
                           reflection_strength: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Render image with reflections from a point light source.
        
        Args:
            cloud: 3D point cloud [B, 4, N] (homogeneous coordinates)
            rgb_vec: RGB values [B, 3, N] or features [B, E, N]
            camera_K: Camera intrinsics [B, 3, 3]
            camera_T: Camera pose [B, 4, 4] or [B, 6]
            light_position: Point light position [B, 3] or [3] (world coordinates)
            light_intensity: Light intensity scalar
            light_color: Light color [3] or [B, 3] (default: white)
            surface_roughness: Surface roughness 0-1 (0=mirror, 1=diffuse)
            light_attenuation: (constant, linear, quadratic) attenuation coefficients
            reflection_strength: Overall reflection strength multiplier (0-1)
            
        Returns:
            Dict containing image with reflections and intermediate results
        """
        B = cloud.shape[0]
        
        # Prepare common inputs
        cloud_xyz, camera_pos, light_color_processed, normals = self._prepare_common_inputs(
            cloud, rgb_vec, camera_T, light_color
        )
        
        # Process point light position
        if light_position.dim() == 1:
            light_position = light_position.unsqueeze(0).expand(B, -1)  # [B, 3]
        light_pos_processed = light_position.unsqueeze(-1)  # [B, 3, 1]
        
        # Calculate reflection intensities for point light
        reflection_intensity = self.calculate_reflection_intensity(
            cloud_xyz=cloud_xyz,
            normals=normals,
            camera_pos=camera_pos,
            light_pos=light_pos_processed,
            light_pose=None,
            light_width=None,
            light_height=None,
            light_samples=1,  # Not used for point light
            light_intensity=light_intensity,
            light_color=light_color_processed,
            surface_roughness=surface_roughness,
            light_attenuation=light_attenuation
        )  # [B, 3, N]
        
        # Light-specific info
        light_info = {
            'light_type': 'point',
            'light_position': light_pos_processed,  # [B, 3, 1]
        }
        
        # Apply reflections and project
        return self._apply_reflections_and_project(
            cloud, rgb_vec, camera_K, camera_T, reflection_intensity,
            normals, camera_pos, light_info, reflection_strength
        )
    
    def forward_planar_light(self,
                            cloud: torch.Tensor,                    # [B, 4, N]
                            rgb_vec: torch.Tensor,                  # [B, 3, N] or [B, E, N]
                            camera_K: torch.Tensor,                 # [B, 3, 3]
                            camera_T: torch.Tensor,                 # [B, 4, 4] or [B, 6]
                            light_pose: torch.Tensor,               # [B, 4, 4]
                            light_width: float,                     # Width of light plane
                            light_height: float,                    # Height of light plane
                            light_samples: int = 16,                # Number of sample points
                            light_intensity: float = 1.0,
                            light_color: Optional[torch.Tensor] = None,     # [3] or [B, 3] (used if no projected_image)
                            projected_image: Optional[torch.Tensor] = None, # [B, 3, H, W] - NEW: RGB image to project
                            surface_roughness: float = 0.1,
                            light_attenuation: Tuple[float, float, float] = (1.0, 0.1, 0.01),
                            reflection_strength: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Render image with reflections from a planar (area) light source with optional image projection.
        
        Args:
            cloud: 3D point cloud [B, 4, N] (homogeneous coordinates)
            rgb_vec: RGB values [B, 3, N] or features [B, E, N]
            camera_K: Camera intrinsics [B, 3, 3]
            camera_T: Camera pose [B, 4, 4] or [B, 6]
            light_pose: Light pose matrix [B, 4, 4] (world-to-light transform)
            light_width: Width of rectangular light plane
            light_height: Height of rectangular light plane
            light_samples: Number of sample points for integration
            light_intensity: Light intensity scalar
            light_color: Light color [3] or [B, 3] (used if no projected_image provided)
            projected_image: RGB image to project [B, 3, H, W] - creates projector effect
            surface_roughness: Surface roughness 0-1 (0=mirror, 1=diffuse)
            light_attenuation: (constant, linear, quadratic) attenuation coefficients
            reflection_strength: Overall reflection strength multiplier (0-1)
            
        Returns:
            Dict containing image with reflections and intermediate results
        """
        # Store projected image for use in reflection calculation
        if projected_image is not None:
            self._current_projected_image = projected_image
        else:
            self._current_projected_image = None
        
        # Prepare common inputs
        cloud_xyz, camera_pos, light_color_processed, normals = self._prepare_common_inputs(
            cloud, rgb_vec, camera_T, light_color
        )
        
        # Calculate reflection intensities for planar light
        reflection_intensity = self.calculate_reflection_intensity(
            cloud_xyz=cloud_xyz,
            normals=normals,
            camera_pos=camera_pos,
            light_pos=None,
            light_pose=light_pose,
            light_width=light_width,
            light_height=light_height,
            light_samples=light_samples,
            light_intensity=light_intensity,
            light_color=light_color_processed,
            surface_roughness=surface_roughness,
            light_attenuation=light_attenuation
        )  # [B, 3, N]
        
        # Clean up temporary attribute
        if hasattr(self, '_current_projected_image'):
            delattr(self, '_current_projected_image')
        
        # Light-specific info
        light_info = {
            'light_type': 'planar',
            'light_pose': light_pose,           # [B, 4, 4]
            'light_width': light_width,         # float
            'light_height': light_height,       # float
            'light_samples': light_samples,     # int
        }
        
        # Add projected image info if provided
        if projected_image is not None:
            light_info.update({
                'projected_image': projected_image,  # [B, 3, H, W]
                'projection_mode': True,
            })
        else:
            light_info.update({
                'projection_mode': False,
            })
        
        # Apply reflections and project
        return self._apply_reflections_and_project(
            cloud, rgb_vec, camera_K, camera_T, reflection_intensity,
            normals, camera_pos, light_info, reflection_strength
        )
    
    def forward(self, *args, **kwargs):
        """
        Deprecated: Use forward_point_light() or forward_planar_light() instead.
        
        This method is kept for backward compatibility but will raise an error
        to encourage using the specific light type methods.
        """
        raise NotImplementedError(
            "The generic forward() method has been deprecated. "
            "Use forward_point_light() for point lights or forward_planar_light() for planar lights instead."
        )