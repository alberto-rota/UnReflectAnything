import torch
import torch.nn as nn
import numpy as np
import cv2
from .keyframing import KeyFrameFinder


class Trajectory:
    def __init__(self, num_frames=None, device=None, min_inlier_ratio=0.5, min_inlier_count=100, max_frames_since_last=20):
        """
        Initialize a trajectory tracker.

        Args:
            num_frames: Number of frames in the sequence
            device: Device to store tensors on
            min_inlier_ratio: Minimum ratio of inliers to consider a frame as tracking well
            min_inlier_count: Minimum absolute number of inliers required
            max_frames_since_last: Maximum frames allowed since last keyframe
        """
        self.num_frames = num_frames
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        if num_frames is not None:
            self.trajectory = (
                torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1).to(self.device)
            )
        else:
            self.trajectory = None
            
        # Initialize keyframe finder
        self.keyframer = KeyFrameFinder(
            min_inlier_ratio=min_inlier_ratio,
            min_inlier_count=min_inlier_count,
            max_frames_since_last=max_frames_since_last
        )
        self.keyframe = None
        self.keyframe_idx = 0

    def initialize(self, num_frames=None, base_pose=None):
        """Initialize trajectory with identity matrices"""
        if num_frames is None:
            num_frames = self.num_frames
        self.trajectory = (
            torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1).to(self.device)
        )
        if base_pose is not None:
            self.trajectory[0] = base_pose
        return self.trajectory

    def get_latest_pose(self):
        """Get the latest pose in the trajectory"""
        return self.trajectory[-1]

    def update_from_keyframe(self, keyframe_idx, current_idx, relative_pose):
        """
        Update trajectory based on relative pose from keyframe to current frame.

        Args:
            keyframe_idx: Index of the keyframe
            current_idx: Index of the current frame
            relative_pose: Relative pose from keyframe to current frame [4, 4]
        """
        # Apply relative transformation based on keyframe
        self.trajectory[current_idx] = self.trajectory[keyframe_idx] @ relative_pose
        return self.trajectory[current_idx]

    def interpolate_between_keyframes(self, idx1, idx2, relative_pose):
        """
        Update trajectory for frames between keyframes using interpolation.

        Args:
            idx1: Index of first keyframe
            idx2: Index of second keyframe
            relative_pose: Relative pose from first to second keyframe
        """
        for j in range(idx1 + 1, idx2 + 1):
            # Apply relative transformation proportionally between keyframes
            weight = (j - idx1) / (idx2 - idx1)
            weighted_pose = self.interpolate_pose(
                torch.eye(4).to(self.device), relative_pose, weight
            )
            self.trajectory[j] = self.trajectory[idx1] @ weighted_pose

    def optimize(self, iterations=10, smoothness_weight=0.1):
        """Apply trajectory optimization to improve smoothness"""
        self.trajectory = optimize_trajectory(
            self.trajectory, iterations, smoothness_weight
        )
        return self.trajectory

    @staticmethod
    def estimate_relative_pose_emd(source_points, target_points, batch_idx, K):
        """
        Estimate the relative pose between two frames using matched points.

        Args:
            source_points: Source points tensor [N, 2]
            target_points: Target points tensor [N, 2]
            batch_idx: Batch indices tensor [N]
            K: Camera intrinsic matrix

        Returns:
            Tensor of shape [B, 4, 4] containing 4x4 transformation matrices
        """

        # Get unique batch indices
        unique_batches = torch.unique(batch_idx)
        batch_size = len(unique_batches)

        # Initialize output transformations
        transformations = (
            torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(source_points.device)
        )

        # Process each batch separately
        for b_idx, batch_id in enumerate(unique_batches):
            # Get points for this batch
            batch_mask = batch_idx == batch_id
            src_pts = source_points[batch_mask].cpu().numpy()
            tgt_pts = target_points[batch_mask].cpu().numpy()
            K = K[b_idx].cpu().numpy()
            # Need at least 5 points for essential matrix estimation
            if len(src_pts) >= 5:
                # Convert to numpy for OpenCV

                # Estimate essential matrix
                E, mask = cv2.findEssentialMat(
                    src_pts, tgt_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
                )

                if E is not None:
                    # Recover rotation and translation from essential matrix
                    _, R, t, _ = cv2.recoverPose(E, src_pts, tgt_pts, K, mask=mask)

                    # Create 4x4 transformation matrix
                    transformation = np.eye(4)
                    transformation[:3, :3] = R
                    transformation[:3, 3] = t.squeeze()

                    # Convert to torch tensor
                    transformations[b_idx] = torch.tensor(
                        transformation, device=source_points.device, dtype=torch.float32
                    )

        return transformations

    @staticmethod
    def estimate_relative_pose_pnp(
        source_points, target_points, batch_idx, K, depth_map_rgb1
    ):
        """
        Estimate the relative pose between two frames using PnP with matched points and depth information.

        Args:
            source_points: Source points tensor [N, 2]
            target_points: Target points tensor [N, 2]
            batch_idx: Batch indices tensor [N]
            K: Camera intrinsic matrix [B, 3, 3]
            depth_map_rgb1: Depth map for RGB1 [B, 1, H, W]

        Returns:
            Tensor of shape [B, 4, 4] containing 4x4 transformation matrices
        """

        # Get unique batch indices
        unique_batches = torch.unique(batch_idx)
        batch_size = len(unique_batches)

        # Initialize output transformations
        transformations = (
            torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(source_points.device)
        )

        # Process each batch separately
        for b_idx, batch_id in enumerate(unique_batches):
            # Get points for this batch
            batch_mask = batch_idx == batch_id
            src_pts = source_points[batch_mask].cpu().numpy()
            tgt_pts = target_points[batch_mask].cpu().numpy()
            K_batch = K[b_idx].cpu().numpy()

            # Convert source points to integer coordinates for indexing the depth map
            src_pts_int = np.round(src_pts).astype(int)

            # Remove the batch dimension
            depth_map_rgb1 = depth_map_rgb1.squeeze(1)
            # Check if points are within the depth map boundaries
            h, w = depth_map_rgb1[b_idx].shape
            valid_mask = (
                (src_pts_int[:, 0] >= 0)
                & (src_pts_int[:, 0] < w)
                & (src_pts_int[:, 1] >= 0)
                & (src_pts_int[:, 1] < h)
            )

            # Need at least 6 points for PnP RANSAC
            if np.sum(valid_mask) >= 6:
                # Get valid points
                valid_src_pts = src_pts[valid_mask]
                valid_tgt_pts = tgt_pts[valid_mask]
                valid_src_pts_int = src_pts_int[valid_mask]

                # Get depth values for valid source points
                depths = (
                    depth_map_rgb1[b_idx]
                    .cpu()
                    .detach()
                    .numpy()[valid_src_pts_int[:, 1], valid_src_pts_int[:, 0]]
                )

                # Filter out zero or invalid depth values
                non_zero_mask = depths > 0
                if np.sum(non_zero_mask) >= 6:
                    # Get the final set of points
                    final_src_pts = valid_src_pts[non_zero_mask]
                    final_tgt_pts = valid_tgt_pts[non_zero_mask]
                    final_depths = depths[non_zero_mask]

                    # Back-project 2D points to 3D using the intrinsic parameters and depth
                    fx, fy = K_batch[0, 0], K_batch[1, 1]
                    cx, cy = K_batch[0, 2], K_batch[1, 2]

                    # Create 3D points
                    src_pts_3d = np.zeros((len(final_src_pts), 3))
                    src_pts_3d[:, 2] = final_depths  # Z = depth
                    src_pts_3d[:, 0] = (
                        (final_src_pts[:, 0] - cx) * final_depths / fx
                    )  # X = (u - cx) * Z / fx
                    src_pts_3d[:, 1] = (
                        (final_src_pts[:, 1] - cy) * final_depths / fy
                    )  # Y = (v - cy) * Z / fy

                    # Estimate pose using PnP RANSAC
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        src_pts_3d,
                        final_tgt_pts,
                        K_batch,
                        None,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                        iterationsCount=1000,
                        reprojectionError=2.0,
                        confidence=0.99,
                    )
                    # rvec *= -1

                    if success:
                        # Convert rotation vector to rotation matrix
                        R_1_to_2, _ = cv2.Rodrigues(rvec)

                        # Invert the transformation to get the pose from camera 2 to camera 1
                        # (to match the convention in the original method)
                        R_2_to_1 = R_1_to_2.T
                        t_2_to_1 = -np.dot(R_2_to_1, tvec)

                        # Create 4x4 transformation matrix
                        transformation = np.eye(4)
                        transformation[:3, :3] = R_2_to_1
                        transformation[:3, 3] = t_2_to_1.squeeze()

                        # Convert to torch tensor
                        transformations[b_idx] = torch.tensor(
                            transformation,
                            device=source_points.device,
                            dtype=torch.float32,
                        )

        return transformations

    @staticmethod
    def interpolate_pose(pose1, pose2, weight):
        """
        Interpolate between two poses.

        Args:
            pose1: First pose tensor [4, 4]
            pose2: Second pose tensor [4, 4]
            weight: Interpolation weight between 0 and 1

        Returns:
            Interpolated pose tensor [4, 4]
        """
        # Extract rotation and translation from poses
        R1 = pose1[:3, :3]
        t1 = pose1[:3, 3]
        R2 = pose2[:3, :3]
        t2 = pose2[:3, 3]

        # Convert rotations to quaternions for proper interpolation
        q1 = rotation_matrix_to_quaternion(R1)
        q2 = rotation_matrix_to_quaternion(R2)

        # Ensure shortest path quaternion interpolation
        if torch.sum(q1 * q2) < 0:
            q2 = -q2

        # Interpolate quaternion and normalize
        q_interp = q1 * (1 - weight) + q2 * weight
        q_interp = q_interp / torch.norm(q_interp)

        # Convert back to rotation matrix
        R_interp = rotation_matrix_to_quaternion(q_interp)

        # Linearly interpolate translation
        t_interp = t1 * (1 - weight) + t2 * weight

        # Construct interpolated pose
        pose_interp = torch.eye(4, device=pose1.device)
        pose_interp[:3, :3] = R_interp
        pose_interp[:3, 3] = t_interp

        return pose_interp

    def needs_keyframe(self, frame_idx, inlier_count, total_points=None):
        """
        Check if a new frame should be a keyframe based on inlier statistics.
        
        Args:
            frame_idx: Current frame index
            inlier_count: Number of inliers matched with previous frame
            total_points: Total number of tracked points (if None, uses last keyframe inliers)
            
        Returns:
            Boolean indicating if this frame should be a keyframe
        """
        return self.keyframer.needsKeyframe(frame_idx, inlier_count, total_points)

    def update_keyframe(self, frame, frame_idx):
        """
        Update the current keyframe.
        
        Args:
            frame: The new keyframe
            frame_idx: Index of the new keyframe
        """
        self.keyframe = frame
        self.keyframe_idx = frame_idx


def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.

    Args:
        R: Rotation matrix tensor [3, 3]

    Returns:
        Quaternion tensor [4] in format [w, x, y, z]
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        S = torch.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return torch.tensor([w, x, y, z], device=R.device)


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.

    Args:
        quaternion: Quaternion tensor [4] in format [w, x, y, z]

    Returns:
        Rotation matrix tensor [3, 3]
    """
    w, x, y, z = quaternion

    # Normalize quaternion
    quaternion = quaternion / torch.norm(quaternion)
    w, x, y, z = quaternion

    # Convert quaternion to rotation matrix
    R = torch.zeros(3, 3, device=quaternion.device)

    R[0, 0] = 1 - 2 * y * y - 2 * z * z
    R[0, 1] = 2 * x * y - 2 * w * z
    R[0, 2] = 2 * x * z + 2 * w * y

    R[1, 0] = 2 * x * y + 2 * w * z
    R[1, 1] = 1 - 2 * x * x - 2 * z * z
    R[1, 2] = 2 * y * z - 2 * w * x

    R[2, 0] = 2 * x * z - 2 * w * y
    R[2, 1] = 2 * y * z + 2 * w * x
    R[2, 2] = 1 - 2 * x * x - 2 * y * y

    return R


def compose_transformations(transforms):
    """
    Compose a sequence of transformations.

    Args:
        transforms: List or tensor of 4x4 transformation matrices

    Returns:
        List of cumulative transformations
    """
    device = transforms[0].device if isinstance(transforms, list) else transforms.device

    # Initialize with identity
    result = [torch.eye(4, device=device)]

    # Compose transformations
    for i in range(len(transforms)):
        result.append(result[-1] @ transforms[i])

    return result[1:]  # Skip the initial identity


def optimize_trajectory(trajectory, iterations=10, smoothness_weight=0.1):
    """
    Apply a simple trajectory optimization to improve smoothness.

    Args:
        trajectory: Tensor of shape [N, 4, 4] with transformation matrices
        iterations: Number of optimization iterations
        smoothness_weight: Weight of the smoothness term

    Returns:
        Optimized trajectory tensor
    """
    # Make a copy to avoid modifying the original
    optimized = trajectory.clone()

    # Simple sliding window optimization
    for _ in range(iterations):
        for i in range(1, len(optimized) - 1):
            # Get neighboring poses
            prev_pose = optimized[i - 1]
            curr_pose = optimized[i]
            next_pose = optimized[i + 1]

            # Calculate relative transforms
            rel_prev = torch.inverse(prev_pose) @ curr_pose
            rel_next = torch.inverse(curr_pose) @ next_pose

            # Calculate the "average" transform
            avg_rel = Trajectory.interpolate_pose(rel_prev, rel_next, 0.5)

            # Update current pose with smoothed version
            new_pose = Trajectory.interpolate_pose(
                curr_pose, prev_pose @ avg_rel, smoothness_weight
            )

            optimized[i] = new_pose

    return optimized
