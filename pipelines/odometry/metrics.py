import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation as R


def absolute_trajectory_error(estimated_trajectory, gt_trajectory):
    """
    Compute the absolute trajectory error (ATE) between estimated and ground truth trajectories.

    Args:
        estimated_trajectory: Estimated camera poses [N, 4, 4]
        gt_trajectory: Ground truth camera poses [N, 4, 4]

    Returns:
        Dictionary with ATE statistics
    """
    # Ensure same device
    if isinstance(estimated_trajectory, torch.Tensor) and isinstance(
        gt_trajectory, torch.Tensor
    ):
        gt_trajectory = gt_trajectory.to(estimated_trajectory.device)

    # Convert to numpy if needed
    if isinstance(estimated_trajectory, torch.Tensor):
        estimated_trajectory = estimated_trajectory.detach().cpu().numpy()
    if isinstance(gt_trajectory, torch.Tensor):
        gt_trajectory = gt_trajectory.detach().cpu().numpy()

    # Extract positions (translation part of the transformation matrices)
    estimated_positions = estimated_trajectory[:, :3, 3]
    gt_positions = gt_trajectory[:, :3, 3]

    # Compute position errors
    position_errors = np.linalg.norm(estimated_positions - gt_positions, axis=1)

    # Compute statistics
    ate_mean = np.mean(position_errors)
    ate_std = np.std(position_errors)
    ate_min = np.min(position_errors)
    ate_max = np.max(position_errors)
    ate_rmse = np.sqrt(np.mean(np.square(position_errors)))

    return {
        "mean": ate_mean,
        "std": ate_std,
        "min": ate_min,
        "max": ate_max,
        "rmse": ate_rmse,
        "errors": position_errors,
    }


def relative_pose_error(estimated_trajectory, gt_trajectory):
    """
    Compute the relative pose error (RPE) between estimated and ground truth trajectories.

    Args:
        estimated_trajectory: Estimated camera poses [N, 4, 4]
        gt_trajectory: Ground truth camera poses [N, 4, 4]

    Returns:
        Dictionary with RPE statistics for rotation and translation
    """
    # Ensure same device
    if isinstance(estimated_trajectory, torch.Tensor) and isinstance(
        gt_trajectory, torch.Tensor
    ):
        gt_trajectory = gt_trajectory.to(estimated_trajectory.device)

    # Convert to numpy if needed
    if isinstance(estimated_trajectory, torch.Tensor):
        estimated_trajectory = estimated_trajectory.detach().cpu().numpy()
    if isinstance(gt_trajectory, torch.Tensor):
        gt_trajectory = gt_trajectory.detach().cpu().numpy()

    # Number of poses
    n_poses = estimated_trajectory.shape[0]

    # Compute relative poses between consecutive frames
    trans_errors = []
    rot_errors = []

    for i in range(n_poses - 1):
        # Relative ground truth transformation
        gt_rel = np.linalg.inv(gt_trajectory[i]) @ gt_trajectory[i + 1]

        # Relative estimated transformation
        est_rel = np.linalg.inv(estimated_trajectory[i]) @ estimated_trajectory[i + 1]

        # Compute error transformation
        error_transform = np.linalg.inv(gt_rel) @ est_rel

        # Extract rotation and translation errors
        trans_error = np.linalg.norm(error_transform[:3, 3])

        # Rotation error as angle in radians
        rot_error = np.arccos((np.trace(error_transform[:3, :3]) - 1) / 2)

        trans_errors.append(trans_error)
        rot_errors.append(rot_error)

    # Convert to numpy arrays
    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)

    # Compute statistics
    rpe_trans = {
        "mean": np.mean(trans_errors),
        "std": np.std(trans_errors),
        "min": np.min(trans_errors),
        "max": np.max(trans_errors),
        "rmse": np.sqrt(np.mean(np.square(trans_errors))),
        "errors": trans_errors,
    }

    rpe_rot = {
        "mean": np.mean(rot_errors),
        "std": np.std(rot_errors),
        "min": np.min(rot_errors),
        "max": np.max(rot_errors),
        "rmse": np.sqrt(np.mean(np.square(rot_errors))),
        "errors": rot_errors,
    }

    return {"translation": rpe_trans, "rotation": rpe_rot}


def trajectory_smoothness(trajectory, step=1):
    """
    Compute the smoothness of a trajectory.

    Args:
        trajectory: Camera poses [N, 4, 4]
        step: Step size for computing differences

    Returns:
        Dictionary with smoothness metrics
    """
    # Convert to torch tensor if numpy
    if isinstance(trajectory, np.ndarray):
        trajectory = torch.from_numpy(trajectory).float()

    # Number of poses
    n_poses = trajectory.shape[0]

    # Initialize arrays for position, rotation, and velocity differences
    position_diffs = []
    rotation_diffs = []
    velocity_diffs = []

    # Reference velocity (initially zero)
    prev_velocity = None

    for i in range(0, n_poses - step, step):
        # Current and next pose
        pose1 = trajectory[i]
        pose2 = trajectory[i + step]

        # Extract positions
        pos1 = pose1[:3, 3]
        pos2 = pose2[:3, 3]

        # Position difference
        pos_diff = torch.norm(pos2 - pos1)
        position_diffs.append(pos_diff)

        # Current velocity
        velocity = pos_diff / step

        # Velocity difference (acceleration)
        if prev_velocity is not None:
            velocity_diff = torch.abs(velocity - prev_velocity)
            velocity_diffs.append(velocity_diff)

        prev_velocity = velocity

        # Rotation difference
        # Using the rotation part of the transformation
        R1 = pose1[:3, :3]
        R2 = pose2[:3, :3]

        # Compute rotation difference (Frobenius norm of difference)
        rot_diff = torch.norm(R2 - R1)
        rotation_diffs.append(rot_diff)

    # Convert to tensors
    position_diffs = torch.stack(position_diffs)
    rotation_diffs = torch.stack(rotation_diffs)

    # Compute metrics
    smoothness_metrics = {
        "position": {
            "mean": position_diffs.mean().item(),
            "std": position_diffs.std().item(),
            "max": position_diffs.max().item(),
        },
        "rotation": {
            "mean": rotation_diffs.mean().item(),
            "std": rotation_diffs.std().item(),
            "max": rotation_diffs.max().item(),
        },
    }

    if velocity_diffs:
        velocity_diffs = torch.stack(velocity_diffs)
        smoothness_metrics["acceleration"] = {
            "mean": velocity_diffs.mean().item(),
            "std": velocity_diffs.std().item(),
            "max": velocity_diffs.max().item(),
        }

    # Overall smoothness score (lower is smoother)
    position_smoothness = position_diffs.std().item() / (
        position_diffs.mean().item() + 1e-8
    )
    rotation_smoothness = rotation_diffs.std().item() / (
        rotation_diffs.mean().item() + 1e-8
    )

    smoothness_metrics["overall_score"] = (
        position_smoothness + rotation_smoothness
    ) / 2

    return smoothness_metrics


def scale_alignment(estimated_trajectory, gt_trajectory):
    """
    Find the optimal scale to align estimated trajectory with ground truth.

    Args:
        estimated_trajectory: Estimated camera poses [N, 4, 4]
        gt_trajectory: Ground truth camera poses [N, 4, 4]

    Returns:
        Scale factor and scaled estimated trajectory
    """
    # Extract positions
    if isinstance(estimated_trajectory, torch.Tensor):
        est_pos = estimated_trajectory[:, :3, 3].cpu().numpy()
    else:
        est_pos = estimated_trajectory[:, :3, 3]

    if isinstance(gt_trajectory, torch.Tensor):
        gt_pos = gt_trajectory[:, :3, 3].cpu().numpy()
    else:
        gt_pos = gt_trajectory[:, :3, 3]

    # Center trajectories
    est_centered = est_pos - est_pos.mean(axis=0)
    gt_centered = gt_pos - gt_pos.mean(axis=0)

    # Compute scale factor
    scale = np.sum(gt_centered * est_centered) / np.sum(est_centered * est_centered)

    # Scale the estimated trajectory
    if isinstance(estimated_trajectory, torch.Tensor):
        scaled_trajectory = estimated_trajectory.clone()
        scaled_trajectory[:, :3, 3] = scaled_trajectory[:, :3, 3] * scale
    else:
        scaled_trajectory = estimated_trajectory.copy()
        scaled_trajectory[:, :3, 3] = scaled_trajectory[:, :3, 3] * scale

    return scale, scaled_trajectory


def align_trajectories(estimated_trajectory, gt_trajectory, align_scale=True):
    """
    Align estimated trajectory to ground truth trajectory.

    Args:
        estimated_trajectory: Estimated camera poses [N, 4, 4]
        gt_trajectory: Ground truth camera poses [N, 4, 4]
        align_scale: Whether to align scale

    Returns:
        Aligned estimated trajectory
    """
    # Extract positions
    if isinstance(estimated_trajectory, torch.Tensor):
        est_pos = estimated_trajectory[:, :3, 3].cpu().numpy()
        is_torch = True
    else:
        est_pos = estimated_trajectory[:, :3, 3]
        is_torch = False

    if isinstance(gt_trajectory, torch.Tensor):
        gt_pos = gt_trajectory[:, :3, 3].cpu().numpy()
    else:
        gt_pos = gt_trajectory[:, :3, 3]

    # Center trajectories
    est_mean = est_pos.mean(axis=0)
    gt_mean = gt_pos.mean(axis=0)

    est_centered = est_pos - est_mean
    gt_centered = gt_pos - gt_mean

    # Find optimal rotation
    H = est_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Find optimal scale if requested
    if align_scale:
        scale = np.trace(R @ H) / np.sum(est_centered**2)
    else:
        scale = 1.0

    # Calculate translation
    t = gt_mean - scale * (R @ est_mean)

    # Create transformation matrix
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t

    # Apply transformation to the estimated trajectory
    aligned_trajectory = np.zeros_like(estimated_trajectory)
    for i in range(len(estimated_trajectory)):
        aligned_trajectory[i] = T @ estimated_trajectory[i]

    # Convert back to torch if needed
    if is_torch:
        aligned_trajectory = torch.from_numpy(aligned_trajectory).to(
            estimated_trajectory.device
        )

    return aligned_trajectory


def compute_position_errors(trajectory, trajectory_gt):
    """
    Compute position errors between estimated and ground truth trajectories.
    
    Args:
        trajectory: Tensor of shape (F, 4, 4) representing estimated trajectory
        trajectory_gt: Tensor of shape (F, 4, 4) representing ground truth trajectory
    
    Returns:
        Dictionary containing position errors
    """
    device = trajectory.device
    F = trajectory.shape[0]
    
    # Initialize error containers
    xyz_errors = torch.zeros((F, 3), device=device)
    translation_errors = torch.zeros(F, device=device)
    cum_xyz_errors = torch.zeros((F, 3), device=device)
    cum_translation_errors = torch.zeros(F, device=device)
    
    for i in range(F):
        # Extract translation (position)
        est_trans = trajectory[i, :3, 3]
        gt_trans = trajectory_gt[i, :3, 3]
        
        # Compute position errors (X, Y, Z)
        xyz_errors[i] = torch.abs(est_trans - gt_trans)
        
        # Compute overall translation error (Euclidean distance)
        translation_errors[i] = torch.norm(est_trans - gt_trans)
        
        # Compute cumulative errors up to this frame
        if i == 0:
            cum_xyz_errors[i] = xyz_errors[i]
            cum_translation_errors[i] = translation_errors[i]
        else:
            cum_xyz_errors[i] = cum_xyz_errors[i-1] + xyz_errors[i]
            cum_translation_errors[i] = cum_translation_errors[i-1] + translation_errors[i]
    
    return {
        'xyz_errors': xyz_errors,
        'translation_errors': translation_errors,
        'cum_xyz_errors': cum_xyz_errors,
        'cum_translation_errors': cum_translation_errors
    }


def compute_rotation_errors(trajectory, trajectory_gt):
    """
    Compute rotation errors between estimated and ground truth trajectories.
    
    Args:
        trajectory: Tensor of shape (F, 4, 4) representing estimated trajectory
        trajectory_gt: Tensor of shape (F, 4, 4) representing ground truth trajectory
    
    Returns:
        Dictionary containing rotation errors
    """
    device = trajectory.device
    F = trajectory.shape[0]
    
    # Initialize error containers
    rpy_errors = torch.zeros((F, 3), device=device)
    rotation_errors = torch.zeros(F, device=device)
    cum_rpy_errors = torch.zeros((F, 3), device=device)
    cum_rotation_errors = torch.zeros(F, device=device)
    
    for i in range(F):
        # Extract rotation matrices
        est_rot = trajectory[i, :3, :3]
        gt_rot = trajectory_gt[i, :3, :3]
        
        # Convert rotation matrices to numpy for scipy rotation
        est_rot_np = est_rot.cpu().numpy()
        gt_rot_np = gt_rot.cpu().numpy()
        
        # Create rotation objects
        est_r = R.from_matrix(est_rot_np)
        gt_r = R.from_matrix(gt_rot_np)
        
        # Get Euler angles (roll, pitch, yaw) in degrees
        est_euler = est_r.as_euler('xyz', degrees=True)
        gt_euler = gt_r.as_euler('xyz', degrees=True)
        
        # Compute angle errors
        euler_errors = np.abs(est_euler - gt_euler)
        # Handle angle wrapping
        for j in range(3):
            if euler_errors[j] > 180:
                euler_errors[j] = 360 - euler_errors[j]
        
        rpy_errors[i] = torch.tensor(euler_errors, device=device)
        
        # Compute overall rotation error (geodesic distance in degrees)
        rel_rot = est_r.inv() * gt_r
        rotation_errors[i] = torch.tensor(rel_rot.magnitude() * 180 / np.pi, device=device)
        
        # Compute cumulative errors up to this frame
        if i == 0:
            cum_rpy_errors[i] = rpy_errors[i]
            cum_rotation_errors[i] = rotation_errors[i]
        else:
            cum_rpy_errors[i] = cum_rpy_errors[i-1] + rpy_errors[i]
            cum_rotation_errors[i] = cum_rotation_errors[i-1] + rotation_errors[i]
    
    return {
        'rpy_errors': rpy_errors,
        'rotation_errors': rotation_errors,
        'cum_rpy_errors': cum_rpy_errors,
        'cum_rotation_errors': cum_rotation_errors
    }


def compute_relative_pose_errors(trajectory, trajectory_gt):
    """
    Compute relative pose errors between estimated and ground truth trajectories.
    
    Args:
        trajectory: Tensor of shape (F, 4, 4) representing estimated trajectory
        trajectory_gt: Tensor of shape (F, 4, 4) representing ground truth trajectory
    
    Returns:
        Dictionary containing relative pose errors
    """
    device = trajectory.device
    F = trajectory.shape[0]
    
    # Initialize error containers
    relative_pose_errors = torch.zeros(F, device=device)
    drift_per_distance = torch.zeros(F, device=device)
    drift_per_time = torch.zeros(F, device=device)
    
    # Store trajectory lengths for calculating drift
    total_distance_gt = 0.0
    previous_position_gt = trajectory_gt[0, :3, 3]
    
    for i in range(F):
        if i > 0:
            # Get relative transformation between consecutive frames
            rel_transform_est = torch.matmul(
                torch.inverse(trajectory[i-1]), trajectory[i]
            )
            rel_transform_gt = torch.matmul(
                torch.inverse(trajectory_gt[i-1]), trajectory_gt[i]
            )
            
            # Error between relative transformations
            rel_error_transform = torch.matmul(
                torch.inverse(rel_transform_gt), rel_transform_est
            )
            
            # Extract translation and rotation components of the error
            rel_error_trans = rel_error_transform[:3, 3]
            rel_error_rot_np = rel_error_transform[:3, :3].cpu().numpy()
            rel_error_r = R.from_matrix(rel_error_rot_np)
            
            # Combined error (weighted sum of translation and rotation)
            trans_component = torch.norm(rel_error_trans)
            rot_component = torch.tensor(rel_error_r.magnitude(), device=device)
            
            # Can adjust weights based on your application (translation vs rotation importance)
            w_trans, w_rot = 1.0, 1.0
            relative_pose_errors[i] = w_trans * trans_component + w_rot * rot_component
            
            # Calculate distance traveled in ground truth trajectory
            current_position_gt = trajectory_gt[i, :3, 3]
            segment_distance = torch.norm(current_position_gt - previous_position_gt)
            total_distance_gt += segment_distance.item()
            previous_position_gt = current_position_gt
            
            # Drift per distance traveled (meters/meter)
            if total_distance_gt > 0:
                drift_per_distance[i] = relative_pose_errors[i] / total_distance_gt
            
            # Drift per time (assuming constant time between frames)
            drift_per_time[i] = relative_pose_errors[i] / i
    
    return {
        'relative_pose_errors': relative_pose_errors,
        'drift_per_distance': drift_per_distance,
        'drift_per_time': drift_per_time
    }


def compute_trajectory_errors(trajectory, trajectory_gt):
    """
    Compute all trajectory errors between estimated and ground truth trajectories.
    
    Args:
        trajectory: Tensor of shape (F, 4, 4) representing estimated trajectory
        trajectory_gt: Tensor of shape (F, 4, 4) representing ground truth trajectory
    
    Returns:
        Dictionary containing all computed errors
    """
    # Compute position errors
    position_errors = compute_position_errors(trajectory, trajectory_gt)
    
    # Compute rotation errors
    rotation_errors = compute_rotation_errors(trajectory, trajectory_gt)
    
    # Compute relative pose errors
    relative_errors = compute_relative_pose_errors(trajectory, trajectory_gt)
    
    # Combine all errors
    errors = {**position_errors, **rotation_errors, **relative_errors}
    
    # Add absolute trajectory error (ATE)
    errors['absolute_trajectory_error'] = position_errors['translation_errors']
    
    return errors


def compute_error_statistics(errors):
    """
    Compute statistics for trajectory errors.
    
    Args:
        errors: Dictionary containing error data
    
    Returns:
        Dictionary containing error statistics
    """
    stats = {}
    
    for key in errors:
        if errors[key].dim() > 1:
            # For XYZ and RPY errors (which are 2D tensors)
            stats[f'{key}_mean'] = torch.mean(errors[key], dim=0)
            stats[f'{key}_std'] = torch.std(errors[key], dim=0)
            stats[f'{key}_max'] = torch.max(errors[key], dim=0)[0]
            stats[f'{key}_min'] = torch.min(errors[key], dim=0)[0]
        else:
            # For translation and rotation errors (which are 1D tensors)
            stats[f'{key}_mean'] = torch.mean(errors[key])
            stats[f'{key}_std'] = torch.std(errors[key])
            stats[f'{key}_max'] = torch.max(errors[key])
            stats[f'{key}_min'] = torch.min(errors[key])
    
    # Compute final cumulative errors (last frame)
    F = errors['xyz_errors'].shape[0]
    stats['final_cum_xyz_errors'] = errors['cum_xyz_errors'][F-1]
    stats['final_cum_rpy_errors'] = errors['cum_rpy_errors'][F-1]
    stats['final_cum_translation_error'] = errors['cum_translation_errors'][F-1]
    stats['final_cum_rotation_error'] = errors['cum_rotation_errors'][F-1]
    
    return stats


def plot_trajectory_errors(errors, trajectory=None, trajectory_gt=None, save_path=None):
    """
    Create visualizations of trajectory errors.
    
    Args:
        errors: Dictionary containing error data
        trajectory: Estimated trajectory tensor (optional, for 3D visualization)
        trajectory_gt: Ground truth trajectory tensor (optional, for 3D visualization)
        save_path: Path to save the plots
    """
    F = errors['xyz_errors'].shape[0]
    frames = np.arange(F)
    
    # Set up the style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("deep")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    plt.rcParams['text.color'] = '#333333'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    
    # Create figure layout with GridSpec for more control
    fig = plt.figure(figsize=(20, 22))
    gs = GridSpec(5, 2, figure=fig)
    
    # Position errors
    ax_xyz = fig.add_subplot(gs[0, 0])
    xyz_errors = errors['xyz_errors'].cpu().numpy()
    ax_xyz.plot(frames, xyz_errors[:, 0], '-', linewidth=2, label='X Error')
    ax_xyz.plot(frames, xyz_errors[:, 1], '-', linewidth=2, label='Y Error')
    ax_xyz.plot(frames, xyz_errors[:, 2], '-', linewidth=2, label='Z Error')
    ax_xyz.set_title('Position Errors Over Time', fontsize=16, fontweight='bold')
    ax_xyz.set_xlabel('Frame', fontsize=12)
    ax_xyz.set_ylabel('Error (units)', fontsize=12)
    ax_xyz.legend(fontsize=10)
    ax_xyz.grid(True, alpha=0.3)
    
    # Orientation errors
    ax_rpy = fig.add_subplot(gs[0, 1])
    rpy_errors = errors['rpy_errors'].cpu().numpy()
    ax_rpy.plot(frames, rpy_errors[:, 0], '-', linewidth=2, label='Roll Error')
    ax_rpy.plot(frames, rpy_errors[:, 1], '-', linewidth=2, label='Pitch Error')
    ax_rpy.plot(frames, rpy_errors[:, 2], '-', linewidth=2, label='Yaw Error')
    ax_rpy.set_title('Orientation Errors Over Time', fontsize=16, fontweight='bold')
    ax_rpy.set_xlabel('Frame', fontsize=12)
    ax_rpy.set_ylabel('Error (degrees)', fontsize=12)
    ax_rpy.legend(fontsize=10)
    ax_rpy.grid(True, alpha=0.3)
    
    # Combined translation and rotation errors
    ax_trans_rot = fig.add_subplot(gs[1, 0])
    trans_errors = errors['translation_errors'].cpu().numpy()
    rot_errors = errors['rotation_errors'].cpu().numpy()
    ax_trans_rot.plot(frames, trans_errors, '-', linewidth=2, label='Translation Error')
    ax_trans_rot_twin = ax_trans_rot.twinx()
    ax_trans_rot_twin.plot(frames, rot_errors, '-', linewidth=2, color='orange', label='Rotation Error')
    
    # Adding legends for both axes
    lines1, labels1 = ax_trans_rot.get_legend_handles_labels()
    lines2, labels2 = ax_trans_rot_twin.get_legend_handles_labels()
    ax_trans_rot.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    
    ax_trans_rot.set_title('Translation and Rotation Errors', fontsize=16, fontweight='bold')
    ax_trans_rot.set_xlabel('Frame', fontsize=12)
    ax_trans_rot.set_ylabel('Translation Error (units)', fontsize=12)
    ax_trans_rot_twin.set_ylabel('Rotation Error (degrees)', fontsize=12)
    ax_trans_rot.yaxis.label.set_color('blue')
    ax_trans_rot_twin.yaxis.label.set_color('orange')
    ax_trans_rot.grid(True, alpha=0.3)
    
    # Cumulative errors
    ax_cum = fig.add_subplot(gs[1, 1])
    cum_trans_errors = errors['cum_translation_errors'].cpu().numpy()
    cum_rot_errors = errors['cum_rotation_errors'].cpu().numpy()
    ax_cum.plot(frames, cum_trans_errors, '-', linewidth=2, label='Cum. Translation Error')
    ax_cum_twin = ax_cum.twinx()
    ax_cum_twin.plot(frames, cum_rot_errors, '-', linewidth=2, color='orange', label='Cum. Rotation Error')
    
    # Adding legends for both axes
    lines1, labels1 = ax_cum.get_legend_handles_labels()
    lines2, labels2 = ax_cum_twin.get_legend_handles_labels()
    ax_cum.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    
    ax_cum.set_title('Cumulative Errors Over Time', fontsize=16, fontweight='bold')
    ax_cum.set_xlabel('Frame', fontsize=12)
    ax_cum.set_ylabel('Cum. Translation Error (units)', fontsize=12)
    ax_cum_twin.set_ylabel('Cum. Rotation Error (degrees)', fontsize=12)
    ax_cum.yaxis.label.set_color('blue')
    ax_cum_twin.yaxis.label.set_color('orange')
    ax_cum.grid(True, alpha=0.3)
    
    # Additional metrics: RPE and ATE
    ax_rpe_ate = fig.add_subplot(gs[2, 0])
    rpe = errors['relative_pose_errors'].cpu().numpy()
    ate = errors['absolute_trajectory_error'].cpu().numpy()
    ax_rpe_ate.plot(frames, rpe, '-', linewidth=2, label='Relative Pose Error (RPE)')
    ax_rpe_ate.plot(frames, ate, '-', linewidth=2, label='Absolute Trajectory Error (ATE)')
    ax_rpe_ate.set_title('RPE and ATE Over Time', fontsize=16, fontweight='bold')
    ax_rpe_ate.set_xlabel('Frame', fontsize=12)
    ax_rpe_ate.set_ylabel('Error', fontsize=12)
    ax_rpe_ate.legend(fontsize=10)
    ax_rpe_ate.grid(True, alpha=0.3)
    
    # Drift metrics
    ax_drift = fig.add_subplot(gs[2, 1])
    drift_distance = errors['drift_per_distance'].cpu().numpy()
    drift_time = errors['drift_per_time'].cpu().numpy()
    # Skip the first few frames for drift_per_time to avoid division by near-zero
    start_idx = 5
    ax_drift.plot(frames[start_idx:], drift_distance[start_idx:], '-', linewidth=2, label='Drift per Distance (m/m)')
    ax_drift.plot(frames[start_idx:], drift_time[start_idx:], '-', linewidth=2, label='Drift per Time (m/frame)')
    ax_drift.set_title('Drift Analysis', fontsize=16, fontweight='bold')
    ax_drift.set_xlabel('Frame', fontsize=12)
    ax_drift.set_ylabel('Drift Ratio', fontsize=12)
    ax_drift.legend(fontsize=10)
    ax_drift.grid(True, alpha=0.3)
    
    # Error distribution histograms
    ax_hist_trans = fig.add_subplot(gs[3, 0])
    sns.histplot(trans_errors, kde=True, ax=ax_hist_trans, bins=30, color='blue')
    ax_hist_trans.set_title('Translation Error Distribution', fontsize=16, fontweight='bold')
    ax_hist_trans.set_xlabel('Translation Error (units)', fontsize=12)
    ax_hist_trans.set_ylabel('Frequency', fontsize=12)
    
    ax_hist_rot = fig.add_subplot(gs[3, 1])
    sns.histplot(rot_errors, kde=True, ax=ax_hist_rot, bins=30, color='orange')
    ax_hist_rot.set_title('Rotation Error Distribution', fontsize=16, fontweight='bold')
    ax_hist_rot.set_xlabel('Rotation Error (degrees)', fontsize=12)
    ax_hist_rot.set_ylabel('Frequency', fontsize=12)
    
    # 3D Trajectory Visualization if trajectories are provided
    if trajectory is not None and trajectory_gt is not None:
        ax_3d = fig.add_subplot(gs[4, :], projection='3d')
        
        # Extract positions from trajectories
        traj_positions = trajectory[:, :3, 3].cpu().numpy()
        traj_gt_positions = trajectory_gt[:, :3, 3].cpu().numpy()
        
        # Plot the trajectories
        ax_3d.plot(traj_positions[:, 0], traj_positions[:, 1], traj_positions[:, 2], 
                  '-', linewidth=2, label='Estimated Trajectory')
        ax_3d.plot(traj_gt_positions[:, 0], traj_gt_positions[:, 1], traj_gt_positions[:, 2], 
                  '-', linewidth=2, label='Ground Truth Trajectory')
        
        # Add keyframes with markers
        keyframe_interval = max(1, F // 20)  # Show about 20 keyframes
        ax_3d.scatter(traj_positions[::keyframe_interval, 0], 
                     traj_positions[::keyframe_interval, 1], 
                     traj_positions[::keyframe_interval, 2], 
                     marker='o', s=30, label='Estimated Keyframes')
        ax_3d.scatter(traj_gt_positions[::keyframe_interval, 0], 
                     traj_gt_positions[::keyframe_interval, 1], 
                     traj_gt_positions[::keyframe_interval, 2], 
                     marker='^', s=30, label='GT Keyframes')
        
        ax_3d.set_title('3D Trajectory Comparison', fontsize=16, fontweight='bold')
        ax_3d.set_xlabel('X (units)', fontsize=12)
        ax_3d.set_ylabel('Y (units)', fontsize=12)
        ax_3d.set_zlabel('Z (units)', fontsize=12)
        ax_3d.legend(fontsize=10)
        
        # Equal aspect ratio
        max_range = np.array([
            traj_positions[:, 0].max() - traj_positions[:, 0].min(),
            traj_positions[:, 1].max() - traj_positions[:, 1].min(),
            traj_positions[:, 2].max() - traj_positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (traj_positions[:, 0].max() + traj_positions[:, 0].min()) * 0.5
        mid_y = (traj_positions[:, 1].max() + traj_positions[:, 1].min()) * 0.5
        mid_z = (traj_positions[:, 2].max() + traj_positions[:, 2].min()) * 0.5
        
        ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        # Create a heatmap of errors over time if no trajectories for 3D visualization
        ax_heatmap = fig.add_subplot(gs[4, :])
        
        # Prepare data for heatmap
        heatmap_data = np.vstack([
            xyz_errors[:, 0], xyz_errors[:, 1], xyz_errors[:, 2],
            rpy_errors[:, 0], rpy_errors[:, 1], rpy_errors[:, 2]
        ])
        
        # Create heatmap
        sns.heatmap(
            heatmap_data, 
            ax=ax_heatmap, 
            cmap='viridis', 
            robust=True,
            yticklabels=['X Error', 'Y Error', 'Z Error', 'Roll Error', 'Pitch Error', 'Yaw Error']
        )
        
        ax_heatmap.set_title('Error Heatmap Over Time', fontsize=16, fontweight='bold')
        ax_heatmap.set_xlabel('Frame', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Create a separate summary figure with key metrics
    fig_summary = plt.figure(figsize=(12, 8))
    
    # Extract summary data
    mean_trans_error = np.mean(trans_errors)
    mean_rot_error = np.mean(rot_errors)
    final_cum_trans = cum_trans_errors[-1]
    final_cum_rot = cum_rot_errors[-1]
    
    # RMSE calculation
    rmse_trans = np.sqrt(np.mean(np.square(trans_errors)))
    rmse_rot = np.sqrt(np.mean(np.square(rot_errors)))
    
    # Create a table-like visualization
    plt.axis('off')
    plt.title('Trajectory Error Summary', fontsize=20, fontweight='bold', pad=20)
    
    metrics = [
        "Mean Translation Error (units)",
        "Mean Rotation Error (degrees)",
        "Translation RMSE (units)",
        "Rotation RMSE (degrees)",
        "Final Cumulative Translation Error (units)",
        "Final Cumulative Rotation Error (degrees)",
        "Mean Drift per Distance (m/m)",
        "Mean Drift per Time (m/frame)"
    ]
    
    values = [
        f"{mean_trans_error:.4f}",
        f"{mean_rot_error:.4f}",
        f"{rmse_trans:.4f}",
        f"{rmse_rot:.4f}",
        f"{final_cum_trans:.4f}",
        f"{final_cum_rot:.4f}",
        f"{np.mean(drift_distance[drift_distance > 0]):.4f}",
        f"{np.mean(drift_time[drift_time > 0]):.4f}"
    ]
    
    cell_text = [[metric, value] for metric, value in zip(metrics, values)]
    
    # Add the table
    table = plt.table(
        cellText=cell_text,
        colWidths=[0.6, 0.3],
        loc='center',
        cellLoc='left'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color the header row
    for (i, j), cell in table.get_celld().items():
        if j == 0:
            cell.set_text_props(fontweight='bold')
    
    if save_path:
        summary_path = save_path.replace('.png', '_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_error_statistics(stats):
    """
    Print summary statistics in a readable format.
    
    Args:
        stats: Dictionary containing error statistics
    """
    print("\n===== TRAJECTORY ERROR STATISTICS =====\n")
    
    print("--- Position Errors (X, Y, Z) ---")
    print(f"Mean: {stats['xyz_errors_mean'].cpu().numpy()}")
    print(f"Std Dev: {stats['xyz_errors_std'].cpu().numpy()}")
    print(f"Max: {stats['xyz_errors_max'].cpu().numpy()}")
    print(f"Min: {stats['xyz_errors_min'].cpu().numpy()}")
    
    print("\n--- Orientation Errors (Roll, Pitch, Yaw) ---")
    print(f"Mean: {stats['rpy_errors_mean'].cpu().numpy()}")
    print(f"Std Dev: {stats['rpy_errors_std'].cpu().numpy()}")
    print(f"Max: {stats['rpy_errors_max'].cpu().numpy()}")
    print(f"Min: {stats['rpy_errors_min'].cpu().numpy()}")
    
    print("\n--- Translation Error ---")
    print(f"Mean: {stats['translation_errors_mean'].item():.4f}")
    print(f"Std Dev: {stats['translation_errors_std'].item():.4f}")
    print(f"Max: {stats['translation_errors_max'].item():.4f}")
    print(f"Min: {stats['translation_errors_min'].item():.4f}")
    
    print("\n--- Rotation Error ---")
    print(f"Mean: {stats['rotation_errors_mean'].item():.4f}")
    print(f"Std Dev: {stats['rotation_errors_std'].item():.4f}")
    print(f"Max: {stats['rotation_errors_max'].item():.4f}")
    print(f"Min: {stats['rotation_errors_min'].item():.4f}")
    
    print("\n--- Cumulative Errors (Final) ---")
    print(f"Translation: {stats['final_cum_translation_error'].item():.4f}")
    print(f"Rotation: {stats['final_cum_rotation_error'].item():.4f}")
    
    print("\n=======================================")
