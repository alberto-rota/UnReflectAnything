import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


def visualize_trajectory(
    trajectory,
    ax=None,
    color="b",
    marker_size=5,
    line_width=1,
    show_axis=True,
    title="Camera Trajectory",
):
    """
    Visualize a camera trajectory in 3D.

    Args:
        trajectory: Tensor or numpy array of shape [N, 4, 4] with transformation matrices
        ax: Matplotlib axis to plot on, creates a new one if None
        color: Color for the trajectory line
        marker_size: Size of the keyframe markers
        line_width: Width of the trajectory line
        show_axis: Whether to show the coordinate axis at each keyframe
        title: Plot title

    Returns:
        Matplotlib figure and axis
    """
    # Convert to numpy if tensor
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach().cpu().numpy()

    # Extract positions (translation parts)
    positions = trajectory[:, :3, 3]

    # Create figure if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Plot trajectory
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=color,
        marker="o",
        markersize=marker_size,
        linewidth=line_width,
    )

    # Draw coordinate axes at keyframes if requested
    if show_axis:
        axis_length = 0.05  # Adjust based on trajectory scale
        for pose in trajectory:
            # Origin of the camera coordinate system
            origin = pose[:3, 3]

            # Axes directions
            x_axis = origin + axis_length * pose[:3, 0]
            y_axis = origin + axis_length * pose[:3, 1]
            z_axis = origin + axis_length * pose[:3, 2]

            # Draw axes
            ax.plot(
                [origin[0], x_axis[0]],
                [origin[1], x_axis[1]],
                [origin[2], x_axis[2]],
                "r-",
            )
            ax.plot(
                [origin[0], y_axis[0]],
                [origin[1], y_axis[1]],
                [origin[2], y_axis[2]],
                "g-",
            )
            ax.plot(
                [origin[0], z_axis[0]],
                [origin[1], z_axis[1]],
                [origin[2], z_axis[2]],
                "b-",
            )

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Make axes equal
    set_axes_equal(ax)

    return fig, ax


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale.

    Args:
        ax: Matplotlib 3D axis
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call it a "sphere".
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_keyframes(video_frames, keyframe_indices, ncols=4, figsize=(15, 10)):
    """
    Visualize selected keyframes from a video.

    Args:
        video_frames: Tensor of shape [seq_len, 3, H, W]
        keyframe_indices: List of keyframe indices
        ncols: Number of columns in the grid
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Convert to numpy if tensor
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.detach().cpu().numpy()
        # Rearrange from [seq_len, 3, H, W] to [seq_len, H, W, 3]
        video_frames = np.transpose(video_frames, (0, 2, 3, 1))

    # Calculate number of rows needed
    nrows = (len(keyframe_indices) + ncols - 1) // ncols

    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each keyframe
    for i, idx in enumerate(keyframe_indices):
        if i < len(axes):
            axes[i].imshow(video_frames[idx])
            axes[i].set_title(f"Frame {idx}")
            axes[i].axis("off")

    # Hide unused subplots
    for i in range(len(keyframe_indices), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig


def visualize_matches(
    framestack, source_pixels, target_pixels, batch_idx=None, scores=None, max_pairs=20
):
    """
    Visualize feature matches between frame pairs.

    Args:
        framestack: Tensor of shape [B, 2, 3, H, W]
        source_pixels: Source keypoint coordinates
        target_pixels: Target keypoint coordinates
        batch_idx: Batch indices for the points
        scores: Optional match confidence scores
        max_pairs: Maximum number of match pairs to show

    Returns:
        Visualization image as numpy array
    """
    # Convert to numpy if tensor
    if isinstance(framestack, torch.Tensor):
        framestack = framestack.detach().cpu().numpy()
        # Rearrange from [B, 2, 3, H, W] to [B, 2, H, W, 3]
        framestack = np.transpose(framestack, (0, 1, 3, 4, 2))

    if isinstance(source_pixels, torch.Tensor):
        source_pixels = source_pixels.detach().cpu().numpy()

    if isinstance(target_pixels, torch.Tensor):
        target_pixels = target_pixels.detach().cpu().numpy()

    if batch_idx is None:
        batch_idx = np.zeros(len(source_pixels), dtype=int)
    elif isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.detach().cpu().numpy()

    # Process each batch
    results = []
    for b in range(framestack.shape[0]):
        # Get source and target images
        source_img = framestack[b, 0]
        target_img = framestack[b, 1]

        # Scale images to [0, 255] uint8
        source_img = (source_img * 255).astype(np.uint8)
        target_img = (target_img * 255).astype(np.uint8)

        # Get points for this batch
        batch_mask = batch_idx == b
        src_pts = source_pixels[batch_mask]
        tgt_pts = target_pixels[batch_mask]

        # Limit number of points for visualization
        if len(src_pts) > max_pairs:
            if scores is not None:
                # Select top-scoring matches
                scores_batch = scores[batch_mask].detach().cpu().numpy()
                top_indices = np.argsort(scores_batch)[-max_pairs:]
                src_pts = src_pts[top_indices]
                tgt_pts = tgt_pts[top_indices]
            else:
                # Random selection
                indices = np.random.choice(len(src_pts), max_pairs, replace=False)
                src_pts = src_pts[indices]
                tgt_pts = tgt_pts[indices]

        # Draw matches
        matches_img = draw_matches(source_img, target_img, src_pts, tgt_pts)
        results.append(matches_img)

    return np.stack(results) if len(results) > 1 else results[0]


def draw_matches(img1, img2, kp1, kp2, color=None, thickness=1, radius=4):
    """
    Draw matches between two images side by side.

    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)
        kp1: Keypoints in the first image
        kp2: Keypoints in the second image
        color: Color for the matches
        thickness: Line thickness
        radius: Keypoint radius

    Returns:
        Image with matches drawn
    """
    # Create output image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    height = max(h1, h2)
    width = w1 + w2
    out = np.zeros((height, width, 3), dtype=np.uint8)

    # Copy images
    out[:h1, :w1] = img1[..., :3] if img1.ndim == 3 else np.stack([img1] * 3, axis=-1)
    out[:h2, w1 : w1 + w2] = (
        img2[..., :3] if img2.ndim == 3 else np.stack([img2] * 3, axis=-1)
    )

    # Draw lines between matches
    if color is None:
        color = (0, 255, 0)  # Default: green

    for i in range(len(kp1)):
        # Get keypoint coordinates
        pt1 = (int(kp1[i, 0]), int(kp1[i, 1]))
        pt2 = (int(kp2[i, 0]) + w1, int(kp2[i, 1]))

        # Draw line
        cv2.line(out, pt1, pt2, color, thickness)

        # Draw circles at keypoints
        cv2.circle(out, pt1, radius, color, thickness)
        cv2.circle(out, pt2, radius, color, thickness)

    return out


def convert_trajectory_to_positions(trajectory):
    """
    Convert a trajectory of transformation matrices to positions.

    Args:
        trajectory: Tensor of shape [N, 4, 4] with transformation matrices

    Returns:
        Tensor of shape [N, 3] with 3D positions
    """
    if isinstance(trajectory, list):
        return [traj[:3, 3] for traj in trajectory]
    return trajectory[:, :3, 3]
