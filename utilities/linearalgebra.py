# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

import torch
import geometry


def vec2skew(t):
    """
    Compute the skew-symmetric matrix of a vector t.
    """
    B = t.shape[0]
    t_x = torch.zeros(B, 3, 3, device=t.device)
    t_x[:, 0, 1] = -t[:, 2]
    t_x[:, 0, 2] = t[:, 1]
    t_x[:, 1, 0] = t[:, 2]
    t_x[:, 1, 2] = -t[:, 0]
    t_x[:, 2, 0] = -t[:, 1]
    t_x[:, 2, 1] = t[:, 0]
    return t_x


def skew2vec(skew_matrix):
    """Convert 3x3 skew symmetric matrix to 3D vector"""
    return torch.stack(
        [skew_matrix[..., 2, 1], skew_matrix[..., 0, 2], skew_matrix[..., 1, 0]], dim=-1
    )


def chain(trans, ignore_rotation=False):
    transc = trans.clone()
    if ignore_rotation:
        transc[:, 3:] = 0
    transm = torch.stack([geometry.euler2mat(tr) for tr in transc]).to(transc.device)
    N = transc.shape[0]
    trajm = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(transc.device)
    trajm[0] = transm[0]
    for i in range(1, N):
        trajm[i] = torch.matmul(trajm[i - 1], transm[i])
    traj = torch.stack([geometry.mat2euler(tr) for tr in trajm])
    return traj


def kfchain(relative_pose, keyframe_idx):

    N = relative_pose.shape[0]
    # Convering Eulers to 4x4 Homogeous matrices
    relative_pose = torch.stack([geometry.euler2mat(tr) for tr in relative_pose]).to(
        relative_pose.device
    )
    keyframe_idx[0] = True  # First frame is always a keyframe

    # Initlizing global trajectory as NxIdentity
    global_pose = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(relative_pose.device)
    # First Keyframe pose is Identity (frame 0)

    last_global_pose_keyframe = global_pose[0]
    for i in range(1, N):

        # If i is not a keyframe
        if not keyframe_idx[i]:
            # GLobal pose of i-th frame is expressed wrt the latest keyframe
            global_pose[i] = torch.matmul(last_global_pose_keyframe, relative_pose[i])

        # last_global_pose_keyframe = global_pose[i - 1]
        # If i is a keyframe
        else:
            global_pose[i] = torch.matmul(last_global_pose_keyframe, relative_pose[i])
            # Update the latest global pose to the one of the keyframe
            last_global_pose_keyframe = global_pose[i - 1]

    global_pose_euler = torch.stack([geometry.mat2euler(tr) for tr in global_pose])
    return global_pose_euler


def rigid(source, target, output_format=None):
    """
    Finds the rigid transformation from source to target, handling different input formats.

    Parameters:
    -----------
    source : torch.Tensor
        Source transformation, either as a 4x4 matrix or 6D Euler vector [x, y, z, roll, pitch, yaw].
        Can be batched [B, 4, 4] or [B, 6] or unbatched [4, 4] or [6].
    target : torch.Tensor
        Target transformation, either as a 4x4 matrix or 6D Euler vector.
        Must match the batch dimension of source but format can differ.
    output_format : str, optional
        If provided, overrides the output format. Options are:
        - "matrix": Return as 4x4 matrix
        - "euler": Return as 6D Euler vector
        - None: Return in the same format as source (default)

    Returns:
    --------
    torch.Tensor
        Transformation from source to target (source^(-1) @ target), in the format
        determined by the source format or the output_format parameter.

    Raises:
    -------
    ValueError
        If inputs have inconsistent dimensions.
    """
    # Check if batched
    source_batched = source.dim() >= 2
    target_batched = target.dim() >= 2

    if source_batched != target_batched:
        raise ValueError("Source and target must both be batched or both unbatched")

    # Determine input formats
    source_is_matrix = source.shape[-1] == 4 and source.shape[-2] == 4
    target_is_matrix = target.shape[-1] == 4 and target.shape[-2] == 4

    # Convert to matrices
    if source_is_matrix:
        source_matrix = source
        if target_is_matrix:
            target_matrix = target
        else:
            # Convert target from euler to matrix
            target_matrix = geometry.euler2mat(target)
    else:
        # Convert source from euler to matrix
        source_matrix = geometry.euler2mat(source)
        if target_is_matrix:
            target_matrix = target
        else:
            # Convert target from euler to matrix
            target_matrix = geometry.euler2mat(target)

    # Compute the transformation: source^(-1) @ target
    source_inv = torch.linalg.inv(source_matrix)
    result_matrix = torch.matmul(source_inv, target_matrix)

    # Determine output format
    if output_format is not None:
        if output_format.lower() == "matrix":
            return result_matrix
        elif output_format.lower() == "euler":
            return geometry.mat2euler(result_matrix)
        else:
            raise ValueError("output_format must be 'matrix', 'euler', or None")
    else:
        # Return in the same format as source
        if source_is_matrix:
            return result_matrix
        else:
            return geometry.mat2euler(result_matrix)


def mmult(A, B, output_format=None):
    """
    Multiplies two transformation matrices A and B, handling different input formats.

    Parameters:
    -----------
    A : torch.Tensor
        First transformation, either as a 4x4 matrix or 6D Euler vector [x, y, z, roll, pitch, yaw].
        Can be batched [B, 4, 4] or [B, 6] or unbatched [4, 4] or [6].
    B : torch.Tensor
        Second transformation, either as a 4x4 matrix or 6D Euler vector.
        Must match the batch dimension of A but format can differ.
    output_format : str, optional
        If provided, overrides the output format. Options are:
        - "matrix": Return as 4x4 matrix
        - "euler": Return as 6D Euler vector
        - None: Return in the same format as A (default)

    Returns:
    --------
    torch.Tensor
        Multiplication of A and B (A @ B), in the format determined by
        A's format or the output_format parameter.

    Raises:
    -------
    ValueError
        If inputs have inconsistent dimensions.
    """
    # Check if batched
    A_batched = A.dim() >= 2
    B_batched = B.dim() >= 2

    if A_batched != B_batched:
        raise ValueError("A and B must both be batched or both unbatched")

    # Determine input formats
    A_is_matrix = A.shape[-1] == 4 and A.shape[-2] == 4
    B_is_matrix = B.shape[-1] == 4 and B.shape[-2] == 4

    # Convert to appropriate format
    if A_is_matrix:
        A_matrix = A
        if B_is_matrix:
            B_matrix = B
        else:
            # Convert B from euler to matrix to match A's format
            B_matrix = geometry.euler2mat(B)
    else:
        # A is euler format
        A_matrix = geometry.euler2mat(A)
        if B_is_matrix:
            B_matrix = B
        else:
            # Both are euler format
            B_matrix = geometry.euler2mat(B)

    # Perform matrix multiplication
    result_matrix = torch.matmul(A_matrix, B_matrix)

    # Determine output format
    if output_format is not None:
        if output_format.lower() == "matrix":
            return result_matrix
        elif output_format.lower() == "euler":
            return geometry.mat2euler(result_matrix)
        else:
            raise ValueError("output_format must be 'matrix', 'euler', or None")
    else:
        # Return in the same format as A
        if A_is_matrix:
            return result_matrix
        else:
            return geometry.mat2euler(result_matrix)


def align_trajectories(A, B):
    # Compute centroids
    A = A[:, :3]
    B = B[:, :3]
    centroid_A = torch.mean(A, dim=0)
    centroid_B = torch.mean(B, dim=0)

    # Translate trajectories to origin
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Compute the covariance matrix
    H = A_centered.T @ B_centered

    # Compute the Singular Value Decomposition (SVD)
    U, S, Vt = torch.linalg.svd(H)

    # Compute the optimal rotation
    R = Vt.T @ U.T

    # Ensure a right-handed coordinate system
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply rotation
    A_rotated = A_centered @ R

    # Translate the rotated trajectory back
    A_aligned = A_rotated + centroid_B

    return A_aligned
