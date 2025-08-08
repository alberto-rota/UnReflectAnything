# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from utilities.tensor_utils import TTensor


def euler2axang(euler: torch.Tensor) -> tuple:
    """
    Convert Euler angles to axis-angle representation.

    Parameters:
    euler (torch.Tensor): A 6-element tensor where the first three elements represent
                          translation (x, y, z) and the last three elements represent
                          rotation (roll, pitch, yaw) in degrees.

    Returns:
    tuple: A tuple containing:
        - translation (torch.Tensor): The translation vector [x, y, z].
        - rotation_axis (torch.Tensor): The rotation axis vector [x, y, z].
        - rotation_angle (float): The rotation angle in degrees.
    """
    euler = euler.cpu().numpy()
    translation, rotation = euler[:3], euler[3:]
    rotvec = Rotation.from_euler("xyz", rotation, degrees=True).as_rotvec()
    rotation_angle = np.linalg.norm(rotvec)
    rotation_axis = rotvec / rotation_angle
    return (
        TTensor(translation),
        TTensor(rotation_axis),
        float(np.degrees(rotation_angle)),
    )


def euler2quat(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles to quaternion representation.

    Parameters:
    euler (torch.Tensor): A 6-element tensor where the first three elements represent
                          translation (x, y, z) and the last three elements represent
                          rotation (roll, pitch, yaw) in degrees.

    Returns:
    tuple: A tuple containing:
        - translation (torch.Tensor): The translation vector [x, y, z].
        - quaternion (torch.Tensor): The quaternion [x, y, z, w (scalar-last)].
    """
    euler = euler.cpu().numpy()
    translation, rotation = euler[:3], euler[3:]
    quat = Rotation.from_euler("xyz", rotation, degrees=True).as_quat()
    return torch.cat((TTensor(translation), TTensor(quat)))


def euler2mat(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles to homogeneous rotation matrices.

    Parameters:
    euler (torch.Tensor): A tensor of shape (N, 6) for batched input or (6,) for unbatched input.
                          The first three elements represent translation (x, y, z), and the
                          last three elements represent rotation (roll, pitch, yaw) in degrees.

    Returns:
    torch.Tensor: A tensor of shape (N, 4, 4) for batched input or (4, 4) for unbatched input
                  containing the homogeneous rotation matrices.
    """
    batched = euler.ndim == 2  # Check if batched
    if not batched:
        euler = euler.unsqueeze(0)  # Add batch dimension for consistency

    translation = euler[:, :3]
    rotation = euler[:, 3:]  # * (torch.pi / 180.0)  # Convert to radians

    roll, pitch, yaw = rotation[:, 0], rotation[:, 1], rotation[:, 2]

    # Compute individual rotation matrices
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    # Rotation matrices
    rot_x = torch.stack(
        [
            torch.stack(
                [torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll)],
                dim=-1,
            ),
            torch.stack([torch.zeros_like(roll), cos_r, -sin_r], dim=-1),
            torch.stack([torch.zeros_like(roll), sin_r, cos_r], dim=-1),
        ],
        dim=-2,
    )

    rot_y = torch.stack(
        [
            torch.stack([cos_p, torch.zeros_like(pitch), sin_p], dim=-1),
            torch.stack(
                [
                    torch.zeros_like(pitch),
                    torch.ones_like(pitch),
                    torch.zeros_like(pitch),
                ],
                dim=-1,
            ),
            torch.stack([-sin_p, torch.zeros_like(pitch), cos_p], dim=-1),
        ],
        dim=-2,
    )

    rot_z = torch.stack(
        [
            torch.stack([cos_y, -sin_y, torch.zeros_like(yaw)], dim=-1),
            torch.stack([sin_y, cos_y, torch.zeros_like(yaw)], dim=-1),
            torch.stack(
                [torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)],
                dim=-1,
            ),
        ],
        dim=-2,
    )

    # Combined rotation matrix: Rz * Ry * Rx
    rotation_matrix = rot_z @ rot_y @ rot_x

    # Create homogeneous transformation matrices
    hom_mat = torch.eye(4, dtype=euler.dtype, device=euler.device).repeat(
        euler.shape[0], 1, 1
    )
    hom_mat[:, :3, :3] = rotation_matrix
    hom_mat[:, :3, 3] = translation

    if not batched:
        hom_mat = hom_mat.squeeze(0)  # Remove batch dimension for single input

    return hom_mat


def quat2euler(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to Euler angles.

    Parameters:
    quat (torch.Tensor): A 7-element tensor where the first three elements represent
                          translation (x, y, z) and the last three elements represent a
                          quaternion tensor [x ,y ,z, w (scalar-last)].

    Returns:
    torch.Tensor: A 6-element tensor containing translation [x, y, z] and Euler angles [roll, pitch, yaw] in degrees.
    """
    quat = quat.cpu().numpy()
    translation, rotation = quat[:3], quat[3:]
    euler = Rotation.from_quat(rotation).as_euler("xyz", degrees=True)
    return torch.cat((TTensor(translation), TTensor(euler)))


def quat2mat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to homogeneous rotation matrix.

    Parameters:
    quat (torch.Tensor): A 7-element tensor where the first three elements represent
                          translation (x, y, z) and the last three elements represent a
                          quaternion tensor [x ,y ,z, w (scalar-last)].

    Returns:
    torch.Tensor: The homogeneous rotation matrix [4x4].
    """
    quat = quat.cpu().numpy()
    translation, rotation = quat[:3], quat[3:]
    mat = Rotation.from_quat(rotation).as_matrix()
    hom_mat = np.eye(4)
    hom_mat[:3, :3] = mat
    hom_mat[:3, 3] = translation
    return TTensor(hom_mat)


def quat2axang(quat: torch.Tensor) -> tuple:
    """
    Convert quaternion to axis-angle representation.

    Parameters:
    quat (torch.Tensor): A 7-element tensor where the first three elements represent
                          translation (x, y, z) and the last three elements represent a
                          quaternion tensor [x ,y ,z, w (scalar-last)].

    Returns:
    tuple: A tuple containing:
        - translation (torch.Tensor): The translation vector [x, y, z].
        - rotation_axis (torch.Tensor): The rotation axis vector [x, y, z].
        - rotation_angle (float): The rotation angle in degrees.
    """
    quat = quat.cpu().numpy()
    translation, rotation = quat[:3], quat[3:]
    rotvec = Rotation.from_quat(rotation).as_rotvec()
    rotation_angle = np.linalg.norm(rotvec)
    rotation_axis = rotvec / rotation_angle
    return translation, TTensor(rotation_axis), float(np.degrees(rotation_angle))


def axang2euler(axang: tuple) -> torch.Tensor:
    """
    Convert axis-angle representation to Euler angles.

    Parameters:
    axang (tuple): A tuple containing:
        - translation (torch.Tensor): The translation vector [x, y, z].
        - rotation_axis (torch.Tensor): The rotation axis vector [x, y, z].
        - rotation_angle (float): The rotation angle in degrees.
    translation (torch.Tensor): The translation vector [x, y, z].

    Returns:
    torch.Tensor: A 6-element tensor containing translation [x, y, z] and Euler angles [roll, pitch, yaw] in degrees.
    """
    translation, rotation_axis, rotation_angle = axang
    rotation_axis = rotation_axis.cpu().numpy()
    rotation_angle = np.radians(rotation_angle)
    rotvec = rotation_axis * rotation_angle
    euler = Rotation.from_rotvec(rotvec).as_euler("xyz", degrees=True)
    return torch.cat((TTensor(translation), TTensor(euler)))


def axang2mat(axang: tuple) -> torch.Tensor:
    """
    Convert axis-angle representation to homogeneous rotation matrix.

    Parameters:
    axang (tuple): A tuple containing:
        - translation (torch.Tensor): The translation vector [x, y, z].
        - rotation_axis (torch.Tensor): The rotation axis vector [x, y, z].
        - rotation_angle (float): The rotation angle in degrees.
    translation (torch.Tensor): The translation vector [x, y, z].

    Returns:
    torch.Tensor: The homogeneous rotation matrix [4x4].
    """
    translation, rotation_axis, rotation_angle = axang
    rotation_axis = rotation_axis.cpu().numpy()
    rotvec = rotation_axis * rotation_angle
    mat = Rotation.from_rotvec(rotvec).as_matrix()
    hom_mat = np.eye(4)
    hom_mat[:3, :3] = mat
    hom_mat[:3, 3] = translation.cpu().numpy()
    return TTensor(hom_mat)


def axang2quat(axang: tuple) -> torch.Tensor:
    """
    Convert axis-angle representation to quaternion.

    Parameters:
    axang (tuple): A tuple containing:
        - translation (torch.Tensor): The translation vector [x, y, z].
        - rotation_axis (torch.Tensor): The rotation axis vector [x, y, z].
        - rotation_angle (float): The rotation angle in degrees.
    translation (torch.Tensor): The translation vector [x, y, z].

    Returns:
    tuple: A tuple containing:
        - translation (torch.Tensor): The translation vector [x, y, z].
        - quaternion (torch.Tensor): The quaternion [x ,y ,z, w (scalar-last)].
    """
    translation, rotation_axis, rotation_angle = axang
    rotation_axis = rotation_axis.cpu().numpy()
    rotation_angle = np.radians(rotation_angle)
    rotvec = rotation_axis * rotation_angle
    quat = Rotation.from_rotvec(rotvec).as_quat()
    return torch.cat((TTensor(translation), TTensor(quat)))


def mat2euler(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert homogeneous rotation matrix to Euler angles while maintaining gradients.
    Uses the math from rotation matrix to Euler angles conversion following XYZ convention.
    Supports both batched and unbatched inputs.

    Parameters:
    mat (torch.Tensor): A homogeneous rotation matrix tensor
                       Either [4x4] or [Bx4x4] where B is batch size

    Returns:
    torch.Tensor: Translation and Euler angles in degrees
                 If unbatched: shape [6] containing [x, y, z, roll, pitch, yaw]
                 If batched: shape [Bx6] containing B sets of [x, y, z, roll, pitch, yaw]
    """
    # Handle unbatched input by adding a batch dimension
    original_ndim = mat.ndim
    if original_ndim == 2:
        mat = mat.unsqueeze(0)

    # Extract rotation matrix [Bx3x3] and translation vector [Bx3]
    rotation_mat = mat[..., :3, :3]
    translation = mat[..., :3, 3]

    # Extract the components needed for conversion
    r11, r12, r13 = (
        rotation_mat[..., 0, 0],
        rotation_mat[..., 0, 1],
        rotation_mat[..., 0, 2],
    )
    r21, r22, r23 = (
        rotation_mat[..., 1, 0],
        rotation_mat[..., 1, 1],
        rotation_mat[..., 1, 2],
    )
    r31, r32, r33 = (
        rotation_mat[..., 2, 0],
        rotation_mat[..., 2, 1],
        rotation_mat[..., 2, 2],
    )

    # Calculate pitch (y-axis rotation)
    # Handle singularity when pitch = ±90°
    pitch = torch.asin(torch.clamp(r13, min=-1.0, max=1.0))

    # Calculate yaw (z-axis rotation) and roll (x-axis rotation)
    cos_pitch = torch.cos(pitch)

    # Threshold for detecting gimbal lock
    thresh = 1e-6

    # Create a mask for gimbal lock cases
    gimbal_lock = torch.abs(cos_pitch) < thresh

    # Regular case (no gimbal lock)
    yaw = torch.where(
        ~gimbal_lock,
        torch.atan2(-r12, r11),
        torch.atan2(r21, r22),  # Arbitrary choice at gimbal lock
    )

    roll = torch.where(
        ~gimbal_lock,
        torch.atan2(-r23, r33),
        torch.zeros_like(pitch),  # At gimbal lock, roll is arbitrary, set to 0
    )

    # Convert to degrees
    euler_angles = torch.stack([roll, pitch, yaw], dim=-1)

    # Combine translation and rotation
    result = torch.cat([translation, euler_angles], dim=-1)

    # Remove batch dimension if input was unbatched
    if original_ndim == 2:
        result = result.squeeze(0)

    return result


def mat2quat(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert homogeneous rotation matrix to quaternion.

    Parameters:
    mat (torch.Tensor): A homogeneous rotation matrix tensor [4x4].

    Returns:
    tuple: A tuple containing:
        - translation (torch.Tensor): The translation vector [x, y, z].
        - quaternion (torch.Tensor): The quaternion [x ,y ,z, w (scalar-last)].
    """
    if isinstance(mat, torch.Tensor):
        mat = mat.cpu().numpy()
    rotation_mat = mat[:3, :3]
    translation = mat[:3, 3]
    quat = Rotation.from_matrix(rotation_mat).as_quat()
    return torch.cat((TTensor(translation), TTensor(quat)))


def mat2axang(mat: torch.Tensor) -> tuple:
    """
    Convert homogeneous rotation matrix to axis-angle representation.

    Parameters:
    mat (torch.Tensor): A homogeneous rotation matrix tensor [4x4].

    Returns:
    tuple: A tuple containing:
        - translation (torch.Tensor): The translation vector [x, y, z].
        - rotation_axis (torch.Tensor): The rotation axis vector [x, y, z].
        - rotation_angle (float): The rotation angle in degrees.
    """
    if isinstance(mat, torch.Tensor):
        mat = mat.cpu().numpy()
    rotation_mat = mat[:3, :3]
    translation = mat[:3, 3]
    rotvec = Rotation.from_matrix(rotation_mat).as_rotvec()
    rotation_angle = np.linalg.norm(rotvec)

    # Handle the case where rotation angle is approximately zero
    if rotation_angle < 1e-10:  # Threshold for considering angle as zero
        # For zero rotation, the axis is arbitrary but should be normalized
        # Using the rotation vector itself maintains any numerical tendencies
        # in the original matrix while avoiding division by near-zero
        rotation_axis = np.array([1.0, 0.0, 0.0])  # Could also use np.zeros(3)
        rotation_angke = 0.0
    else:
        # Normal case: normalize the rotation vector to get the axis
        rotation_axis = rotvec / rotation_angle

    return (TTensor(translation), TTensor(rotation_axis), rotation_angle)


import geometry


def compute_global_pose(
    prev_pose, current_transformation, is_keyframe, last_keyframe_pose
):
    """
    Compute the global pose based on the previous pose and current transformation.

    Args:
        prev_pose (torch.Tensor): The pose matrix from the previous frame (4x4)
        current_transformation (torch.Tensor): The current local transformation in euler angles (6-dim)
        is_keyframe (bool): Boolean indicating if the current frame is a keyframe
        last_keyframe_pose (torch.Tensor): The pose of the last keyframe (4x4)

    Returns:
        tuple: (updated_global_pose, updated_keyframe_pose)
            - updated_global_pose (torch.Tensor): The updated global pose matrix (4x4)
            - updated_keyframe_pose (torch.Tensor): The updated keyframe reference pose (4x4)
    """
    # Convert euler angles to transformation matrix
    transformation_matrix = geometry.euler2mat(current_transformation)

    if not is_keyframe:
        # Standard frames - relative to last keyframe
        updated_global_pose = torch.matmul(last_keyframe_pose, transformation_matrix)
        return updated_global_pose, last_keyframe_pose
    else:
        # Keyframes - update reference poses
        new_keyframe_pose = (
            prev_pose  # The previous pose becomes our new keyframe reference
        )
        updated_global_pose = torch.matmul(new_keyframe_pose, transformation_matrix)
        return updated_global_pose, new_keyframe_pose
