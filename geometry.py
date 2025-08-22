import torch


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
    rotation = euler[:, 3:]  # Convert to radians

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

    # Calculate Euler angles
    pitch = torch.atan2(-r31, torch.sqrt(r11**2 + r21**2))
    yaw = torch.atan2(r21, r11)
    roll = torch.atan2(r32, r33)

    # Convert to degrees
    euler_angles = torch.stack([roll, pitch, yaw], dim=-1)

    # Combine translation and rotation
    result = torch.cat([translation, euler_angles], dim=-1)

    # Remove batch dimension if input was unbatched
    if original_ndim == 2:
        result = result.squeeze(0)

    return result


def Tdist(T1: torch.Tensor, T2: torch.Tensor, angle_mode: str = "radians"):
    """
    Computes the rotation angle and translation distance between two 4x4 transformation matrices.
    Supports batched inputs.

    Args:
        T1: (N, 4, 4) or (4, 4) tensor, first transformation matrix
        T2: (N, 4, 4) or (4, 4) tensor, second transformation matrix
        angle_mode: str, either "radians" or "degrees"

    Returns:
        angle: (N,) or scalar tensor, rotation angle in specified mode
        distance: (N,) or scalar tensor, Euclidean distance between translations
    """

    # Ensure input is a batch
    if T1.dim() == 2:
        T1 = T1.unsqueeze(0)
        T2 = T2.unsqueeze(0)

    # Extract translation components
    t1 = T1[:, :3, 3]  # (N, 3)
    t2 = T2[:, :3, 3]  # (N, 3)

    # Compute Euclidean distance
    distance = torch.norm(t2 - t1, dim=1)  # (N,)

    # Extract rotation components
    R1 = T1[:, :3, :3]  # (N, 3, 3)
    R2 = T2[:, :3, :3]  # (N, 3, 3)

    # Compute relative rotation matrix
    R_rel = R2 @ R1.transpose(-1, -2)  # (N, 3, 3)

    # Compute rotation angle using trace formula: cos(theta) = (trace(R) - 1) / 2
    trace_R = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)  # (N,)
    angle = torch.acos(torch.clamp((trace_R - 1) / 2, -1.0, 1.0))  # (N,)

    # Convert to degrees if required
    if angle_mode == "degrees":
        angle = torch.rad2deg(angle)

    # If input was not batched, return scalars instead of tensors
    if angle.shape[0] == 1:
        return angle.item(), distance.item()
    return angle, distance
