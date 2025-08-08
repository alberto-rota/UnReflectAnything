# -------------------------------------------------------------------------------------------------#

""" Copyright (c) 2024 Asensus Surgical """

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

# Standard library imports
import io
import json
import math
import os
import re
import socket
import subprocess
from contextlib import redirect_stdout
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import cv2 as cv
import lovely_tensors as lt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paramiko
import rich.traceback
from PIL import Image, ImageDraw
from rich import print as nativeprint
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation

# PyTorch imports
import torch
import torchvision
from torch import Tensor

# Configuration
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
rich.traceback.install()


def get_hostname() -> str:

    hostname = socket.gethostname()
    if "alberto-vm" in hostname:
        return "AsensusVM"
    elif "rk018445" in hostname:
        return "MUFASA"


def highlight_image_patch(
    image: torch.Tensor,
    patch_size: int,
    row_idx: int,
    col_idx: int,
    color: str = "green",
    width: int = 2,
) -> Image:
    """
    Highlights a specific patch in an image tensor by drawing a red rectangle around it.

    Args:
        image (torch.Tensor): The input image tensor with shape [C, H, W].
        patch_size (int): The size of each patch (default is 14).
        row_idx (int): The row index of the patch to be highlighted.
        col_idx (int): The column index of the patch to be highlighted.

    Returns:
        Image: A PIL Image with the specified patch highlighted.
    """
    # Convert tensor to numpy array and transpose to HWC format (Height, Width, Channels)
    image_np = image.numpy().transpose(1, 2, 0)  # --> [H, W, C]
    img_h, img_w, _ = image_np.shape  # Image height and width

    # Calculate number of patches in both dimensions
    num_patches_x = img_w // patch_size
    num_patches_y = img_h // patch_size

    # Create a blank canvas for the mosaic with the same size as the original image
    mosaic = Image.new("RGB", (img_w, img_h))

    # Iterate through each patch
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Extract the patch from the numpy array
            patch = image_np[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
                :,
            ]  # --> [patch_size, patch_size, C]
            patch_img = Image.fromarray(
                (patch * 255).astype(np.uint8)
            )  # Convert the patch to a PIL Image

            # Paste the patch into the mosaic at the correct position
            mosaic.paste(patch_img, (j * patch_size, i * patch_size))

    # Highlight the selected patch by drawing a red rectangle around it
    draw = ImageDraw.Draw(mosaic)
    x0, y0 = col_idx * patch_size, row_idx * patch_size  # Top-left corner of the patch
    x1, y1 = x0 + patch_size, y0 + patch_size  # Bottom-right corner of the patch
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)  # Draw the rectangle

    return mosaic


def highlight_pixel_region(
    image: torch.Tensor,
    patch_size: int,
    pixel_x: int,
    pixel_y: int,
    color: str = "green",
    width: int = 2,
) -> Image.Image:
    """
    Highlights a square region centered around specific pixel coordinates in an image.

    Args:
        image (torch.Tensor): Input image tensor with shape [C, H, W]
        patch_size (int): Size of the square region to highlight
        pixel_x (int): X coordinate of the center pixel
        pixel_y (int): Y coordinate of the center pixel
        color (str, optional): Color of the highlight rectangle. Defaults to "green"
        width (int, optional): Width of the highlight rectangle border. Defaults to 2

    Returns:
        Image.Image: PIL Image with the region around the specified pixel highlighted

    Note:
        If the region would extend beyond image boundaries, the highlight rectangle
        is clipped to fit within the image.
    """
    # Convert tensor to numpy array and transpose to HWC format
    image_np = image.numpy().transpose(1, 2, 0)  # [H, W, C]
    img_h, img_w, _ = image_np.shape

    # Calculate the region boundaries centered on the pixel
    half_size = patch_size // 2

    # Ensure pixel coordinates are integers
    pixel_x, pixel_y = int(pixel_y), int(pixel_x)
    # Calculate rectangle coordinates, ensuring they stay within image bounds
    x0 = max(0, pixel_x - half_size)
    y0 = max(0, pixel_y - half_size)
    x1 = min(img_w, pixel_x + half_size)
    y1 = min(img_h, pixel_y + half_size)

    # Create PIL Image from numpy array
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

    # Draw the highlight rectangle
    draw = ImageDraw.Draw(image_pil)
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

    # Optionally, draw a small cross at the center pixel for precision
    cross_size = 2
    draw.line(
        [(pixel_x - cross_size, pixel_y), (pixel_x + cross_size, pixel_y)],
        fill=color,
        width=1,
    )
    draw.line(
        [(pixel_x, pixel_y - cross_size), (pixel_x, pixel_y + cross_size)],
        fill=color,
        width=1,
    )

    return image_pil


def channels(t: torch.Tensor, **kwargs: Any) -> None:
    """
    Display tensor channels using lovely_tensors.

    Args:
        t (torch.Tensor): The tensor to display.
        **kwargs (Any): Additional keyword arguments passed to lt.chans().
    """
    # Normalize
    t = (t - t.mean()) / t.std()

    # Display the tensor channels using lovely_tensors
    display(lt.chans(t, **kwargs))


def rgb(t: torch.Tensor, **kwargs: Any) -> None:
    """
    Display tensor as RGB image using lovely_tensors.

    Args:
        t (torch.Tensor): The tensor to display.
        **kwargs (Any): Additional keyword arguments passed to lt.rgb().
    """
    # Display the tensor as an RGB image using lovely_tensors
    if len(t.shape) == 4:
        t = panelize(t)
    if len(t.shape) == 3 and t.shape[0] == 1:
        t = plasma(t.repeat(3, 1, 1))
    if len(t.shape) == 2:
        t = t.unsqueeze(0)
        t = plasma(t.repeat(3, 1, 1))
    t = (t - t.min()) / (t.max() - t.min())
    display(lt.rgb(t, **kwargs))


def tprint(args, shape=False, dtype=False, device=False, grad_fn=False, **kwargs):

    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    output = []
    np.set_printoptions(precision=4, suppress=True)

    def tensor_to_string(tensor):
        return str(tensor.cpu().detach().numpy())

    for arg in args:
        if isinstance(arg, torch.Tensor):

            infos = ""
            if shape:
                infos += f"Shape: {tuple(arg.shape)}"
            if dtype:
                infos += f"Dtype {str(arg.dtype).split('torch.')[1]}"
            if device:
                infos += f"Device: {arg.device}"
            if grad_fn:
                infos += (
                    f"Grad_fn: {arg.grad_fn}" if arg.grad_fn is not None else "NOGRAD"
                )
            if shape or dtype or device or grad_fn:
                infos += "\n"
            infos += tensor_to_string(arg)

            output.append(infos)
        elif (isinstance(arg, list) or isinstance(arg, tuple)) and all(
            isinstance(x, torch.Tensor) for x in arg
        ):
            print(f"{len(arg)} elements:", [x.shape for x in arg])
        else:
            output.append(str(arg))
    nativeprint(sep.join(output), end=end)


def print(*args: Any, **kwargs: Any) -> None:
    """
    Custom print function to handle both tensors and regular objects.

    If the argument is a torch.Tensor, use lovely_tensors to print.
    Otherwise, use the rich print function.

    Args:
        *args (Any): The arguments to print.
        **kwargs (Any): Additional keyword arguments passed to the print function.
    """
    # Check if any argument is a torch.Tensor
    if any(isinstance(arg, torch.Tensor) for arg in args):
        tprint(args, **kwargs)
    else:
        # Use the original print function for non-tensor arguments
        nativeprint(*args, **kwargs)


def TTensor(obj: object) -> torch.Tensor:
    """
    Converts an object to a torch.Tensor if it is not already one.

    Args:
        obj (object): The input object to be converted to a torch.Tensor.

    Returns:
        torch.Tensor: The input object converted to a torch.Tensor, or the original
                      object if it is already a torch.Tensor.
    """
    if not isinstance(obj, torch.Tensor):
        # Convert the object to a torch.Tensor if it is not already a tensor
        return torch.Tensor(obj)
    # Return the object as is if it is already a tensor
    return obj


def hwc(image: torch.Tensor) -> torch.Tensor:
    """
    Reshapes a sensor image from CxHxW to HxWxC.

    Args:
        image (torch.Tensor): Input CxHxW image.

    Returns:
        torch.Tensor: Output HxWxC image.
    """
    return image.permute(2, 1, 0)


def tp(tensor: torch.Tensor) -> np.ndarray:
    """
    Transposes and.cpu().detach()es a tensor from channels-first to channels-last format.

    Args:
        tensor (torch.Tensor): Input tensor in channels-first format.

    Returns:
        np.ndarray: Output tensor in channels-last format.
    """
    return tensor.cpu().detach().permute(1, 2, 0).numpy()


def sp(size: tuple) -> tuple:
    """
    Converts a size tuple to a tuple of the same elements.

    Args:
        size (tuple): Input size tuple.

    Returns:
        tuple: Output size tuple.
    """
    return tuple(size)


def show(tensor: torch.Tensor) -> None:
    """
    Displays a tensor as an image using matplotlib.

    Args:
        tensor (torch.Tensor): Input tensor representing an image.
    """
    with torch.no_grad():
        plt.imshow(tp(tensor))
        plt.show()


def estrinsics(R: np.ndarray, t: np.ndarray) -> torch.Tensor:
    """
    Constructs a 4x4 extrinsics matrix from a rotation matrix (R) and translation vector (t).

    Args:
        R (np.ndarray): Rotation matrix.
        t (np.ndarray): Translation vector.

    Returns:
        torch.Tensor: 4x4 extrinsics matrix.
    """
    E = np.zeros((4, 4))
    E[0:3, 0:3] = R
    E[0:3, 3] = t
    E[3, 3] = 1
    return TTensor(E).float()


def intrinsics(M: np.ndarray, DOWNSAMPLE: int = 1) -> torch.Tensor:
    """
    Constructs a 4x4 intrinsics matrix from a camera matrix (M) and an optional downsampling factor.

    Args:
        M (np.ndarray): Camera matrix.
        DOWNSAMPLE (int, optional): Downsampling factor. Defaults to 1.

    Returns:
        torch.Tensor: 4x4 intrinsics matrix.
    """
    I = np.zeros((4, 4))
    I[0:3, 0:3] = M
    I[3, 3] = 1
    # Accounting for image resizing
    I[0, 0] = I[0, 0] * DOWNSAMPLE
    I[1, 1] = I[1, 1] * DOWNSAMPLE
    I[0, 2] = I[0, 2] * DOWNSAMPLE
    I[1, 2] = I[1, 2] * DOWNSAMPLE
    return TTensor(I).float()


def cvcalib_fromyaml(path: str = "assets/camera/endoscope_calibration.yaml") -> dict:
    """
    Reads camera calibration parameters from a YAML file and returns them as a dictionary.

    Args:
        path (str, optional): Path to the YAML file. Defaults to "endoscope_calibration.yaml".

    Returns:
        dict: Dictionary containing calibration parameters.
    """
    fs = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    return {
        "R": TTensor(fs.getNode("R").mat()).float(),
        "T": TTensor(fs.getNode("T").mat()).float(),
        "D1": TTensor(fs.getNode("D1").mat()).float(),
        "M1": TTensor(fs.getNode("M1").mat()).float(),
        "M2": TTensor(fs.getNode("M2").mat()).float(),
    }


def Tplot(ax: plt.Axes, T: np.ndarray) -> None:
    """
    Plots a transformation matrix as arrows in a 3D plot.

    Args:
        ax (plt.Axes): A matplotlib 3D axis.
        T (np.ndarray): 4x4 transformation matrix.
    """
    X, Y, Z = T[:3, 0], T[:3, 1], T[:3, 2]
    x, y, z = T[0, -1], T[1, -1], T[2, -1]

    ax.quiver(x, y, z, X[0], X[1], X[2], color="r", normalize=True)
    ax.quiver(x, y, z, Y[0], Y[1], Y[2], color="g", normalize=True)
    ax.quiver(x, y, z, Z[0], Z[1], Z[2], color="b", normalize=True)


def fig3d() -> plt.Axes:
    """
    Creates a 3D plot figure.

    Returns:
        plt.Axes: 3D plot axis.
    """
    return plt.figure().add_subplot(projection="3d")


def set_axes_equal(ax: plt.Axes) -> None:
    """
    Makes axes of a 3D plot have equal scale.

    Args:
        ax (plt.Axes): A matplotlib 3D axis.
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

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def depth_sine(h: int, w: int, f: int = 20) -> torch.Tensor:
    """
    Generates a depth image as a 2D sine wave.

    Args:
        h (int): Height of the depth image.
        w (int): Width of the depth image.
        f (int, optional): Frequency of the sine wave. Defaults to 20.

    Returns:
        torch.Tensor: Generated depth image.
    """
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y, sparse=True)
    if f is None:
        f = w / 10

    z = 2 * np.sin(xx / (w / f)) + 2 * np.sin(yy / (h / f)) + 8
    z = np.expand_dims(z, 0)
    return torch.from_numpy(z).float()


### ROTATION TRANSFORMATION FUNCTIONS ###


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
    Convert Euler angles to homogeneous rotation matrix.

    Parameters:
    euler (torch.Tensor): A 6-element tensor where the first three elements represent
                          translation (x, y, z) and the last three elements represent
                          rotation (roll, pitch, yaw) in degrees.

    Returns:
    torch.Tensor: The homogeneous rotation matrix [4x4].
    """
    euler = euler.cpu().numpy()
    translation, rotation = euler[:3], euler[3:]
    mat = Rotation.from_euler("xyz", rotation, degrees=True).as_matrix()
    hom_mat = np.eye(4)
    hom_mat[:3, :3] = mat
    hom_mat[:3, 3] = translation
    return TTensor(hom_mat)


def euler2mat_attached(euler: torch.Tensor) -> torch.Tensor:
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
    rotation = euler[:, 3:] * (torch.pi / 180.0)  # Convert to radians

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
    rotation_angle = np.radians(rotation_angle)
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


def mat2euler_attached(mat: torch.Tensor) -> torch.Tensor:
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


def mat2euler(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert homogeneous rotation matrix to Euler angles.

    Parameters:
    mat (torch.Tensor): A homogeneous rotation matrix tensor [4x4].

    Returns:
    torch.Tensor: A 6-element tensor containing translation [x, y, z] and Euler angles [roll, pitch, yaw] in degrees.
    """
    if isinstance(mat, torch.Tensor):
        mat = mat.cpu().numpy()
    rotation_mat = mat[:3, :3]
    translation = mat[:3, 3]
    euler = Rotation.from_matrix(rotation_mat).as_euler("xyz", degrees=True)
    return torch.cat((TTensor(translation), TTensor(euler)))


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
    rotation_axis = rotvec / rotation_angle
    return (
        TTensor(translation),
        TTensor(rotation_axis),
        float(np.degrees(rotation_angle)),
    )


#########################################################


def posrot2v6(pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """
    Converts position and rotation tensors into a 6-vector.

    Args:
        pos (torch.Tensor): Position tensor.
        rot (torch.Tensor): Rotation tensor.

    Returns:
        torch.Tensor: 6-vector.
    """
    return torch.cat([pos, rot], dim=0)


def scatterpptk(cloud: torch.Tensor, rgb_vec: torch.Tensor = None) -> None:
    """
    Displays a 3D point cloud using the pptk viewer.

    Args:
        cloud (torch.Tensor): Input point cloud.
        rgb_vec (torch.Tensor, optional): RGB color information for the point cloud. Defaults to None.
    """
    rgbcloud = pptk.viewer(cloud[:3, :].cpu().detach()().permute(1, 0))
    if rgb_vec is not None:
        rgbcloud.attributes(rgb_vec.permute(1, 0))
    rgbcloud.set(show_axis=True)


def scattero3d(cloud: torch.Tensor, rgb_vec: torch.Tensor = None) -> None:
    """
    Displays a 3D point cloud using the Open3D viewer.

    Args:
        cloud (torch.Tensor): Input point cloud.
        rgb_vec (torch.Tensor, optional): RGB color information for the point cloud. Defaults to None.
    """
    pass


def imshow_batch(batch: torch.Tensor) -> None:
    """
    Display a batch of images using matplotlib.

    Args:
    batch (torch.Tensor): A batch of images.

    Returns:
    None
    """
    imggrid = torchvision.utils.make_grid(batch)
    # Create figure of aspect ratio equal to the batch size
    plt.figure(figsize=(20, 20 * batch.shape[0]))
    plt.imshow(tp(imggrid))
    plt.show()
    pass


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of dictionaries into a dictionary of tensors.

    Args:
    batch (List[Dict]): List of dictionaries containing tensors.

    Returns:
    Dict[str, torch.Tensor]: A dictionary containing tensors stacked along the 0th dimension.
    """
    return {key: torch.stack([d[key] for d in batch]) for key in batch[0]}


def plasma(depthmap: torch.Tensor) -> torch.Tensor:
    """
    Apply the plasma colormap to a depth map.

    Args:
    depthmap (torch.Tensor): A tensor representing the depth map.

    Returns:
    torch.Tensor: A tensor representing the depth map with the plasma colormap applied.
    """
    # Convert torch tensor to numpy
    if len(depthmap.shape) == 2:
        depthmap = depthmap.unsqueeze(0)
    if len(depthmap.shape) == 3:
        depthmap = (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min())
        depthmap_np = depthmap.permute(2, 1, 0).cpu().detach().numpy()[..., 0]
        # Apply plasma colormap
        depthmap_np_plasma = plt.get_cmap("magma")(depthmap_np)[:, :, :3]
        # Convert back to torch tensor
        depthmap_rgb = torch.from_numpy(depthmap_np_plasma).permute(2, 1, 0)
    if len(depthmap.shape) == 4:
        depthmap = (depthmap - depthmap.min(dim=1).min(dim=2).min(dim=3)) / (
            depthmap.max(dim=1).min(dim=2).min(dim=3)
            - depthmap.min(dim=1).min(dim=2).min(dim=3)
        )
        depthmap_np = depthmap.permute(0, 2, 3, 1).cpu().detach().numpy()[..., 0]
        # Apply plasma colormap
        depthmap_np_plasma = plt.get_cmap("magma")(depthmap_np)[:, :, :, :3]
        # Convert back to torch tensor
        depthmap_rgb = torch.from_numpy(depthmap_np_plasma).permute(0, 3, 2, 1)
    return depthmap_rgb


def dinotransform(height, width) -> torchvision.transforms.Compose:
    """
    Define the preprocessing transformation for DINOv2.

    Returns:
    torchvision.transforms.Compose: A composition of torchvision transformations.
    """
    assert height % 16 == 0 and width % 16 == 0
    return torchvision.transforms.Compose(
        [
            lambda x: 255.0 * x,  # Discard alpha component and scale by 255
            torchvision.transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
            torchvision.transforms.Resize((height, width)),
        ]
    )


def dinotransform_inv(inp="rgbvec") -> torchvision.transforms.Compose:
    """
    Define the postprocessing transformation for DINOv2

    Returns:
    torchvision.transforms.Compose: A composition of inverse torchvision transformations.
    """
    if inp == "image":
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((512, 640)),
                torchvision.transforms.Normalize(
                    mean=(0.0, 0.0, 0.0), std=(1 / 58.395, 1 / 57.12, 1 / 57.375)
                ),
                torchvision.transforms.Normalize(
                    mean=(-123.675, -116.28, -103.53), std=(1.0, 1.0, 1.0)
                ),
                lambda x: 1 / 255.0 * x,
            ]
        )
    elif inp == "rbgvec":
        return torchvision.transforms.Compose(
            [
                # torchvision.transforms.Resize((512,640)),
                torchvision.transforms.Normalize(
                    mean=(0.0, 0.0, 0.0), std=(1 / 58.395, 1 / 57.12, 1 / 57.375)
                ),
                torchvision.transforms.Normalize(
                    mean=(-123.675, -116.28, -103.53), std=(1.0, 1.0, 1.0)
                ),
                lambda x: 1 / 255.0 * x,
            ]
        )


def titlescreen() -> None:
    """
    Prints the title screen from a text file.
    """
    with open("assets/banner_giant.txt", "r") as f:
        print(f"[white]{f.read()}[/white]")


def coloredbar(parts: list, colors: list, num_blocks: int) -> str:
    """
    Creates a colored bar using unicode block characters.


    Args:
        parts (list): A list of numbers representing the size of each part of the bar.
        colors (list): A list of colors for each part of the bar.
        num_blocks (int): The total number of blocks in the bar.

    Returns:
        str: A string representing the colored bar.
    """
    assert len(parts) == len(colors)
    total = sum(parts)
    block = "\u25A0"
    bar = ""
    for part, color in zip(parts, colors):
        num_part_blocks = round(part / total * num_blocks)
        bar += f"[{color}]{block * num_part_blocks}[/]"
    return bar


def millify(n: float) -> str:
    """
    Converts a number into a string with a suffix that indicates its scale (thousands, millions, etc.)

    Parameters:
    n (int, float): The number to be converted.

    Returns:
    str: The converted string.
    """
    millnames = ["", " Th", " M", " B", " T"]

    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )

    return "{:.1f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


def gt2string(tensor: torch.Tensor) -> str:
    """
    Converts a tensor into a string representation.

    Parameters:
    tensor (torch.Tensor): The tensor to be converted.

    Returns:
    str: The string representation of the tensor.
    """
    append = ["X", "Y", "Z", "R", "P", "Y"]
    return " ".join([f"{a}:{x.item():+.4f}  " for x, a in zip(tensor, append)])


def pred2string(pos: torch.Tensor, rot: torch.Tensor) -> str:
    """
    Converts position and rotation tensors into a string representation.

    Parameters:
    pos (torch.Tensor): The position tensor to be converted.
    rot (torch.Tensor): The rotation tensor to be converted.

    Returns:
    str: The string representation of the position and rotation tensors.
    """
    tensor = torch.cat([pos, rot], dim=0)
    append = ["X", "Y", "Z", "R", "P", "Y"]
    return " ".join([f"{a}:{x.item():+.4f}  " for x, a in zip(tensor, append)])


def asymptote(y_data):
    # Define the exponential function
    def exponential_function(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Fit the data to the exponential function
    y_data = np.array(y_data)
    x_data = np.arange(len(y_data))
    try:
        params, covariance = curve_fit(exponential_function, x_data, y_data)
    except:
        return None
    # Extract the parameters
    a_fit, b_fit, c_fit = params

    # The horizontal asymptote is given by y = c
    horizontal_asymptote = c_fit

    return horizontal_asymptote


def improvement(y_data):
    # Fit the data to the exponential function
    y_data = np.array(y_data)
    x_data = np.arange(len(y_data))

    # Calculate the improvement
    first_value = y_data[0]
    last_value = y_data[-1]
    improvement = ((last_value - first_value) / first_value) * 100

    return improvement


def RdGr(value):
    # Ensure the value is between 0 and 1
    value = max(0.0, min(1.0, value))

    # Map the value to a red (0) to green (1) gradient
    red = int(255 * (1 - value))
    green = int(255 * value)

    # Create a hex color code
    color = f"#{red:02x}{green:02x}00"

    # Create the text object with the specified color
    return f"[{color}]{(value*100):.2f}[/{color}]"


def perc_req_grad(model):
    return (
        sum(param.requires_grad for param in model.parameters())
        / len(list(model.parameters()))
        * 100
    )


def perc_grad_finite(model):
    return (
        sum(
            param.grad is not None and torch.isfinite(param.grad).all()
            for param in model.parameters()
            if param.requires_grad
        )
        / sum(param.requires_grad for param in model.parameters())
        * 100
    )


def detect_aval_cpus():
    """
    Detects the number of available CPUs.
    """
    try:
        currentjobid = os.environ["SLURM_JOB_ID"]
        currentjobid = int(currentjobid)
        command = f"squeue --Format=JobID,cpus-per-task | grep {currentjobid}"
        # Run the command as a subprocess and capture the output
        output = subprocess.check_output(command, shell=True)[5:-4].replace(b" ", b"")
        cpus = output.decode("utf-8")
        cpus2 = len(os.sched_getaffinity(0))
        cpus = min(int(cpus), cpus2)
    except:
        cpus = 1  # os.cpu_count()
    return cpus


def greenred(values):
    """
    Assigns a color to each position in an array of integers based on a green-to-red colorscale.

    Args:
    - values (np.ndarray): An array of integers.

    Returns:
    - colors (np.ndarray): An array of colors corresponding to the input values, in RGB format.
    """
    min_val = 0
    values = values.cpu().numpy()
    max_val = np.max(values) / 2
    normalized = np.clip(
        (
            (values - min_val) / (max_val - min_val)
            if max_val != min_val
            else np.zeros_like(values)
        ),
        0,
        1,
    )

    # Initialize the color array
    colors = np.zeros((len(values), 3))

    # Assign colors based on the normalized value
    # Green to Red: (0,1,0) to (1,0,0), capping values at red_sat to red
    colors[:, 0] = normalized  # Red channel increases with value
    colors[:, 1] = 1 - normalized  # Green channel decreases with value
    colors[:, 2] = 0  # Blue channel is always 0

    return TTensor(colors)


def copy_layer(layer):
    # Get the class of the layer
    layer_class = type(layer)
    # Retrieve the __init__ method signature
    init_signature = inspect.signature(layer_class.__init__)
    # Extract the names of the __init__ parameters (excluding 'self')
    init_param_names = list(init_signature.parameters.keys())[1:]

    # Build a dictionary of parameter names and values
    init_kwargs = {
        name: getattr(layer, name) for name in init_param_names if hasattr(layer, name)
    }

    # Create a new instance of the layer class with the extracted parameters
    layer_copy = layer_class(**init_kwargs)

    return layer_copy


def chain(trans, ignore_rotation=False):
    transc = trans.clone()
    if ignore_rotation:
        transc[:, 3:] = 0
    transm = torch.stack([euler2mat(tr) for tr in transc]).to(transc.device)
    N = transc.shape[0]
    trajm = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(transc.device)
    trajm[0] = transm[0]
    for i in range(1, N):
        trajm[i] = torch.matmul(trajm[i - 1], transm[i])
    traj = torch.stack([mat2euler(tr) for tr in trajm])
    return traj


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


from math import log, floor, ceil


def embedding2chw(embedding: torch.Tensor, embed_dim_last=True) -> torch.Tensor:
    """
    Reorganizes the embedding output of DINOv2 into CHW form.

    Args:
        embedding (torch.Tensor): Input embedding tensor of shape (N, D).

    Returns:
        torch.Tensor: Tensor of shape (D, H, W).
    """
    # Validate input tensor shape
    if len(embedding.shape) == 2:
        embedding = embedding.unsqueeze(0)

    if embed_dim_last == False:
        embedding = embedding.permute(0, 2, 1)
    B, N, D = embedding.shape

    # Calculate height and width from sequence length
    side_length = int(N**0.5)

    if side_length * side_length != N:
        raise ValueError("The sequence length must be a perfect square")

    # Reshape the embedding from (B,N D) to (B,H, W, D)
    chw_tensor = embedding.view(B, side_length, side_length, D).permute(0, 3, 1, 2)

    return chw_tensor


def closest_multiple(number: int, base: int, mode: str = "closest") -> int:
    """
    Find the multiple of `base` closest to `number`, with options for ceiling or floor.

    Parameters:
    - number (int): The target number to find the closest multiple for.
    - base (int): The base multiple to use.
    - mode (str): Determines the method for finding the closest multiple.
      Can be "closest" for the nearest multiple, "inf" for the floor (next lower multiple),
      or "sup" for the ceiling (next higher multiple). Default is "closest".

    Returns:
    int: The closest multiple of `base` to `number` according to the selected mode.
    """
    if mode == "inf":
        return (number // base) * base
    elif mode == "sup":
        return ((number + base - 1) // base) * base
    else:  # mode == "closest"
        lower_multiple = (number // base) * base
        upper_multiple = lower_multiple + base
        return (
            lower_multiple
            if (number - lower_multiple) <= (upper_multiple - number)
            else upper_multiple
        )


def closest_power(number: float, base: int, mode: str = "closest") -> float:
    """
    Find the power of `base` closest to `number`, with options for ceiling or floor.

    Parameters:
    - number (float): The target number to approximate with a power of `base`.
    - base (int): The base for the exponentiation.
    - mode (str): Determines the method for finding the closest power.
      Can be "closest" for the nearest power, "inf" for the floor (next lower power),
      or "sup" for the ceiling (next higher power). Default is "closest".

    Returns:
    float: The closest power of `base` to `number` according to the selected mode.
    """
    if mode == "inf":
        return base ** floor(log(number, base))
    elif mode == "sup":
        return base ** ceil(log(number, base))
    else:  # mode == "closest"
        lower_power = base ** floor(log(number, base))
        upper_power = base ** ceil(log(number, base))
        return (
            lower_power
            if (number - lower_power) <= (upper_power - number)
            else upper_power
        )


def dup_indexes(arr):
    unique, counts = torch.unique(arr, return_counts=True)
    duplicates = unique[counts > 1]
    return [torch.where(arr == dup)[0].tolist() for dup in duplicates]


def hessian_trace(model, loss_fn, data, target):
    """
    Estimate the trace of the Hessian matrix for a given model and loss function.

    This function estimates the trace of the Hessian matrix of the loss function
    with respect to the model's parameters. It uses Hutchinson's estimator, which
    is an efficient way to estimate the trace without computing the full Hessian.
    The method is useful for analyzing the curvature of the loss surface.

    Parameters:
    - model (torch.nn.Module): The model for which the Hessian trace is computed.
    - loss_fn (callable): The loss function used during the model's training.
    - data (torch.Tensor): The input data batch.
    - target (torch.Tensor): The target outputs for the given input data.

    Returns:
    - float: The estimated trace of the Hessian matrix.

    """

    # Reset gradients to zero to avoid accumulation from previous operations
    model.zero_grad()

    # Forward pass: compute the model's output given the input data
    output = model(data)

    # Compute the loss using the model's output and the target values
    loss = loss_fn(output, target)

    # Filter model parameters to only those that require gradients
    params_with_grad = [p for p in model.parameters() if p.requires_grad]

    # Compute gradients of the loss with respect to model parameters
    grads = torch.autograd.grad(loss, params_with_grad, create_graph=True)

    # Flatten the gradients to a single vector (grad_vector)
    grad_vector = torch.cat([grad.reshape(-1) for grad in grads if grad is not None])

    # Generate a random vector (v) with the same shape as grad_vector
    v = torch.randn(grad_vector.shape, device=grad_vector.device)

    # Compute the Hessian-vector product (Hv) for the gradient vector
    Hv = torch.autograd.grad(grad_vector @ v, params_with_grad)

    # Flatten the Hessian-vector product to a single vector (Hv_vector)
    Hv_vector = torch.cat([hv.reshape(-1) for hv in Hv]).detach()

    # Estimate the trace of the Hessian using Hutchinson's estimator
    trace_estimate = v @ Hv_vector

    # Return the estimated trace as a scalar value
    return trace_estimate.item()


def spatial_attention_maps(
    attention_output: torch.Tensor, reference_patch: int = 0, patch_size: int = 14
) -> torch.Tensor:
    """
    Generate spatial attention maps from the given attention output.

    Parameters:
    attention_output (torch.Tensor): The output tensor from the attention layer with shape [batch_size, num_heads, num_patches, num_patches].
    reference_patch (int): The reference patch index to use for extracting attention. Default is 0.
    patch_size (int): The size to which the attention map patches will be upsampled. Default is 14.

    Returns:
    torch.Tensor: The upsampled attention maps with shape [num_heads, patch_size * h_featmap, patch_size * w_featmap].
    """
    # Extract attention for the reference patch and reshape
    attentions = attention_output[reference_patch, 1:]

    # For example, if attention_output shape is [batch_size, num_heads, num_patches, num_patches]
    # and if num_patches is square of some integer then
    w_featmap = h_featmap = int((attention_output.shape[-1] - 1) ** 0.5)

    # Reshape attentions to spatial dimensions
    attentions = attentions.reshape(
        nh, w_featmap, h_featmap
    )  # --> [num_heads, h_featmap, w_featmap]

    # Upsample the attention maps to the desired patch size
    attentions = nn.functional.interpolate(
        attentions.unsqueeze(0), scale_factor=patch_size, mode="bicubic"
    )[0]
    # --> [num_heads, patch_size * h_featmap, patch_size * w_featmap]

    return attentions


def get_image_patch_from_idx(
    image: np.ndarray, idx: Union[int, Tuple[int, int]], patch_size: int
) -> np.ndarray:
    """
    Extracts a patch from the given image based on the specified index.

    Parameters:
    image (np.ndarray): The input image from which the patch will be extracted.
    idx (Union[int, Tuple[int, int]]): The index of the patch. Can be an integer or a tuple of integers.
        - If an integer, it is interpreted as a linear index.
        - If a tuple, it is interpreted as (row, column) index.
    patch_size (int): The size of the patch to be extracted (patch will be of size patch_size x patch_size).

    Returns:
    np.ndarray: The extracted image patch.
    """
    if isinstance(idx, int):
        h, w = image.shape[-2:]  # Extract the height and width of the image
        y = idx // (w // patch_size)  # Calculate the row index
        x = idx % (h // patch_size)  # Calculate the column index
        y0 = y * patch_size  # Calculate the starting row coordinate of the patch
        y1 = (y + 1) * patch_size  # Calculate the ending row coordinate of the patch
        x0 = x * patch_size  # Calculate the starting column coordinate of the patch
        x1 = (x + 1) * patch_size  # Calculate the ending column coordinate of the patch
    elif isinstance(idx, tuple):
        y, x = idx  # Unpack the tuple into row and column indices
        y0 = y * patch_size  # Calculate the starting row coordinate of the patch
        y1 = (y + 1) * patch_size  # Calculate the ending row coordinate of the patch
        x0 = x * patch_size  # Calculate the starting column coordinate of the patch
        x1 = (x + 1) * patch_size  # Calculate the ending column coordinate of the patch
    else:
        raise ValueError("idx must be either an integer or a tuple of integers")

    return image[..., y0:y1, x0:x1]  # Extract and return the patch from the image


def sftp_transfer(local_file_path: str, remote_file_path: str) -> None:
    """
    Transfers a file from a local directory to a remote SFTP server.

    Args:
        local_file_path (str): The path to the local file to be transferred.
        remote_file_path (str): The destination path on the remote SFTP server.

    Returns:
        None
    """

    # Load the SFTP credentials from a JSON file
    try:
        with open("/home/alberto/MONO3D/secrets/keys.json") as f:
            sftp_credentials = json.load(f)

        # Extract SFTP credentials
        sftp_host = sftp_credentials["sftp_host"]
        sftp_port = sftp_credentials["sftp_port"]
        sftp_username = sftp_credentials["sftp_username"]
        sftp_password = sftp_credentials["sftp_password"]

        # Initialize the SFTP client using the provided credentials
        transport = paramiko.Transport((sftp_host, sftp_port))
        transport.connect(username=sftp_username, password=sftp_password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Upload the file from the local path to the remote path
        sftp.put(local_file_path, remote_file_path)

        # Close the SFTP connection
        sftp.close()
        transport.close()
        print(f" [green]-->OK-->[/green] Saved to NAS:/{remote_file_path}")

    except Exception as e:
        print(f" [red]-->ERROR-->[/red] Cound not upload file to NAS : ", end="")
        print(e)
        return


def sampleinspect(sample: tuple):
    """
    Visualizes a sample from the dataset, including the source and target frames, and prints out the transformation.

    Optionally, an index can be specified. If not provided, a random sample is chosen.

    Args:
        idx (int, optional): The index of the sample to inspect. Defaults to None.
    """
    # Choose a random index if none is provided

    # Unpack the framestack and Ts2t from the sample

    if len(sample) == 2:
        framestack, Ts2t = sample[0], sample[1]
    else:
        framestack, Ts2t, paths = sample[0], sample[1], sample[2]

    if len(framestack.shape) == 5:
        framestack = framestack[0]
    if len(Ts2t.shape) == 2:
        Ts2t = Ts2t[0]

    fstack = torch.cat([framestack[0], framestack[1]], dim=-1)
    # Assuming source and target are the first and last in the framestack respectively
    if len(sample) == 3:
        paths = sample[2]
        print(f"Paths: \n-{paths[0]}\n-{paths[1]}")
    # print(f"Frameskip: {self.frameskip}")
    # Create a subplot for source and target
    rgb(fstack)

    # Print transformation information
    print(f"Transformation:")
    tprint(Ts2t.cpu().numpy())

    # Interpret and print the directional information based on the transformation vector
    direction = [
        "OBJ Right / CAM Left" if Ts2t[0] < 0 else "OBJ Left / CAM Right",
        "OBJ Down / CAM UP" if Ts2t[1] < 0 else "OBJ Up / CAM Down",
        "OBJ Farther / CAM Backward" if Ts2t[2] < 0 else "OBJ Closer / CAM Forward",
    ]
    print(" ".join(direction))
    translation, rax, rang = euler2axang(Ts2t)
    print(f"L\th: {translation.norm():.2f} mm")
    print(f"Rotation: {rang:.2f} deg")


def tensor2wandb(tensor):
    return tensor.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()


def panelize(
    *images: Union[Tensor, Image.Image],
    mode: str = "horizontal",
    grid_size: Optional[Tuple[int, int]] = None,
    output_type: str = "tensor",
) -> Union[Tensor, Image.Image]:
    """
    Combine multiple images into a panel horizontally, vertically, or in a grid.

    Args:
        images: Input images (torch tensors or PIL images)
        mode: 'horizontal', 'vertical', or 'grid'
        grid_size: (rows, cols) for grid mode
        output_type: 'tensor' or 'pil'

    Returns:
        Combined image as tensor or PIL image
    """

    # Convert PIL images to tensors if needed
    def to_tensor(img):
        if isinstance(img, Image.Image):
            return (
                torch.from_numpy(np.array(img)).permute(2, 0, 1).detach().cpu() / 255.0
            )
        return img.detach().cpu()

    tensors = [to_tensor(img) for img in images]

    if mode == "horizontal":
        # Ensure same height
        max_height = max(t.shape[1] for t in tensors)
        padded = [
            torch.nn.functional.pad(t, (0, 0, 0, max_height - t.shape[1]))
            for t in tensors
        ]
        result = torch.cat(padded, dim=2)

    elif mode == "vertical":
        # Ensure same width
        max_width = max(t.shape[2] for t in tensors)
        padded = [
            torch.nn.functional.pad(t, (0, max_width - t.shape[2], 0, 0))
            for t in tensors
        ]
        result = torch.cat(padded, dim=1)

    else:  # grid
        if not grid_size:
            # Auto-calculate grid size
            n = len(tensors)
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            grid_size = (rows, cols)

        max_height = max(t.shape[1] for t in tensors)
        max_width = max(t.shape[2] for t in tensors)

        # Pad all images to max size
        padded = [
            torch.nn.functional.pad(
                t, (0, max_width - t.shape[2], 0, max_height - t.shape[1])
            )
            for t in tensors
        ]

        # Pad with empty images if needed
        while len(padded) < grid_size[0] * grid_size[1]:
            padded.append(torch.zeros_like(padded[0]))

        # Create grid
        rows = []
        for i in range(0, len(padded), grid_size[1]):
            rows.append(torch.cat(padded[i : i + grid_size[1]], dim=2))
        result = torch.cat(rows, dim=1)

    if output_type == "pil":
        result *= 255
        result = Image.fromarray(result.permute(1, 2, 0).numpy().astype("uint8"))

    return result


def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)


def check_rerun_output(func, *args, **kwargs):
    # Create a buffer to capture output
    buffer = io.StringIO()

    # Run the function while capturing its output
    with redirect_stdout(buffer):
        func(*args, **kwargs)

    # Get the captured output
    output = buffer.getvalue()

    # Define the pattern to search for
    pattern = r"WARN  re_sdk_comms"

    # Check if the output matches the pattern
    if re.search(pattern, output):
        print(
            "[RERUN] >> [orange3]Cannot communicate with server. Will not log[/orange3]"
        )
        return False
    else:
        # If no pattern match, print the original output
        print("[RERUN] >> [green]Connection established[/green]")
        return True


def overlay(image, heatmap, alpha=0.4, cmap="plasma"):
    """
    Overlay a heatmap on an image with a given 5opacity.

    :param image: Input image as a numpy array or PIL Image.
    :param heatmap: Heatmap as a numpy array.
    :param alpha: Opacity of the heatmap overlay (between 0 and 1).
    :param cmap: Color map to use for the heatmap.
    :return: The overlaid image as a numpy array.
    """
    # Ensure image is a numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Normalize the heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Apply colormap to the heatmap
    colored_heatmap = plt.get_cmap(cmap)(heatmap)[:, :, :3]  # Take RGB channels
    # Resize heatmap to match the image size if necessary
    if heatmap.shape[:2] != image.shape[:2]:
        colored_heatmap = np.array(
            Image.fromarray((colored_heatmap * 255).astype(np.uint8)).resize(
                (image.shape[1], image.shape[2])
            )
        )
        colored_heatmap = colored_heatmap / 255.0
    # Overlay the heatmap on the image
    overlayed_image = (1 - alpha) * image.permute(1, 2, 0) + alpha * colored_heatmap
    # overlayed_image = image.permute(1,2,0) * colored_heatmap.mean(axis=2, keepdims=True)
    overlayed_image = np.clip(overlayed_image * 255, 0, 255)
    return overlayed_image.permute(2, 0, 1)


def embedding2chw(embedding: torch.Tensor) -> torch.Tensor:
    """
    Reorganizes the embedding output of DINOv2 into CHW form.

    Args:
        embedding (torch.Tensor): Input embedding tensor of shape (N, D).

    Returns:
        torch.Tensor: Tensor of shape (D, H, W).
    """
    # Validate input tensor shape
    if len(embedding.shape) == 2:
        embedding = embedding.unsqueeze(0)

    B, N, D = embedding.shape

    # Calculate height and width from sequence length
    side_length = int(N**0.5)

    if side_length * side_length != N:
        raise ValueError("The sequence length must be a perfect square")

    # Reshape the embedding from (B,N D) to (B,H, W, D)
    chw_tensor = embedding.view(B, side_length, side_length, D).permute(0, 3, 1, 2)

    return chw_tensor


def resizeTransform(height=384, width=384) -> torchvision.transforms.Compose:
    """
    Define the postprocessing transformation for DINOv2

    Returns:
    torchvision.transforms.Compose: A composition of inverse torchvision transformations.
    """
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((height, width), antialias=True),
        ]
    )


# Compute the skew-symmetric matrix of t for each batch
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


def viewPixelsMatches(
    img1: Union[Image.Image, torch.Tensor],
    img2: Union[Image.Image, torch.Tensor],
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    scores: torch.Tensor,
    topk: int = 20,
) -> Image.Image:

    def to_pil(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.dim() == 3:
                img = img.permute(1, 2, 0)
            return Image.fromarray(img.numpy().astype("uint8"))
        return img

    img1, img2 = to_pil(img1 * 255), to_pil(img2 * 255)

    # Select evenly spaced points
    n = len(scores)
    stride = max(1, n // topk)
    selected_indices = torch.arange(0, n, stride)[:topk]

    pts1, pts2 = pts1[selected_indices], pts2[selected_indices]
    scores = scores[selected_indices]

    # Calculate patch size based on image dimensions
    w1, h1 = img1.size
    w2, h2 = img2.size
    patch_size = int(
        min(w1, h1) * 2 / topk
    )  # Reasonable default relative to image size
    half_patch = patch_size // 2

    # Create canvas with space for patches
    h = max(h1, h2) + 2 * patch_size  # Add space for two rows of patches
    w = w1 + w2
    canvas = Image.new("RGB", (w, h))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (w1, 0))

    # Draw matches
    draw = ImageDraw.Draw(canvas)
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min())

    # Extract and draw patches
    for i, ((x1, y1), (x2, y2), score) in enumerate(zip(pts1, pts2, norm_scores)):
        # Draw match line
        color = (int(255 * (1 - score)), int(255 * score), 0)
        draw.line([x1, y1, x2 + w1, y2], fill=color, width=2)

        # Extract and paste patches
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # First image patch
        left1 = max(0, x1 - half_patch)
        top1 = max(0, y1 - half_patch)
        patch1 = img1.crop(
            (left1, top1, min(w1, left1 + patch_size), min(h1, top1 + patch_size))
        )
        patch_x = i * patch_size
        canvas.paste(patch1, (patch_x, h - 2 * patch_size))

        # Second image patch
        left2 = max(0, x2 - half_patch)
        top2 = max(0, y2 - half_patch)
        patch2 = img2.crop(
            (left2, top2, min(w2, left2 + patch_size), min(h2, top2 + patch_size))
        )
        canvas.paste(patch2, (patch_x, h - patch_size))

        # Draw boxes around patches
        draw.rectangle(
            [patch_x, h - 2 * patch_size, patch_x + patch_size, h - patch_size],
            outline=color,
            width=2,
        )
        draw.rectangle(
            [patch_x, h - patch_size, patch_x + patch_size, h], outline=color, width=2
        )

    return canvas


def viewPatchMatches(
    img1: Union[Image.Image, torch.Tensor],
    img2: Union[Image.Image, torch.Tensor],
    similarity_matrix: torch.Tensor,
    topk: int = 20,
) -> Image.Image:
    def to_pil(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.dim() == 3:
                img = img.permute(1, 2, 0)
            return Image.fromarray(img.numpy().astype("uint8"))
        return img

    img1, img2 = to_pil(img1 * 255), to_pil(img2 * 255)
    w1, h1 = img1.size
    w2, h2 = img2.size

    num_patches = similarity_matrix.shape[0]
    patch_size_h = h1 // int(math.sqrt(num_patches))
    patch_size_w = w1 // int(math.sqrt(num_patches))

    # Select evenly spaced indices
    n = similarity_matrix.shape[0]
    stride = max(1, n // topk)
    src_indices = torch.arange(0, n, stride)[:topk]

    # Get best match for each selected source patch
    values = torch.zeros(topk, dtype=torch.float32)
    tgt_indices = torch.zeros(topk, dtype=torch.int64)
    for i, src_idx in enumerate(src_indices):
        values[i], tgt_indices[i] = similarity_matrix[src_idx].max(dim=0)

    patches_per_row = w1 // patch_size_w

    def get_patch_center(idx):
        row = int(idx // patches_per_row)
        col = int(idx % patches_per_row)
        return (
            col * patch_size_w + patch_size_w // 2,
            row * patch_size_h + patch_size_h // 2,
        )

    w = w1 + w2
    display_patch_size = w // topk  # Width for bottom display patches
    h = max(h1, h2) + 2 * display_patch_size
    canvas = Image.new("RGB", (w, h))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (w1, 0))

    draw = ImageDraw.Draw(canvas)
    norm_scores = (values - values.min()) / (values.max() - values.min())

    display_patch_size = w // topk  # Width for bottom display patches
    h = max(h1, h2) + 2 * display_patch_size

    for i, (src_idx, tgt_idx, score) in enumerate(
        zip(src_indices, tgt_indices, norm_scores)
    ):
        src_x, src_y = get_patch_center(src_idx)
        tgt_x, tgt_y = get_patch_center(tgt_idx)

        color = (int(255 * (1 - score)), int(255 * score), 0)
        draw.line([src_x, src_y, tgt_x + w1, tgt_y], fill=color, width=2)

        draw.rectangle(
            [
                src_x - patch_size_w // 2,
                src_y - patch_size_h // 2,
                src_x + patch_size_w // 2,
                src_y + patch_size_h // 2,
            ],
            outline=color,
            width=2,
        )
        draw.rectangle(
            [
                tgt_x + w1 - patch_size_w // 2,
                tgt_y - patch_size_h // 2,
                tgt_x + w1 + patch_size_w // 2,
                tgt_y + patch_size_h // 2,
            ],
            outline=color,
            width=2,
        )

        # Extract patches
        src_patch = img1.crop(
            (
                src_x - patch_size_w // 2,
                src_y - patch_size_h // 2,
                src_x + patch_size_w // 2,
                src_y + patch_size_h // 2,
            )
        )
        tgt_patch = img2.crop(
            (
                tgt_x - patch_size_w // 2,
                tgt_y - patch_size_h // 2,
                tgt_x + patch_size_w // 2,
                tgt_y + patch_size_h // 2,
            )
        )

        # Resize patches to fill width
        src_patch = src_patch.resize(
            (display_patch_size, display_patch_size), resample=Image.NEAREST
        )
        tgt_patch = tgt_patch.resize(
            (display_patch_size, display_patch_size), resample=Image.NEAREST
        )

        patch_x = i * display_patch_size
        canvas.paste(src_patch, (patch_x, h - 2 * display_patch_size))
        canvas.paste(tgt_patch, (patch_x, h - display_patch_size))

        draw.rectangle(
            [
                patch_x,
                h - 2 * display_patch_size,
                patch_x + display_patch_size,
                h - display_patch_size,
            ],
            outline=color,
            width=2,
        )
        draw.rectangle(
            [patch_x, h - display_patch_size, patch_x + display_patch_size, h],
            outline=color,
            width=2,
        )

    return canvas
