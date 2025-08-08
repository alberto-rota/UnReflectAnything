# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

from typing import Tuple, Union
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw
import torch


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
