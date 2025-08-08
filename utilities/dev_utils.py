# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

import math
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torch
from sklearn.decomposition import PCA
import numpy as np
import inspect
from .tensor_utils import *
from google.cloud import storage
import tempfile
import os


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


def tensor2wandb(tensor):
    return tensor.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()


def panelize(
    *images: Union[Tensor, Image.Image],
    mode: str = "horizontal",
    grid_size: Optional[Tuple[int, int]] = None,
    output_type: Optional[str] = "tensor",
    resize_to_match: Optional[bool] = True,
) -> Union[Tensor, Image.Image]:
    """
    Combine multiple images into a panel horizontally, vertically, or in a grid.

    Args:
        images: Input images (torch tensors or PIL images)
        mode: 'horizontal', 'vertical', or 'grid'
        grid_size: (rows, cols) for grid mode
        output_type: 'tensor' or 'pil'
        resize_to_match: If True, resize images to match the smallest dimension
                        in the concatenation direction

    Returns:
        Combined image as tensor or PIL image
    """

    # Convert PIL images to tensors if needed
    def to_tensor(img):
        if isinstance(img, Image.Image):
            return (
                torch.from_numpy(np.array(img.convert("RGBA")))
                .permute(2, 0, 1)
                .detach()
                .cpu()
                / 255.0
            )
        return img.detach().cpu()

    tensors = [to_tensor(img) for img in images]
    # tensors = [t.unsqueeze(0) if len(t.shape) == 3 else t for t in tensors]
    # print(tensors[0].shape)
    if mode == "horizontal":
        if resize_to_match:
            # Find minimum width and resize all images to that width
            min_height = min(t.shape[1] for t in tensors)
            tensors = [
                torch.nn.functional.interpolate(
                    t.unsqueeze(0),
                    size=(min_height, t.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                for t in tensors
            ]
        # Ensure same height
        max_height = max(t.shape[1] for t in tensors)
        padded = [
            torch.nn.functional.pad(t, (0, 0, 0, max_height - t.shape[1]))
            for t in tensors
        ]
        result = torch.cat(padded, dim=2)

    elif mode == "vertical":
        if resize_to_match:
            # Find minimum height and resize all images to that height
            min_width = min(t.shape[2] for t in tensors)
            tensors = [
                torch.nn.functional.interpolate(
                    t.unsqueeze(0),
                    size=(t.shape[1], min_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                for t in tensors
            ]
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

        if resize_to_match:
            # Find minimum dimensions
            min_height = min(t.shape[1] for t in tensors)
            min_width = min(t.shape[2] for t in tensors)
            # Resize all images to minimum dimensions
            tensors = [
                torch.nn.functional.interpolate(
                    t.unsqueeze(0),
                    size=(min_height, min_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                for t in tensors
            ]

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


def embedding2color(
    embedding_map: torch.Tensor,
    color_space: str = "RGB",
    normalize_method: str = "minmax",
    pca: Optional[PCA] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, PCA]]:
    """
    Convert high-dimensional embedding map to RGB visualization using PCA.
    
    Supports multiple input tensor shapes:
    - BxCxHxW (feature map): Standard feature maps
    - BxCxHxWxL (feature volume): 3D feature volumes  
    - BxSxC (transformer output): Sequence embeddings
    
    Args:
        embedding_map (torch.Tensor): Embedding tensor with shapes:
            - (B, C, H, W): Feature map
            - (B, C, H, W, L): 3D feature volume
            - (B, S, C): Transformer sequence output
        color_space (str): Color space to use ('RGB', 'HSV', 'LAB')
        normalize_method (str): Method to normalize PCA components ('minmax', 'robust', 'standard')
        pca (Optional[PCA]): Pre-fitted PCA object. If None, a new PCA will be computed

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, PCA]]:
            - If pca is None: returns (visualization tensor, fitted PCA object)
            - If pca is provided: returns only visualization tensor
            Visualization tensor has same shape as input with C dimension = 3, values in [0, 1]
    """
    device = embedding_map.device
    original_shape = embedding_map.shape
    ndim = len(original_shape)
    
    # Determine tensor format and extract dimensions
    if ndim == 4:  # BxCxHxW (feature map)
        B, C, H, W = original_shape
        # Reshape: B, C, H, W -> B, H, W, C -> (B*H*W), C
        embeddings_reshaped = embedding_map.permute(0, 2, 3, 1)  # B, H, W, C  
        embeddings_2d = embeddings_reshaped.reshape(-1, C).cpu().numpy()
        output_shape = (B, 3, H, W)
        
    elif ndim == 5:  # BxCxHxWxL (feature volume)
        B, C, H, W, L = original_shape
        # Reshape: B, C, H, W, L -> B, H, W, L, C -> (B*H*W*L), C
        embeddings_reshaped = embedding_map.permute(0, 2, 3, 4, 1)  # B, H, W, L, C
        embeddings_2d = embeddings_reshaped.reshape(-1, C).cpu().numpy()
        output_shape = (B, 3, H, W, L)
        
    elif ndim == 3:  # BxSxC (transformer output)
        B, S, C = original_shape
        # Reshape: B, S, C -> (B*S), C
        embeddings_2d = embedding_map.reshape(-1, C).cpu().numpy()
        output_shape = (B, S, 3)
        
    else:
        raise ValueError(f"Unsupported tensor shape: {original_shape}. "
                        f"Expected shapes: BxCxHxW, BxCxHxWxL, or BxSxC")

    # Handle PCA computation or application
    if pca is None:
        # Compute new PCA
        pca = PCA(n_components=3)
        embeddings_pca = pca.fit_transform(embeddings_2d)
        compute_pca = True
    else:
        # Use provided PCA
        embeddings_pca = pca.transform(embeddings_2d)
        compute_pca = False

    # Reshape back to appropriate format based on input type
    if ndim == 4:  # BxCxHxW
        viz = embeddings_pca.reshape(B, H, W, 3)
        normalize_axes = (1, 2)  # H, W axes
        
    elif ndim == 5:  # BxCxHxWxL  
        viz = embeddings_pca.reshape(B, H, W, L, 3)
        normalize_axes = (1, 2, 3)  # H, W, L axes
        
    elif ndim == 3:  # BxSxC
        viz = embeddings_pca.reshape(B, S, 3)
        normalize_axes = (1,)  # S axis

    # Normalize components based on specified method
    if normalize_method == "minmax":
        # Scale to [0, 1] using min-max normalization
        viz_min = viz.min(axis=normalize_axes, keepdims=True)
        viz_max = viz.max(axis=normalize_axes, keepdims=True)
        viz = (viz - viz_min) / (viz_max - viz_min + 1e-8)  # Add epsilon to avoid division by zero
        
    elif normalize_method == "robust":
        # Robust scaling using percentiles
        for b in range(B):
            p_low, p_high = np.percentile(viz[b], [1, 99], axis=normalize_axes[:-1] if len(normalize_axes) > 1 else 0)
            viz[b] = np.clip((viz[b] - p_low) / (p_high - p_low + 1e-8), 0, 1)
            
    elif normalize_method == "standard":
        # Standardize and then clip to [0, 1]
        mean = viz.mean(axis=normalize_axes, keepdims=True)
        std = viz.std(axis=normalize_axes, keepdims=True)
        viz = np.clip((viz - mean) / (std + 1e-6), -3, 3)
        viz = (viz + 3) / 6  # Scale from [-3, 3] to [0, 1]

    # Convert to tensor and move to original device
    viz = torch.from_numpy(viz).float().to(device)

    # Convert to specified color space (only applies to last dimension = 3)
    if color_space == "HSV":
        # Convert RGB to HSV - reshape to work with existing function
        orig_viz_shape = viz.shape
        viz_flat = viz.reshape(-1, 3)
        viz_flat = rgb_to_hsv(viz_flat.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        viz = viz_flat.reshape(orig_viz_shape)
        
    elif color_space == "LAB":
        # Convert RGB to LAB - reshape to work with existing function  
        orig_viz_shape = viz.shape
        viz_flat = viz.reshape(-1, 3)
        viz_flat = rgb_to_lab(viz_flat.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        viz = viz_flat.reshape(orig_viz_shape)

    # Permute to final output format based on input type
    if ndim == 4:  # BxCxHxW -> B, 3, H, W
        viz = viz.permute(0, 3, 1, 2)
    elif ndim == 5:  # BxCxHxWxL -> B, 3, H, W, L  
        viz = viz.permute(0, 4, 1, 2, 3)
    # ndim == 3 (BxSxC) stays as B, S, 3

    if compute_pca:
        return viz, pca
    else:
        return viz


def rgb_to_hsv(rgb):
    """Convert RGB to HSV color space."""
    # Assume input is B, H, W, 3 in range [0, 1]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    max_rgb, _ = torch.max(rgb, dim=-1)
    min_rgb, _ = torch.min(rgb, dim=-1)
    diff = max_rgb - min_rgb

    # Hue calculation
    hue = torch.zeros_like(max_rgb)

    # Mask for diff != 0 to avoid division by zero
    mask = diff != 0

    # Red is maximum
    idx = (mask) & (rgb[..., 0] == max_rgb)
    hue[idx] = (60 * (g[idx] - b[idx]) / diff[idx]) % 360

    # Green is maximum
    idx = (mask) & (rgb[..., 1] == max_rgb)
    hue[idx] = 60 * (2.0 + (b[idx] - r[idx]) / diff[idx])

    # Blue is maximum
    idx = (mask) & (rgb[..., 2] == max_rgb)
    hue[idx] = 60 * (4.0 + (r[idx] - g[idx]) / diff[idx])

    # Normalize hue to [0, 1]
    hue = hue / 360.0

    # Saturation calculation
    sat = torch.zeros_like(max_rgb)
    idx = max_rgb != 0
    sat[idx] = diff[idx] / max_rgb[idx]

    # Value calculation
    val = max_rgb

    return torch.stack([hue, sat, val], dim=-1)


def rgb_to_lab(rgb):
    """Convert RGB to LAB color space."""
    # Basic implementation - for more accurate conversion, consider using colorspace libraries
    # This is a simplified version for visualization purposes

    # Convert RGB to XYZ
    xyz = torch.zeros_like(rgb)

    # Matrix multiplication for RGB to XYZ conversion
    xyz[..., 0] = 0.4124 * rgb[..., 0] + 0.3576 * rgb[..., 1] + 0.1805 * rgb[..., 2]
    xyz[..., 1] = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    xyz[..., 2] = 0.0193 * rgb[..., 0] + 0.1192 * rgb[..., 1] + 0.9505 * rgb[..., 2]

    # XYZ to LAB conversion
    # Using D65 illuminant
    xyz_n = torch.tensor([0.95047, 1.0, 1.08883]).to(rgb.device)

    # Scale XYZ values
    xyz = xyz / xyz_n

    # Apply cube root transformation
    mask = xyz > 0.008856
    xyz[mask] = torch.pow(xyz[mask], 1 / 3)
    xyz[~mask] = 7.787 * xyz[~mask] + 16 / 116

    # Calculate LAB values
    L = (116 * xyz[..., 1]) - 16
    a = 500 * (xyz[..., 0] - xyz[..., 1])
    b = 200 * (xyz[..., 1] - xyz[..., 2])

    # Normalize to [0, 1] for visualization
    L = L / 100  # L is normally in [0, 100]
    a = (a + 128) / 255  # a is normally in [-128, 127]
    b = (b + 128) / 255  # b is normally in [-128, 127]

    return torch.stack([L, a, b], dim=-1)


def download_from_gcs(
    model_name, bucket_name="alberto-bucket", destination_file_path=None
):
    """
    Downloads a file from Google Cloud Storage bucket

    Args:
        bucket_name (str): Name of the GCS bucket
        model_name (str): Path to the file in GCS bucket
        destination_file_path (str, optional): Local path to save the file.
                                             If None, uses a temporary file

    Returns:
        str: Path to the downloaded file, or None if download failed
    """

    # Initialize the GCS client
    storage_client = storage.Client()

    # Get the bucker
    bucket = storage_client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(f"ARTIFACTS/{model_name}.pt")

    # If no destination specified, create a temporary file
    if destination_file_path is None:
        temp_dir = tempfile.gettempdir()
        destination_file_path = os.path.join(temp_dir, os.path.basename(model_name))

    # Download the file
    blob.download_to_filename(f"{destination_file_path}.pt")

    # print(f"Downloaded {model_name} to {destination_file_path}")
    return f"{destination_file_path}.pt"
