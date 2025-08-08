# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

from typing import Dict, List
import numpy as np
from rich import print as nativeprint
import torch
import torch.nn.functional as F

def closest_multiple(value, factor, mode="closest"):
    """
    Find the closest multiple of a factor to a given value.

    Args:
        value (int): The value to find the closest multiple for
        factor (int): The factor to use
        mode (str): One of 'closest', 'inf', or 'sup'
            - 'closest': returns the closest multiple (rounding)
            - 'inf': returns the largest multiple not exceeding value (floor)
            - 'sup': returns the smallest multiple not less than value (ceiling)

    Returns:
        int: The closest multiple according to the specified mode
    """
    if mode == "closest":
        return round(value / factor) * factor
    elif mode == "inf":
        return (value // factor) * factor
    elif mode == "sup":
        return ((value + factor - 1) // factor) * factor
    else:
        raise ValueError("Mode must be one of 'closest', 'inf', or 'sup'")


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


def tprint(args, shape=False, dtype=False, device=False, grad_fn=False, **kwargs):

    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    output = []
    np.set_printoptions(precision=4, suppress=True)

    def tensor_to_string(tensor):
        return str(tensor.cpu().detach().numpy())

    for arg in args:
        if isinstance(arg, torch.Tensor):

            infos = "\n"
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


def sp(size: tuple) -> tuple:
    """
    Converts a size tuple to a tuple of the same elements.

    Args:
        size (tuple): Input size tuple.

    Returns:
        tuple: Output size tuple.
    """
    return tuple(size)


def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)


def embedding2chw(
    embedding: torch.Tensor, embed_dim_last=True, aspect_ratio: float = None
) -> torch.Tensor:
    """
    Reorganizes the embedding output into CHW form, handling both square and non-square sequence lengths.

    Args:
        embedding (torch.Tensor): Input embedding tensor of shape (N, D).
        embed_dim_last (bool): If True, expects embedding dimension to be last, otherwise expects it second.
        aspect_ratio (float, optional): Desired width/height ratio. If None, will try to make as square as possible.

    Returns:
        torch.Tensor: Tensor of shape (D, H, W).
    """
    # Validate input tensor shape
    if len(embedding.shape) == 2:
        embedding = embedding.unsqueeze(0)

    if not embed_dim_last:
        embedding = embedding.permute(0, 2, 1)
    B, N, D = embedding.shape

    # Calculate dimensions based on sequence length and aspect ratio
    if aspect_ratio is None:
        # Find factors closest to square
        def get_factors(n):
            factors = []
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    factors.append((i, n // i))
            return min(factors, key=lambda x: abs(x[0] - x[1]))

        height, width = get_factors(N)
    else:
        # Calculate dimensions to match desired aspect ratio as closely as possible
        height = int((N / aspect_ratio) ** 0.5)
        # Find the closest factor pair that maintains the desired aspect ratio
        while N % height != 0:
            height -= 1
        width = N // height

        # Verify we haven't deviated too far from desired aspect ratio
        actual_ratio = width / height
        if abs(actual_ratio - aspect_ratio) > 0.5:  # You can adjust this threshold
            print(
                f"Warning: Actual aspect ratio ({actual_ratio:.2f}) differs significantly from requested ({aspect_ratio:.2f})"
            )

    # Reshape the embedding from (B,N,D) to (B,H,W,D)
    chw_tensor = embedding.view(B, height, width, D).permute(0, 3, 1, 2)

    return chw_tensor


def chw2embedding(chw_tensor: torch.Tensor, embed_dim_last=False) -> torch.Tensor:
    """
    Converts a CHW-formatted tensor back into an embedding format.

    Args:
        chw_tensor (torch.Tensor): Input tensor of shape (B, D, H, W).
        embed_dim_last (bool): If True, returns embedding with dimension last, else second.

    Returns:
        torch.Tensor: Embedding tensor of shape (B, N, D) or (B, D, N).
    """
    # Validate input tensor shape
    if chw_tensor.ndim != 4:
        raise ValueError(
            f"Expected a 4D tensor (B, D, H, W), but got shape {chw_tensor.shape}."
        )

    B, D, H, W = chw_tensor.shape
    N = H * W

    # Reshape from (B, D, H, W) to (B, N, D)
    embedding = chw_tensor.permute(0, 2, 3, 1).reshape(B, N, D)

    if not embed_dim_last:
        embedding = embedding.permute(0, 2, 1)

    return embedding


def append_cls(
    featuremap: torch.Tensor, 
    cls_token: torch.Tensor, 
    prepend: bool = True
) -> torch.Tensor:
    """
    Appends or prepends a CLS token to a feature map, handling various input formats.
    
    Args:
        featuremap (torch.Tensor): Input feature map of shape:
            - (B, E, H, W) - CHW format
            - (B, H*W, E) - embedding format with embed_dim_last=True
            - (B, E, H*W) - embedding format with embed_dim_last=False
            - (E, H, W) - unbatched CHW format
            - (H*W, E) - unbatched embedding format
        cls_token (torch.Tensor): CLS token of shape:
            - (B, 1, E) - batched with singleton sequence dimension
            - (B, E) - batched without sequence dimension
            - (1, E) - unbatched with singleton sequence dimension
            - (E,) - unbatched without sequence dimension
        prepend (bool): If True, prepend CLS token, else append it.
    
    Returns:
        torch.Tensor: Embedding sequence of shape (B, H*W+1, E) or (H*W+1, E) for unbatched.
    """
    # Handle unbatched inputs by adding batch dimension
    original_shape = featuremap.shape
    
    # Determine if inputs are batched or unbatched
    is_batched = featuremap.ndim == 4 or (featuremap.ndim == 3 and cls_token.ndim >= 2 and cls_token.shape[0] > 1)
    
    if not is_batched:
        # Add batch dimension for unbatched inputs
        if featuremap.ndim == 3:  # (E, H, W) -> (1, E, H, W)
            featuremap = featuremap.unsqueeze(0)
            cls_token = cls_token.unsqueeze(0) if cls_token.ndim == 2 else cls_token
        elif featuremap.ndim == 2:  # (H*W, E) -> (1, H*W, E)
            featuremap = featuremap.unsqueeze(0)
            cls_token = cls_token.unsqueeze(0) if cls_token.ndim == 1 else cls_token
    
    # Handle CLS token shape variations
    if cls_token.ndim == 1:  # (E,) -> (1, 1, E)
        cls_token = cls_token.unsqueeze(0).unsqueeze(0)
    elif cls_token.ndim == 2:  # (B, E) -> (B, 1, E)
        cls_token = cls_token.unsqueeze(1)
    
    # Detect featuremap format and convert to embedding format
    if featuremap.ndim == 4:  # (B, E, H, W) - CHW format
        # Convert to embedding format: (B, H*W, E)
        embedding = chw2embedding(featuremap, embed_dim_last=True)
    elif featuremap.ndim == 3:  # (B, N, E) or (B, E, N)
        # Check if it's already in embedding format
        B, dim1, dim2 = featuremap.shape
        if dim1 == cls_token.shape[-1]:  # (B, E, N) format - compare with embedding dimension
            embedding = featuremap.permute(0, 2, 1)  # (B, N, E)
        else:  # (B, N, E) format
            embedding = featuremap
    else:
        raise ValueError(f"Unexpected featuremap shape: {original_shape}")
    
    # Ensure CLS token has correct batch size
    if cls_token.shape[0] == 1 and embedding.shape[0] > 1:
        cls_token = cls_token.expand(embedding.shape[0], -1, -1)
    elif cls_token.shape[0] != embedding.shape[0]:
        raise ValueError(f"Batch size mismatch: featuremap {embedding.shape[0]}, cls_token {cls_token.shape[0]}")
    
    # Concatenate CLS token with embedding
    if prepend:
        # Prepend CLS token: (B, 1+seq_len, E)
        result = torch.cat([cls_token, embedding], dim=1)
    else:
        # Append CLS token: (B, seq_len+1, E)
        result = torch.cat([embedding, cls_token], dim=1)
    
    # Remove batch dimension if input was unbatched
    if len(original_shape) <= 3:
        result = result.squeeze(0)
    
    return result


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of dictionaries into a dictionary of tensors.

    Args:
    batch (List[Dict]): List of dictionaries containing tensors.

    Returns:
    Dict[str, torch.Tensor]: A dictionary containing tensors stacked along the 0th dimension.
    """
    return {key: torch.stack([d[key] for d in batch]) for key in batch[0]}


def dup_indexes(arr):
    unique, counts = torch.unique(arr, return_counts=True)
    duplicates = unique[counts > 1]
    return [torch.where(arr == dup)[0].tolist() for dup in duplicates]


def embedding_mask_from_pixels(
    pixel_mask: torch.Tensor, patch_size: int = 8, embedding_dim: int = 768
) -> torch.Tensor:
    """
    Convert a pixel-wise binary mask to a patch-wise binary mask for embeddings.
    A patch is considered valid only if all pixels within it are valid.

    Args:
        pixel_mask (torch.Tensor): Binary mask of shape (B, 3, H, W) where all channels are equal
        patch_size (int): Size of the patches for embeddings
        embedding_dim (int): Number of embedding features

    Returns:
        torch.Tensor: Binary mask for embeddings of shape (B, embedding_dim, H/patch_size, W/patch_size)
    """
    B, _, H, W = pixel_mask.shape
    device = pixel_mask.device

    # Verify that all channels are equal
    assert torch.all(pixel_mask[:, 0] == pixel_mask[:, 1]) and torch.all(
        pixel_mask[:, 1] == pixel_mask[:, 2]
    ), "All channels in the mask must be equal"

    # Verify dimensions are divisible by patch_size
    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), f"Image dimensions ({H}, {W}) must be divisible by patch_size {patch_size}"

    # Take just one channel since they're all equal
    single_channel_mask = pixel_mask[:, 0]  # Shape: (B, H, W)

    # Reshape into patches
    H_p = H // patch_size
    W_p = W // patch_size

    # Unfold into patches
    patches = single_channel_mask.unfold(1, patch_size, patch_size).unfold(
        2, patch_size, patch_size
    )
    # Shape after unfold: (B, H_p, W_p, patch_size, patch_size)

    # Check if all pixels in each patch are valid (1)
    valid_patches = patches.sum(dim=(-2, -1)) == patch_size * patch_size
    # Shape: (B, H_p, W_p)

    # Expand to match embedding dimensions
    embedding_mask = valid_patches.unsqueeze(1).expand(B, embedding_dim, H_p, W_p)

    return embedding_mask.float()


def generate_random_pose_tensor(
    translation_minmax=None,
    euler_minmax=None,
    angle_unit="degrees",
    device=None,
    dtype=torch.float32,
):
    """
    Generates a random 6x1 pose tensor with translation and Euler angles.

    Parameters:
        translation_minmax (list or tuple of tuples): Specifies the (min, max) for each translation component.
            Format: [(t_x_min, t_x_max), (t_y_min, t_y_max), (t_z_min, t_z_max)]
            Example: [(-5, 5), (-10, 10), (-15, 15)]
            If None, defaults to [(-10, 10)] for all components.

        euler_minmax (list or tuple of tuples): Specifies the (min, max) for each Euler angle component.
            Format: [(alpha_min, alpha_max), (beta_min, beta_max), (gamma_min, gamma_max)]
            Example: [(-180, 180), (-90, 90), (-180, 180)]
            If None, defaults to [(-180, 180)] for all components.

        angle_unit (str): Unit for Euler angles. 'degrees' or 'radians'. Defaults to 'degrees'.

        device (torch.device, optional): The device on which to create the tensor.
            Defaults to None, which means the tensor is created on the CPU.

        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Defaults to torch.float32.

    Returns:
        torch.Tensor: A 6x1 tensor where:
            - Elements 0-2 are translations (t_x, t_y, t_z)
            - Elements 3-5 are Euler angles (alpha, beta, gamma)

    Raises:
        ValueError: If input ranges are not properly specified.
    """
    # Set default ranges if not provided
    if translation_minmax is None:
        translation_minmax = [(-10, 10)] * 3
    if euler_minmax is None:
        euler_minmax = [(-180, 180)] * 3  # Default in degrees

    # Validate translation_minmax
    if (not isinstance(translation_minmax, (list, tuple))) or len(
        translation_minmax
    ) != 3:
        raise ValueError(
            "translation_minmax must be a list or tuple of three (min, max) tuples."
        )
    for idx, t_range in enumerate(translation_minmax):
        if (not isinstance(t_range, (list, tuple))) or len(t_range) != 2:
            raise ValueError(
                f"Each translation range must be a tuple/list of two values. Error at index {idx}."
            )
        if t_range[0] > t_range[1]:
            raise ValueError(
                f"Translation range min must be <= max. Error at index {idx}."
            )

    # Validate euler_minmax
    if (not isinstance(euler_minmax, (list, tuple))) or len(euler_minmax) != 3:
        raise ValueError(
            "euler_minmax must be a list or tuple of three (min, max) tuples."
        )
    for idx, e_range in enumerate(euler_minmax):
        if (not isinstance(e_range, (list, tuple))) or len(e_range) != 2:
            raise ValueError(
                f"Each Euler angle range must be a tuple/list of two values. Error at index {idx}."
            )
        if e_range[0] > e_range[1]:
            raise ValueError(
                f"Euler angle range min must be <= max. Error at index {idx}."
            )

    # Generate random translations
    translations = []
    for idx, (t_min, t_max) in enumerate(translation_minmax):
        t = torch.empty(1, device=device, dtype=dtype).uniform_(t_min, t_max)
        translations.append(t)

    # Generate random Euler angles
    euler_angles = []
    for idx, (e_min, e_max) in enumerate(euler_minmax):
        angle = torch.empty(1, device=device, dtype=dtype).uniform_(e_min, e_max)
        euler_angles.append(angle)

    # Concatenate translations and Euler angles
    pose_vector = torch.cat(translations + euler_angles, dim=0).view(6, 1)

    # Convert angles to radians if necessary
    if angle_unit.lower() == "degrees":
        pose_vector[3:6] = torch.deg2rad(pose_vector[3:6])
    elif angle_unit.lower() == "radians":
        pass  # No conversion needed
    else:
        raise ValueError("angle_unit must be either 'degrees' or 'radians'.")

    return pose_vector.T

def interpolate_featuremap(tensor, mask, mode="nearest", k=4):
    """
    Fully vectorized interpolation for filling invalid positions.
    
    Args:
        tensor: Input tensor of shape (C, H, W)
        mask: Binary mask of shape (H, W) where True indicates valid positions
        mode: Interpolation mode - "nearest" or "bilinear"
        k: Number of nearest neighbors for bilinear interpolation (ignored for nearest)
        
    Returns:
        Interpolated tensor of same shape as input
    """
    C, H, W = tensor.shape
    device = tensor.device
    
    # Ensure mask is boolean
    mask = mask.bool()
    
    # Early returns for edge cases
    if mask.all():
        return tensor.clone()
    
    if not mask.any():
        # No valid positions - return original or handle as needed
        return tensor.clone()
    
    # Create output tensor
    output = tensor.clone()
    
    # Get coordinates of valid and invalid positions
    valid_positions = torch.where(mask)  # Returns (y_coords, x_coords)
    invalid_positions = torch.where(~mask)
    
    # Convert to coordinate tensors [y, x] format
    valid_coords = torch.stack(valid_positions, dim=1).float()  # Shape: (num_valid, 2)
    invalid_coords = torch.stack(invalid_positions, dim=1).float()  # Shape: (num_invalid, 2)
    
    # Vectorized distance calculation using broadcasting
    # Shape: (num_invalid, 1, 2) - (1, num_valid, 2) = (num_invalid, num_valid, 2)
    coord_diff = invalid_coords.unsqueeze(1) - valid_coords.unsqueeze(0)
    
    # Calculate L2 distances
    distances = torch.norm(coord_diff, dim=2)  # Shape: (num_invalid, num_valid)
    
    # Extract coordinates for indexing
    invalid_y, invalid_x = invalid_positions
    
    if mode == "nearest":
        # Find nearest valid position for each invalid position
        nearest_indices = torch.argmin(distances, dim=1)  # Shape: (num_invalid,)
        
        # Get coordinates of nearest valid positions
        nearest_valid_coords = valid_coords[nearest_indices]  # Shape: (num_invalid, 2)
        nearest_y = nearest_valid_coords[:, 0].long()
        nearest_x = nearest_valid_coords[:, 1].long()
        
        # Vectorized assignment for all channels simultaneously
        output[:, invalid_y, invalid_x] = tensor[:, nearest_y, nearest_x]
        
    elif mode == "bilinear":
        # Ensure k doesn't exceed number of valid positions
        k = min(k, len(valid_coords))
        
        # Find k nearest neighbors for distance-weighted interpolation
        k_distances, k_indices = torch.topk(distances, k, dim=1, largest=False)
        
        # Calculate weights using inverse distance weighting
        # Add small epsilon to avoid division by zero for exact matches
        epsilon = 1e-8
        weights = 1.0 / (k_distances + epsilon)
        
        # Handle case where distance is exactly zero (avoid inf weights)
        exact_matches = k_distances < epsilon
        if exact_matches.any():
            # For exact matches, use weight of 1.0 for closest match, 0.0 for others
            weights = torch.where(exact_matches, 
                                torch.where(k_distances == k_distances.min(dim=1, keepdim=True)[0], 
                                          1.0, 0.0), 
                                weights)
        
        # Normalize weights so they sum to 1 for each invalid position
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # Get coordinates of k nearest neighbors
        k_nearest_coords = valid_coords[k_indices]  # Shape: (num_invalid, k, 2)
        k_nearest_y = k_nearest_coords[:, :, 0].long()
        k_nearest_x = k_nearest_coords[:, :, 1].long()
        
        # Vectorized gathering of values from k nearest neighbors
        num_invalid = len(invalid_y)
        
        # Expand indices for all channels
        batch_idx = torch.arange(num_invalid, device=device).view(1, num_invalid, 1).expand(C, num_invalid, k)
        channel_idx = torch.arange(C, device=device).view(C, 1, 1).expand(C, num_invalid, k)
        
        # Gather values from k nearest neighbors for all channels
        # Shape: (C, num_invalid, k)
        neighbor_values = tensor[
            channel_idx.reshape(-1),
            k_nearest_y.unsqueeze(0).expand(C, -1, -1).reshape(-1),
            k_nearest_x.unsqueeze(0).expand(C, -1, -1).reshape(-1)
        ].reshape(C, num_invalid, k)
        
        # Apply distance-weighted interpolation
        # weights shape: (num_invalid, k) -> (1, num_invalid, k) for broadcasting
        interpolated_values = (neighbor_values * weights.unsqueeze(0)).sum(dim=2)
        
        # Assign interpolated values
        output[:, invalid_y, invalid_x] = interpolated_values
        
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}. Use 'nearest' or 'bilinear'.")
    
    return output


def inpaint(image, missing_value, median_kernel_size=3, iterations=10):
    """
    Inpaint missing values in BxWxW RGB image using median filtering
    
    Args:
        image: torch.Tensor of shape (B, W, W) where each pixel value represents RGB
        missing_value: value that indicates missing pixels
        median_kernel_size: size of median filter kernel (should be odd)
        iterations: number of inpainting iterations
    
    Returns:
        inpainted_img: torch.Tensor of shape (B, W, W) with missing values filled
    """
    
    B, W1, W2 = image.shape
    device = image.device
    
    # Create mask for missing values
    base_mask = (image != missing_value).float()  # shape: (B, W, W)
    
    # Initialize inpainted image
    inpainted_img = image.clone()
    
    pad = median_kernel_size // 2
    
    for _ in range(iterations):
        # Pad the image using reflection
        padded = F.pad(inpainted_img, (pad, pad, pad, pad), mode="reflect")
        
        # Extract patches using unfold
        patches = padded.unfold(1, median_kernel_size, 1).unfold(2, median_kernel_size, 1)
        # patches shape after unfold: (B, H_out, W_out, kernel_size, kernel_size)
        # where H_out = W_out = W (since we're unfolding a square image)
        
        # Reshape for median computation
        patches = patches[:,:W2,:W2]
        patches = patches.contiguous().view(B, W1, W2, median_kernel_size * median_kernel_size)
        # patches shape: (B, W, W, kernel_size^2)
        
        # Compute median
        median_img, _ = patches.median(dim=-1)
        # median_img shape: (B, W, W)
        
        # Update only missing pixels
        final_img = inpainted_img.clone()
        final_img[base_mask == 0] = median_img[base_mask == 0]
        inpainted_img = final_img
    
    return inpainted_img

