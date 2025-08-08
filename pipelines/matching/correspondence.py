import torch
import torch.nn.functional as F


def get_matching_points_with_patches(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    threshold: float = 0.9,
    patch_size: int = 16,
    patch_size_enlarged: int = 32,
    min_matches: int = 20,
    max_matches: int = 1000,
    embedding_mask: torch.Tensor = None,
    knn: int = 1,
    use_cycle_consistency: bool = True,
) -> tuple:
    """
    Identify matching points between two sets of embeddings and extract corresponding pixel patches,
    using vectorized operations to avoid per-patch for loops.

    Args:
        embeddings1 (torch.Tensor): Source embeddings of shape (B, C, N).
        embeddings2 (torch.Tensor): Target embeddings of shape (B, C, N).
        source_image (torch.Tensor): Source image tensor of shape (B, 3, H, W).
        target_image (torch.Tensor): Target image tensor of shape (B, 3, H, W).
        threshold (float): Similarity threshold for matching points.
        patch_size (int): Size of the patches to extract (used for computing grid centers).
        patch_size_enlarged (int): Size of the enlarged patch (actual extracted patch will be of this size).
        min_matches (int): Minimum number of matches required per batch.
        max_matches (int): Maximum number of matches to process per batch.
        embedding_mask (torch.Tensor, optional): Mask for the second set of embeddings.
        knn (int): Number of nearest neighbors to find for each source point (default=1)
        use_cycle_consistency (bool): Whether to apply cycle consistency check (default=True)

    Returns:
        tuple: Contains batch indices, source and target pixel coordinates (in the original image),
               source and target patches (extracted via grid_sample), scores, and the similarity matrix.
    """
    B, C, N = embeddings1.shape
    device = embeddings1.device
    grid_size = int(N**0.5)
    assert grid_size**2 == N, "Sequence length must be a perfect square."
    half_patch = patch_size // 2
    half_patch_enlarged = patch_size_enlarged // 2

    # Normalize embeddings along channel dimension
    embeddings1_norm = F.normalize(embeddings1, dim=1)
    embeddings2_norm = F.normalize(embeddings2, dim=1)

    # Compute similarity matrix between all pairs (B, N, N)
    sim_matrix = torch.bmm(embeddings1_norm.transpose(1, 2), embeddings2_norm)

    # Optionally apply mask to embeddings2
    if embedding_mask is not None:
        embedding_mask = embedding_mask.mean(dim=1).view(B, 1, N)
        sim_matrix = sim_matrix * embedding_mask

    # Get top-k matches from source to target
    scores_12, indices_12 = torch.topk(sim_matrix, k=knn, dim=2)  # (B, N, k)

    if use_cycle_consistency:
        # Get top-k matches from target to source (transpose the similarity matrix)
        scores_21, indices_21 = torch.topk(
            sim_matrix.transpose(1, 2), k=knn, dim=2
        )  # (B, N, k)

        # Initialize cycle consistency mask
        cycle_mask = torch.zeros_like(scores_12)

        # Here's a much more efficient implementation with minimal loops
        # Create tensors to hold indices
        src_indices = torch.arange(N, device=device)

        # For each batch and source point (we still need these two loops)
        for b in range(B):
            # This can be vectorized: Convert source indices to one-hot encoding
            src_one_hot = torch.eye(N, device=device)[src_indices]  # Shape: (N, N)

            # For each source's target match, find if it maps back to the source
            # by comparing the matched source indices with our one-hot encoding
            for i in range(N):
                # Get this source's target matches
                tgt_matches = indices_12[b, i]  # Shape: (k)

                # For each target match, check if it maps back to this source
                for k_idx, j in enumerate(tgt_matches):
                    # Get sources this target maps back to
                    back_src_indices = indices_21[b, j]  # Shape: (k)

                    # Check if our source i is in the back-mapping
                    is_consistent = (back_src_indices == i).any()

                    # If cycle-consistent, keep the score
                    if is_consistent:
                        cycle_mask[b, i, k_idx] = 1.0

        # Apply mask to get cycle-consistent scores
        final_scores = scores_12 * cycle_mask
    else:
        final_scores = scores_12

    # Pad the images once; pad size equals half_patch_enlarged so that patch extraction is valid
    pad = half_patch_enlarged
    source_image_padded = F.pad(
        source_image, (pad, pad, pad, pad), mode="constant", value=0
    )
    target_image_padded = F.pad(
        target_image, (pad, pad, pad, pad), mode="constant", value=0
    )

    # Containers for match information across batches
    all_batch_indices = []
    all_src_centers_padded = (
        []
    )  # Centers in padded image coordinates for patch extraction
    all_tgt_centers_padded = []
    all_src_pixel_points = []  # Original image coordinates (without padding)
    all_tgt_pixel_points = []
    all_scores = []

    # Loop over batches to select valid matching points
    for b in range(B):
        # Reshape scores and indices for this batch to handle k matches
        batch_scores = final_scores[b].view(-1)  # Flatten to (N*k,)
        batch_src_indices = torch.arange(N, device=device).repeat_interleave(knn)
        batch_tgt_indices = indices_12[b].view(-1)  # Flatten to (N*k,)

        # Filter based on threshold
        valid_mask = batch_scores > threshold
        if valid_mask.sum() < min_matches:
            # If too few matches, take top min_matches regardless of threshold
            topk_scores, topk_indices = torch.topk(
                batch_scores, min_matches, sorted=True
            )
            selected_scores = topk_scores
            selected_src_indices = batch_src_indices[topk_indices]
            selected_tgt_indices = batch_tgt_indices[topk_indices]
        else:
            # Get indices of valid matches
            valid_indices = torch.where(valid_mask)[0]
            # If we have more than max_matches valid matches, take the top max_matches by score
            if valid_indices.shape[0] > max_matches:
                # Get scores of valid matches
                valid_scores = batch_scores[valid_indices]
                # Get top max_matches indices
                _, top_indices = torch.topk(valid_scores, max_matches, sorted=True)
                # Select the top max_matches indices
                selected_indices = valid_indices[top_indices]
                selected_scores = batch_scores[selected_indices]
                selected_src_indices = batch_src_indices[selected_indices]
                selected_tgt_indices = batch_tgt_indices[selected_indices]
            else:
                # Use all valid matches if below max_matches
                selected_scores = batch_scores[valid_mask]
                selected_src_indices = batch_src_indices[valid_mask]
                selected_tgt_indices = batch_tgt_indices[valid_mask]

        # Compute grid positions in the embedding space (each index corresponds to a cell in the grid)
        src_y = selected_src_indices // grid_size
        src_x = selected_src_indices % grid_size
        tgt_y = selected_tgt_indices // grid_size
        tgt_x = selected_tgt_indices % grid_size

        # Compute the center positions in the original image (before padding)
        src_center_x = src_x * patch_size + half_patch
        src_center_y = src_y * patch_size + half_patch
        tgt_center_x = tgt_x * patch_size + half_patch
        tgt_center_y = tgt_y * patch_size + half_patch

        # For patch extraction, convert these centers to padded coordinates
        src_center_x_padded = src_center_x + pad
        src_center_y_padded = src_center_y + pad
        tgt_center_x_padded = tgt_center_x + pad
        tgt_center_y_padded = tgt_center_y + pad

        # Store results
        num_matches = selected_src_indices.size(0)
        all_batch_indices.append(
            torch.full((num_matches,), b, dtype=torch.long, device=device)
        )
        all_src_centers_padded.append(
            torch.stack(
                [src_center_x_padded.float(), src_center_y_padded.float()], dim=1
            )
        )
        all_tgt_centers_padded.append(
            torch.stack(
                [tgt_center_x_padded.float(), tgt_center_y_padded.float()], dim=1
            )
        )
        # Save original pixel coordinates (to be returned)
        all_src_pixel_points.append(torch.stack([src_center_x, src_center_y], dim=1))
        all_tgt_pixel_points.append(torch.stack([tgt_center_x, tgt_center_y], dim=1))
        all_scores.append(selected_scores)

    # If no matches are found, return empty tensors
    if len(all_batch_indices) == 0:
        empty_patch = torch.zeros(
            (0, 3, patch_size_enlarged, patch_size_enlarged),
            dtype=source_image.dtype,
            device=device,
        )
        return (
            torch.zeros((0,), dtype=torch.long, device=device),
            torch.zeros((0, 2), dtype=torch.long, device=device),
            torch.zeros((0, 2), dtype=torch.long, device=device),
            empty_patch,
            empty_patch,
            torch.zeros((0,), dtype=embeddings1.dtype, device=device),
            sim_matrix,
        )

    # Concatenate matches from all batches
    batch_indices = torch.cat(all_batch_indices, dim=0)  # (M,)
    src_centers_padded = torch.cat(all_src_centers_padded, dim=0)  # (M, 2)
    tgt_centers_padded = torch.cat(all_tgt_centers_padded, dim=0)  # (M, 2)
    src_pixel_points = torch.cat(all_src_pixel_points, dim=0)  # (M, 2)
    tgt_pixel_points = torch.cat(all_tgt_pixel_points, dim=0)  # (M, 2)
    scores = torch.cat(all_scores, dim=0)  # (M,)

    # --- Vectorized patch extraction using grid_sample ---
    # We want to extract patches of size (patch_size_enlarged, patch_size_enlarged)
    patch_dim = patch_size_enlarged

    # Create a base grid of pixel offsets.
    # We sample at pixel centers: generate indices 0,...,patch_dim-1 and shift by (-half_patch_enlarged + 0.5)
    dx = torch.arange(patch_dim, device=device, dtype=torch.float) - (
        half_patch_enlarged - 0.5
    )
    dy = torch.arange(patch_dim, device=device, dtype=torch.float) - (
        half_patch_enlarged - 0.5
    )
    grid_y, grid_x = torch.meshgrid(
        dy, dx, indexing="ij"
    )  # shape: (patch_dim, patch_dim)
    base_offsets = torch.stack(
        (grid_x, grid_y), dim=-1
    )  # shape: (patch_dim, patch_dim, 2)

    M = batch_indices.shape[0]
    # For source patches: add base offsets to each center to form a sampling grid
    src_centers_exp = src_centers_padded.view(M, 1, 1, 2)  # (M,1,1,2)
    src_grid = src_centers_exp + base_offsets.unsqueeze(
        0
    )  # (M, patch_dim, patch_dim, 2)

    # Normalize grid coordinates to [-1, 1] for grid_sample.
    H_pad, W_pad = source_image_padded.shape[-2:]
    src_grid_norm = src_grid.clone()
    src_grid_norm[..., 0] = (src_grid[..., 0] / (W_pad - 1)) * 2 - 1
    src_grid_norm[..., 1] = (src_grid[..., 1] / (H_pad - 1)) * 2 - 1

    # Do the same for target patches
    tgt_centers_exp = tgt_centers_padded.view(M, 1, 1, 2)
    tgt_grid = tgt_centers_exp + base_offsets.unsqueeze(0)
    H_pad_t, W_pad_t = target_image_padded.shape[-2:]
    tgt_grid_norm = tgt_grid.clone()
    tgt_grid_norm[..., 0] = (tgt_grid[..., 0] / (W_pad_t - 1)) * 2 - 1
    tgt_grid_norm[..., 1] = (tgt_grid[..., 1] / (H_pad_t - 1)) * 2 - 1

    # Gather the corresponding padded images for each match using the batch indices.
    src_images_selected = source_image_padded[batch_indices]  # (M, 3, H_pad, W_pad)
    tgt_images_selected = target_image_padded[batch_indices]  # (M, 3, H_pad_t, W_pad_t)

    # Extract patches in a vectorized way.
    src_patches = F.grid_sample(src_images_selected, src_grid_norm, align_corners=True)
    tgt_patches = F.grid_sample(tgt_images_selected, tgt_grid_norm, align_corners=True)

    return (
        batch_indices,  # (M,) batch indices for each match
        src_pixel_points,  # (M, 2) source pixel coordinates (original image)
        tgt_pixel_points,  # (M, 2) target pixel coordinates (original image)
        src_patches,  # (M, 3, patch_size_enlarged, patch_size_enlarged)
        tgt_patches,  # (M, 3, patch_size_enlarged, patch_size_enlarged)
        scores,  # (M,) matching scores
        sim_matrix,  # (B, N, N) similarity matrix
    )
