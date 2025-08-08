import torch
import torch.nn.functional as F
import numpy as np


def points_to_patches(points, embedding_map, patch_size=8, mask=None):
    B, N, _ = points.shape
    _, C, H_p, W_p = embedding_map.shape
    H, W = H_p * patch_size, W_p * patch_size

    points = torch.clamp(points, min=0, max=float(H))
    patch_x = torch.clamp((points[..., 0] / patch_size).long(), 0, W_p - 1)
    patch_y = torch.clamp((points[..., 1] / patch_size).long(), 0, H_p - 1)

    batch_idx = torch.arange(B, device=points.device)[:, None, None]
    channel_idx = torch.arange(C, device=points.device)[None, :, None]
    patch_y_exp = patch_y.unsqueeze(1).expand(B, C, N)
    patch_x_exp = patch_x.unsqueeze(1).expand(B, C, N)

    output = embedding_map[batch_idx, channel_idx, patch_y_exp, patch_x_exp]

    if mask is not None:
        mask_reduced = mask[:, 0]
        output_mask = mask_reduced[
            torch.arange(B, device=points.device)[:, None], patch_y, patch_x
        ]
        output_mask = output_mask.unsqueeze(1).expand(B, C, N)
    else:
        output_mask = torch.ones((B, C, N), device=points.device)

    flat_indices = patch_y * W_p + patch_x
    return output, flat_indices, output_mask


def filter_inliers(points_data, inliers_mask):
    """
    Filter all point-related data based on inlier mask.

    Args:
        points_data: Dictionary containing all point-related data
        inliers_mask: Boolean mask indicating inliers

    Returns:
        Dictionary with filtered point data
    """
    filtered_data = {}
    for key, value in points_data.items():
        if value is not None:
            filtered_data[key] = value[inliers_mask]

    return filtered_data


def filter_scores(points_data, scores, threshold=0):
    """
    Filter all point-related data based on score threshold.

    Args:
        points_data: Dictionary containing all point-related data
        scores: Confidence scores for matches
        threshold: Minimum score to keep

    Returns:
        Dictionary with filtered point data and valid scores
    """
    valid_mask = scores != threshold
    filtered_data = filter_inliers(points_data, valid_mask)
    filtered_scores = scores[valid_mask]

    return filtered_data, filtered_scores


def apply_refinement_offsets(
    source_pixels, target_pixels, source_offsets, target_offsets, patch_size
):
    """
    Apply refinement offsets to the matched pixel coordinates.

    Args:
        source_pixels: Source frame pixel coordinates
        target_pixels: Target frame pixel coordinates
        source_offsets: Offset values for source pixels
        target_offsets: Offset values for target pixels
        patch_size: Size of the patch used for refinement

    Returns:
        Updated source and target pixel coordinates
    """
    half_patch = patch_size / 2

    source_pixels_refined = source_pixels + source_offsets - half_patch + 0.5
    target_pixels_refined = target_pixels + target_offsets - half_patch + 0.5

    return source_pixels_refined, target_pixels_refined


def filter_boundary_matches(
    source_pixels,
    target_pixels,
    scores,
    batch_idx_match,
    boundary=20,
    image_width=384,
    image_height=384,
):
    """
    Remove matches where either source or target points fall outside image boundaries.

    Args:
        source_pixels (torch.Tensor): Source pixel coordinates of shape (N, 2)
        target_pixels (torch.Tensor): Target pixel coordinates of shape (N, 2)
        scores (torch.Tensor): Matching scores of shape (N,)
        batch_idx_match (torch.Tensor): Batch indices for each match of shape (N,)
        width (int): Image width
        height (int): Image height

    Returns:
        tuple: Filtered tensors (source_pixels_filtered, target_pixels_filtered,
               scores_filtered, batch_idx_match_filtered)
    """
    # Compute valid masks for points inside image boundaries

    valid_source = (
        (source_pixels[:, 0] >= boundary)
        & (source_pixels[:, 0] < image_width - boundary)
        & (source_pixels[:, 1] >= boundary)
        & (source_pixels[:, 1] < image_height - boundary)
    )
    valid_target = (
        (target_pixels[:, 0] >= boundary)
        & (target_pixels[:, 0] < image_width - boundary)
        & (target_pixels[:, 1] >= boundary)
        & (target_pixels[:, 1] < image_height - boundary)
    )
    valid_mask = valid_source & valid_target

    # Filter out matches that fall outside the image boundaries
    source_pixels_filtered = source_pixels[valid_mask]
    target_pixels_filtered = target_pixels[valid_mask]
    scores_filtered = scores[valid_mask]
    batch_idx_match_filtered = batch_idx_match[valid_mask]

    return (
        source_pixels_filtered,
        target_pixels_filtered,
        scores_filtered,
        batch_idx_match_filtered,
        valid_mask,
    )


def filter_matches_by_unified_score(match_data, score_threshold=0.5, max_matches=None):
    """
    Filter matches based on the unified scoring system, returning only high-quality matches.

    Args:
        match_data: Dictionary containing match data from match_images():
            - source_pixels_matched: Source points (N, 2)
            - target_pixels_matched: Target points (N, 2)
            - scores: Unified match scores (N,)
            - batch_idx_match: Batch indices (N,)
            - inliers: Optional boolean tensor (N,) indicating RANSAC inliers
        score_threshold: Minimum score to keep a match (default: 0.5)
        max_matches: Maximum number of matches to return per batch (default: None for all matches)

    Returns:
        Dictionary with filtered match data
    """
    device = match_data["scores"].device
    batch_size = (
        match_data["batch_idx_match"].max().item() + 1
        if match_data["batch_idx_match"].numel() > 0
        else 0
    )

    # Create a mask for matches with scores above threshold
    score_mask = match_data["scores"] >= score_threshold

    # If we also have inliers information from RANSAC, consider it (optional)
    if "inliers" in match_data and match_data["inliers"] is not None:
        inlier_mask = match_data["inliers"].bool()
        # Keep matches that have high scores OR are inliers
        final_mask = score_mask | inlier_mask
    else:
        final_mask = score_mask

    # Extract matches that pass the filters
    filtered_data = {}

    # If we want to limit the number of matches per batch
    if max_matches is not None and batch_size > 0:
        all_indices = []

        # Process each batch separately
        for b in range(batch_size):
            batch_mask = match_data["batch_idx_match"] == b
            batch_final_mask = final_mask & batch_mask

            # Get indices of matches in this batch that pass the filter
            batch_indices = torch.nonzero(batch_final_mask).squeeze(-1)

            if batch_indices.numel() > 0:
                # Sort by score (highest first)
                batch_scores = match_data["scores"][batch_indices]
                sorted_indices = torch.argsort(batch_scores, descending=True)
                batch_indices = batch_indices[sorted_indices]

                # Limit to max_matches
                batch_indices = batch_indices[:max_matches]
                all_indices.append(batch_indices)

        # Combine indices from all batches
        if all_indices:
            filtered_indices = torch.cat(all_indices)

            # Apply the indices to filter the data
            for key in match_data:
                if key in [
                    "source_pixels_matched",
                    "target_pixels_matched",
                    "scores",
                    "batch_idx_match",
                    "inliers",
                ]:
                    if match_data[key] is not None and match_data[key].numel() > 0:
                        filtered_data[key] = match_data[key][filtered_indices]
                else:
                    filtered_data[key] = match_data[key]
        else:
            # No matches pass the filters, return empty tensors
            for key in match_data:
                if key in [
                    "source_pixels_matched",
                    "target_pixels_matched",
                    "scores",
                    "batch_idx_match",
                    "inliers",
                ]:
                    filtered_data[key] = torch.zeros(
                        (0,) + match_data[key].shape[1:],
                        device=device,
                        dtype=match_data[key].dtype,
                    )
                else:
                    filtered_data[key] = match_data[key]
    else:
        # No max_matches limit, just apply the filter mask
        for key in match_data:
            if key in [
                "source_pixels_matched",
                "target_pixels_matched",
                "scores",
                "batch_idx_match",
                "inliers",
            ]:
                if match_data[key] is not None and match_data[key].numel() > 0:
                    filtered_data[key] = match_data[key][final_mask]
            else:
                filtered_data[key] = match_data[key]

    return filtered_data
