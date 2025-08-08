import numpy as np
import torch
import losses as loss
import torch
from sklearn.metrics import precision_recall_curve, auc


def f1_score(pred_points, true_points, threshold):
    precision, recall = precision_recall(pred_points, true_points, threshold)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def inlier_ratio(pred_points, true_points, threshold):
    distances = torch.norm(pred_points - true_points, dim=2)
    correct_matches = distances < threshold

    inlier_ratios = correct_matches.sum(dim=1) / correct_matches.size(1)
    return inlier_ratios.mean().item()  # Average across batche


def descriptor_matching_rate(pred_points, true_points, threshold):
    distances = torch.norm(pred_points - true_points, dim=2)
    correct_matches = distances < threshold

    matching_rate = correct_matches.sum(dim=1) / pred_points.size(1)
    return matching_rate.mean().item()


def NCM(pred_points, true_points, threshold):
    """
    Calculates the Number of Correct Matches (NCM).

    Args:
        pred_points (torch.Tensor): Predicted points of shape (B, N, 2).
        true_points (torch.Tensor): Ground truth points of shape (B, N, 2).
        threshold (float): Distance threshold for considering a match correct.

    Returns:
        int: Total number of correct matches across all batches.
    """
    distances = torch.norm(pred_points - true_points, dim=2)  # Euclidean distance
    correct_matches = (distances < threshold).sum()  # Count correct matches
    return correct_matches.item()  # Return as integer


def success_rate(pred_points, true_points, threshold, success_threshold=0.5):
    """
    Calculates the Success Rate (SR).

    Args:
        pred_points (torch.Tensor): Predicted points of shape (B, N, 2).
        true_points (torch.Tensor): Ground truth points of shape (B, N, 2).
        threshold (float): Distance threshold for considering a match correct.
        success_threshold (float): Proportion of correct matches required to consider a batch successful (default=0.5).

    Returns:
        float: Success rate as a fraction of successful batches.
    """
    distances = torch.norm(pred_points - true_points, dim=2)  # Euclidean distance
    correct_matches_per_batch = (distances < threshold).sum(
        dim=1
    )  # Count correct matches per batch
    match_ratios = correct_matches_per_batch / pred_points.size(
        1
    )  # Proportion of correct matches per batch

    successful_batches = (
        match_ratios >= success_threshold
    ).sum()  # Count successful batches
    success_rate = successful_batches / pred_points.size(
        0
    )  # Fraction of successful batches
    return success_rate.item()  # Return as a float


def fundamental_error(F1, F2, error_type="percentage", reduction="mean"):
    """
    Calculate the error between two batches of fundamental matrices.
    Handles sign ambiguity, scale differences, and batching.

    Args:
        F1 (torch.Tensor): First batch of 3x3 fundamental matrices (shape: Bx3x3)
        F2 (torch.Tensor): Second batch of 3x3 fundamental matrices (shape: Bx3x3)
        error_type (str): Type of error to return ('percentage' or 'absolute')
        reduction (str): Reduction method ('mean' or 'none')

    Returns:
        If reduction='mean':
            float: Average error across the batch (percentage or absolute)
            torch.Tensor: Individual errors for each matrix in the batch
        If reduction='none':
            torch.Tensor: Individual errors for each matrix in the batch
    """
    # Check shapes
    if F1.shape != F2.shape or len(F1.shape) != 3:
        raise ValueError(f"Input shapes must be Bx3x3. Got {F1.shape} and {F2.shape}")

    batch_size = F1.shape[0]

    # Normalize matrices to handle scale differences
    # Compute Frobenius norm for each matrix in the batch
    F1_norms = torch.norm(F1.reshape(batch_size, -1), dim=1) + 1e-6
    F2_norms = torch.norm(F2.reshape(batch_size, -1), dim=1) + 1e-6

    # Reshape norms for broadcasting and normalize
    F1_norm = F1 / F1_norms.view(-1, 1, 1)
    F2_norm = F2 / F2_norms.view(-1, 1, 1)

    # Calculate errors for both positive and negative F2 for each pair in the batch
    error_positive = torch.norm((F1_norm - F2_norm).reshape(batch_size, -1), dim=1)
    error_negative = torch.norm((F1_norm + F2_norm).reshape(batch_size, -1), dim=1)

    # Get the minimum error for each pair
    batch_errors = torch.minimum(error_positive, error_negative)

    if error_type == "percentage":
        # Convert to percentage (relative to the normalized matrices)
        # Maximum possible error between normalized matrices is sqrt(2)
        max_possible_error = torch.sqrt(torch.tensor(2.0, device=F1.device))
        batch_errors = batch_errors / max_possible_error

    # Fallback in case F1_norms or F2_norms are zero. Set the error to NaN
    if torch.any(F1_norms <= 5e-6) or torch.any(F2_norms <= 5e-6):
        batch_errors = torch.full((batch_size,), float("nan"), device=F1.device)
        avg_error = float("nan")
    else:
        # Compute average error across the batch if reduction is mean
        avg_error = torch.mean(batch_errors).item() if reduction == "mean" else None

    if reduction == "mean":
        return avg_error, batch_errors
    else:
        return batch_errors


def epipolar_error(points1, points2, F, batch_index, reduction="mean"):
    """
    Calculate epipolar error between matched points using fundamental matrices.

    Args:
        points1 (torch.Tensor): First set of points (shape: Nx2)
        points2 (torch.Tensor): Second set of points (shape: Nx2)
        F (torch.Tensor): Batch of fundamental matrices (shape: Bx3x3)
        batch_index (torch.Tensor): Batch indices for each point pair (shape: Nx1)
        reduction (str): Reduction method ('mean' or 'none')

    Returns:
        If reduction='mean':
            float: Mean epipolar error
        If reduction='none':
            torch.Tensor: Per-point epipolar errors (shape: N)
    """
    N = points1.shape[0]
    ones = torch.ones(N, 1, device=points1.device, dtype=points1.dtype)
    p1_h = torch.cat([points1, ones], dim=1)  # shape Nx3
    p2_h = torch.cat([points2, ones], dim=1)  # shape Nx3

    # Select appropriate fundamental matrices for each match
    batch_idx = batch_index.squeeze(-1).int()  # shape N
    F_sel = F[batch_idx]  # shape Nx3x3

    # Reshape points for batched multiplication
    p1_h_exp = p1_h.unsqueeze(-1).float()  # shape Nx3x1
    p2_h_exp = p2_h.unsqueeze(-1).float()  # shape Nx3x1

    # Compute epipolar lines
    # l2 = F * p1 for each match
    l2 = torch.bmm(F_sel, p1_h_exp).squeeze(-1)  # shape Nx3
    # l1 = F^T * p2 for each match
    l1 = torch.bmm(F_sel.transpose(1, 2), p2_h_exp).squeeze(-1)  # shape Nx3

    # Unbind line coefficients and point coordinates
    a2, b2, c2 = l2.unbind(dim=1)
    a1, b1, c1 = l1.unbind(dim=1)
    x2, y2 = points2.unbind(dim=1)
    x1, y1 = points1.unbind(dim=1)

    # Compute distances from points to their epipolar lines
    dist2 = torch.abs(a2 * x2 + b2 * y2 + c2) / torch.sqrt(a2**2 + b2**2 + 1e-8)
    dist1 = torch.abs(a1 * x1 + b1 * y1 + c1) / torch.sqrt(a1**2 + b1**2 + 1e-8)
    error = (dist1 + dist2) / 2.0

    if reduction == "mean":
        return error.mean().item()
    else:
        return error


def mean_matching_distance(pred_points, true_points, batch_index, reduction="mean"):
    """
    Calculate mean distance between predicted and true points with batch support.

    Args:
        pred_points (torch.Tensor): Predicted points, shape (N, 2)
        true_points (torch.Tensor): Ground truth points, shape (N, 2)
        batch_index (torch.Tensor): Batch indices for each point pair, shape (N,) or (N, 1)
        reduction (str): Reduction method ('mean' or 'none')

    Returns:
        If reduction='mean':
            float: Mean distance across all batches
        If reduction='none':
            torch.Tensor: Mean distance for each batch, shape (B,) where B is the number of unique batches
    """
    # Ensure batch_index is the right shape
    if batch_index.dim() == 2:
        batch_index = batch_index.squeeze(-1)  # Convert from (N, 1) to (N)
    batch_index = batch_index.int()
    # Calculate per-point distances
    point_distances = torch.norm(pred_points.float() - true_points.float(), dim=1)

    if reduction == "mean":
        # Return the mean distance across all points
        return point_distances.mean().item()
    else:
        # Calculate mean distance per batch
        unique_batches = torch.unique(batch_index)
        batch_distances = torch.zeros(len(unique_batches), device=pred_points.device)

        for i, batch_id in enumerate(unique_batches):
            # Create mask for points in this batch
            batch_mask = batch_index == batch_id
            # Calculate mean distance for this batch
            if batch_mask.sum() > 0:  # Avoid division by zero
                batch_distances[i] = point_distances[batch_mask].mean()

        return batch_distances


def precision_recall(
    source_pixels_matched,
    target_pixels_matched,
    true_pixels_matched,
    batch_indexes,
    confidence_scores,
    threshold,
    fundamental=None,
    reduction="mean",
):
    """
    Computes Precision, Recall, and AUC-PR for matched points across batches.

    Args:
        source_pixels_matched (torch.Tensor): Source points, shape (N, 2).
        target_pixels_matched (torch.Tensor): Predicted target points, shape (N, 2).
        true_pixels_matched (torch.Tensor): Ground truth target points, shape (N, 2) or None.
        batch_indexes (torch.Tensor): Batch indices for each point, shape (N,).
        confidence_scores (torch.Tensor): Confidence scores for each prediction, shape (N,).
        threshold (float): Distance threshold in pixels to consider a prediction as True Positive.
        fundamental (torch.Tensor, optional): Batch of fundamental matrices (shape: Bx3x3).
                                                      Required if true_pixels_matched is None.
        reduction (str): Reduction method ('mean' or 'none')

    Returns:
        If reduction='mean':
            precision (float): Mean Precision across all batches.
            recall (float): Mean Recall across all batches.
            auc_pr (float): Area Under the Precision-Recall Curve.
        If reduction='none':
            precision (torch.Tensor): Precision for each batch.
            recall (torch.Tensor): Recall for each batch.
            auc_pr (float): Overall Area Under the Precision-Recall Curve.
    """
    # Determine if we're using epipolar error or Euclidean distance
    if true_pixels_matched is not None:
        # Use Euclidean distance between predicted and ground truth points
        distances = torch.norm(target_pixels_matched - true_pixels_matched, dim=1)
    else:
        # Use epipolar error when ground truth is not available
        if fundamental is None:
            raise ValueError(
                "Fundamental matrices must be provided when true_pixels_matched is None"
            )

        # Calculate epipolar errors (per-point)
        distances = epipolar_error(
            source_pixels_matched,
            target_pixels_matched,
            fundamental,
            batch_indexes,
            reduction="none",
        )

    # Assign labels based on the distance/error threshold
    # Label = 1 if distance/error < threshold (True Positive), else 0 (False Positive)
    labels = (distances < threshold).int()

    if reduction == "mean":
        # Sort confidence scores and labels in descending order
        sorted_scores, indices = torch.sort(confidence_scores, descending=True)
        sorted_labels = labels[indices]

        # Calculate cumulative sums
        cumsum_tp = torch.cumsum(sorted_labels, dim=0)
        total_positives = labels.sum()
        total_predictions = len(labels)

        # Calculate precision and recall curves
        precision_curve = cumsum_tp / torch.arange(
            1, total_predictions + 1, device=source_pixels_matched.device
        )
        recall_curve = cumsum_tp / (total_positives + 1e-8)

        # Compute AUC-PR using trapezoidal rule
        auc_pr = torch.trapz(precision_curve, recall_curve)

        # Compute overall Precision and Recall
        mean_precision = precision_curve.mean()
        mean_recall = recall_curve.mean()

        return mean_precision.item(), mean_recall.item(), auc_pr.item()
    else:
        # Get unique batch indices
        unique_batches = torch.unique(batch_indexes)
        precisions = []
        recalls = []

        # Compute precision and recall for each batch separately
        for batch_id in unique_batches:
            batch_mask = batch_indexes == batch_id
            batch_labels = labels[batch_mask]
            batch_scores = confidence_scores[batch_mask]

            # Skip batches with no positive examples
            if batch_labels.sum() == 0:
                precisions.append(0.0)
                recalls.append(0.0)
                continue

            # Sort confidence scores and labels in descending order
            sorted_scores, indices = torch.sort(batch_scores, descending=True)
            sorted_labels = batch_labels[indices]

            # Calculate cumulative sums
            cumsum_tp = torch.cumsum(sorted_labels, dim=0)
            total_positives = batch_labels.sum()
            total_predictions = len(batch_labels)

            # Calculate precision and recall curves
            precision_curve = cumsum_tp / torch.arange(
                1, total_predictions + 1, device=source_pixels_matched.device
            )
            recall_curve = cumsum_tp / (total_positives + 1e-8)

            # Store mean precision and recall for this batch
            precisions.append(precision_curve.mean().item())
            recalls.append(recall_curve.mean().item())

        # Convert lists to tensors
        precision_tensor = torch.tensor(precisions, device=source_pixels_matched.device)
        recall_tensor = torch.tensor(recalls, device=source_pixels_matched.device)

        # Calculate overall AUC-PR using all points
        sorted_scores, indices = torch.sort(confidence_scores, descending=True)
        sorted_labels = labels[indices]
        cumsum_tp = torch.cumsum(sorted_labels, dim=0)
        total_positives = labels.sum()
        total_predictions = len(labels)
        precision_curve = cumsum_tp / torch.arange(
            1, total_predictions + 1, device=source_pixels_matched.device
        )
        recall_curve = cumsum_tp / (total_positives + 1e-8)
        auc_pr = torch.trapz(precision_curve, recall_curve)

        return precision_tensor, recall_tensor, auc_pr.item()


def f1_score(points1, points2, F, batch_index, threshold=1.0, reduction="mean"):
    """
    Calculate F1-score for point matches based on epipolar constraint.

    A match is considered a True Positive if its distance to the
    corresponding ground truth epipolar line is less than the threshold (default: 1 pixel).

    Args:
        points1 (torch.Tensor): First set of points (shape: Nx2)
        points2 (torch.Tensor): Second set of points (shape: Nx2)
        F (torch.Tensor): Batch of fundamental matrices (shape: Bx3x3)
        batch_index (torch.Tensor): Batch indices for each point pair (shape: Nx1)
        threshold (float, optional): Threshold for considering a match as correct (default: 1.0)
        reduction (str): Reduction method ('mean' or 'none')

    Returns:
        If reduction='mean':
            float: Mean F1-score across all batches
        If reduction='none':
            torch.Tensor: Per-batch F1-scores (shape: B)
    """
    import torch

    # Calculate epipolar errors for each point pair
    errors = epipolar_error(points1, points2, F, batch_index, reduction="none")

    # Consider a match as correct (TP) if error < threshold
    correct_matches = errors < threshold

    # Get batch size from fundamental matrix
    batch_size = F.shape[0]

    if reduction == "none":
        # Initialize tensors to store batch-wise metrics
        batch_f1 = torch.zeros(batch_size, device=points1.device)

        # Compute F1-score for each batch
        for b in range(batch_size):
            # Get mask for current batch
            batch_mask = batch_index.squeeze(-1) == b

            # Skip if no points in this batch
            if not batch_mask.any():
                batch_f1[b] = 0.0
                continue

            # Count true positives and total predictions in this batch
            batch_correct = correct_matches[batch_mask]
            batch_tp = batch_correct.sum().float()
            batch_total = batch_mask.sum().float()

            # Calculate precision (TP / total predicted)
            precision = (
                batch_tp / batch_total
                if batch_total > 0
                else torch.tensor(0.0, device=points1.device)
            )

            # In this context, recall equals precision since we're evaluating all predictions
            # against the epipolar constraint
            recall = precision

            # Calculate F1-score for this batch
            if precision + recall > 0:
                batch_f1[b] = 2 * (precision * recall) / (precision + recall)

        return batch_f1

    else:  # reduction == "mean"
        # Count true positives and total predictions
        true_positives = correct_matches.sum().item()
        total_predictions = len(errors)

        # Calculate precision
        precision = true_positives / total_predictions if total_predictions > 0 else 0.0

        # Calculate recall (simplified approach based on available information)
        recall = precision

        # Calculate F1-score
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return f1


def compute_metrics(
    mp,
    source_pixels_matched,
    target_pixels_matched,
    true_pixels_matched,
    batch_idx_match,
    scores,
    fundamental_pred,
    fundamental_gt,
):
    """
    Compute various metrics for evaluating matching performance.

    Args:
        target_pixels_matched: Predicted target pixel coordinates
        true_pixels_matched: Ground truth target pixel coordinates
        batch_idx_match: Batch indices for each match
        scores: Confidence scores for each match
        fundamental_pred: Predicted fundamental matrix
        fundamental_pred8: Predicted fundamental matrix from 8-point algorithm
        fundamental_gt: Ground truth fundamental matrix
        inliers: Inlier mask
        triplets_dict: Dictionary containing triplet information
        loss_tensor: Loss tensor
        lossdfe: DFE loss tensor
        patch_size: Patch size used for matching
        inlier_patch_ratio: Ratio for determining inliers
        gradient_accumulation_steps: Number of gradient accumulation steps
        batch_size: Batch size

    Returns:
        dict: Dictionary containing all computed metrics
    """
    # Compute precision, recall, AUCPR
    precision, recall, AUCPR = precision_recall(
        source_pixels_matched.detach(),
        target_pixels_matched.detach(),
        true_pixels_matched.detach() if true_pixels_matched is not None else None,
        batch_idx_match.detach(),
        scores.detach(),
        mp.config.RESAMPLED_PATCH_SIZE * mp.config.MAX_EPIPOLAR_DISTANCE * (2**0.5),
        fundamental_pred,
    )

    # Compute epipolar error
    epipolar = epipolar_error(
        source_pixels_matched.cpu(),
        target_pixels_matched.cpu(),
        fundamental_gt.cpu(),
        batch_idx_match.cpu(),
    )

    # Compute fundamental matrix errors
    fundamental = fundamental_error(fundamental_pred.cpu(), fundamental_gt.cpu())[0]
    # Compute mean matching distance
    if true_pixels_matched is not None:
        mean_match_distance = mean_matching_distance(
            target_pixels_matched.cpu(),
            true_pixels_matched.cpu(),
            batch_idx_match.cpu(),
        )
    else:
        mean_match_distance = None

    # Return all metrics in a dictionary
    return {
        "Precision": precision,
        "Recall": recall,
        "AUCPR": AUCPR,
        "EpipolarError": epipolar,
        "FundamentalError": fundamental,
        "MDistMean": mean_match_distance,
    }


def compute_unified_match_score(
    source_pixels,
    target_pixels,
    batch_idx,
    fundamental_matrix=None,
    confidence_scores=None,
    config=None,
):
    """
    Computes a unified match quality score that combines multiple metrics.

    Args:
        source_pixels (torch.Tensor): Source points, shape (N, 2)
        target_pixels (torch.Tensor): Target points, shape (N, 2)
        batch_idx (torch.Tensor): Batch indices for each point pair, shape (N,)
        fundamental_matrix (torch.Tensor, optional): Fundamental matrix of shape (B, 3, 3)
        true_targets (torch.Tensor, optional): Ground truth target points if available, shape (N, 2)
        confidence_scores (torch.Tensor, optional): Initial confidence scores from descriptor matching, shape (N,)
        config (dict, optional): Configuration parameters for scoring weights

    Returns:
        torch.Tensor: Unified match scores of shape (N,)
    """
    device = source_pixels.device
    n_matches = source_pixels.shape[0]

    # Handle empty tensor case
    if n_matches == 0:
        return torch.tensor([], device=device)

    # Get batch size safely
    batch_size = batch_idx.max().item() + 1 if n_matches > 0 else 0

    # Default config if not provided
    if config is None:
        config = {
            "w_descriptor": 0.3,  # Weight for descriptor similarity score
            "w_epipolar": 0.4,  # Weight for epipolar constraint
            "w_refinement": 0.2,  # Weight for pixel distance (from refinement)
            "w_mutual": 0.1,  # Weight for mutual consistency
            "epipolar_threshold": 1.0,  # Threshold for epipolar distance (in pixels)
        }

    # Initialize the final scores tensor
    final_scores = torch.ones(n_matches, device=device)

    # 1. Descriptor similarity score component
    if confidence_scores is not None:
        descriptor_scores = confidence_scores
    else:
        # Default to 1.0 if not provided
        descriptor_scores = torch.ones(n_matches, device=device)
    # 2. Epipolar constraint score component
    if fundamental_matrix is not None:
        # Calculate how well matches satisfy the epipolar constraint
        epipolar_errors = epipolar_error(
            source_pixels,
            target_pixels,
            fundamental_matrix,
            batch_idx,
            reduction="none",
        )
        # Convert errors to scores (1.0 for error=0, decreasing as error increases)
        epipolar_scores = torch.exp(-epipolar_errors / config["epipolar_threshold"])
    else:
        epipolar_scores = torch.ones(n_matches, device=device)
    # 4. Mutual consistency score component (fully vectorized)
    mutual_scores = torch.ones(n_matches, device=device)
    # Compute mutual consistency if we have enough matches
    if n_matches > 8:
        # Process all batches at once using tensor operations
        unique_batch_ids = torch.unique(batch_idx)

        for b in unique_batch_ids:
            batch_mask = batch_idx == b
            n_batch = batch_mask.sum()

            if n_batch <= 8:
                continue

            # Get points for current batch
            src_pts_batch = source_pixels[batch_mask]  # Shape: [n_batch, 2]
            tgt_pts_batch = target_pixels[batch_mask]  # Shape: [n_batch, 2]
            batch_indices = torch.nonzero(batch_mask).squeeze(-1)

            # Compute all pairwise distances in a vectorized way
            # Shape: [n_batch, n_batch]
            src_dist = torch.cdist(src_pts_batch, src_pts_batch, p=2)
            tgt_dist = torch.cdist(tgt_pts_batch, tgt_pts_batch, p=2)

            # Avoid division by zero and tiny distances
            # Shape: [n_batch, n_batch]
            mask = (
                (src_dist > 1.0)
                & (tgt_dist > 1.0)
                & ~torch.eye(n_batch, dtype=torch.bool, device=device)
            )

            # Skip if no valid pairs found
            if not mask.any():
                mutual_scores[batch_indices] = torch.ones(n_batch, device=device)
                continue

            # Compute distance ratios (always <= 1.0)
            # Shape: [n_batch, n_batch]
            ratio_mat = torch.where(
                mask,
                torch.minimum(
                    src_dist / torch.clamp(tgt_dist, min=1e-8),
                    tgt_dist / torch.clamp(src_dist, min=1e-8),
                ),
                torch.ones_like(src_dist),
            )

            # Convert to scores (1.0 is perfect)
            # Shape: [n_batch, n_batch]
            score_mat = torch.exp(-torch.abs(1.0 - ratio_mat) * 5.0)

            # Replace non-valid pairs with 1.0 (neutral element for multiplication)
            score_mat = torch.where(mask, score_mat, torch.ones_like(score_mat))

            # Product of all pairwise scores for each point, with normalization
            # Shape: [n_batch]
            point_scores = torch.prod(score_mat, dim=1) ** (1.0 / (n_batch - 1))

            # Update the global mutual scores tensor
            mutual_scores[batch_indices] = point_scores

    # Combine all score components with their weights
    # Check that all tensor shapes match before combining
    n_descriptor = descriptor_scores.shape[0]
    n_epipolar = epipolar_scores.shape[0]
    n_mutual = mutual_scores.shape[0]

    # If tensor sizes don't match, truncate to the smallest size or pad as needed
    min_size = min(n_descriptor, n_epipolar, n_mutual)
    if not (n_descriptor == n_epipolar == n_mutual):
        descriptor_scores = descriptor_scores[:min_size]
        epipolar_scores = epipolar_scores[:min_size]
        mutual_scores = mutual_scores[:min_size]

    # Now combine with weights after ensuring all tensors have the same size
    final_scores = (
        config["w_descriptor"] * descriptor_scores
        + config["w_epipolar"] * epipolar_scores
        + config["w_mutual"] * mutual_scores
    )
    # Normalize to [0, 1] range
    final_scores = torch.clamp(final_scores, 0.0, 1.0)
    return final_scores
