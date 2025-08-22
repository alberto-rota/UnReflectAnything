import numpy as np
import torch
import losses as loss


def evaluate_odometry(gt, pred, prepend_tonames="Test/"):
    """Computation of error metrics between predicted and ground truth odometry"""
    # Compute the Absolute Trajectory Error (ATE) between ground truth and predicted trajectories
    ate = compute_ate(gt, pred)
    # re = compute_re(gt, pred)
    return {
        prepend_tonames + "_ATE": ate,
        # prepend_tonames + "_tRPE": trpe,
        # prepend_tonames + "_rRPE": rrpe,
    }


def compute_re(gtruth_r, pred_r):
    RE = 0
    gt = gtruth_r
    pred = pred_r
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose @ np.linalg.inv(pred_pose)
        s = np.linalg.norm([R[0, 1] - R[1, 0], R[1, 2] - R[2, 1], R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return RE / gtruth_r.shape[0]


def compute_ate(gtruth_xyz: np.ndarray, pred_xyz_o: np.ndarray) -> float:
    """
    Computes the Absolute Trajectory Error (ATE) between ground truth and predicted trajectories.

    Parameters:
        gtruth_xyz (np.ndarray): Ground truth trajectory points.
        pred_xyz_o (np.ndarray): Predicted trajectory points.

    Returns:
        float: The RMSE of the alignment error.
    """
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = torch.sum(gtruth_xyz * pred_xyz) / torch.sum(pred_xyz**2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = torch.sqrt(torch.sum(alignment_error**2)) / gtruth_xyz.shape[0]
    return rmse


def evaluate_depth(gt, pred, prepend_tonames="Test/"):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = torch.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).float().mean()

    rmse = torch.sqrt(((gt - pred) ** 2).mean())

    rmse_log = torch.sqrt(((torch.log(gt) - torch.log(pred)) ** 2).mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return {
        prepend_tonames + "_AbsRel": abs_rel.item(),
        prepend_tonames + "_SqRel": sq_rel.item(),
        prepend_tonames + "_RMSE": rmse.item(),
        prepend_tonames + "_RMSELog": rmse_log.item(),
        prepend_tonames + "_delta": a1.item(),
    }


def evaluate_reprojection(framestack, warped, prepend_tonames="Test/"):
    # Accumulated photometric loss
    lossfn = loss.WeightedCombinationLoss(
        [
            (
                "SSIM",
                loss.SSIMLoss(),
                0.85,
                {"target", "warped"},
            ),
            (
                "L1",
                loss.L1Loss(),
                0.15,
                {"target", "warped"},
            ),
        ]
    ).to(framestack.device)
    final_reprojection_error = lossfn(
        target=framestack[-1, 1].unsqueeze(0), warped=warped[-1].unsqueeze(0)
    )
    return {prepend_tonames + "_ReprojectionError": final_reprojection_error.item()}


def horn_transformation(A, B):
    """
    Aligns two sets of points in 3D space A and B using Horn's method.

    Parameters:
    - A: Nx3 numpy array of points.
    - B: Nx3 numpy array of points, where B is aligned to A.

    Returns:
    - R: 3x3 rotation matrix.
    - t: 3x1 translation vector.
    """

    # Converting a list of matrices to a list of 6D vectors
    A = np.array([mat2euler(A[i]).numpy() for i in range(0, len(A))])
    B = np.array([mat2euler(B[i]).numpy() for i in range(0, len(B))])
    # Keep only the 3 positional elements of the vector
    A = A[:, :3]
    B = B[:, :3]
    # Center of the trajectory
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the trajectories on a common center of mass
    AA = A - centroid_A
    BB = B - centroid_B
    # Obtain the rotation matrix through SVD
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Special reflection case if det(R) < 0
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Obtain the translation vector based on the centroids
    t = centroid_B.T - R @ centroid_A.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def ATE(groundtruth: np.ndarray, predicted: np.ndarray) -> float:
    """
    Computes the Absolute Trajectory Error (ATE) between ground truth and predicted trajectories.

    ATE quantifies the overall discrepancy between two trajectories, considering the position
    of each pose. This function aligns the predicted trajectory with the ground truth using
    a similarity transformation and then computes the mean Euclidean distance between corresponding
    points.

    Parameters:
        groundtruth (np.ndarray): An array of ground truth poses, each represented by a 4x4 transformation matrix.
        predicted (np.ndarray): An array of predicted poses, each represented by a 4x4 transformation matrix.

    Returns:
        float: The mean ATE over all pose pairs.
    """
    # Align the predicted trajectory to the ground truth
    S = horn_transformation(groundtruth, predicted)
    # Compute the error for each pose
    E = [
        np.linalg.inv(groundtruth[i]) @ S @ predicted[i]
        for i in range(len(groundtruth))
    ]
    # Extract the translational component and compute its norm
    tnormE = np.array([np.linalg.norm(E[i][:, 3]) for i in range(len(E))])
    # Return the mean error
    return np.mean(tnormE)


def tRPE(groundtruth: np.ndarray, predicted: np.ndarray, delta: int = 1) -> float:
    """
    Computes the Translational Relative Pose Error (tRPE) between ground truth and predicted trajectories.

    tRPE evaluates the translational component of the relative pose error over a sequence of poses,
    considering pairs of poses that are 'delta' steps apart. It provides insight into the local accuracy
    of the trajectory over short distances.

    Parameters:
        groundtruth (np.ndarray): An array of ground truth poses, each represented by a 4x4 transformation matrix.
        predicted (np.ndarray): An array of predicted poses, each represented by a 4x4 transformation matrix.
        delta (int): The step size between pose pairs for computing the relative error.

    Returns:
        float: The mean translational RPE over all evaluated pose pairs.
    """
    # Compute the relative pose error for each pair of poses 'delta' steps apart
    F = [
        np.linalg.inv(groundtruth[i])
        @ groundtruth[i + delta]
        @ np.linalg.inv(predicted[i])
        @ predicted[i + delta]
        for i in range(len(predicted) - delta)
    ]
    # Extract the translational component and compute its norm
    return np.mean([np.linalg.norm(F[i][:, 3]) for i in range(len(F))])


def rRPE(groundtruth: np.ndarray, predicted: np.ndarray, delta: int = 1) -> float:
    """
    Computes the Rotational Relative Pose Error (rRPE) between ground truth and predicted trajectories.

    rRPE evaluates the rotational component of the relative pose error over a sequence of poses,
    considering pairs of poses that are 'delta' steps apart. It assesses how well the algorithm
    maintains rotational consistency over short distances.

    Parameters:
        groundtruth (np.ndarray): An array of ground truth poses, each represented by a 4x4 transformation matrix.
        predicted (np.ndarray): An array of predicted poses, each represented by a 4x4 transformation matrix.
        delta (int): The step size between pose pairs for computing the relative error.

    Returns:
        float: The mean rotational RPE over all evaluated pose pairs.
    """
    # Compute the relative pose error for each pair of poses 'delta' steps apart
    F = [
        np.linalg.inv(groundtruth[i])
        @ groundtruth[i + delta]
        @ np.linalg.inv(predicted[i])
        @ predicted[i + delta]
        for i in range(len(predicted) - delta)
    ]
    # Extract the rotational component, compute the angle of rotation, and return the mean
    return np.mean([np.arccos((np.trace(F[i][:3, :3]) - 1) / 2) for i in range(len(F))])


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
    return inlier_ratios.mean().item()  # Average across batches


import torch


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
        F1 (numpy.ndarray): First batch of 3x3 fundamental matrices (shape: Bx3x3)
        F2 (numpy.ndarray): Second batch of 3x3 fundamental matrices (shape: Bx3x3)
        error_type (str): Type of error to return ('percentage' or 'absolute')
        reduction (str): Reduction method ('mean' or 'none')

    Returns:
        If reduction='mean':
            float: Average error across the batch (percentage or absolute)
            numpy.ndarray: Individual errors for each matrix in the batch
        If reduction='none':
            numpy.ndarray: Individual errors for each matrix in the batch
    """
    # Ensure inputs are numpy arrays
    F1 = np.array(F1)
    F2 = np.array(F2)

    # Check shapes
    if F1.shape != F2.shape or len(F1.shape) != 3:
        raise ValueError(f"Input shapes must be Bx3x3. Got {F1.shape} and {F2.shape}")

    batch_size = F1.shape[0]

    # Normalize matrices to handle scale differences
    # Compute Frobenius norm for each matrix in the batch
    F1_norms = np.linalg.norm(F1.reshape(batch_size, -1), axis=1) + 1e-6
    F2_norms = np.linalg.norm(F2.reshape(batch_size, -1), axis=1) + 1e-6

    # Reshape norms for broadcasting and normalize
    F1_norm = F1 / F1_norms[:, None, None]
    F2_norm = F2 / F2_norms[:, None, None]

    # Calculate errors for both positive and negative F2 for each pair in the batch
    error_positive = np.linalg.norm((F1_norm - F2_norm).reshape(batch_size, -1), axis=1)
    error_negative = np.linalg.norm((F1_norm + F2_norm).reshape(batch_size, -1), axis=1)

    # Get the minimum error for each pair
    batch_errors = np.minimum(error_positive, error_negative)

    if error_type == "percentage":
        # Convert to percentage (relative to the normalized matrices)
        # Maximum possible error between normalized matrices is sqrt(2)
        max_possible_error = np.sqrt(2)
        batch_errors = batch_errors / max_possible_error

    # Fallback in case F1_norms or F2_norms are zero. Set the error to NaN
    if np.any(F1_norms <= 5e-6) or np.any(F2_norms <= 5e-6):
        batch_errors = np.full(batch_size, np.nan)
        avg_error = np.nan
    else:
        # Compute average error across the batch if reduction is mean
        avg_error = np.mean(batch_errors) if reduction == "mean" else None

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
    batch_idx = batch_index.squeeze(-1)  # shape N
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
    pred_points,
    true_points,
    batch_indexes,
    confidence_scores,
    threshold,
    reduction="mean",
):
    """
    Computes Precision, Recall, and AUC-PR for matched points across batches.

    Args:
        pred_points (torch.Tensor): Predicted points, shape (N, 2).
        true_points (torch.Tensor): Ground truth points, shape (N, 2).
        batch_indexes (torch.Tensor): Batch indices for each point, shape (N,).
        confidence_scores (torch.Tensor): Confidence scores for each prediction, shape (N,).
        threshold (float): Distance threshold in pixels to consider a prediction as True Positive.
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

    # Ensure inputs are on CPU for sklearn compatibility
    pred_points = pred_points.cpu()
    true_points = true_points.cpu()
    batch_indexes = batch_indexes.cpu()
    confidence_scores = confidence_scores.cpu()

    # 1. Compute Euclidean distances between predicted and true points
    # Shape: (N,)
    distances = torch.norm(pred_points - true_points, dim=1)

    # 2. Assign labels based on the distance threshold
    # Label = 1 if distance < threshold (True Positive), else 0 (False Positive)
    labels = (distances < threshold).int()

    if reduction == "mean":
        # Convert tensors to NumPy arrays for scikit-learn functions
        labels_np = labels.numpy()
        scores_np = confidence_scores.numpy()

        # Compute Precision-Recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            labels_np, scores_np
        )

        # Compute AUC-PR
        auc_pr = auc(recall_curve, precision_curve)

        # Compute overall Precision and Recall
        mean_precision = precision_curve.mean()
        mean_recall = recall_curve.mean()

        return mean_precision, mean_recall, auc_pr
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

            # Convert to numpy for scikit-learn
            batch_labels_np = batch_labels.numpy()
            batch_scores_np = batch_scores.numpy()

            # Compute precision-recall curve for this batch
            batch_precision, batch_recall, _ = precision_recall_curve(
                batch_labels_np, batch_scores_np
            )

            # Store mean precision and recall for this batch
            precisions.append(batch_precision.mean())
            recalls.append(batch_recall.mean())

        # Convert lists to tensors
        precision_tensor = torch.tensor(precisions)
        recall_tensor = torch.tensor(recalls)

        # Calculate overall AUC-PR using all points
        labels_np = labels.numpy()
        scores_np = confidence_scores.numpy()
        precision_curve, recall_curve, _ = precision_recall_curve(labels_np, scores_np)
        auc_pr = auc(recall_curve, precision_curve)

        return precision_tensor, recall_tensor, auc_pr


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
