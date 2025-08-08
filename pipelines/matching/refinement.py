import torch
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor


def process_patch(patch_data):
    s_patch, t_patch, ratio_thresh, patch_size = patch_data

    # Convert source patch: (3,H,W) → (H,W,3), scale to uint8, grayscale
    s_patch = np.transpose(s_patch, (2, 1, 0))
    s_patch_8u = (s_patch * 255).clip(0, 255).astype(np.uint8)
    s_patch_gray = cv2.cvtColor(s_patch_8u, cv2.COLOR_RGB2GRAY)

    # Convert target patch similarly
    t_patch = np.transpose(t_patch, (2, 1, 0))
    t_patch_8u = (t_patch * 255).clip(0, 255).astype(np.uint8)
    t_patch_gray = cv2.cvtColor(t_patch_8u, cv2.COLOR_RGB2GRAY)

    # Create SIFT and BF matcher (using CPU)
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    kp_s, desc_s = sift.detectAndCompute(s_patch_gray, None)
    kp_t, desc_t = sift.detectAndCompute(t_patch_gray, None)

    # Fallback if no keypoints/descriptors found
    if desc_s is None or desc_t is None or len(desc_s) == 0 or len(desc_t) == 0:
        return (
            [patch_size / 2, patch_size / 2],
            [patch_size / 2, patch_size / 2],
            1 - ratio_thresh,
        )

    # kNN matching and Lowe's ratio test
    knn_matches = bf.knnMatch(desc_s, desc_t, k=2)
    good_matches = []
    good_scores = []
    for match_pair in knn_matches:
        if len(match_pair) >= 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                good_scores.append(
                    1 - (1 - ratio_thresh) / ratio_thresh * (m.distance / n.distance)
                )

    if len(good_matches) == 0:
        return (
            [patch_size / 2, patch_size / 2],
            [patch_size / 2, patch_size / 2],
            1 - ratio_thresh,
        )

    # Select best match (lowest distance)
    best_match = min(good_matches, key=lambda x: x.distance)
    best_score = min(good_scores)
    s_kp = kp_s[best_match.queryIdx].pt  # (x, y)
    t_kp = kp_t[best_match.trainIdx].pt  # (x, y)

    return ([s_kp[1], s_kp[0]], [t_kp[1], t_kp[0]], best_score)


def SIFT_patch_refiner(
    spatches: torch.Tensor,
    tpatches: torch.Tensor,
    ratio_thresh=0.75,
    patch_size=16,
    num_workers=8,
):
    N = spatches.shape[0]
    # Convert all patches to numpy arrays once (CPU-side)
    spatches_np = spatches.cpu().numpy()
    tpatches_np = tpatches.cpu().numpy()
    # Prepare data for parallel processing
    patch_data = [
        (spatches_np[i], tpatches_np[i], ratio_thresh, patch_size) for i in range(N)
    ]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_patch, patch_data))
    src_coords_list, tgt_coords_list, scores_list = zip(*results)
    device = spatches.device
    best_src_xy = torch.tensor(src_coords_list, dtype=torch.float, device=device)
    best_tgt_xy = torch.tensor(tgt_coords_list, dtype=torch.float, device=device)
    best_scores = torch.tensor(scores_list, dtype=torch.float, device=device)
    return best_src_xy, best_tgt_xy, best_scores


def FFT_patch_refiner(
    spatches: torch.Tensor,
    tpatches: torch.Tensor,
    patch_size: int = 16,
):
    """
    Refines patch correspondences using phase correlation with torch tensors on GPU.
    Incorporates original matching scores and produces refined scores based on correlation confidence.

    Inputs:
      spatches: Source patches tensor of shape (N, C, patch_size, patch_size)
      tpatches: Target patches tensor of shape (N, C, patch_size, patch_size)
      scores: Original confidence scores for each match of shape (N,)
      confidence_threshold: Minimum correlation value to consider a refinement valid
      patch_size: The size of each square patch
      alpha: Weight for blending original and correlation scores (higher alpha gives more weight to original scores)

    Outputs:
      best_src_xy: Tensor of refined source patch centers (N, 2)
      best_tgt_xy: Tensor of corresponding target patch centers (N, 2)
      refined_scores: Tensor of refined confidence scores (N,)
    """
    N = spatches.shape[0]
    eps = 1e-8

    # Compute FFT for each patch pair (batch-wise)
    F1 = torch.fft.fft2(spatches, dim=(-2, -1))
    F2 = torch.fft.fft2(tpatches, dim=(-2, -1))

    # Compute cross-power spectrum and sum over channels
    cross_power = (F2 * torch.conj(F1)).sum(dim=1, keepdim=True)
    cross_power_normalized = cross_power / (torch.abs(cross_power) + eps)

    # Compute the inverse FFT to obtain the correlation surface
    corr = torch.fft.ifft2(cross_power_normalized, dim=(-2, -1))
    corr_abs = torch.abs(corr).squeeze(1)  # Shape: (N, patch_size, patch_size)

    # Find the peak correlation value and its index for each patch
    best_values, best_indices = corr_abs.view(N, -1).max(dim=1)  # Shapes: (N,), (N,)

    # Convert flat indices into 2D indices (peak_y, peak_x)
    peak_y = best_indices // patch_size
    peak_x = best_indices % patch_size

    # Compute offsets, adjusting for wrap-around
    offset_x = torch.where(peak_x > (patch_size // 2), peak_x - patch_size, peak_x)
    offset_y = torch.where(peak_y > (patch_size // 2), peak_y - patch_size, peak_y)
    offsets = torch.stack((offset_x, offset_y), dim=1).float()  # shape: (N, 2)

    # Define the center of a patch
    center = (patch_size - 1) / 2.0
    best_src_xy = torch.tensor([center, center], device=spatches.device).repeat(N, 1)
    best_tgt_xy = best_src_xy + offsets

    # Create refined scores by blending original scores with correlation values
    # Normalize correlation values to [0, 1] range (they should already be in this range)
    normalized_corr = best_values.clamp(0, 1)
    # Create a mask for matches with correlation below threshold
    # low_confidence_mask = normalized_corr > confidence_threshold
    # # Blend scores using weighted average for matches with sufficient confidence
    # refined_scores = torch.where(
    #     low_confidence_mask,
    #     # torch.zeros_like(scores),  # Set score to 0 for low confidence refinements
    #     alpha * scores + (1 - alpha) * normalized_corr,  # Weighted blend for others
    #     normalized_corr,
    # )

    return best_src_xy, best_tgt_xy, normalized_corr
