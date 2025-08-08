import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
import geometry


class FundamentalEstimator8PA(nn.Module):
    def __init__(self, alpha=50.0, epsilon=1e-6):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, pts1, pts2, scores, batch_idx):
        """
        pts1: [N, 2]
        pts2: [N, 2]
        scores: [N, 1]
        batch_idx: [N, 1] with integer batch indices
        """
        batch_idx = batch_idx.squeeze(-1)
        unique_batches = torch.unique(batch_idx)
        F_list = []
        for b in unique_batches:
            mask = batch_idx == b
            pts1_b = pts1[mask].unsqueeze(0)  # [1, n_b, 2]
            pts2_b = pts2[mask].unsqueeze(0)  # [1, n_b, 2]
            # Force scores_b to shape [1, n_b, 1] explicitly
            scores_b = scores[mask].view(1, -1, 1)  # [1, n_b, 1]
            if pts1_b.shape[1] < 8:
                raise ValueError(
                    f"At least 8 point correspondences required for batch element {int(b)}"
                )
            F_b = self.forward_single(pts1_b, pts2_b, scores_b)
            F_list.append(F_b)
        return torch.cat(F_list, dim=0)  # [B, 3, 3]

    def forward_single(self, pts1, pts2, scores):
        # Apply softmax to scores over the match dimension and use sqrt(weighting)
        weights = torch.softmax(scores, dim=1)  # expected shape: [1, n, 1]
        pts1_norm, T1 = self.normalize_points(pts1)
        pts2_norm, T2 = self.normalize_points(pts2)
        x1, y1 = pts1_norm[..., 0], pts1_norm[..., 1]
        x2, y2 = pts2_norm[..., 0], pts2_norm[..., 1]

        A = torch.stack(
            [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, torch.ones_like(x1)],
            dim=2,  # Resulting shape: [1, n, 9]
        )

        # Multiply each row of A by sqrt(weight)
        A_weighted = A * torch.sqrt(weights)  # shapes: [1, n, 9] * [1, n, 1]
        U_A, S_A, Vh_A = torch.linalg.svd(
            A_weighted, full_matrices=False, driver="gesvd"
        )
        F = Vh_A[:, -1, :].view(1, 3, 3)

        U_f, S_f, Vh_f = torch.linalg.svd(F)
        S_f = S_f + self.epsilon  # stabilization

        # Smoothly suppress the smallest singular value via a sigmoid
        s_small = S_f[:, -1]
        s_small_corrected = s_small * torch.sigmoid(
            -self.alpha * (s_small - self.epsilon)
        )
        S_corrected = torch.stack([S_f[:, 0], S_f[:, 1], s_small_corrected], dim=1)
        F_rank2 = U_f.bmm(torch.diag_embed(S_corrected)).bmm(Vh_f)
        F_rank2 = F_rank2 / (torch.norm(F_rank2, dim=(1, 2), keepdim=True) + 1e-8)

        F_out = self.normalize_fundamental_matrix(
            T2.transpose(1, 2).bmm(F_rank2).bmm(T1)
        )
        return F_out

    @staticmethod
    def normalize_points(pts):
        # pts: [B, n, 2]
        B, n, _ = pts.shape
        mean = pts.mean(dim=1, keepdim=True)
        std = pts.std(dim=1, keepdim=True) + 1e-8
        scale = torch.sqrt(torch.tensor(2.0, device=pts.device)) / std.mean(
            dim=2, keepdim=True
        )
        zeros = torch.zeros(B, 1, 1, device=pts.device)
        ones = torch.ones(B, 1, 1, device=pts.device)
        T = torch.cat(
            [
                scale,
                zeros,
                -scale * mean[:, :, 0:1],
                zeros,
                scale,
                -scale * mean[:, :, 1:2],
                zeros,
                zeros,
                ones,
            ],
            dim=1,
        ).view(B, 3, 3)

        pts_h = torch.cat([pts, ones.expand(B, n, 1)], dim=2)
        pts_norm_h = T.bmm(pts_h.transpose(1, 2)).transpose(1, 2)
        return pts_norm_h[:, :, :2] / pts_norm_h[:, :, 2:3], T

    @staticmethod
    def normalize_fundamental_matrix(F):
        return F / (torch.norm(F, p="fro", dim=(1, 2), keepdim=True) + 1e-8)


class FundamentalEstimatorRANSAC(nn.Module):
    """
    Neural network module to estimate the fundamental matrix from point correspondences
    using a differentiable 8-point algorithm with match quality scores.
    Handles multiple point pairs per batch element using batch_indexes to group points.

    Returns:
        F: Tensor of shape (batch_size, 3, 3) containing the fundamental matrices,
           normalized by their Frobenius norm.
        mask: Tensor of shape (N,) where N is the total number of correspondences.
              Each element is 1 for an inlier and 0 for an outlier, matching the
              order of pts1/pts2 and indexable via batch_indexes.
    """

    def __init__(self):
        super(FundamentalEstimatorRANSAC, self).__init__()

    def forward(self, pts1, pts2, batch_indexes, max_epipolar_distance=1):
        device = pts1.device
        pts1_np = pts1.detach().cpu().numpy()
        pts2_np = pts2.detach().cpu().numpy()
        batch_indexes_np = batch_indexes.detach().cpu().numpy()
        max_batch_idx = batch_indexes_np.max()
        batch_size = max_batch_idx + 1

        fundamental_matrices = []
        # Create a final mask array that will have one value per correspondence.
        inliers_batched = []
        inliers_batch_idx = []
        for batch_idx in range(batch_size.astype(np.int32)):
            # Get points for the current batch
            batch_inliers = batch_indexes_np == batch_idx
            pts1_batch = pts1_np[batch_inliers].astype(np.float32)
            pts2_batch = pts2_np[batch_inliers].astype(np.float32)

            if (
                len(pts1_batch) < 8
            ):  # Minimum 8 points required for the 8-point algorithm
                F = np.zeros((3, 3), dtype=np.float32)
                inliers = np.zeros(len(pts1_batch), dtype=np.float32)
            else:
                # Estimate fundamental matrix using RANSAC; inliers indicates inliers (1) and outliers (0)
                try:
                    F, inliers = cv2.findFundamentalMat(
                        pts1_batch,
                        pts2_batch,
                        method=cv2.FM_RANSAC,
                        ransacReprojThreshold=max_epipolar_distance,
                        confidence=0.99,
                    )
                except Exception as e:
                    F = None

                if F is None or F.shape != (3, 3):
                    F = np.zeros((3, 3), dtype=np.float32)
                    inliers = np.zeros(len(pts1_batch), dtype=np.float32)
                else:
                    # Ensure inliers is a flat array.
                    inliers = inliers.ravel().astype(np.float32)

            fundamental_matrices.append(F)
            # Place the inlier/outlier inliers values in the corresponding positions of final_inliers.
            inliers_batched.append(torch.Tensor(inliers))
            inliers_batch_idx.append(torch.Tensor(len(inliers) * [batch_idx]))

        # Convert results back to PyTorch tensors
        F = torch.tensor(
            np.stack(fundamental_matrices), device=device, dtype=torch.float32
        )
        # Normalize F to prevent division by zero issues
        F = F / (F.norm(dim=(1, 2), keepdim=True) + 1e-8)
        inliers = torch.cat(inliers_batched, dim=0).bool()
        inliers_batch_idx = torch.cat(inliers_batch_idx, dim=0).bool()
        return F, inliers, inliers_batch_idx
