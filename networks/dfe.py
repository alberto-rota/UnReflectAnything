import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, inplace=True, has_bias=True, learn_affine=True):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1, bias=has_bias)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=has_bias)
        self.inorm1 = nn.InstanceNorm1d(128)
        self.bnorm1 = nn.BatchNorm1d(128)
        self.inorm2 = nn.InstanceNorm1d(128)
        self.bnorm2 = nn.BatchNorm1d(128)

    def forward(self, data):
        x = self.bnorm1(self.inorm1(self.conv1(data)))
        x = F.relu(self.bnorm2(self.inorm2(self.conv2(x))))
        return data + x


class WeightEstimatorNet(nn.Module):
    """Network for weight estimation."""

    def __init__(self, input_size, inplace=True, has_bias=True, learn_affine=True):
        """Init.

        Args:
            input_size (float): size of input
            inplace (bool, optional): Defaults to True. LeakyReLU inplace?
            has_bias (bool, optional): Defaults to True. Conv1d bias?
            learn_affine (bool, optional): Defaults to True. InstanceNorm1d affine?
        """

        super(WeightEstimatorNet, self).__init__()

        track = False
        has_bias = True
        learn_affine = True
        self.model = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(64, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(64, 128, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(128, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(128, 1024, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(1024, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(1024, 512, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(512, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(512, 256, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(256, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(256, 1, kernel_size=1, bias=has_bias),
        )

    def forward(self, data):
        """Forward pass.

        Args:
            data (tensor): input data

        Returns:
            tensor: forward pass
        """

        return self.model(data)


class RescaleAndExpand(nn.Module):
    """Normalizes the input points to [-1, 1]^2 and transforms them to homogenous coordinates.
    Modified to handle variable number of points per batch.
    """

    def __init__(self):
        """Init."""
        super(RescaleAndExpand, self).__init__()
        self.register_buffer("ones", torch.ones((1, 1)))

    def normalize(self, pts, batch_indices, batch_size):
        """Normalizes the input points to [-1, 1]^2 and transforms them to homogenous coordinates.
        Handles variable number of points per batch.

        Args:
            pts (tensor): input points [N, 2]
            batch_indices (tensor): batch indices for each point [N]
            batch_size (int): total batch size

        Returns:
            tensor: transformed points
            tensor: transformation matrices
        """
        # Create homogeneous coordinates
        ones = self.ones.expand(pts.size(0), 1)
        pts_homog = torch.cat((pts, ones), 1)  # [N, 3]

        # Initialize transformations
        transform = torch.zeros((batch_size, 3, 3), device=pts.device)
        transformed_pts = []

        # Process each batch separately
        for b in range(batch_size):
            # Get points for this batch
            mask = batch_indices == b
            if not mask.any():
                # Handle empty batch (no points)
                transform[b, 0, 0] = 1
                transform[b, 1, 1] = 1
                transform[b, 2, 2] = 1
                continue

            pts_batch = pts_homog[mask]

            # Calculate center of points
            center = torch.mean(pts_batch[:, :2], 0)

            # Calculate distances from center
            dist = pts_batch[:, :2] - center
            meandist = dist.pow(2).sum(1).sqrt().mean()

            # Calculate scale
            scale = 1.0 / meandist if meandist > 0 else 1.0

            # Build transformation matrix
            transform[b, 0, 0] = scale
            transform[b, 1, 1] = scale
            transform[b, 2, 2] = 1
            transform[b, 0, 2] = -center[0] * scale
            transform[b, 1, 2] = -center[1] * scale

            # Apply transformation
            pts_transformed = torch.matmul(transform[b], pts_batch.t())
            transformed_pts.append(pts_transformed)

        return transformed_pts, transform

    def forward(self, source_pts, target_pts, batch_indices, batch_size):
        """Forward pass.

        Args:
            source_pts (tensor): points in first image [N, 2]
            target_pts (tensor): points in second image [N, 2]
            batch_indices (tensor): batch indices for each point [N]
            batch_size (int): total batch size

        Returns:
            list: transformed points in first image (list of tensors)
            list: transformed points in second image (list of tensors)
            tensor: transformation (first image)
            tensor: transformation (second image)
        """
        pts1_transformed, transform1 = self.normalize(
            source_pts, batch_indices, batch_size
        )
        pts2_transformed, transform2 = self.normalize(
            target_pts, batch_indices, batch_size
        )

        return pts1_transformed, pts2_transformed, transform1, transform2


class ModelEstimator(nn.Module):
    """Estimator for model. Modified to handle variable number of points per batch."""

    def __init__(self):
        """Init."""
        super(ModelEstimator, self).__init__()
        self.register_buffer("mask", torch.ones(3))
        self.mask[-1] = (
            0  # Set the last value to 0 to enforce rank-2 constraint for fundamental matrix
        )

    def weighted_svd(
        self,
        pts1_list,
        pts2_list,
        weights,
        batch_indices,
        batch_size,
        transforms1,
        transforms2,
    ):
        """Solve homogeneous least squares problem and extract model.
        Modified to handle variable number of points per batch.

        Args:
            pts1_list (list): list of transformed points in first image
            pts2_list (list): list of transformed points in second image
            weights (tensor): estimated weights with shape [1, 1, N]
            batch_indices (tensor): batch indices for each point [N]
            batch_size (int): total batch size
            transforms1 (tensor): transformation matrices for first image
            transforms2 (tensor): transformation matrices for second image

        Returns:
            tensor: estimated fundamental matrices for each batch
        """
        # Initialize output tensor
        out = torch.zeros((batch_size, 3, 3), device=weights.device)

        # Reshape weights to match batch_indices
        weights_reshaped = weights.squeeze().unsqueeze(1)  # Shape: [N, 1]

        # Process each batch separately
        for b in range(batch_size):
            # Get points for this batch
            mask = batch_indices == b
            if not mask.any() or mask.sum() < 8:
                # Not enough points for F-matrix estimation (need at least 8)
                continue

            batch_weights = weights_reshaped[mask]

            # For variable point count, we need to gather the points from the list
            if pts1_list and pts2_list:
                pts1 = pts1_list[b]
                pts2 = pts2_list[b]

                # Construct the constraint matrix
                # p = [x'x, x'y, x', y'x, y'y, y', x, y, 1]
                p = torch.zeros((mask.sum(), 9), device=weights.device)

                for i in range(pts1.shape[1]):
                    x1, y1, _ = pts1[:, i]
                    x2, y2, _ = pts2[:, i]
                    p[i, 0] = x1 * x2
                    p[i, 1] = x1 * y2
                    p[i, 2] = x1
                    p[i, 3] = y1 * x2
                    p[i, 4] = y1 * y2
                    p[i, 5] = y1
                    p[i, 6] = x2
                    p[i, 7] = y2
                    p[i, 8] = 1.0

                # Apply weights
                X = p * batch_weights

                # Solve homogeneous least squares problem
                _, _, V = torch.svd(X)
                F = V[:, -1].view(3, 3)

                # Project to rank 2
                U, S, V = torch.svd(F)
                S_modified = S.clone()  # Create a new tensor
                S_modified[-1] = 0  # Modify the new tensor
                F_projected = U @ torch.diag(S_modified) @ V.t()

                # Denormalize
                F_final = transforms1[b].t() @ F_projected @ transforms2[b]

                # Scale to have unit Frobenius norm
                F_final = F_final / torch.norm(F_final)

                out[b] = F_final

        return out

    def forward(
        self,
        pts1_list,
        pts2_list,
        weights,
        batch_indices,
        batch_size,
        transforms1,
        transforms2,
    ):
        """Forward pass.

        Args:
            pts1_list (list): list of transformed points in first image
            pts2_list (list): list of transformed points in second image
            weights (tensor): estimated weights
            batch_indices (tensor): batch indices for each point
            batch_size (int): total batch size
            transforms1 (tensor): transformation matrices for first image
            transforms2 (tensor): transformation matrices for second image

        Returns:
            tensor: estimated fundamental matrix
        """
        out = self.weighted_svd(
            pts1_list,
            pts2_list,
            weights,
            batch_indices,
            batch_size,
            transforms1,
            transforms2,
        )
        return out


def compute_residuals(
    source_pts, target_pts, batch_indices, batch_size, fundamental_mats
):
    """Compute robust symmetric epipolar distance for variable point count.

    Args:
        source_pts (tensor): points in first image [N, 2]
        target_pts (tensor): points in second image [N, 2]
        batch_indices (tensor): batch indices for each point [N]
        batch_size (int): total batch size
        fundamental_mats (tensor): fundamental matrices [batch_size, 3, 3]

    Returns:
        tensor: robust symmetric epipolar distances [N]
    """
    # Create homogeneous coordinates
    ones = torch.ones((source_pts.size(0), 1), device=source_pts.device)
    source_pts_homog = torch.cat((source_pts, ones), 1)  # [N, 3]
    target_pts_homog = torch.cat((target_pts, ones), 1)  # [N, 3]

    # Initialize output
    residuals = torch.zeros(source_pts.size(0), device=source_pts.device)

    # Compute for each point
    for i in range(source_pts.size(0)):
        b = batch_indices[i].item()
        F = fundamental_mats[b]

        # Compute epipolar lines
        line1 = F @ source_pts_homog[i]
        line2 = target_pts_homog[i] @ F

        # Compute symmetric epipolar distance
        scalar_product = (target_pts_homog[i] * line1).sum()

        # Compute normalization factors
        norm1 = torch.norm(line1[:2])
        norm2 = torch.norm(line2[:2])

        # Compute symmetric distance
        if norm1 > 1e-8 and norm2 > 1e-8:
            dist = scalar_product.abs() * (1.0 / norm1 + 1.0 / norm2)
            # Apply robust function (clamp at gamma=0.5)
            residuals[i] = torch.clamp(dist, max=0.5)

    return residuals


class ModifiedNormalizedEightPointNet(nn.Module):
    """Modified NormalizedEightPointNet for fundamental matrix estimation with variable point count.

    The input format is different from the original:
    - source_pts: Nx2 tensor of points in first image
    - target_pts: Nx2 tensor of points in second image
    - batch_indices: Nx1 tensor indicating which batch each point belongs to
    - batch_size: integer indicating total number of batches
    """

    def __init__(self, depth=1, side_info_size=0):
        """Init.

        Args:
            depth (int, optional): Defaults to 1. Iteration depth for weight refinement
            side_info_size (int, optional): Defaults to 0. Additional point info dimension
        """
        super(ModifiedNormalizedEightPointNet, self).__init__()

        self.depth = depth

        # Data processing
        self.rescale_and_expand = RescaleAndExpand()

        # Model estimator
        self.model = ModelEstimator()

        # Weight estimator networks
        self.weights_init = WeightEstimatorNet(4 + side_info_size)
        self.weights_iter = WeightEstimatorNet(6 + side_info_size)

    def forward(self, source_pts, target_pts, batch_indices, side_info=None):
        """Forward pass with variable point count support.

        Args:
            source_pts (tensor): Nx2 tensor of points in first image
            target_pts (tensor): Nx2 tensor of points in second image
            batch_indices (tensor): Nx1 tensor indicating batch membership
            batch_size (int): total number of batches
            side_info (tensor, optional): Additional point information

        Returns:
            dict: Dictionary containing fundamental matrices and other outputs
        """
        # Ensure batch_indices has the right shape (flatten if it's Nx1)size =
        batch_size = batch_indices.max().item() + 1
        if batch_indices.dim() > 1:
            batch_indices = batch_indices.squeeze()

        # Handle side_info
        if side_info is None:
            side_info = torch.zeros((source_pts.size(0), 0), device=source_pts.device)

        # Normalize points and get transformation matrices
        pts1_list, pts2_list, transform1, transform2 = self.rescale_and_expand(
            source_pts, target_pts, batch_indices, batch_size
        )

        # Prepare input for weight estimation
        # Normalize coordinates to [0,1] for network input
        normalized_source = (source_pts - source_pts.min(0)[0]) / (
            source_pts.max(0)[0] - source_pts.min(0)[0] + 1e-8
        )
        normalized_target = (target_pts - target_pts.min(0)[0]) / (
            target_pts.max(0)[0] - target_pts.min(0)[0] + 1e-8
        )

        # Concatenate inputs for weight estimation
        input_features = torch.cat(
            (normalized_source, normalized_target, side_info), dim=1
        )
        input_features = input_features.t().unsqueeze(0)  # Shape: [1, features, N]

        # Initial weight estimation
        weights = F.softmax(self.weights_init(input_features), dim=2)

        # First model estimation
        out_depth = self.model(
            pts1_list,
            pts2_list,
            weights,
            batch_indices,
            batch_size,
            transform1,
            transform2,
        )
        out = [out_depth]

        # Iterative refinement
        for _ in range(1, self.depth):
            # Compute residuals
            residual = compute_residuals(
                source_pts, target_pts, batch_indices, batch_size, out_depth
            )

            # Update weights
            input_features_with_weights = torch.cat(
                (input_features, weights, residual.unsqueeze(0).unsqueeze(0)), dim=1
            )
            weights = F.softmax(self.weights_iter(input_features_with_weights), dim=2)

            # Re-estimate model
            out_depth = self.model(
                pts1_list,
                pts2_list,
                weights,
                batch_indices,
                batch_size,
                transform1,
                transform2,
            )
            out.append(out_depth)

        # Prepare output

        return {
            "fundamental": out[-1],
            "transform1": transform1,
            "transform2": transform2,
            "weights": weights,  # Shape: [N, 1]
        }


"""Loss functions.
"""
import torch


def symmetric_epipolar_distance(pts1, pts2, fundamental_mat):
    """Symmetric epipolar distance.

    Args:
        pts1 (tensor): points in first image
        pts2 (tensor): point in second image
        fundamental_mat (tensor): fundamental matrix

    Returns:
        tensor: symmetric epipolar distance
    """

    line_1 = torch.bmm(pts1, fundamental_mat)
    line_2 = torch.bmm(pts2, fundamental_mat.permute(0, 2, 1))

    scalar_product = (pts2 * line_1).sum(2)

    ret = scalar_product.abs() * (
        1 / line_1[:, :, :2].norm(2, 2) + 1 / line_2[:, :, :2].norm(2, 2)
    )

    return ret


# def robust_symmetric_epipolar_distance(pts1, pts2, fundamental_mat, gamma=1.0):
def robust_symmetric_epipolar_distance(pts1, pts2, fundamental_mat, gamma=0.5):
    """Robust symmetric epipolar distance.

    Args:
        pts1 (tensor): points in first image
        pts2 (tensor): point in second image
        fundamental_mat (tensor): fundamental matrix
        gamma (float, optional): Defaults to 0.5. robust parameter

    Returns:
        tensor: robust symmetric epipolar distance
    """

    sed = symmetric_epipolar_distance(pts1, pts2, fundamental_mat)
    ret = torch.clamp(sed, max=gamma)

    return ret


def vectorized_symmetric_epipolar_distance(
    pts1, pts2, fundamental_mats, batch_indices, batch_size
):
    """Vectorized symmetric epipolar distance for variable point count data structure.

    Args:
        pts1 (tensor): points in first image [N, 3]
        pts2 (tensor): points in second image [N, 3]
        fundamental_mats (tensor): fundamental matrices [batch_size, 3, 3]
        batch_indices (tensor): batch indices for each point [N]
        batch_size (int): total number of batches

    Returns:
        tensor: symmetric epipolar distance [N]
    """
    # Ensure batch_indices is properly shaped
    batch_indices = batch_indices.squeeze()

    # Initialize output tensor
    N = pts1.size(0)
    distances = torch.zeros(N, device=pts1.device)

    # Create a mapping from each point to its corresponding fundamental matrix
    # This is the key to vectorization
    point_F_matrices = fundamental_mats[batch_indices]  # Shape: [N, 3, 3]

    # Calculate epipolar lines for all points simultaneously
    # For each point i: line_1[i] = point_F_matrices[i] @ pts1[i]
    # Reshape pts1 to [N, 3, 1] for matmul with [N, 3, 3]
    pts1_reshaped = pts1.unsqueeze(2)  # Shape: [N, 3, 1]
    line_1 = torch.matmul(point_F_matrices, pts1_reshaped).squeeze(2)  # Shape: [N, 3]

    # For line_2, we need F^T for each point
    # Transpose each F matrix in the batch
    point_F_matrices_transposed = point_F_matrices.transpose(1, 2)  # Shape: [N, 3, 3]

    # Calculate line_2 for all points
    pts2_reshaped = pts2.unsqueeze(2)  # Shape: [N, 3, 1]
    line_2 = torch.matmul(point_F_matrices_transposed, pts2_reshaped).squeeze(
        2
    )  # Shape: [N, 3]

    # Calculate scalar product for all points
    # This is the dot product between pts2 and line_1
    scalar_product = (pts2 * line_1).sum(dim=1)  # Shape: [N]

    # Calculate normalization factors for all points
    norm_line_1 = torch.norm(line_1[:, :2], dim=1)  # Shape: [N]
    norm_line_2 = torch.norm(line_2[:, :2], dim=1)  # Shape: [N]

    # Create combined normalization factor with safety for division
    safe_norm_factor = 1.0 / (norm_line_1 + 1e-10) + 1.0 / (norm_line_2 + 1e-10)

    # Where norms are too small, set factor to zero
    valid_norms = (norm_line_1 > 1e-8) & (norm_line_2 > 1e-8)
    safe_norm_factor = safe_norm_factor * valid_norms.float()

    # Calculate final distances
    distances = scalar_product.abs() * safe_norm_factor

    return distances


def modified_robust_symmetric_epipolar_distance(
    pts1, pts2, fundamental_mats, batch_indices, batch_size, gamma=10
):
    """Robust symmetric epipolar distance for variable point count.

    Args:
        pts1 (tensor): points in first image [N, 3]
        pts2 (tensor): points in second image [N, 3]
        fundamental_mats (tensor): fundamental matrices [batch_size, 3, 3]
        batch_indices (tensor): batch indices for each point [N]
        batch_size (int): total number of batches
        gamma (float, optional): robust parameter (default: 0.5)

    Returns:
        tensor: mean robust symmetric epipolar distance
    """
    # Calculate standard symmetric epipolar distance
    sed = vectorized_symmetric_epipolar_distance(
        pts1, pts2, fundamental_mats, batch_indices, batch_size
    )

    # Apply robust function (clamping)
    robust_sed = torch.clamp(sed, max=gamma)

    # Return mean for loss computation
    return robust_sed.mean()
