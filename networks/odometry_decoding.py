# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#


import torch
import torch.nn as nn
import math


class FUND_Predictor(nn.Module):
    def __init__(self, size="loftr", loftr_coarse=True):
        super(FUND_Predictor, self).__init__()

        self.loftr_coarse = loftr_coarse
        self.size = size
        if self.size == "loftr":
            in_channels = 256
        elif self.size == "dino":
            in_channels = 768
        elif self.size == "loftr+dino":
            in_channels = 256 + 768

        if loftr_coarse:
            self.loftr2px = LoFTR2px(
                input_channels=(in_channels),  # + 256),
                img_height=384,  # Replace with actual image height
                img_width=384,  # Replace with actual image width
            )
            self.match = CorrespondenceFinder()
            self.retrieve = CorrespondingPointsRetriever()
        self.fundamental_estimator = FundamentalEstimator()
        self.epipolar_refinement = EpipolarRefinement()
        # self.fundamental2essential = proj.Fundamental2Essential()
        # self.essential2candidates = proj.Essential2PoseCandidates()
        # self.candidates2pose = proj.DisambiguateCandidates()

    def forward(self, patchembeddings):
        if self.loftr_coarse:
            source_embedding = patchembeddings[:, 0]  # Shape: (B, C, H, W)
            target_embedding = patchembeddings[:, 1]  # Shape: (B, C, H, W)

            # Get pixel coordinates from embeddings
            source_pixels = self.loftr2px(source_embedding)  # Shape: (B, 2, H * W)
            target_pixels = self.loftr2px(target_embedding)  # Shape: (B, 2, H * W)

            # Flatten embeddings
            B, C, H, W = source_embedding.shape
            N = H * W
            source_embedding_flat = source_embedding.view(B, C, N)  # Shape: (B, C, N)
            target_embedding_flat = target_embedding.view(B, C, N)  # Shape: (B, C, N)

            # Find correspondences
            indices_target, initial_scores, sim_matrix = self.match(
                source_embedding_flat, target_embedding_flat
            )
            # Retrieve corresponding pixel coordinates
            source_matched, target_matched = self.retrieve(
                source_pixels, target_pixels, indices_target
            )
        else:
            source_matched, target_matched, initial_scores = patchembeddings

        F_greedy = self.fundamental_estimator(
            source_matched, target_matched, initial_scores
        )

        # Epipolar Refinement
        weights = self.epipolar_refinement(F_greedy, source_matched, target_matched)
        # Recompute Fundamental Matrix with refined weights
        F_refined = self.fundamental_estimator(
            source_matched, target_matched, weights.permute(1, 0)
        )
        initial_scores[:, 0] = 0
        return {
            "fundamental": F_greedy,
            "fundamental_greedy": F_greedy,
            "source_matches": source_matched,
            "target_matches": target_matched,
            "scores": initial_scores,
            # "similarity_matrix": sim_matrix,
        }


class LoFTR2px(nn.Module):
    """
    Revised LoFTR2px module that applies sinusoidal positional encoding to aid in mapping
    embeddings back to pixel coordinates.

    Args:
        input_channels (int): Number of channels in the backbone output (C).
        img_height (int): Height of the original image.
        img_width (int): Width of the original image.
        hidden_dim (int): Dimension of the hidden layers in the MLP.
        pos_dim (int): Dimensionality of the sinusoidal positional encoding.
    """

    def __init__(
        self, input_channels, img_height, img_width, hidden_dim=256, pos_dim=64
    ):
        super(LoFTR2px, self).__init__()
        self.input_channels = input_channels
        self.img_height = img_height
        self.img_width = img_width
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim

        # MLP input: embeddings (C) + basic coordinates (2) + positional encoding (pos_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_channels + 2 + pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),  # Outputs normalized offsets within [0, 1]
        )

    def forward(self, features):
        """
        Predict pixel coordinates for each embedding patch.

        Args:
            features (torch.Tensor): Embeddings of shape (B, C, H, W).

        Returns:
            torch.Tensor: Pixel coordinates of shape (B, 2, H * W).
        """
        B, C, H, W = features.shape
        device = features.device

        stride_y = self.img_height / H
        stride_x = self.img_width / W

        # Basic grid coordinates
        y_coords = torch.arange(0, H, device=device).unsqueeze(1).expand(H, W)
        x_coords = torch.arange(0, W, device=device).unsqueeze(0).expand(H, W)
        x_coords_flat = x_coords.reshape(-1)  # (H*W)
        y_coords_flat = y_coords.reshape(-1)  # (H*W)

        # Basic positional features (just integers)
        pos_2d = (
            torch.stack([x_coords_flat, y_coords_flat], dim=1)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )
        # pos_2d: (B, H*W, 2)

        # Sinusoidal positional encoding
        pos_enc = self.sinusoidal_pos_encoding(H, W, self.pos_dim, device)
        # pos_enc: (H, W, pos_dim)
        pos_enc_flat = pos_enc.view(-1, self.pos_dim).unsqueeze(0).expand(B, -1, -1)
        # pos_enc_flat: (B, H*W, pos_dim)

        # Flatten features and concatenate
        features_flat = features.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        # Combine embeddings, basic coordinates, and sinusoidal PE
        features_with_pos = torch.cat([features_flat, pos_2d, pos_enc_flat], dim=2)

        # Predict offset within each patch
        offsets = self.mlp(features_with_pos)  # (B, H*W, 2)

        # Compute top-left corner of each patch in the original image space
        x_top_left = x_coords_flat * stride_x
        y_top_left = y_coords_flat * stride_y
        top_left = (
            torch.stack([x_top_left, y_top_left], dim=1).unsqueeze(0).expand(B, -1, -1)
        )

        # Final pixel coordinates
        pixel_coords = top_left + offsets * torch.tensor(
            [stride_x, stride_y], device=device
        )
        # (B, H*W, 2) -> (B, 2, H*W)
        pixel_coords = pixel_coords.permute(0, 2, 1)

        return pixel_coords

    def sinusoidal_pos_encoding(self, H, W, d_model, device):
        """
        Create a 2D sinusoidal positional encoding of shape (H, W, d_model).
        Assumes d_model is divisible by 4 for simplicity.
        """
        assert d_model % 4 == 0
        d_half = d_model // 2
        d_quarter = d_half // 2  # Since we split between x and y, and then sin/cos

        # Create position tensors
        y_positions = (
            torch.arange(H, dtype=torch.float32, device=device)
            .unsqueeze(1)
            .unsqueeze(2)
        )  # (H,1,1)
        x_positions = (
            torch.arange(W, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .unsqueeze(2)
        )  # (1,W,1)

        # Create div_terms
        div_term = torch.exp(
            torch.arange(0, d_quarter, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / d_quarter)
        )  # (d_quarter,)

        # Add extra dimension for broadcasting
        div_term = div_term.unsqueeze(0).unsqueeze(0)  # (1,1,d_quarter)

        # Compute sinusoidal signals
        pe_y = torch.zeros(H, W, d_half, device=device)
        pe_x = torch.zeros(H, W, d_half, device=device)

        # Calculate y encodings
        sin_y = torch.sin(y_positions * div_term)  # (H,1,d_quarter)
        cos_y = torch.cos(y_positions * div_term)  # (H,1,d_quarter)
        pe_y[:, :, :d_quarter] = sin_y.expand(-1, W, -1)
        pe_y[:, :, d_quarter:] = cos_y.expand(-1, W, -1)

        # Calculate x encodings
        sin_x = torch.sin(x_positions * div_term)  # (1,W,d_quarter)
        cos_x = torch.cos(x_positions * div_term)  # (1,W,d_quarter)
        pe_x[:, :, :d_quarter] = sin_x.expand(H, -1, -1)
        pe_x[:, :, d_quarter:] = cos_x.expand(H, -1, -1)

        # Concatenate y and x encodings
        pe = torch.cat([pe_y, pe_x], dim=2)  # (H, W, d_model)

        return pe


class CorrespondenceFinder(nn.Module):
    """
    Module to find correspondences between two sets of embeddings.

    Args:
        ratio_threshold: The ratio threshold for filtering correspondences. Default is 0.8.
        embedding_mask: (Optional) A binary mask for embeddings2. If provided, invalid entries in embeddings2 will be ignored.
    """

    def __init__(self, ratio_threshold=0.8):
        super(CorrespondenceFinder, self).__init__()
        self.ratio_threshold = ratio_threshold

    def forward(self, embeddings1, embeddings2, embedding_mask=None):
        B, C, N = embeddings1.shape
        device = embeddings1.device

        # Normalize embeddings
        embeddings1_norm = nn.functional.normalize(embeddings1, p=2, dim=1)
        embeddings2_norm = nn.functional.normalize(embeddings2, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.bmm(embeddings1_norm.transpose(1, 2), embeddings2_norm)

        # Apply mask to similarity matrix (if embedding_mask is provided)
        if embedding_mask is not None:
            embedding_mask = embedding_mask.mean(dim=1)  # Collapse channels
            embedding_mask = embedding_mask.view(B, 1, N)  # Reshape for broadcasting
            sim_matrix = sim_matrix * embedding_mask  # Set invalid regions to 0

        # Forward matching (1->2)
        scores_12, indices_12 = torch.max(sim_matrix, dim=2)  # Shape: (B, N)

        # Backward matching (2->1)
        scores_21, indices_21 = torch.max(
            sim_matrix.transpose(1, 2), dim=2
        )  # Shape: (B, N)

        # Get second-best scores for ratio test
        sim_matrix_clone = sim_matrix.clone()
        # Set best scores to -inf to get second best
        sim_matrix_clone.scatter(2, indices_12.unsqueeze(2), float("-inf"))
        scores_12_second, _ = torch.max(sim_matrix_clone, dim=2)

        # Compute ratio scores
        ratio_scores = scores_12 / (scores_12_second + 1e-8)

        # Cycle consistency check
        cycle_consistent = torch.gather(indices_21, 1, indices_12) == torch.arange(
            N, device=device
        ).unsqueeze(0).expand(B, N)

        # Combined confidence score
        valid_matches = cycle_consistent
        final_scores = torch.where(
            valid_matches, scores_12, torch.zeros_like(scores_12)
        )

        # Normalize scores (optional)
        # final_scores = nn.functional.softmax(final_scores, dim=1)

        return indices_12, final_scores, sim_matrix


class CorrespondingPointsRetriever(nn.Module):
    """
    Module to retrieve corresponding pixel coordinates based on matching indices.
    """

    def __init__(self):
        super(CorrespondingPointsRetriever, self).__init__()

    def forward(self, coords1, coords2, indices2):
        """
        Retrieve corresponding pixel coordinates.

        Args:
            coords1 (torch.Tensor): Coordinates from image 1, shape (B, 2, N)
            coords2 (torch.Tensor)f: Coordinates from image 2, shape (B, 2, N)
            indices2 (torch.Tensor): Indices of matching coordinates in image 2, shape (B, N)

        Returns:
            pts1 (torch.Tensor): Corresponding points from image 1, shape (B, N, 2)
            pts2 (torch.Tensor): Corresponding points from image 2, shape (B, N, 2)
        """
        # Gather matching coordinates from image 2
        indices_expanded = indices2.unsqueeze(1).expand(-1, 2, -1)  # Shape: (B, 2, N)
        coords2_matched = torch.gather(coords2, 2, indices_expanded)  # Shape: (B, 2, N)

        # Transpose to shape (B, N, 2)
        pts1 = coords1.permute(0, 2, 1)  # Shape: (B, N, 2)
        pts2 = coords2_matched.permute(0, 2, 1)  # Shape: (B, N, 2)

        return pts1, pts2


import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FundamentalEstimator(nn.Module):
    def __init__(self, alpha=50.0, epsilon=1e-6):
        """
        alpha: Controls the steepness of the sigmoid for smoothing.
        epsilon: Small positive value to stabilize near-degenerate cases.
        """
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, pts1, pts2, scores):
        B, N, _ = pts1.shape
        if N < 8:
            raise ValueError("At least 8 point correspondences required.")

        # Use a soft weighting via softmax on the input scores
        normalized_scores = nn.functional.softmax(scores, dim=1)

        pts1_norm, T1 = self.normalize_points(pts1)
        pts2_norm, T2 = self.normalize_points(pts2)

        x1, y1 = pts1_norm[:, :, 0], pts1_norm[:, :, 1]
        x2, y2 = pts2_norm[:, :, 0], pts2_norm[:, :, 1]
        A = torch.stack(
            [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, torch.ones_like(x1)],
            dim=2,
        )

        # Weighted SVD
        U_A, S_A, Vh_A = torch.linalg.svd(A, full_matrices=False, driver="gesvd")
        F = Vh_A[:, -1, :].view(B, 3, 3)

        # Compute SVD of initial F
        U_f, S_f, Vh_f = torch.linalg.svd(F)

        # Add epsilon to avoid exact zero singular values that cause instability
        S_f = S_f + self.epsilon

        # Apply a smooth approximation to enforce rank-2:
        # We want to smoothly set the smallest singular value to ~0.
        # Use a sigmoid-based "soft threshold":
        # s_small_corrected = s_small * sigmoid(-alpha*(s_small - epsilon))
        # For s_small close to epsilon, this pushes it closer to zero.
        s_small = S_f[:, -1]
        s_small_corrected = s_small * torch.sigmoid(
            -self.alpha * (s_small - self.epsilon)
        )

        # Construct corrected singular values
        S_corrected = torch.stack([S_f[:, 0], S_f[:, 1], s_small_corrected], dim=1)

        # Reconstruct F with smoothed singular values
        F_rank2 = U_f.bmm(torch.diag_embed(S_corrected)).bmm(Vh_f)

        # Normalize fundamental matrix
        F_rank2 = F_rank2 / (torch.norm(F_rank2, dim=(1, 2), keepdim=True) + 1e-8)

        F_out = self.normalize_fundamental_matrix(
            T2.transpose(1, 2).bmm(F_rank2).bmm(T1)
        )
        return F_out

    @staticmethod
    def normalize_points(pts):
        B, N, _ = pts.shape
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

        pts_h = torch.cat([pts, ones.expand(B, N, 1)], dim=2)
        pts_norm_h = T.bmm(pts_h.transpose(1, 2)).transpose(1, 2)
        return pts_norm_h[:, :, :2] / pts_norm_h[:, :, 2:3], T

    @staticmethod
    def normalize_fundamental_matrix(F):
        return F / (torch.norm(F, p="fro", dim=(1, 2), keepdim=True) + 1e-8)


class EpipolarRefinement(nn.Module):
    def __init__(self):
        super(EpipolarRefinement, self).__init__()

    def forward(self, F, source_points, target_points):
        # Compute epipolar lines for source and target points

        source_points = torch.cat(
            [source_points, torch.ones_like(source_points[:, :, 0:1])], dim=-1
        )
        target_points = torch.cat(
            [target_points, torch.ones_like(target_points[:, :, 0:1])], dim=-1
        )

        lines_target = torch.bmm(F, source_points.transpose(1, 2))  # Shape: (B, 3, N)
        lines_source = torch.bmm(F.transpose(1, 2), target_points.transpose(1, 2))

        # Compute epipolar distances
        denom_target = torch.sqrt(
            lines_target[:, 0, :] ** 2 + lines_target[:, 1, :] ** 2 + 1e-8
        )
        denom_source = torch.sqrt(
            lines_source[:, 0, :] ** 2 + lines_source[:, 1, :] ** 2 + 1e-8
        )

        dist_target = (
            torch.sum(target_points.transpose(1, 2) * lines_target, dim=1).abs()
            / denom_target
        ).transpose(1, 0)
        dist_source = (
            torch.sum(source_points.transpose(1, 2) * lines_source, dim=1).abs()
            / denom_source
        ).transpose(1, 0)

        epipolar_dist = dist_target + dist_source  # Shape: (B, N)

        # Compute weights using a robust loss function
        weights = torch.nn.functional.softmax(epipolar_dist, dim=1)

        return weights


class TranslationScaler(nn.Module):
    def __init__(self):
        super(TranslationScaler, self).__init__()
        # Embed CLS tokens into 3D space to match the translation vector's dimensionality
        self.cls_embed = nn.Linear(768, 3)

    def forward(self, translation_vector, cls_token1, cls_token2):
        """
        Args:
            translation_vector (torch.Tensor): Normalized translation vectors of shape (B, 3)
            cls_token1 (torch.Tensor): First CLS token of shape (B, 768)
            cls_token2 (torch.Tensor): Second CLS token of shape (B, 768)
        Returns:
            torch.Tensor: Scale factors of shape (B,), can be negative
        """
        # Project CLS tokens into 3D space
        cls_vec1 = self.cls_embed(cls_token1)  # Shape: (B, 3)
        cls_vec2 = self.cls_embed(cls_token2)  # Shape: (B, 3)

        # Compute the difference between the embedded CLS vectors
        cls_diff = cls_vec2 - cls_vec1  # Shape: (B, 3)

        # Calculate the scale as the dot product between cls_diff and translation_vector
        # This inherently accounts for direction and can be negative
        scale = (
            torch.sum(cls_diff * translation_vector, dim=1)
            .unsqueeze(-1)
            .expand(cls_vec1.shape[0], 3)
        )  # Shape: (B,)
        return scale
