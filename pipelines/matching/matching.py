import torch
import torch.nn as nn

from utilities import *
from networks.base import MONO3DModel
import networks.backbones as backbones
import networks.depth_decoding as depth_decoding
import copy

from pipelines.features.featureextractor import FeatureExtractor
import pipelines.matching as matching
from pipelines.matching.helpers import *
from pipelines.matching import learning, epipolar, refinement, correspondence, metrics

import warnings

torch.autograd.set_detect_anomaly(True)


class MatchingPipeline:
    def __init__(
        self,
        config,
        model=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        ### INITIALIZATIONS
        self.config = config
        self.device = device

        ### MODULES
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            if isinstance(model, MatcherBackbone):  # If model passed directly, use it
                self.model = model.to(self.device)
            elif (
                isinstance(model, str) and model != ""
            ):  # If model passed as a name str, load checkpoint
                self.model = MatcherBackbone(shared_key="asdjnasljkn").to(self.device)
                self.model.fromArtifact(model)
            elif (
                config.get("RUN") is None
                or config.get("RUN") == ""
                or model == ""
                or model == None
            ):  # If no model or model name provided, initalize from scratch.
                self.model = MatcherBackbone(
                    backbone_brand=config["BACKBONE_BRAND"],
                    size=config["BACKBONE_SIZE"],
                    resampled_patch_size=config["RESAMPLED_PATCH_SIZE"],
                ).to(self.device)

        self.warp = matching.projections.Warp(
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
        ).to(self.device)
        self.RANSAC_unwrapped = epipolar.FundamentalEstimatorRANSAC()
        self.EIGHPA_raw = epipolar.FundamentalEstimator8PA()

        ### DIMENSIONS
        self.height, self.width, self.batch_size = (
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
            config.BATCH_SIZE,
        )
        _, self.embed_dim, self.seq_len = self.model.embed(
            torch.randn(
                self.batch_size,
                3,
                self.height,
                self.width,
            ).to(self.device),
            mode="seq",
        ).shape
        self.patch_size = int((self.height * self.width / self.seq_len) ** 0.5)

    def synthethize_ground_truth(
        self,
        framestack,
        K,
        camera_pose_gt,
        depthstack=None,
        source_matched_points=None,
        batch_idx_match=None,
    ):
        if depthstack is None:
            depthstack = (
                1 / self.model.depth(framestack) * self.config.DEPTH_SCALE_FACTOR
            )
        if source_matched_points is None:
            source_matched_points = matching.learning.generate_grid(
                num_points=self.config.TRIPLETS_TO_MINE,
                batch_size=framestack.shape[0],
                framestack=framestack,  # or image_height, image_width
                device=self.device,
            )

        warping_output = self.warp(
            framestack[:, 0],  # Source image
            depthstack,  # Depth map
            K,  # Camera intrinsics
            camera_pose_gt,  # Camera pose
            points_to_match=source_matched_points,
            batch_idx_match=batch_idx_match,
            return_mask=True,
            return_artifacts=True,
        )
        warped, target_matched_points_true, mask = (
            warping_output["warped"],
            warping_output["matches"],
            warping_output["mask"],
        )

        # Build an embedding mask from the projector's occlusion mask
        embedding_mask = chw2embedding(
            embedding_mask_from_pixels(
                mask,
                patch_size=self.config.RESAMPLED_PATCH_SIZE,
                embedding_dim=self.embed_dim,
            )
        )
        return {
            "warped": warped,
            "source_matched_points": source_matched_points,
            "target_matched_points_true": target_matched_points_true,
            "embedding_mask": embedding_mask,
            # "cloud": warping_output["cloud"],
            # "rgb_vec": warping_output["rgb_vec"],
        }

    def mine_triplets(
        self,
        modeloutput,
        source_matched_points,
        target_matched_points_true,
        embedding_mask,
    ):
        source_embs = modeloutput["source_embedding_match"]  # [B, C, HW?]
        target_embs = modeloutput["target_embedding_match"]
        source_matched_embs, source_matched_embs_idx, _ = points_to_patches(
            source_matched_points,
            embedding2chw(source_embs, embed_dim_last=False),
            patch_size=self.patch_size,
        )
        target_matched_embs, target_matched_embs_idx, embedding_match_mask = (
            points_to_patches(
                target_matched_points_true,
                embedding2chw(target_embs, embed_dim_last=False),
                patch_size=self.patch_size,
                mask=embedding2chw(embedding_mask, embed_dim_last=False),
            )
        )
        triplets_dict = learning.mine_triplets_optimized(
            sourceembs=source_matched_embs,
            targetembs=target_matched_embs,
            sourceembs_idx=source_matched_embs_idx,
            targetembs_idx=target_matched_embs_idx,
            target_mask=embedding_match_mask,
        )
        return triplets_dict

    def compute_correspondences(
        self,
        modeloutput,
        framestack,
        embedding_mask=None,
        knn=1,
    ):
        """
        Compute correspondences between source and target patches using the model output and kNN.
        Args:
            modeloutput: Dictionary containing model outputs
            framestack: Stack of frames to process
            embedding_mask: Optional mask for embeddings (e.g., for occlusion)
            knn: Number of nearest neighbors to consider (default=1)
        """
        # Extract embeddings and get initial matches
        (
            batch_idx_match,
            source_pixels_matched,
            target_pixels_matched,
            source_patches_matched,
            target_patches_matched,
            descriptor_scores,  # Renamed to descriptor_scores to clarify
            sim_matrix,
        ) = correspondence.get_matching_points_with_patches(
            modeloutput["source_embedding_match"],
            modeloutput["target_embedding_match"],
            framestack[:, 0],
            framestack[:, 1],
            threshold=self.config.PATCH_MATCHING_SCORE_THRESHOLD,
            min_matches=self.config.MIN_MATCHES_TO_COLLECT,
            max_matches=self.config.MAX_MATCHES_TO_PROCESS,
            embedding_mask=embedding_mask,
            patch_size=self.config.RESAMPLED_PATCH_SIZE,
            patch_size_enlarged=self.config.REFINEMENT_AREA,
            knn=knn,
        )
        source_pixel_offset, target_pixel_offset, refinement_scores = (
            refinement.FFT_patch_refiner(
                source_patches_matched,
                target_patches_matched,
                patch_size=self.config.REFINEMENT_AREA,
            )
        )

        points_data = {
            "source_pixels": source_pixels_matched,
            "target_pixels": target_pixels_matched,
            "batch_idx": batch_idx_match,
            "source_patches": source_patches_matched,
            "target_patches": target_patches_matched,
            "source_offset": source_pixel_offset,
            "target_offset": target_pixel_offset,
        }

        # # Filter based on scores
        # filtered_points, scores = filter_scores(points_data, scores)

        # # Extract filtered data
        # source_pixels_matched = filtered_points["source_pixels"]
        # target_pixels_matched = filtered_points["target_pixels"]
        # batch_idx_match = filtered_points["batch_idx"]
        # source_patches_matched = filtered_points["source_patches"]
        # target_patches_matched = filtered_points["target_patches"]
        # source_pixel_offset = filtered_points["source_offset"]
        # target_pixel_offset = filtered_points["target_offset"]

        # Apply refinement offsets to the matched pixel coordinates
        patch_size = self.config.REFINEMENT_AREA
        source_pixels_matched, target_pixels_matched = apply_refinement_offsets(
            source_pixels_matched,
            target_pixels_matched,
            source_pixel_offset,
            target_pixel_offset,
            patch_size,
        )
        # Return the correspondence results
        return {
            "source_pixels_matched": source_pixels_matched,
            "target_pixels_matched": target_pixels_matched,
            "batch_idx_match": batch_idx_match,
            "descriptor_scores": descriptor_scores,  # Use our unified scores
            "refinement_scores": refinement_scores,  # Use our unified scores
            "sim_matrix": sim_matrix,
        }

    def RANSAC(
        self,
        source_pixels_matched,
        target_pixels_matched,
        batch_idx_match,
    ):
        """
        Run RANSAC to estimate fundamental matrix and identify inliers.

        Args:
            source_pixels_matched: Source pixel coordinates of shape [N, 2]
            target_pixels_matched: Target pixel coordinates of shape [N, 2]
            batch_idx_match: Batch indices of shape [N]

        Returns:
            F: Estimated fundamental matrix of shape [B, 3, 3]
            inliers: Boolean tensor of shape [N] indicating inliers
            updated_scores: If scores provided, returns scores updated with epipolar information
        """
        # Run RANSAC to estimate fundamental matrix
        F, inliers, _ = self.RANSAC_unwrapped(
            source_pixels_matched,
            target_pixels_matched,
            batch_idx_match,
            max_epipolar_distance=self.config.MAX_EPIPOLAR_DISTANCE,
        )

        epipolar_errors = metrics.epipolar_error(
            source_pixels_matched,
            target_pixels_matched,
            F,
            batch_idx_match,
            reduction="none",
        )
        # Convert errors to scores (1.0 for error=0, decreasing as error increases)
        epipolar_scores = torch.exp(
            -epipolar_errors / self.config.MAX_EPIPOLAR_DISTANCE
        )

        return {
            "F": F,
            "inliers": inliers,
            "scores": epipolar_scores,
        }

    def EightPointAlgorithm(
        self, source_pixels_matched, target_pixels_matched, batch_idx_match, scores
    ):
        F = self.eightPA(
            source_pixels_matched.float(),
            target_pixels_matched.float(),
            scores.float(),
            batch_idx_match,
        )
        return {"F": F}

    def compute_metrics(
        self,
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
        precision, recall, AUCPR = metrics.precision_recall(
            source_pixels_matched.detach(),
            target_pixels_matched.detach(),
            true_pixels_matched.detach() if true_pixels_matched is not None else None,
            batch_idx_match.detach(),
            scores.detach(),
            self.config.MAX_EPIPOLAR_DISTANCE,
            fundamental_pred,
        )

        # Compute epipolar error
        epipolar_error = metrics.epipolar_error(
            source_pixels_matched.cpu(),
            target_pixels_matched.cpu(),
            fundamental_gt.cpu(),
            batch_idx_match.cpu(),
        )

        # Compute fundamental matrix errors
        fundamental_error = metrics.fundamental_error(
            fundamental_pred.cpu(), fundamental_gt.cpu()
        )[0]
        # Compute mean matching distance
        if true_pixels_matched is not None:
            mean_match_distance = metrics.mean_matching_distance(
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
            "EpipolarError": epipolar_error,
            "FundamentalError": fundamental_error,
            "MDistMean": mean_match_distance,
        }

    def match_images(self, image1, image2=None, knn=1):
        """
        Compute correspondences between two images using the unified scoring system.

        Args:
            image1: First image of shape [3, H, W] or [B, 3, H, W]
            image2: Second image of shape [3, H, W] or [B, 3, H, W]. If None, uses the second frame from image1.
            knn: Number of nearest neighbors to consider

        Returns:
            Dictionary containing match information including source/target pixels, scores, etc.
        """
        # Adjust dimensions if missing batch dimension
        if len(image1.shape) == 3:
            image1 = image1.unsqueeze(0)
        if image2 is not None and len(image2.shape) == 3:
            image2 = image2.unsqueeze(0)

        # Create framestack
        if image2 is None:
            framestack = image1
        else:
            framestack = torch.stack([image1, image2], dim=1)

        # Compute model outputs
        with torch.no_grad():
            modeloutput = self.model(framestack)

        # Get initial correspondences with unified scores (without epipolar component)
        correspondence_data = self.compute_correspondences(
            modeloutput, framestack, knn=knn
        )

        # Compute fundamental matrix using RANSAC and update scores with epipolar information
        F, inliers, epipolar_scores = self.RANSAC(
            correspondence_data["source_pixels_matched"],
            correspondence_data["target_pixels_matched"],
            correspondence_data["batch_idx_match"],
        ).values()

        # Update correspondence data with RANSAC results
        correspondence_data.update(
            {
                "F": F,
                "inliers": inliers,
                "epipolar_scores": epipolar_scores,
            }
        )

        # Compute final unified scores with all components
        scores = self.combine_scores(
            correspondence_data["descriptor_scores"],
            correspondence_data["refinement_scores"],
            correspondence_data["epipolar_scores"],
            config=self.config.SCORE_WEIGHTS,
        )

        # Update the final scores
        correspondence_data["scores"] = scores

        return correspondence_data

    def combine_scores(
        self, descriptor_scores, refinement_scores, epipolar_scores, config
    ):
        """
        Combine different scores using a linear combination with configurable weights.

        Args:
            descriptor_scores: tensor of shape (N,) - feature descriptor matching scores
            refinement_scores: tensor of shape (N,) - refinement matching scores
            epipolar_scores: tensor of shape (N,) - epipolar geometry consistency scores
            config: dictionary of weights for each score component

        Returns:
            combined_scores: tensor of shape (N,) - weighted combination of input scores
        """
        # Ensure all inputs are on the same device
        device = descriptor_scores.device

        # Get weights from config
        descriptor_weight = torch.tensor(config.get("DESCRIPTOR", 1.0), device=device)
        refinement_weight = torch.tensor(config.get("REFINEMENT", 1.0), device=device)
        epipolar_weight = torch.tensor(config.get("EPIPOLAR", 1.0), device=device)

        # Normalize weights to sum to 1
        total_weight = descriptor_weight + refinement_weight + epipolar_weight
        descriptor_weight = descriptor_weight / total_weight
        refinement_weight = refinement_weight / total_weight
        epipolar_weight = epipolar_weight / total_weight

        # Vectorized linear combination
        combined_scores = (
            descriptor_weight * descriptor_scores
            + refinement_weight * refinement_scores
            + epipolar_weight * epipolar_scores
        )
        # Check for NaN or non-numeric values in scores
        invalid_mask = torch.isnan(combined_scores) | torch.isinf(combined_scores)
        if invalid_mask.any():
            # Replace invalid scores with zeros
            combined_scores = torch.where(invalid_mask, torch.zeros_like(combined_scores), combined_scores)
        return combined_scores

    # Wrapper for compute_metrics
    def compute_metrics(self, *args):
        return metrics.compute_metrics(*args)

    def show_match(
        self,
        match_data,
        image1,
        image2,
        show="both",  # "both", "compare", "epipolar"
        pts2_true=None,  # Ground truth target points for comparison view
        pose_6d=None,  # Camera pose for epipolar view
        topk=20,
        use_actual_topk=False,
        **kwargs
    ):
        """
        Visualize matching results using the output from match_images.

        Args:
            match_data (dict): Output from match_images method
            image1 (torch.Tensor): First image [3, H, W] or [B, 3, H, W]
            image2 (torch.Tensor): Second image [3, H, W] or [B, 3, H, W]  
            show (str): What to display - "both", "compare", "epipolar"
            pts2_true (torch.Tensor, optional): Ground truth target points for comparison
            pose_6d (torch.Tensor or dict, optional): Camera pose for epipolar visualization
            topk (int): Number of top matches to display
            use_actual_topk (bool): Whether to use actual top-k matches or evenly spaced
            **kwargs: Additional arguments passed to visualization functions
        """
        from utilities.visualization import viewComparePixelMatches, viewEpipolarGeometry, rgb
        
        # Ensure images have batch dimension removed for visualization
        if len(image1.shape) == 4:
            image1 = image1[0]
        if len(image2.shape) == 4:
            image2 = image2[0]
            
        # Extract data from match_data
        pts1 = match_data["source_pixels_matched"]  # [N, 2]
        pts2 = match_data["target_pixels_matched"]  # [N, 2]
        scores = match_data["scores"]  # [N]
        F = match_data["F"]  # [B, 3, 3]
        
        # Handle batch dimension in F matrix
        if len(F.shape) == 3:
            F = F[0]  # Take first batch element
            
        # Filter to single batch if batch indices are provided
        if "batch_idx_match" in match_data:
            batch_idx = match_data["batch_idx_match"]
            # Filter for first batch (index 0)
            batch_mask = batch_idx == 0
            pts1 = pts1[batch_mask]
            pts2 = pts2[batch_mask]
            scores = scores[batch_mask]
            if pts2_true is not None:
                pts2_true = pts2_true[batch_mask]
            else:
                pts2_true = pts2
        
        if show in ["both", "compare"]:
            # Show comparison with ground truth
            comparison_img = viewComparePixelMatches(
                image1, 
                image2,
                pts1,
                pts2, 
                pts2,
                scores,
                topk=topk,
                use_actual_topk=use_actual_topk,
                as_tensor=True,
                **kwargs
            )
            display(rgb(comparison_img))
                
        if show in ["both", "epipolar"]:
            # Show epipolar geometry
            epipolar_img = viewEpipolarGeometry(
                image1,
                image2, 
                pts1,
                pts2,
                scores,
                F,
                pose_6d=pose_6d,
                topk=topk,
                use_actual_topk=use_actual_topk,
                as_tensor=True,
                **kwargs
            )
            display(rgb(epipolar_img))


class MatcherBackbone(FeatureExtractor):
    def __init__(
        self,
        backbone_brand="intel",
        size="beit-base-384",
        resampled_patch_size=8,
        shared_key=None,  # New parameter for sharing feature extractor
    ):
        super().__init__(backbone_brand=backbone_brand, size=size, shared_key=shared_key)
        self.resampled_patch_size = resampled_patch_size

        self.depthpredictor = depth_decoding.DPT_Predictor(
            backbone_brand=backbone_brand,
            size=size,
            out_h=384,
            out_w=384,
        )
        # Patchsize resampler
        self.patchsize_resampler = lambda x: nn.functional.interpolate(
            x,
            size=(384 // self.resampled_patch_size, 384 // self.resampled_patch_size),
            mode="bilinear",
            align_corners=False,
        )

        # Extract backbone output indices (if needed for depth, etc.)
        # Note: backbone_out_indices is now set in parent class
        # self.backbone_out_indices = self.backbone_out_indices  # Already available

    def forward(self, framestack):
        """
        Args:
            framestack: torch.Tensor, shape [B, 2, 3, H, W]
        Returns:
            dict with source/target embeddings for matching
        """
        source = framestack[:, 0]  # [B, 3, H, W]
        target = framestack[:, -1] # [B, 3, H, W]

        # Extract DINO features
        with torch.no_grad():
            source_features = self.extract_features(source)  # list of [B, N+1, C]
            target_features = self.extract_features(target)

        # Matcher head: pass last layer tokens through lastvitlayer
        source_features_matcher = self.lastvitlayer(source_features[-1])[0]  # [B, N+1, C]
        target_features_matcher = self.lastvitlayer(target_features[-1])[0]  # [B, N+1, C]

        # Prepare output dict (all shapes [B, C, HW] after permute)
        mono3doutput = {
            "source_embedding": source_features[-1][:, 1:, :].permute(0, 2, 1),  # [B, C, HW]
            "target_embedding": target_features[-1][:, 1:, :].permute(0, 2, 1),  # [B, C, HW]
            "source_embedding_match": source_features_matcher[:, 1:, :].permute(0, 2, 1),  # [B, C, HW]
            "target_embedding_match": target_features_matcher[:, 1:, :].permute(0, 2, 1),  # [B, C, HW]
            "source_cls": source_features[-1][:, 0, :],  # [B, C]
            "target_cls": target_features[-1][:, 0, :],  # [B, C]
        }

        # Resample if needed
        if self.resampled_patch_size != 16:
            for key in [
                "source_embedding",
                "target_embedding",
                "source_embedding_match",
                "target_embedding_match",
            ]:
                # [B, C, HW] -> [B, C, h, w] -> resample -> [B, C, h', w'] -> flatten -> [B, C, HW']
                mono3doutput[key] = chw2embedding(
                    self.patchsize_resampler(embedding2chw(mono3doutput[key], False))
                )

        return mono3doutput

    def _compute_depth(self, source_features_dino):
        """Compute depth maps from source features."""
        depthstack = self.depthpredictor(
            source_features_dino,
            self.backbone_out_indices,
        )
        return depthstack.view(-1, 1, depthstack.shape[-2], depthstack.shape[-1])

    def _extract_embeddings(self, features, loftr_shape=True):
        """Extract embeddings for a single image."""
        if loftr_shape:
            return self.patchsize_resampler(embedding2chw(features[-1][:, 1:]))
        else:
            return embedding2chw(features[-1][:, 1:])

    def depth(self, framestack):
        """Inference-only method to return depth maps for one or more images or frame stacks."""

        source_features_dino = self.extract_features(
            framestack[:, 0] if len(framestack.shape) == 5 else framestack
        )
        return self._compute_depth(source_features_dino)

    def embed(self, *frames, mode="chw", loftr_shape=False):
        """Inference-only method to return embeddings for one or more frames."""
        assert mode in ["chw", "seq"], "Invalid mode. Must be 'chw' or 'seq'."
        embs = [
            self._extract_embeddings(self.extract_features(image), loftr_shape)
            for image in frames
        ]
        embs = [chw2embedding(emb) if mode == "seq" else emb for emb in embs]
        if len(embs) == 1:
            return embs[0]
        return embs

    def fromArtifact(
        self,
        model_name,
        bucket=None,
        device="cuda",
    ):
        """
        Loads a PyTorch model from Google Cloud Storage or local checkpoint

        Args:
            bucket (str, optional): Name of the GCS bucket. If None, uses GCS_BUCKET_NAME environment variable
            model_name (str): Path to the model file in the file bucket
            device (str): Device to load the model on ('cuda' or 'cpu')

        Returns:
            torch.nn.Module: Loaded model, or None if loading failed
        """
        # Try downloading from GCS first
        try:
            local_path = download_from_gcs(model_name=model_name, bucket_name=bucket)
        except Exception as e:
            logger.warning(f"Failed to download from GCS: {str(e)}")
            local_path = None

        # If GCS download fails, check local checkpoint directory
        if local_path is None:
            local_path = os.path.join("checkpoints", "weights_best.pt")
            if not os.path.exists(local_path):
                raise FileNotFoundError(
                    f"Model not found in GCS bucket {bucket} or locally at {local_path}"
                )

        # Load the model state dict
        loaded_dict = torch.load(local_path, map_location=device, weights_only=True)
        self.load_state_dict(loaded_dict)

        # Clean up temporary file if it exists
        if os.path.dirname(local_path) == tempfile.gettempdir():
            os.remove(local_path)
