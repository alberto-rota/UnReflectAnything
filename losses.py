"""
Main loss module for UnReflectAnything model.

This module contains the composite UnReflectLoss class that combines multiple
loss terms for training the reflection removal and highlight inpainting model.
"""

import torch
import torch.nn as nn
from loss_utils import (
    SSIMLoss,
    MaskedL1Loss,
    SeamLoss,
    HighlightRegressionLoss,
    TokenInpaintLoss,
    DiffuseHighlightPenaltyLoss,
    reconstruct_image_from_components,
    saturation_ring_blob_consistency,
    _total_variation,
)
from models_utils import pixel_mask_to_patch_mask


class UnReflectLoss(nn.Module):
    """
    Composite loss function for UnReflectAnything model.

    Combines multiple loss terms:
    - Component losses: specular, diffuse, highlight regression
    - Reconstruction loss: composite image matching
    - Context identity: preserve non-hole regions
    - Seam loss: boundary consistency for inpainting
    - TV loss: smoothness regularization in holes
    - Saturation ring loss: color consistency between holes and context
    - Token inpaint loss: feature-space distillation for masked patches

    Args:
        # Component loss weights
        weight_specular_loss: Weight for specular component loss (default: 1.0)
        weight_diffuse_loss: Weight for diffuse component loss (default: 1.0)
        weight_highlight_loss: Weight for highlight regression loss (default: 1.0)
        weight_image_reconstruction: Weight for image reconstruction loss (default: 0.5)
        weight_alpha_regularization: Weight for alpha channel regularization (default: 0.0, unused)

        # Saturation ring loss parameters
        weight_saturation_ring: Weight for saturation ring consistency loss (default: 0.0)
        ring_kernel_size: Odd kernel size for ring dilation (default: 7)
        ring_var_weight: Weight for variance matching vs mean matching in ring loss (default: 0.5)
        ring_texture_weight: Weight for texture consistency term in ring loss (default: 1.0)

        # Highlight regression config
        hlreg_w_l1: Weight for L1/Charbonnier term in highlight regression (default: 1.0)
        hlreg_use_charb: Use Charbonnier loss instead of L1 for highlight regression (default: True)
        hlreg_w_dice: Weight for Dice loss in highlight regression (default: 0.2)
        hlreg_w_ssim: Weight for SSIM loss in highlight regression (default: 0.0)
        hlreg_w_grad: Weight for gradient loss in highlight regression (default: 0.0)
        hlreg_w_tv: Weight for TV loss in highlight regression (default: 0.0)
        hlreg_balance_mode: Class balancing mode for highlight regression: "none", "auto", "pos_weight" (default: "none")
        hlreg_pos_weight: Weight for positive class when balance_mode="pos_weight" (default: 1.0)
        hlreg_focal_gamma: Focal loss gamma for highlight regression (0 = disabled) (default: 0.0)

        # Highlight rendering
        highlight_color: RGB color for highlights (default: (1.0, 1.0, 1.0))
        clamp_reconstruction: Clamp reconstructed images to [0, 1] (default: True)

        # Context and seam loss parameters
        weight_context_identity: Weight for L1 loss outside holes (default: 1.0)
        weight_seam: Weight for gradient matching on ring/seam (default: 0.5)
        weight_tv_in_hole: Weight for TV loss only inside holes (default: 1e-3)
        ring_dilate_kernel: Kernel size for ring mask dilation (default: 7)

        # Token-space loss parameters
        weight_token_inpaint: Weight for token-space feature distillation loss (default: 1.0)
        token_feat_alpha: Mixing weight: alpha * L1 + (1-alpha) * (1-cosine) in feature space (default: 0.5)

        # Seam loss parameters
        seam_use_charb: Use Charbonnier loss in seam loss (default: True)
        seam_weight_grad: Weight for gradient term in seam loss (default: 0.2)

        # Diffuse highlight penalty parameters
        weight_diffuse_highlight_penalty: Weight for penalty loss on highlights in diffuse output (default: 0.0)
        diffuse_hl_threshold: Brightness/luminance threshold for detecting highlights in diffuse (default: 0.7)
        diffuse_hl_use_charb: Use Charbonnier loss for diffuse highlight penalty (default: True)
        diffuse_hl_penalty_mode: Penalty mode: "brightness" or "pixel" (default: "brightness")
        diffuse_hl_target_brightness: Target brightness/luminance for penalized pixels (default: threshold)
        diffuse_hl_use_luminance: If True, use perceptually-weighted luminance; if False, use mean brightness (default: False)
    """

    def __init__(
        self,
        # Loss weights for specular, diffuse and highlight reconstruction
        weight_specular_loss=1.0,
        weight_diffuse_loss=1.0,
        weight_highlight_loss=1.0,
        weight_image_reconstruction=0.5,
        weight_alpha_regularization=0.0,
        weight_saturation_ring: float = 0.0,
        ring_kernel_size: int = 7,  # odd; dilation size for surrounding ring
        ring_var_weight: float = 0.5,  # weight on variance matching vs mean matching
        ring_texture_weight: float = 1.0,  # weight on texture consistency term
        # Highlight regression config
        hlreg_w_l1=1.0,
        hlreg_use_charb=True,
        hlreg_w_dice=0.2,
        hlreg_w_ssim=0.0,
        hlreg_w_grad=0.0,
        hlreg_w_tv=0.0,
        hlreg_balance_mode: str = "none",
        hlreg_pos_weight: float = 1.0,
        hlreg_focal_gamma: float = 0.0,
        # Highlight rendering
        highlight_color=(1.0, 1.0, 1.0),
        clamp_reconstruction=True,
        weight_context_identity: float = 1.0,  # L1 outside holes
        weight_seam: float = 0.5,  # gradient match on ring
        weight_tv_in_hole: float = 1e-3,  # TV only inside holes
        ring_dilate_kernel: int = 7,  # for ring mask
        # Token-space loss parameters
        weight_token_inpaint: float = 1.0,  # λ for token-space loss
        token_feat_alpha: float = 0.5,  # mix L1 vs (1-cosine) in feature space
        seam_use_charb: bool = True,
        seam_weight_grad: float = 0.2,
        # Diffuse highlight penalty parameters
        weight_diffuse_highlight_penalty: float = 0.0,  # Weight for penalty on highlights in diffuse
        diffuse_hl_threshold: float = 0.7,  # Brightness/luminance threshold for highlight detection
        diffuse_hl_use_charb: bool = True,  # Use Charbonnier loss
        diffuse_hl_penalty_mode: str = "brightness",  # "brightness" or "pixel"
        diffuse_hl_target_brightness: float = None,  # Target brightness/luminance (None = use threshold)
        diffuse_hl_use_luminance: bool = False,  # Use perceptually-weighted luminance instead of mean brightness
        **kwargs,
    ):
        super().__init__()

        # ====================================================================
        # Store loss weights
        # ====================================================================
        self.weight_specular_loss = weight_specular_loss
        self.weight_diffuse_loss = weight_diffuse_loss
        self.weight_highlight_loss = weight_highlight_loss
        self.weight_image_reconstruction = weight_image_reconstruction
        self.weight_saturation_ring = weight_saturation_ring
        self.ring_kernel_size = ring_kernel_size
        self.ring_var_weight = ring_var_weight
        self.ring_texture_weight = ring_texture_weight

        # Highlight rendering parameters
        self.highlight_color = torch.tensor(highlight_color, dtype=torch.float32)
        self.clamp_reconstruction = clamp_reconstruction

        # Context and seam loss weights
        self.weight_context_identity = weight_context_identity
        self.weight_tv_in_hole = weight_tv_in_hole
        self.weight_seam = weight_seam
        self.ring_dilate_kernel = ring_dilate_kernel

        # Token-space loss parameters
        self.weight_token_inpaint = weight_token_inpaint
        self.token_feat_alpha = token_feat_alpha

        # Diffuse highlight penalty parameters
        self.weight_diffuse_highlight_penalty = weight_diffuse_highlight_penalty

        # Component weights dictionary for easy access
        self.component_weights = {
            "specular": weight_specular_loss,
            "diffuse": weight_diffuse_loss,
            "highlight": weight_highlight_loss,
        }

        # ====================================================================
        # Initialize sub-loss modules
        # ====================================================================
        self.masked_ssim_loss = SSIMLoss()
        self.masked_l1_loss = MaskedL1Loss()

        self.highlight_regression_loss = HighlightRegressionLoss(
            w_l1=hlreg_w_l1,
            use_charbonnier=hlreg_use_charb,
            w_dice=hlreg_w_dice,
            w_ssim=hlreg_w_ssim,
            w_grad=hlreg_w_grad,
            w_tv=hlreg_w_tv,
            balance_mode=hlreg_balance_mode,
            pos_weight=hlreg_pos_weight,
            focal_gamma=hlreg_focal_gamma,
        )

        self.seam_loss_fn = SeamLoss(
            ring_kernel=self.ring_dilate_kernel,  # defines boundary ring thickness
            use_charbonnier=seam_use_charb,  # Charbonnier vs L1
            weight_l1=1.0,  # pixel diff along seam
            weight_grad=seam_weight_grad,  # gradient match along seam
            eps=1e-6,
            reduction="mean",
        )

        self.token_inpaint_loss = TokenInpaintLoss(
            token_feat_alpha=token_feat_alpha,
        )

        self.diffuse_highlight_penalty_loss = DiffuseHighlightPenaltyLoss(
            brightness_threshold=diffuse_hl_threshold,
            use_charbonnier=diffuse_hl_use_charb,
            penalty_mode=diffuse_hl_penalty_mode,
            target_brightness=diffuse_hl_target_brightness,
            use_luminance=diffuse_hl_use_luminance,
        )

        # ====================================================================
        # Initialize helper modules and buffers
        # ====================================================================
        # Max-pooling for ring mask dilation (cheap morphological dilation)
        self._ring_pool = nn.MaxPool2d(
            self.ring_dilate_kernel, stride=1, padding=self.ring_dilate_kernel // 2
        )

        # Small gradient kernels for texture consistency (applied per-channel)
        kx = torch.tensor([[-1.0, 1.0]], dtype=torch.float32).view(1, 1, 1, 2)
        ky = torch.tensor([[-1.0], [1.0]], dtype=torch.float32).view(1, 1, 2, 1)
        self.register_buffer("_ring_kx", kx)
        self.register_buffer("_ring_ky", ky)

    def forward(
        self, prediction, ground_truth, pixel_supervision_mask, pixel_inpaint_mask, patch_supervision_mask, patch_inpaint_mask
    ):
        """
        Compute composite loss for UnReflectAnything model.

        Args:
            prediction: Dict with predicted components:
                       - 'diffuse': (B, C, H, W) where C ∈ {3, 4}
                       - 'specular': (B, C, H, W) where C ∈ {3, 4} (optional)
                       - 'highlight': (B, 1, H, W) (optional)
                       - 'tokens_completed': List[L] of (B, N, C) if token loss enabled
            ground_truth: Dict with ground truth components:
                          - 'diffuse': (B, C, H, W) (optional)
                          - 'specular': (B, C, H, W) (optional)
                          - 'highlight': (B, 1, H, W) (optional)
                          - 'rgb_highlighted': (B, 3, H, W) input image with highlights
                          - 'hole_mask': (B, 1, H, W) optional hole mask
                          - 'patch_mask_sup': (B, N) optional patch mask for token loss
                          - 'patch_size': int patch size for token loss (if patch_mask_sup not provided)
                          - 'tokens_teacher': List[L] of (B, N, C) if token loss enabled
            pixel_supervision_mask: (B, 1, H, W) pixels that contribute to the supervision loss (supervised region)
            pixel_inpaint_mask: (B, 1, H, W) pixels that need to be inpainted (inpainting region)
            patch_supervision_mask: (B, N) patches that contribute to the supervision loss (supervised region)
            patch_inpaint_mask: (B, N) patches that need to be inpainted (inpainting region)
        Returns:
            losses: Dict with individual loss terms and 'total' key
        """

        losses = {}

        # ====================================================================
        # Component supervision losses
        # ====================================================================
        # Supervise individual components (specular, diffuse, highlight) on pixel_supervision_mask
        available_components = [
            k for k in prediction.keys() if k in ground_truth and k != "rgb_highlighted"
        ]
        if not available_components:
            raise ValueError(
                "No matching components found between predictions and ground truth"
            )

        for comp_name in available_components:
            pred_comp = prediction[comp_name]
            gt_comp = ground_truth[comp_name]
            
            # Highlight prediciton is a custom loss. Also there is no masking here
            if comp_name.lower() == "highlight":
                pred_h = pred_comp.clamp(0, 1)
                gt_h = gt_comp.clamp(0, 1)
                hl_loss = self.highlight_regression_loss(
                    pred_h, gt_h
                )  # <--- Mask = None
                losses["HighlightRegression"] = hl_loss
                continue
            
            # Diffuse/specular: supervise RGB channels ONLY on pixel_supervision_mask
            pred_rgb = pred_comp[:, :3]
            gt_rgb = gt_comp[:, :3]
            rgb_l1 = self.masked_l1_loss(pred_rgb, gt_rgb, pixel_supervision_mask)
            rgb_ssim = self.masked_ssim_loss(pred_rgb, gt_rgb, pixel_supervision_mask)
            comp_loss = rgb_l1 + (1.0 - rgb_ssim)

            # Optional alpha channel supervision
            if pred_comp.shape[1] == 4 and gt_comp.shape[1] == 4:
                alpha_l1 = self.masked_l1_loss(
                    pred_comp[:, 3:4], gt_comp[:, 3:4], pixel_supervision_mask
                )
                comp_loss = comp_loss + alpha_l1

            losses[f"{comp_name.capitalize()}"] = comp_loss

        # ====================================================================
        # Image reconstruction loss
        # ====================================================================
        # Supervise reconstructed composite image on pixel_supervision_mask
        if "rgb_highlighted" in ground_truth and "diffuse" in prediction:
            pred_recon = reconstruct_image_from_components(
                prediction=prediction,
                mask=pixel_supervision_mask,
                clamp_reconstruction=self.clamp_reconstruction,
            )
            input_rgb = ground_truth["rgb_highlighted"]
            recon_l1 = self.masked_l1_loss(
                pred_recon, input_rgb, pixel_supervision_mask
            )
            recon_ssim = self.masked_ssim_loss(
                pred_recon, input_rgb, pixel_supervision_mask
            )
            losses["Reconstruction"] = recon_l1 + (1.0 - recon_ssim)
        else:
            losses["Reconstruction"] = None

        # ====================================================================
        # Context identity loss
        # ====================================================================
        # Preserve non-hole regions: diffuse should match GT outside holes
        if "diffuse" in prediction and self.weight_context_identity > 0:
            diffuse_pred_rgb = prediction["diffuse"][:, :3]
            if "diffuse" in ground_truth:
                diffuse_gt_rgb = ground_truth["diffuse"][:, :3]
            else:
                # If diffuse GT not provided, identity w.r.t. input RGB (safe fallback)
                diffuse_gt_rgb = ground_truth["rgb_highlighted"]

            ctx_l1 = self.masked_l1_loss(
                diffuse_pred_rgb, diffuse_gt_rgb, pixel_supervision_mask
            )
            losses["ContextIdentity"] = ctx_l1
        else:
            losses["ContextIdentity"] = torch.tensor(0.0)

        # ====================================================================
        # Seam loss and TV in hole
        # ====================================================================
        # Boundary consistency: match gradients and pixels along seam
        if "diffuse" in prediction:
            D_hat = prediction["diffuse"][:, :3]
            # Prefer clean diffuse GT if present; otherwise fallback to input
            D_ref = ground_truth.get("diffuse")[:, :3]

            # m_hole is the *inpainting* region; SeamLoss computes a ring = dilate(m_hole) - m_hole
            # so pass m_hole here (NOT m_sup)
            seam_loss = self.seam_loss_fn(D_hat, D_ref, mask=pixel_inpaint_mask)

            # Optional: TV loss inside the hole for smoothness
            if self.weight_tv_in_hole > 0:
                tv_loss = _total_variation(D_hat * pixel_inpaint_mask)
            else:
                tv_loss = torch.tensor(0.0, device=D_hat.device)
        else:
            seam_loss = torch.tensor(0.0)
            tv_loss = torch.tensor(0.0)

        losses["Seam"] = seam_loss if seam_loss is not None else torch.tensor(0.0)
        losses["TVinHole"] = tv_loss if tv_loss is not None else torch.tensor(0.0)

        # ====================================================================
        # Token-space inpainting loss
        # ====================================================================
        
        # Feature-space distillation on masked patches
        if self.weight_token_inpaint > 0:
            if "tokens_completed" not in prediction:
                raise KeyError(
                    "prediction must include 'tokens_completed' when weight_token_inpaint > 0"
                )
            if "tokens_teacher" not in ground_truth:
                raise KeyError(
                    "ground_truth must include 'tokens_teacher' when weight_token_inpaint > 0"
                )

            tokens_completed = prediction[
                "tokens_completed"
            ]  # Tokens predicted by inpainting model
            tokens_teacher = ground_truth[
                "tokens_teacher"
            ]  # Desired tokens from ground truth diffuse
            # The token inpainting loss should be computed on the inpainted tokens that are also supervised
            patch_inpaint_supervision_mask = patch_supervision_mask * patch_inpaint_mask
            
            l_token = self.token_inpaint_loss(tokens_completed, tokens_teacher, patch_inpaint_supervision_mask)
            # l_token = self.token_inpaint_loss(tokens_completed, tokens_teacher, patch_inpaint_mask)
            # l_token = self.token_inpaint_loss(tokens_completed, tokens_teacher, patch_inpaint_supervision_mask)
            losses["TokenInpaint"] = l_token
        else:
            losses["TokenInpaint"] = torch.zeros(())

        # ====================================================================
        # Diffuse highlight penalty loss
        # ====================================================================
        # Explicitly penalize highlights in diffuse decoder output
        if self.weight_diffuse_highlight_penalty > 0:
            if "diffuse" in prediction:
                diffuse_rgb = prediction["diffuse"][:, :3]  # (B, 3, H, W)
                # Optionally apply supervision mask to focus on supervised regions
                hl_penalty = self.diffuse_highlight_penalty_loss(
                    diffuse_rgb, mask=pixel_supervision_mask
                )
                losses["HPenalty"] = hl_penalty
            else:
                losses["HPenalty"] = torch.zeros(())
        else:
            losses["HPenalty"] = torch.zeros(())

        # ====================================================================
        # Compute total loss
        # ====================================================================
        total = 0.0

        # Component losses
        if "Specular" in losses:
            total = total + self.component_weights["specular"] * losses["Specular"]
        if "Diffuse" in losses:
            total = total + self.component_weights["diffuse"] * losses["Diffuse"]
        if "HighlightRegression" in losses:
            total = (
                total
                + self.component_weights["highlight"] * losses["HighlightRegression"]
            )

        # Reconstruction loss
        if losses.get("Reconstruction") is not None:
            total = total + self.weight_image_reconstruction * losses["Reconstruction"]

        # Regularization and consistency losses
        total = total + self.weight_context_identity * losses["ContextIdentity"]
        total = total + self.weight_seam * losses["Seam"]
        total = total + self.weight_tv_in_hole * losses["TVinHole"]
        # total = total + self.weight_saturation_ring * losses["SaturationRing"]
        total = total + self.weight_token_inpaint * losses["TokenInpaint"]
        total = total + self.weight_diffuse_highlight_penalty * losses["HPenalty"]

        losses["total"] = total
        return losses

    def reconstruct_image(self, prediction, mask=None):
        """
        Reconstruct RGB image from predicted components.

        Args:
            prediction: Dict with keys like 'diffuse', 'specular', 'highlight'
                       - diffuse: (B, C, H, W) where C ∈ {3, 4}
                       - specular: (B, C, H, W) where C ∈ {3, 4} (optional)
                       - highlight: (B, 1, H, W) (optional)
            mask: (B, 1, H, W) optional mask to apply before composition

        Returns:
            reconstructed_rgb: (B, 3, H, W) reconstructed RGB image
        """
        return reconstruct_image_from_components(
            prediction=prediction,
            mask=mask,
            clamp_reconstruction=self.clamp_reconstruction,
        )
