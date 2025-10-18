import torch
import torch.nn as nn
from loss_utils import (
    SSIMLoss,
    MaskedL1Loss,
    HighlightRegressionLoss,
    reconstruct_image_from_components,
    saturation_ring_blob_consistency,
)


class UnReflectLoss(nn.Module):
    def __init__(
        self,
        # Loss weights fot specular, diffuse and highlight reconstruction
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
    ):
        super().__init__()
        # Loading loss weights from params
        self.weight_specular_loss = weight_specular_loss
        self.weight_diffuse_loss = weight_diffuse_loss
        self.weight_highlight_loss = weight_highlight_loss
        self.weight_image_reconstruction = weight_image_reconstruction
        self.weight_saturation_ring = weight_saturation_ring
        self.ring_kernel_size = ring_kernel_size
        self.ring_var_weight = ring_var_weight
        self.ring_texture_weight = ring_texture_weight

        # Highlight rendering
        self.highlight_color = torch.tensor(highlight_color, dtype=torch.float32)
        self.clamp_reconstruction = clamp_reconstruction

        # Subloss function initialization
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

        # Small gradient kernels for texture consistency (applied per-channel)
        kx = torch.tensor([[-1.0, 1.0]], dtype=torch.float32).view(1, 1, 1, 2)
        ky = torch.tensor([[-1.0], [1.0]], dtype=torch.float32).view(1, 1, 2, 1)
        self.register_buffer("_ring_kx", kx)
        self.register_buffer("_ring_ky", ky)

        self.component_weights = {
            "specular": weight_specular_loss,
            "diffuse": weight_diffuse_loss,
            "highlight": weight_highlight_loss,
        }

    def forward(self, prediction, ground_truth, mask=None):
        """
        Args:
            prediction: dict with keys like 'diffuse', 'specular', 'highlight'
                       - diffuse: [B,C,H,W] where C∈{3,4}
                       - specular: [B,C,H,W] where C∈{3,4}
                       - highlight: [B,1,H,W]
            ground_truth: dict with keys matching prediction + 'rgb_highlighted'
                         - diffuse: [B,3,H,W]
                         - specular: [B,3,H,W]
                         - highlight: [B,1,H,W]
                         - rgb_highlighted: [B,3,H,W]
            mask: [B,1,H,W] or [B,C,H,W] - binary mask (1=include, 0=ignore)
                 If None, uses all pixels
        """

        # We return a loss dict with all the sub-losses for logging purposes. The loss used for backprop is the total loss
        losses = {}

        # If mask is not provided, nothing gets masked
        if mask is None:
            B, _, H, W = ground_truth["rgb_highlighted"].shape
            mask = torch.ones(B, 1, H, W, device=ground_truth["rgb_highlighted"].device)

        ### Establishing available components to be used for loss computation
        # Models can be dinamically initialized to use different decoders, and not all three of the 
        # outputs "diffuse", "specular" and "highlight" might be available. Moreover, the training dataset might not 
        # contain all three of the components (e.g. no "specular" from rgb-only datasets). We establish 
        #  the available components by checking which keys are present in both the prediction and ground truth dictionaries.
        available_components = [
            k for k in prediction.keys() if k in ground_truth and k != "rgb_highlighted"
        ]
        if not available_components:
            raise ValueError(
                "No matching components found between predictions and ground truth"
            )

        # Iterating over all available components and computing the corresponding loss
        for comp_name in available_components:
            pred_comp = prediction[comp_name]
            gt_comp = ground_truth[comp_name]

            ### Highlight predictions are soft binary masks, so we need to compute the regression loss on them
            # Not computing the L1 and SSIM loss
            if comp_name.lower() == "highlight":
                if pred_comp.shape[1] == 1 and gt_comp.shape[1] == 1:
                    # Outputs should already be clamped to [0,1]. Guardrails in case.
                    pred_h = torch.clamp(pred_comp, 0.0, 1.0)
                    gt_h = torch.clamp(gt_comp, 0.0, 1.0)

                    ### Computing highlight regression loss
                    hl_loss = self.highlight_regression_loss(pred_h, gt_h)
                    losses["HighlightRegression"] = hl_loss
                    # Continuing to the next component, we skip L1 and SSIM losses for highlights
                continue

            ### Computing L1 and SSIM losses for the other components ("diffuse" and "specular")
            # Diffuse and specular images might either be RGB or RGBA. We extract the RGB channels for these losses.
            pred_rgb = pred_comp[:, :3]  # [B,3,H,W]
            gt_rgb = gt_comp[:, :3]  # [B,3,H,W]
            rgb_l1 = self.masked_l1_loss(pred_rgb, gt_rgb, mask)
            rgb_ssim = self.masked_ssim_loss(pred_rgb, gt_rgb, mask)
            comp_loss = rgb_l1 + (1.0 - rgb_ssim)

            # If both GT and prediction have and alpha channel, supervise it too (only L1 loss=)
            if pred_comp.shape[1] == 4 and gt_comp.shape[1] == 4:
                alpha_l1 = self.masked_l1_loss(pred_comp[:, 3:4], gt_comp[:, 3:4], mask)
                comp_loss = comp_loss + alpha_l1
            losses[f"{comp_name.capitalize()}"] = comp_loss

        ### Computing the image reconstruction loss
        # "diffuse", "specular" and "highlight" should composite to form the original image.
        # We only compute the reconstruction loss if the ground truth and prediction contain 
        # the "rgb_highlighted" key and the "diffuse" key, respectively. Diffuse is required for the reconstruction.
        if "rgb_highlighted" in ground_truth and "diffuse" in prediction:
            # Reconstructing the image from the available components, masking the reconstruction to the highlight mask
            pred_reconstruction = reconstruct_image_from_components(
                prediction=prediction,
                mask=mask,
                clamp_reconstruction=self.clamp_reconstruction,
            )
            input_rgb = ground_truth["rgb_highlighted"]  # [B,3,H,W]
            # Computing the L1 and SSIM losses for the reconstructed image
            recon_l1 = self.masked_l1_loss(
                pred_reconstruction, input_rgb, mask
            )
            recon_ssim = self.masked_ssim_loss(
                pred_reconstruction, input_rgb, mask
            ) 
            reconstruction_loss = recon_l1 + (1.0 - recon_ssim)
            losses["Reconstruction"] = reconstruction_loss
        else:
            losses["Reconstruction"] = None

      ### Computing the combined per-blob saturation vs ring consistency loss
        if "diffuse" in prediction:
            diffuse_pred_rgb = prediction["diffuse"][:, :3]
            include_mask = (
                mask if mask is not None else torch.ones_like(diffuse_pred_rgb[:, :1])
            )
            if self.weight_saturation_ring > 0.0:
                sat_ring = saturation_ring_blob_consistency(
                    diffuse_rgb=diffuse_pred_rgb,
                    include_mask=include_mask,
                    ring_kernel_size=self.ring_kernel_size,
                )
                losses["SaturationRing"] = sat_ring
            else:
                losses["SaturationRing"] = torch.tensor(
                    0.0, device=diffuse_pred_rgb.device, dtype=diffuse_pred_rgb.dtype
                )
        else:
            losses["SaturationRing"] = torch.tensor(0.0)

        # ===== Total Loss =====
        total_loss = 0.0
        if "Specular" in losses:
            total_loss = total_loss + self.component_weights["specular"] * losses["Specular"]
        if "Diffuse" in losses:
            total_loss = total_loss + self.component_weights["diffuse"] * losses["Diffuse"]
        if "HighlightRegression" in losses:
            total_loss = total_loss + self.component_weights["highlight"] * losses["HighlightRegression"]
        if losses.get("Reconstruction") is not None:
            total_loss = total_loss + self.weight_image_reconstruction * losses["Reconstruction"]
        total_loss = total_loss + self.weight_saturation_ring * losses["SaturationRing"]

        losses["total"] = total_loss
        return losses

    def reconstruct_image(self, prediction, mask=None):
        """
        Args:
            prediction: dict with keys like 'diffuse', 'specular', 'highlight'
                       - diffuse: [B,C,H,W] where C∈{3,4}
                       - specular: [B,C,H,W] where C∈{3,4}
                       - highlight: [B,1,H,W]
        """
        return reconstruct_image_from_components(
            prediction=prediction,
            mask=mask,
            clamp_reconstruction=self.clamp_reconstruction,
        )