import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_utils import (
    SSIMLoss,
    MaskedL1Loss,
    HighlightRegressionLoss,
    reconstruct_image_from_components,
    saturation_ring_blob_consistency,
    _total_variation,
    _grad_mag,
    _pixel_to_patch_mask
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
        weight_context_identity: float = 1.0,   # L1 outside holes
        weight_ring_grad: float = 0.5,          # gradient match on ring
        weight_tv_in_hole: float = 1e-3,        # TV only inside holes
        ring_dilate_kernel: int = 7,            # for ring mask
        # Token-space loss parameters
        weight_token_inpaint: float = 1.0,   # λ for token-space loss
        token_feat_alpha: float = 0.5,       # mix L1 vs (1-cosine) in feature space

        **kwargs
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
        self.weight_context_identity = weight_context_identity
        self.weight_ring_grad = weight_ring_grad
        self.weight_tv_in_hole = weight_tv_in_hole
        self.ring_dilate_kernel = ring_dilate_kernel

        # quick, cheap dilation kernel for ring (max-pool)
        self._ring_pool = nn.MaxPool2d(self.ring_dilate_kernel, stride=1, padding=self.ring_dilate_kernel//2)
        self.weight_token_inpaint = weight_token_inpaint
        self.token_feat_alpha = token_feat_alpha
    
    def _token_inpaint_loss(
        self,
        tokens_completed: list[torch.Tensor],
        tokens_teacher: list[torch.Tensor],
        patch_mask_sup: torch.Tensor,  # (B,N) boolean — supervised masked patches (synthetic ∧ ¬dataset)
    ) -> torch.Tensor:
        """
        Feature-space distillation on masked patches.
        tokens_*: List[L] of (B, N, C)
        patch_mask_sup: (B, N) bool — where to supervise (masked & supervised)
        """
        if not isinstance(tokens_completed, (list, tuple)) or not isinstance(tokens_teacher, (list, tuple)):
            raise ValueError("tokens_completed and tokens_teacher must be lists of tensors.")

        if len(tokens_completed) != len(tokens_teacher):
            raise ValueError("tokens_completed and tokens_teacher must have same length (layers).")

        l_total = 0.0
        cos = nn.CosineSimilarity(dim=-1)

        any_mask = patch_mask_sup.any()
        if not any_mask:
            return torch.zeros((), device=tokens_teacher[0].device)

        for Tc, Tt in zip(tokens_completed, tokens_teacher):
            # Tc, Tt: (B, N, C)
            # Index only masked-supervised positions
            idx = patch_mask_sup.unsqueeze(-1).expand_as(Tc)  # (B,N,C) bool
            Tc_m = Tc[idx].view(-1, Tc.shape[-1])            # (M, C)
            Tt_m = Tt[idx].view(-1, Tt.shape[-1])            # (M, C)
            if Tc_m.numel() == 0:
                continue
            l1   = (Tc_m - Tt_m).abs().mean()
            cosd = 1.0 - cos(Tc_m, Tt_m).mean()
            l_total = l_total + (self.token_feat_alpha * l1 + (1.0 - self.token_feat_alpha) * cosd)

        return l_total / max(1, len(tokens_completed))

    def _make_masks(self, mask_sup: torch.Tensor, hole_mask_opt: torch.Tensor | None):
        """
        mask_sup: m_sup (B,1,H,W): where we have GT (synthetic highlight minus dataset highlight)
        hole_mask_opt: m_hole (B,1,H,W) if provided: union of synthetic ∪ dataset highlights
        returns: m_sup, m_hole, ring
        """
        m_sup = (mask_sup > 0.5).float()
        if hole_mask_opt is None:
            m_hole = m_sup.clone()   # fallback: if no explicit hole mask, use supervised mask
        else:
            m_hole = (hole_mask_opt > 0.5).float()

        with torch.no_grad():
            dil = self._ring_pool(m_hole)
            ring = (dil - m_hole).clamp_(0, 1)
        return m_sup, m_hole, ring

    def forward(self, prediction, ground_truth, mask=None):
        """
        mask: interpreted as m_sup (supervised region).
        Optionally, pass ground_truth["hole_mask"] (m_hole = synthetic ∪ dataset highlights).
        """
        # === MASKS ===
        # m_sup: where pixel GT is reliable (your synthetic highlight not overlapping dataset highlight)
        # m_hole: what we want to "repair" (synthetic ∪ dataset highlights)
        m_sup, m_hole, ring = self._make_masks(
            mask, ground_truth.get("hole_mask", None)
        )

        losses = {}
        # === existing component supervision (unchanged), just make sure we use m_sup for pixelwise image matching ===
        available_components = [k for k in prediction.keys() if k in ground_truth and k != "rgb_highlighted"]
        if not available_components:
            raise ValueError("No matching components found between predictions and ground truth")

        for comp_name in available_components:
            pred_comp = prediction[comp_name]
            gt_comp = ground_truth[comp_name]

            if comp_name.lower() == "highlight":
                # your existing highlight regression (e.g., to synthetic highlight, or any target you provide)
                pred_h = pred_comp.clamp(0, 1)
                gt_h = gt_comp.clamp(0, 1)
                hl_loss = self.highlight_regression_loss(pred_h, gt_h)
                losses["HighlightRegression"] = hl_loss
                continue

            # Diffuse/specular: supervise RGB channels ONLY on m_sup (do NOT supervise dataset highlights)
            pred_rgb = pred_comp[:, :3]
            gt_rgb   = gt_comp[:, :3]
            rgb_l1   = self.masked_l1_loss(pred_rgb, gt_rgb, m_sup)
            rgb_ssim = self.masked_ssim_loss(pred_rgb, gt_rgb, m_sup)
            comp_loss = rgb_l1 + (1.0 - rgb_ssim)

            if pred_comp.shape[1] == 4 and gt_comp.shape[1] == 4:
                alpha_l1 = self.masked_l1_loss(pred_comp[:, 3:4], gt_comp[:, 3:4], m_sup)
                comp_loss = comp_loss + alpha_l1

            losses[f"{comp_name.capitalize()}"] = comp_loss

        # === Reconstruction term (unchanged) but supervise on m_sup ===
        if "rgb_highlighted" in ground_truth and "diffuse" in prediction:
            pred_recon = reconstruct_image_from_components(
                prediction=prediction, mask=m_sup, clamp_reconstruction=self.clamp_reconstruction
            )
            input_rgb = ground_truth["rgb_highlighted"]
            recon_l1   = self.masked_l1_loss(pred_recon, input_rgb, m_sup)
            recon_ssim = self.masked_ssim_loss(pred_recon, input_rgb, m_sup)
            losses["Reconstruction"] = recon_l1 + (1.0 - recon_ssim)
        else:
            losses["Reconstruction"] = None

        # === NEW: Context identity outside holes ===
        if "diffuse" in prediction and self.weight_context_identity > 0:
            diffuse_pred_rgb = prediction["diffuse"][:, :3]
            if "diffuse" in ground_truth:
                diffuse_gt_rgb = ground_truth["diffuse"][:, :3]
            else:
                # if diffuse GT not provided, identity w.r.t. input RGB (safe fallback)
                diffuse_gt_rgb = ground_truth["rgb_highlighted"]

            ctx_mask = (1.0 - m_hole).detach()
            ctx_l1 = self.masked_l1_loss(diffuse_pred_rgb, diffuse_gt_rgb, ctx_mask)
            losses["ContextIdentity"] = ctx_l1
        else:
            losses["ContextIdentity"] = torch.tensor(0.0)

        # === NEW: Seam (ring) gradient consistency + TV in hole ===
        seam_loss = torch.tensor(0.0)
        tv_loss   = torch.tensor(0.0)
        if "diffuse" in prediction:
            D_hat = prediction["diffuse"][:, :3]
            # choose GT for gradient comparison (prefer clean diffuse GT; else input)
            D_ref = ground_truth.get("diffuse", ground_truth["rgb_highlighted"])[:, :3]

            if self.weight_ring_grad > 0:
                grad = _grad_mag(D_hat - D_ref).mean(dim=1, keepdim=True)  # (B,1,H,W)
                seam_loss = (grad * ring).sum() / (ring.sum() + 1e-6)

            if self.weight_tv_in_hole > 0:
                tv_loss = _total_variation(D_hat * m_hole)

        losses["RingGrad"] = seam_loss
        losses["TVinHole"] = tv_loss

        # === Your existing saturation/texture ring loss (kept as-is) ===
        if "diffuse" in prediction:
            diffuse_pred_rgb = prediction["diffuse"][:, :3]
            include_mask = m_hole if m_hole is not None else torch.ones_like(diffuse_pred_rgb[:, :1])
            if self.weight_saturation_ring > 0.0:
                sat_ring = saturation_ring_blob_consistency(
                    diffuse_rgb=diffuse_pred_rgb,
                    include_mask=include_mask,
                    ring_kernel_size=self.ring_kernel_size,
                )
                losses["SaturationRing"] = sat_ring
            else:
                losses["SaturationRing"] = torch.tensor(0.0, device=diffuse_pred_rgb.device, dtype=diffuse_pred_rgb.dtype)
        else:
            losses["SaturationRing"] = torch.tensor(0.0)
            
        if self.weight_token_inpaint > 0:
            if "tokens_completed" not in prediction:
                raise KeyError("prediction must include 'tokens_completed' when weight_token_inpaint > 0")
            if "tokens_teacher" not in prediction:
                raise KeyError("prediction must include 'tokens_teacher' when weight_token_inpaint > 0")

            tokens_completed = prediction["tokens_completed"]
            tokens_teacher   = prediction["tokens_teacher"]

            # Prefer directly provided patch_mask_sup; else try to derive from pixel supervised mask
            if "patch_mask_sup" in ground_truth and ground_truth["patch_mask_sup"] is not None:
                pm_sup = ground_truth["patch_mask_sup"].bool()  # (B,N)
            else:
                # Derive from pixel m_sup using patch_size (must be provided)
                if m_sup is None:
                    raise KeyError("Need pixel 'mask' (m_sup) or ground_truth['patch_mask_sup'] for token loss.")
                patch_size = int(ground_truth.get("patch_size", 16))
                pm_sup = _pixel_to_patch_mask(m_sup, patch=patch_size)  # (B,N) bool

            l_token = self._token_inpaint_loss(tokens_completed, tokens_teacher, pm_sup)
            losses["TokenInpaint"] = l_token
        else:
            losses["TokenInpaint"] = torch.zeros(())

        # === TOTAL ===
        total = 0.0
        if "Specular" in losses:
            total = total + self.component_weights["specular"] * losses["Specular"]
        if "Diffuse" in losses:
            total = total + self.component_weights["diffuse"] * losses["Diffuse"]
        if "HighlightRegression" in losses:
            total = total + self.component_weights["highlight"] * losses["HighlightRegression"]
        if losses.get("Reconstruction") is not None:
            total = total + self.weight_image_reconstruction * losses["Reconstruction"]

        # NEW weights
        total = total + self.weight_context_identity * losses["ContextIdentity"]
        total = total + self.weight_ring_grad * losses["RingGrad"]
        total = total + self.weight_tv_in_hole * losses["TVinHole"]

        # existing texture/saturation ring term
        total = total + self.weight_saturation_ring * losses["SaturationRing"]

        # new token-space loss
        total = total + self.weight_token_inpaint * losses["TokenInpaint"]

        losses["total"] = total
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