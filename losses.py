import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
#   Low-level loss pieces
# -------------------------
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", self._create_window())

    def _create_window(self):
        coords = torch.arange(self.window_size, dtype=torch.float32)
        coords -= self.window_size // 2
        g = torch.exp(-(coords**2) / (2 * self.sigma**2))
        g /= g.sum()
        window = g.unsqueeze(1) @ g.unsqueeze(0)  # [w,w]
        return window.unsqueeze(0).unsqueeze(0)    # [1,1,w,w]

    def forward(self, x, y):
        # x, y: [B, C, H, W]
        B, C, H, W = x.shape
        window = self.window.to(device=x.device, dtype=x.dtype).expand(C, 1, -1, -1)

        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=C)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=C)

        mu_x_sq = mu_x**2
        mu_y_sq = mu_y**2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, window, padding=self.window_size // 2, groups=C) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=self.window_size // 2, groups=C) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=C) - mu_xy

        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
            (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        )
        return ssim_map.mean()


class CharbonnierLoss(nn.Module):
    """Smooth L1:  sqrt((x)^2 + eps^2).mean()  — zero at equality."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps**2).mean()


class SquaredSoftDiceLoss(nn.Module):
    """
    Soft-Dice with squared denominator:
    Dice = (2 * <p,y>) / (||p||^2 + ||y||^2)  ->  Loss = 1 - Dice
    => Loss == 0 when pred == target, even for soft labels.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        B = pred.size(0)
        p = pred.view(B, -1)
        y = target.view(B, -1)
        inter = (p * y).sum(dim=1)
        denom = (p.pow(2).sum(dim=1) + y.pow(2).sum(dim=1))
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class GradientLoss(nn.Module):
    """Match finite-difference gradients — zero at equality."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 1]], dtype=torch.float32).view(1, 1, 1, 2)
        ky = torch.tensor([[-1], [1]], dtype=torch.float32).view(1, 1, 2, 1)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, pred, target):
        dx_p = F.conv2d(pred, self.kx)
        dy_p = F.conv2d(pred, self.ky)
        dx_t = F.conv2d(target, self.kx)
        dy_t = F.conv2d(target, self.ky)
        return (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()


class TVLoss(nn.Module):
    """Total variation on the prediction."""
    def forward(self, x):
        tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w


# -------------------------------------
#   Highlight regression (alpha in [0,1])
# -------------------------------------
class HighlightRegressionLoss(nn.Module):
    """
    Per-pixel regression loss for soft highlight fraction alpha ∈ [0,1].
    All selected terms are 0 when pred == gt.
    """
    def __init__(
        self,
        w_l1=1.0,              # Charbonnier/L1 main term
        use_charbonnier=True,
        w_dice=0.0,            # squared soft-Dice
        w_ssim=0.0,            # SSIM on alpha
        w_grad=0.0,            # gradient consistency
        w_tv=0.0,              # TV on pred (regularizer)
        ssim_impl=None,        # pass SSIMLoss() if using SSIM
        dice_smooth=1e-6,
        charbonnier_eps=1e-6,
        clamp_to_unit=True,
        # New: class-imbalance and stabilization options (backward compatible)
        balance_mode: str = "none",   # 'none' | 'auto' | 'pos_weight'
        pos_weight: float = 1.0,       # used when balance_mode == 'pos_weight'
        focal_gamma: float = 0.0,      # >0 to focus large errors, 0 keeps old behavior
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_dice = w_dice
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.w_tv = w_tv
        self.clamp_to_unit = clamp_to_unit

        self.l_main = CharbonnierLoss(eps=charbonnier_eps) if use_charbonnier else nn.L1Loss()
        self.l_dice = SquaredSoftDiceLoss(smooth=dice_smooth)
        self.l_grad = GradientLoss()
        self.l_tv = TVLoss()
        self.ssim = ssim_impl
        self.balance_mode = balance_mode
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma

    def forward(self, pred, target):
        if self.clamp_to_unit:
            pred = torch.clamp(pred, 0.0, 1.0)
            target = torch.clamp(target, 0.0, 1.0)

        loss = 0.0
        if self.w_l1 > 0:
            # Optional focal modulation on the per-pixel residual (keeps grads for small errors too)
            if self.focal_gamma > 0.0:
                # detach target to avoid second-order effects; keep pred in graph
                resid = (pred - target).abs()
                focal_w = torch.pow(resid.clamp_min(1e-6), self.focal_gamma)
            else:
                focal_w = 1.0

            if self.balance_mode == "none":
                main_term = self.l_main(pred * focal_w, target * focal_w)
            else:
                # Compute per-pixel weights
                if self.balance_mode == "auto":
                    # Balance positives/negatives to contribute equally
                    # target assumed in [0,1]; threshold at 0.5 for positives
                    with torch.no_grad():
                        pos_frac = (target >= 0.5).float().mean().clamp_min(1e-6)
                        w_pos = 0.5 / pos_frac
                        w_neg = 0.5 / (1.0 - pos_frac)
                        pixel_w = torch.where(target >= 0.5, w_pos, w_neg)
                elif self.balance_mode == "pos_weight":
                    pixel_w = torch.where(target >= 0.5, self.pos_weight, 1.0)
                else:
                    pixel_w = 1.0

                if isinstance(self.l_main, CharbonnierLoss):
                    # Inline charbonnier to support per-pixel weights
                    eps = self.l_main.eps
                    main_term = torch.sqrt((pred - target) ** 2 + eps**2)
                    main_term = (main_term * pixel_w * focal_w).mean()
                else:
                    main_term = ((pred - target).abs() * pixel_w * focal_w).mean()

            loss = loss + self.w_l1 * main_term
        if self.w_dice > 0:
            loss = loss + self.w_dice * self.l_dice(pred, target)
        if self.w_ssim > 0 and self.ssim is not None:
            loss = loss + self.w_ssim * (1.0 - self.ssim(pred, target))
        if self.w_grad > 0:
            loss = loss + self.w_grad * self.l_grad(pred, target)
        if self.w_tv > 0:
            loss = loss + self.w_tv * self.l_tv(pred)
        return loss


# -------------------------
#   Compositing utilities
# -------------------------
def alpha_composite(components):
    """
    Alpha composite multiple RGBA components, over black.
    components: list of [B,4,H,W] tensors
    returns: [B,3,H,W]
    """
    result = torch.zeros_like(components[0][:, :3])
    for comp in components:
        rgb = comp[:, :3]
        a = comp[:, 3:4]
        result = a * rgb + (1 - a) * result
    return result


def compose_diffuse_highlight_and_layers(
    diffuse_rgb, additive_highlights_rgb, layered_rgba, clamp_after_add=True
):
    """
    diffuse_rgb: [B,3,H,W]
    additive_highlights_rgb: list of [B,3,H,W] tensors to ADD (e.g., alpha * color)
    layered_rgba: list of [B,4,H,W] layers to alpha-over on top
    """
    result = diffuse_rgb
    for h in additive_highlights_rgb:
        result = result + h
    if clamp_after_add:
        result = torch.clamp(result, 0.0, 1.0)
    # Alpha-over any actual layers on top (rare for this use-case)
    for comp in layered_rgba:
        rgb = comp[:, :3]
        a = comp[:, 3:4]
        result = a * rgb + (1 - a) * result
    return result


# -------------------------
#   Main Decomposition loss
# -------------------------
class DecompositionLoss(nn.Module):
    """
    Flexible intrinsic decomposition objective.

    - Diffuse, Specular: supervised with L1 + (1 - SSIM) (+ optional alpha L1 if RGBA).
    - Highlight (1-ch): treated as per-pixel regression alpha ∈ [0,1].
    - Reconstruction: ADDITIVE highlight composition:
        I_recon = diffuse + alpha_highlight * color_highlight  (clamped)
      so perfect diffuse & highlight -> zero reconstruction loss.
    - Alpha regularization: configurable (disabled by default to allow 0 total at perfection).
    """
    def __init__(
        self,
        component_weights=None,
        # Individual component loss weights
        weight_specular_loss=1.0,
        weight_diffuse_loss=1.0,
        weight_highlight_loss=1.0,

        # Global loss term weights
        weight_component_matching=1.0,      # How well components match their ground truth
        weight_image_reconstruction=0.5,    # How well reconstructed image matches input
        weight_alpha_regularization=0.0,    # Default 0.0 to allow exact-0 totals in debug
        weight_spatial_consistency=0.0,     # Reserved hook (unused here)

        # highlight regression config
        hlreg_w_l1=1.0,
        hlreg_use_charb=True,
        hlreg_w_dice=0.2,
        hlreg_w_ssim=0.0,
        hlreg_w_grad=0.0,
        hlreg_w_tv=0.0,
        # New knobs (forwarded to HighlightRegressionLoss)
        hlreg_balance_mode: str = "none",
        hlreg_pos_weight: float = 1.0,
        hlreg_focal_gamma: float = 0.0,

        # highlight rendering
        highlight_color=(1.0, 1.0, 1.0),  # C_highlight
        clamp_after_add=True,

        # alpha regularization behavior
        alpha_reg_mode="none",    # 'none' | 'variance' | 'match_gt'
    ):
        super().__init__()

        # component weights
        if component_weights is not None:
            self.component_weights = dict(component_weights)
        else:
            self.component_weights = {
                "specular": weight_specular_loss,
                "diffuse":  weight_diffuse_loss,
                "highlight": weight_highlight_loss,
            }
        self.default_component_weight = 1.0

        # global weights
        self.weight_component_matching = weight_component_matching
        self.weight_image_reconstruction = weight_image_reconstruction
        self.weight_alpha_regularization = weight_alpha_regularization
        self.weight_spatial_consistency = weight_spatial_consistency

        # highlight rendering
        self.highlight_color = highlight_color
        self.clamp_after_add = clamp_after_add

        # alpha reg behavior
        assert alpha_reg_mode in ("none", "variance", "match_gt")
        self.alpha_reg_mode = alpha_reg_mode

        # losses
        self.ssim_loss = SSIMLoss()
        self.highlight_regression_loss = HighlightRegressionLoss(
            w_l1=hlreg_w_l1,
            use_charbonnier=hlreg_use_charb,
            w_dice=hlreg_w_dice,
            w_ssim=hlreg_w_ssim,
            w_grad=hlreg_w_grad,
            w_tv=hlreg_w_tv,
            ssim_impl=self.ssim_loss,
            balance_mode=hlreg_balance_mode,
            pos_weight=hlreg_pos_weight,
            focal_gamma=hlreg_focal_gamma,
        )

    # --- helpers ---
    def _single_to_rgb_highlight(self, highlight_single):
        """[B,1,H,W] -> [B,3,H,W] as alpha * color."""
        r, g, b = self.highlight_color
        return torch.cat(
            [
                highlight_single * r,
                highlight_single * g,
                highlight_single * b,
            ],
            dim=1,
        )

    def _single_to_rgba_highlight(self, highlight_single):
        """If you ever want to alpha-over highlights (not recommended for energy), keep this."""
        rgb = self._single_to_rgb_highlight(highlight_single)
        return torch.cat([rgb, highlight_single], dim=1)  # [B,4,H,W]

    # --- main forward ---
    def forward(self, pred_components, gt_components):
        """
        pred_components: dict mapping name -> [B,C,H,W]
        gt_components:   dict mapping name -> [B,C,H,W], requires key 'rgb'
        """
        losses = {}
        input_rgb = gt_components["rgb"]

        # components present on both sides (except 'rgb')
        available_components = [k for k in pred_components.keys()
                                if (k in gt_components and k != "rgb")]
        if not available_components:
            raise ValueError("No matching components found between predictions and ground truth")

        decomposition_loss = 0.0

        diffuse_rgb = None
        additive_highlights = []     # list of [B,3,H,W] added to diffuse
        layered_rgba = []            # optional layers to alpha-over

        for comp_name in available_components:
            pred_comp = pred_components[comp_name]
            gt_comp = gt_components[comp_name]
            comp_weight = self.component_weights.get(comp_name, self.default_component_weight)

            # ---- Highlight: 1 channel regression ----
            if comp_name.lower() == "highlight" and pred_comp.shape[1] == 1 and gt_comp.shape[1] == 1:
                pred_h = torch.clamp(pred_comp, 0.0, 1.0)
                gt_h   = torch.clamp(gt_comp,   0.0, 1.0)

                hl_loss = self.highlight_regression_loss(pred_h, gt_h)
                losses["HighlightRegression"] = hl_loss
                decomposition_loss = decomposition_loss + comp_weight * hl_loss

                # for reconstruction: ADDITIVE energy
                additive_highlights.append(self._single_to_rgb_highlight(pred_h))
                continue

            # ---- Diffuse / Specular / others ----
            if pred_comp.shape[1] >= 3 and gt_comp.shape[1] >= 3:
                rgb_l1 = F.l1_loss(pred_comp[:, :3], gt_comp[:, :3])
                rgb_ssim = self.ssim_loss(pred_comp[:, :3], gt_comp[:, :3])
            else:
                rgb_l1 = F.l1_loss(pred_comp[:, :1], gt_comp[:, :1])
                rgb_ssim = 0.0

            alpha_l1 = 0.0
            if pred_comp.shape[1] == 4 and gt_comp.shape[1] == 4:
                alpha_l1 = F.l1_loss(pred_comp[:, 3:4], gt_comp[:, 3:4])

            comp_loss = rgb_l1 + (1 - rgb_ssim) + alpha_l1
            decomposition_loss = decomposition_loss + comp_weight * comp_loss
            losses[f"{comp_name.capitalize()}"] = comp_loss

            # For reconstruction:
            if comp_name.lower() == "diffuse":
                diffuse_rgb = pred_comp[:, :3]
            else:
                # If you truly have layered components, push RGBA layers here.
                # If component is inherently additive (specular), you can add it to additive_highlights instead.
                if pred_comp.shape[1] == 4:
                    layered_rgba.append(pred_comp)
                # elif pred_comp.shape[1] == 3:
                #     additive_highlights.append(pred_comp)  # uncomment if you want additive behavior

        losses["Decomposition"] = decomposition_loss

        # ---- Reconstruction: additive highlight + optional layered alpha-over ----
        reconstruction_loss = 0.0
        if diffuse_rgb is not None:
            pred_reconstruction = compose_diffuse_highlight_and_layers(
                diffuse_rgb, additive_highlights, layered_rgba, clamp_after_add=self.clamp_after_add
            )
            recon_l1 = F.l1_loss(pred_reconstruction, input_rgb)
            recon_ssim = self.ssim_loss(pred_reconstruction, input_rgb)
            reconstruction_loss = recon_l1 + (1 - recon_ssim)
        losses["Reconstruction"] = reconstruction_loss

        # ---- Alpha regularization (disabled by default; designed to be 0 when 'match_gt') ----
        alpha_reg_loss = 0.0
        if self.alpha_reg_mode != "none":
            # For 'variance', penalize near-constant alphas (old behavior).
            # For 'match_gt', encourage pred highlight == gt highlight (==0 at equality).
            if self.alpha_reg_mode == "variance":
                # variance penalty across any RGBA layer alphas (rarely used here)
                for comp_name in available_components:
                    if comp_name.lower() == "highlight":
                        alpha = torch.clamp(pred_components[comp_name], 0, 1)
                        alpha_var = torch.var(alpha.view(alpha.size(0), -1), dim=1).mean()
                        alpha_reg_loss = alpha_reg_loss + torch.exp(-alpha_var)
            elif self.alpha_reg_mode == "match_gt":
                if "highlight" in pred_components and "highlight" in gt_components:
                    pred_h = torch.clamp(pred_components["highlight"], 0, 1)
                    gt_h   = torch.clamp(gt_components["highlight"], 0, 1)
                    alpha_reg_loss = F.l1_loss(pred_h, gt_h)

        losses["AlphaReg"] = alpha_reg_loss

        total_loss = (
            self.weight_component_matching * losses["Decomposition"]
            + self.weight_image_reconstruction * losses["Reconstruction"]
            + self.weight_alpha_regularization * losses["AlphaReg"]
        )

        losses["total"] = total_loss
        return losses

