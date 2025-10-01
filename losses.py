import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x, y, mask=None):
        # x, y: [B, C, H, W]
        # mask: [B, 1, H, W] or [B, C, H, W] - binary mask (1 = include, 0 = ignore)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        B, C, H, W = x.shape
        window = self.window.to(device=x.device, dtype=x.dtype).expand(C, 1, -1, -1)
        
        # Ensure mask has same number of channels as input
        if mask.shape[1] == 1:
            mask = mask.expand(-1, C, -1, -1)  # [B, C, H, W]

        # Apply mask to inputs
        x_masked = x * mask
        y_masked = y * mask

        # Compute local statistics with masked inputs
        mu_x = F.conv2d(x_masked, window, padding=self.window_size // 2, groups=C)
        mu_y = F.conv2d(y_masked, window, padding=self.window_size // 2, groups=C)
        
        # Compute mask weights (sum of mask values in each window)
        mask_weights = F.conv2d(mask.float(), window, padding=self.window_size // 2, groups=C)
        
        # Normalize means by mask weights (avoid division by zero)
        epsilon = 1e-8
        mu_x = mu_x / (mask_weights + epsilon)
        mu_y = mu_y / (mask_weights + epsilon)

        mu_x_sq = mu_x**2
        mu_y_sq = mu_y**2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x_masked * x_masked, window, padding=self.window_size // 2, groups=C) / (mask_weights + epsilon) - mu_x_sq
        sigma_y_sq = F.conv2d(y_masked * y_masked, window, padding=self.window_size // 2, groups=C) / (mask_weights + epsilon) - mu_y_sq
        sigma_xy = F.conv2d(x_masked * y_masked, window, padding=self.window_size // 2, groups=C) / (mask_weights + epsilon) - mu_xy

        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
            (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        )  # [B, C, H, W]
        
        # Apply mask to SSIM map and compute weighted mean
        ssim_map_masked = ssim_map * (mask_weights > 0).float()  # Zero out regions with no valid mask
        return ssim_map_masked.sum() / ((mask_weights > 0).float().sum() + epsilon)

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, mask=None):
        # x, y: [B, C, H, W]
        # mask: [B, 1, H, W] or [B, C, H, W] - binary mask (1 = include, 0 = ignore)
        
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        # Ensure mask has same number of channels as input
        if mask.shape[1] == 1:
            mask = mask.expand_as(x)  # [B, C, H, W]
        
        # Compute L1 loss only on masked regions
        l1_map = torch.abs(x - y) * mask  # [B, C, H, W]
        
        # Compute mean over valid (non-zero mask) pixels
        epsilon = 1e-8
        return l1_map.sum() / (mask.sum() + epsilon)

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


# -------------------------
#   Compositing utilities
# -------------------------
def alpha_composite(components, output_format='rgb'):
    """
    Alpha composite multiple components over a diffuse base layer.
    
    Handles variable channel counts:
    - 1 channel: grayscale-alpha (value represents both intensity and opacity)
    - 3 channels: RGB with implicit full opacity
    - 4 channels: RGBA
    
    Args:
        components: dict of [B,C,H,W] tensors where C ∈ {1, 3, 4}
                   Must contain 'diffuse' key as the base layer
        output_format: 'rgb' or 'rgba' (default: 'rgb')
    
    Returns:
        [B,C_out,H,W] where C_out is 3 for 'rgb' or 4 for 'rgba'
    """
    # Start with diffuse as base layer
    diffuse = components['diffuse']  # [B,C,H,W]
    B, C, H, W = diffuse.shape
    
    # Convert diffuse to RGBA [B,4,H,W]
    if C == 1:  # Grayscale-alpha: value is both intensity and alpha
        rgb = diffuse.expand(-1, 3, -1, -1)  # [B,3,H,W]
        alpha = diffuse  # [B,1,H,W]
        result_rgba = torch.cat([rgb, alpha], dim=1)  # [B,4,H,W]
    elif C == 3:  # RGB -> add full opacity
        alpha = torch.ones_like(diffuse[:, :1])  # [B,1,H,W]
        result_rgba = torch.cat([diffuse, alpha], dim=1)  # [B,4,H,W]
    else:  # C == 4, already RGBA
        result_rgba = diffuse
    
    # Composite other layers on top in arbitrary order
    for key, comp in components.items():
        if key == 'diffuse':
            continue
        
        C = comp.shape[1]  # [B,C,H,W]
        
        # Convert component to RGBA [B,4,H,W]
        if C == 1:  # Grayscale-alpha
            rgb = comp.expand(-1, 3, -1, -1)  # [B,3,H,W]
            alpha = comp  # [B,1,H,W]
            comp_rgba = torch.cat([rgb, alpha], dim=1)  # [B,4,H,W]
        elif C == 3:  # RGB -> add full opacity
            alpha = torch.ones_like(comp[:, :1])  # [B,1,H,W]
            comp_rgba = torch.cat([comp, alpha], dim=1)  # [B,4,H,W]
        else:  # C == 4, already RGBA
            comp_rgba = comp
        
        # Alpha composite: foreground over background
        fg_rgb = comp_rgba[:, :3]  # [B,3,H,W]
        fg_a = comp_rgba[:, 3:4]  # [B,1,H,W]
        bg_rgb = result_rgba[:, :3]  # [B,3,H,W]
        bg_a = result_rgba[:, 3:4]  # [B,1,H,W]
        
        # Standard "over" operator
        out_rgb = fg_a * fg_rgb + (1 - fg_a) * bg_rgb  # [B,3,H,W]
        out_a = fg_a + (1 - fg_a) * bg_a  # [B,1,H,W]
        
        result_rgba = torch.cat([out_rgb, out_a], dim=1)  # [B,4,H,W]
    
    # Return in requested format
    if output_format.lower() == 'rgba':
        return result_rgba  # [B,4,H,W]
    else:  # 'rgb'
        return result_rgba[:, :3]  # [B,3,H,W]
    

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


class UnReflectLoss(nn.Module):
    """
    Flexible intrinsic decomposition objective with masked losses.
    
    - Components (diffuse, specular): supervised with masked L1 + (1 - SSIM)
    - Highlight (1-ch): per-pixel regression alpha ∈ [0,1]
    - Reconstruction: alpha_composite(diffuse + specular) + additive highlight
    """
    def __init__(
        self,
        component_weights=None,
        # Individual component loss weights
        weight_specular_loss=1.0,
        weight_diffuse_loss=1.0,
        weight_highlight_loss=1.0,

        # Global loss term weights
        weight_component_matching=1.0,
        weight_image_reconstruction=0.5,
        weight_alpha_regularization=0.0,
        weight_spatial_consistency=0.0,

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

        # Alpha regularization behavior
        alpha_reg_mode="none",  # 'none' | 'variance' | 'match_gt'
    ):
        super().__init__()

        # Component weights
        if component_weights is not None:
            self.component_weights = dict(component_weights)
        else:
            self.component_weights = {
                "specular": weight_specular_loss,
                "diffuse": weight_diffuse_loss,
                "highlight": weight_highlight_loss,
            }
        self.default_component_weight = 1.0

        # Global weights
        self.weight_component_matching = weight_component_matching
        self.weight_image_reconstruction = weight_image_reconstruction
        self.weight_alpha_regularization = weight_alpha_regularization
        self.weight_spatial_consistency = weight_spatial_consistency

        # Highlight rendering
        self.highlight_color = torch.tensor(highlight_color, dtype=torch.float32)
        self.clamp_reconstruction = clamp_reconstruction

        # Alpha reg behavior
        assert alpha_reg_mode in ("none", "variance", "match_gt")
        self.alpha_reg_mode = alpha_reg_mode

        # Losses
        self.ssim_loss = SSIMLoss()
        self.masked_l1_loss = MaskedL1Loss()
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

    def _single_to_rgb_highlight(self, highlight_single):
        """Convert [B,1,H,W] highlight alpha to [B,3,H,W] additive RGB."""
        # highlight_single: [B,1,H,W]
        color = self.highlight_color.to(highlight_single.device)  # [3]
        # Broadcast: [B,1,H,W] * [3,1,1] -> [B,3,H,W]
        return highlight_single * color.view(1, 3, 1, 1)

    def reconstruct_image(self, prediction):
        """
        Reconstruct rgb_highlighted from predicted components using alpha_composite.
        All components (diffuse, specular, highlight) are composited together.
    
        Returns: [B,3,H,W]
        """
        # Build dict for alpha_composite
        composite_dict = {}
    
        # Always start with diffuse as base
        if 'diffuse' not in prediction:
            raise ValueError("Diffuse component is required for reconstruction")
    
        composite_dict['diffuse'] = prediction['diffuse']  # [B,C,H,W] where C∈{3,4}
    
        # Add specular if present
        if 'specular' in prediction:
            composite_dict['specular'] = prediction['specular']  # [B,C,H,W] where C∈{3,4}
    
        # Add highlight if present (alpha_composite handles 1-channel as grayscale-alpha)
        if 'highlight' in prediction:
            highlight = torch.clamp(prediction['highlight'], 0.0, 1.0)  # [B,1,H,W]
            composite_dict['highlight'] = highlight
    
        # Composite all components using alpha_composite
        composed_rgb = alpha_composite(composite_dict, output_format='rgb')  # [B,3,H,W]
    
        # Optional clamping (alpha_composite should keep values in [0,1], but just in case)
        if self.clamp_reconstruction:
            composed_rgb = torch.clamp(composed_rgb, 0.0, 1.0)
    
        return composed_rgb

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
        losses = {}
        
        # Default mask: all ones
        if mask is None:
            B, _, H, W = ground_truth['rgb_highlighted'].shape
            mask = torch.ones(B, 1, H, W, device=ground_truth['rgb_highlighted'].device)
        
        # Find available components (exclude 'rgb_highlighted')
        available_components = [
            k for k in prediction.keys()
            if k in ground_truth and k != 'rgb_highlighted'
        ]
        
        if not available_components:
            raise ValueError("No matching components found between predictions and ground truth")

        # ===== Component Matching Loss =====
        decomposition_loss = 0.0

        for comp_name in available_components:
            pred_comp = prediction[comp_name]
            gt_comp = ground_truth[comp_name]
            comp_weight = self.component_weights.get(comp_name, self.default_component_weight)

            # ---- Highlight: 1-channel regression ----
            if comp_name.lower() == "highlight":
                if pred_comp.shape[1] == 1 and gt_comp.shape[1] == 1:
                    pred_h = torch.clamp(pred_comp, 0.0, 1.0)
                    gt_h = torch.clamp(gt_comp, 0.0, 1.0)
                    
                    hl_loss = self.highlight_regression_loss(pred_h, gt_h)
                    losses["HighlightRegression"] = hl_loss
                    decomposition_loss = decomposition_loss + comp_weight * hl_loss
                continue

            # ---- Diffuse / Specular: RGB or RGBA ----
            # Extract RGB channels for comparison
            pred_rgb = pred_comp[:, :3]  # [B,3,H,W]
            gt_rgb = gt_comp[:, :3]  # [B,3,H,W]
            
            # Masked losses on RGB
            rgb_l1 = self.masked_l1_loss(pred_rgb, gt_rgb, mask)
            rgb_ssim = self.ssim_loss(pred_rgb, gt_rgb, mask)
            
            comp_loss = rgb_l1 + (1.0 - rgb_ssim)
            
            # If both have alpha channel, supervise it too
            if pred_comp.shape[1] == 4 and gt_comp.shape[1] == 4:
                alpha_l1 = self.masked_l1_loss(
                    pred_comp[:, 3:4], gt_comp[:, 3:4], mask
                )
                comp_loss = comp_loss + alpha_l1
            
            decomposition_loss = decomposition_loss + comp_weight * comp_loss
            losses[f"{comp_name.capitalize()}"] = comp_loss

        losses["Decomposition"] = decomposition_loss

        # ===== Image Reconstruction Loss =====
        reconstruction_loss = 0.0
        if 'rgb_highlighted' in ground_truth:
            # Reconstruct from ALL available predicted components
            try:
                pred_reconstruction = self.reconstruct_image(prediction)  # [B,3,H,W]
                input_rgb = ground_truth['rgb_highlighted']  # [B,3,H,W]
        
                recon_l1 = self.masked_l1_loss(pred_reconstruction, input_rgb)
                recon_ssim = self.ssim_loss(pred_reconstruction, input_rgb)
                reconstruction_loss = recon_l1 + (1.0 - recon_ssim)
            except ValueError:
                # If diffuse is missing, we can't reconstruct - skip reconstruction loss
                pass

        losses["Reconstruction"] = reconstruction_loss

        # ===== Alpha Regularization =====
        alpha_reg_loss = 0.0
        if self.alpha_reg_mode != "none" and 'highlight' in prediction:
            pred_h = torch.clamp(prediction['highlight'], 0.0, 1.0)
            
            if self.alpha_reg_mode == "variance":
                # Penalize uniform alphas
                alpha_var = torch.var(pred_h.view(pred_h.size(0), -1), dim=1).mean()
                alpha_reg_loss = torch.exp(-alpha_var)
            elif self.alpha_reg_mode == "match_gt" and 'highlight' in ground_truth:
                # Encourage matching GT (redundant with HighlightRegression, but kept for API compat)
                gt_h = torch.clamp(ground_truth['highlight'], 0.0, 1.0)
                alpha_reg_loss = F.l1_loss(pred_h, gt_h)
        
        losses["AlphaRegularization"] = alpha_reg_loss

        # ===== Total Loss =====
        total_loss = (
            self.weight_component_matching * losses["Decomposition"]
            + self.weight_image_reconstruction * losses["Reconstruction"]
            + self.weight_alpha_regularization * losses["AlphaRegularization"]
        )
        
        losses["total"] = total_loss
        return losses

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
