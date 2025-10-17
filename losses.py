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
#   Color-space utilities
# -------------------------
def rgb_hsv_saturation(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute HSV saturation from RGB in [0,1].

    Args:
        x: [B,3,H,W] tensor in [0,1]
        eps: numerical stability for division

    Returns:
        [B,1,H,W] saturation in [0,1]
    """
    # [B,1,H,W]
    maxc, _ = x.max(dim=1, keepdim=True)
    minc, _ = x.min(dim=1, keepdim=True)
    delta = maxc - minc
    return delta / (maxc + eps)


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
        weight_spatial_consistency=0.0,  # legacy placeholder (unused)

        # New: diffuse saturation and ring-consistency regularizers (default off)
        weight_diffuse_saturation: float = 0.0,
        diffuse_saturation_max: float = 0.35,
        weight_diffuse_ring: float = 0.0,
        ring_kernel_size: int = 7,           # odd; dilation size for surrounding ring
        ring_var_weight: float = 0.5,        # weight on variance matching vs mean matching
        ring_texture_weight: float = 1.0,    # weight on texture consistency term

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
        self.weight_diffuse_saturation = weight_diffuse_saturation
        self.diffuse_saturation_max = diffuse_saturation_max
        self.weight_diffuse_ring = weight_diffuse_ring
        self.ring_kernel_size = ring_kernel_size
        self.ring_var_weight = ring_var_weight
        self.ring_texture_weight = ring_texture_weight

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

        # Small gradient kernels for texture consistency (applied per-channel)
        kx = torch.tensor([[-1.0, 1.0]], dtype=torch.float32).view(1, 1, 1, 2)
        ky = torch.tensor([[-1.0], [1.0]], dtype=torch.float32).view(1, 1, 2, 1)
        self.register_buffer("_ring_kx", kx)
        self.register_buffer("_ring_ky", ky)

    @staticmethod
    def _to_single_channel_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        Ensure binary mask [B,1,H,W]. Accepts [B,1,H,W] or [B,C,H,W]. Values in {0,1}.
        """
        if mask.dim() != 4:
            raise ValueError("mask must be 4D [B,C,H,W]")
        if mask.size(1) == 1:
            return mask
        return (mask.sum(dim=1, keepdim=True) > 0).float()

    @staticmethod
    def _safe_mean(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # x: [B,C,H,W], m: [B,1,H,W]
        m = m.clamp(0.0, 1.0)
        denom = m.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
        return (x * m).sum(dim=(2, 3), keepdim=True) / denom

    @staticmethod
    def _safe_var(x: torch.Tensor, m: torch.Tensor, mean: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
        m = m.clamp(0.0, 1.0)
        if mean is None:
            mean = UnReflectLoss._safe_mean(x, m, eps)
        denom = m.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
        return (((x - mean) ** 2) * m).sum(dim=(2, 3), keepdim=True) / denom

    def _hole_and_ring_masks(self, include_mask: torch.Tensor) -> tuple:
        """
        From include mask [B,1,H,W] (1=valid/context), build hole mask (complement)
        and a thin ring around the hole using max-pooling dilation.
        Returns (hole_mask, ring_mask) both [B,1,H,W].
        """
        base = self._to_single_channel_mask(include_mask)
        hole = (1.0 - base).clamp(0.0, 1.0)
        if self.ring_kernel_size <= 1:
            return hole, torch.zeros_like(hole)
        pad = self.ring_kernel_size // 2
        dilated = F.max_pool2d(hole, kernel_size=self.ring_kernel_size, stride=1, padding=pad)
        ring = (dilated - hole).clamp(0.0, 1.0)
        return hole, ring

    def _saturation_hinge(self, diffuse_rgb: torch.Tensor, hole_mask: torch.Tensor) -> torch.Tensor:
        # diffuse_rgb: [B,3,H,W], hole_mask: [B,1,H,W]
        sat = rgb_hsv_saturation(diffuse_rgb)
        excess = (sat - self.diffuse_saturation_max).clamp_min(0.0)
        eps = 1e-8
        return (excess * hole_mask).sum() / (hole_mask.sum() + eps)

    def _texture_map(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-channel gradient magnitude map using small finite differences."""
        B, C, H, W = x.shape
        # Forward finite differences with padding to preserve size
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dx = F.pad(dx, (0, 1, 0, 0))  # pad one column on the right -> [B,C,H,W]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = F.pad(dy, (0, 0, 0, 1))  # pad one row at the bottom -> [B,C,H,W]
        return torch.sqrt(dx * dx + dy * dy + 1e-8)

    def _ring_consistency_loss(self, diffuse_rgb: torch.Tensor, hole_mask: torch.Tensor, ring_mask: torch.Tensor) -> torch.Tensor:
        # Color statistics alignment (mean + variance)
        mean_h = self._safe_mean(diffuse_rgb, hole_mask)   # [B,C,1,1]
        mean_r = self._safe_mean(diffuse_rgb, ring_mask)   # [B,C,1,1]
        var_h = self._safe_var(diffuse_rgb, hole_mask, mean_h)
        var_r = self._safe_var(diffuse_rgb, ring_mask, mean_r)
        color_term = (mean_h - mean_r).abs().mean() + self.ring_var_weight * (var_h - var_r).abs().mean()

        # Texture alignment via gradient magnitude statistics
        tex = self._texture_map(diffuse_rgb)
        mean_tex_h = self._safe_mean(tex, hole_mask)
        mean_tex_r = self._safe_mean(tex, ring_mask)
        texture_term = (mean_tex_h - mean_tex_r).abs().mean()

        return color_term + self.ring_texture_weight * texture_term

    def _single_to_rgb_highlight(self, highlight_single):
        """Convert [B,1,H,W] highlight alpha to [B,3,H,W] additive RGB."""
        # highlight_single: [B,1,H,W]
        color = self.highlight_color.to(highlight_single.device)  # [3]
        # Broadcast: [B,1,H,W] * [3,1,1] -> [B,3,H,W]
        return highlight_single * color.view(1, 3, 1, 1)

    def reconstruct_image(self, prediction, mask=None):
        """
        Reconstruct rgb_highlighted from predicted components using alpha_composite.
    
        Args:
            mask: [B,1,H,W] - if provided, components are masked before compositing
    
        Returns: [B,3,H,W]
        """ 
        composite_dict = {}

        # Apply mask to components BEFORE compositing
        if mask is not None:
            mask_3ch = mask.expand(-1, 3, -1, -1)  # [B,3,H,W]
    
        # Diffuse (always required)
        if 'diffuse' not in prediction:
            raise ValueError("Diffuse component is required for reconstruction")
    
        diffuse = prediction['diffuse']
        if mask is not None:
            # Mask RGB channels
            if diffuse.shape[1] == 3:
                diffuse = diffuse * mask_3ch
            elif diffuse.shape[1] == 4:
                diffuse = torch.cat([
                    diffuse[:, :3] * mask_3ch,
                    diffuse[:, 3:4] * mask  # Mask alpha too
                ], dim=1)
        composite_dict['diffuse'] = diffuse

        # Specular (if present)
        if 'specular' in prediction:
            specular = prediction['specular']
            if mask is not None:
                if specular.shape[1] == 3:
                    specular = specular * mask_3ch
                elif specular.shape[1] == 4:
                    specular = torch.cat([
                        specular[:, :3] * mask_3ch,
                        specular[:, 3:4] * mask
                    ], dim=1)
            composite_dict['specular'] = specular

        # Highlight (if present)
        if 'highlight' in prediction:
            highlight = torch.clamp(prediction['highlight'], 0.0, 1.0)
            if mask is not None:
                highlight = highlight * mask  # [B,1,H,W]
            composite_dict['highlight'] = highlight

        # Composite (now only unmasked pixels contribute)
        composed_rgb = alpha_composite(composite_dict, output_format='rgb')
    
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
                pred_reconstruction = self.reconstruct_image(prediction,mask)  # [B,3,H,W]
                input_rgb = ground_truth['rgb_highlighted']  # [B,3,H,W]
        
                # IMPORTANT: Apply mask to reconstruction loss too!
                # In real highlight regions, we have NO ground truth for clean diffuse
                # Reconstruction would force: pred_diffuse + pred_highlight = rgb (with real highlights)
                # This would teach wrong diffuse values in those regions
                # So we MUST mask reconstruction loss in real highlight regions
                recon_l1 = self.masked_l1_loss(pred_reconstruction, input_rgb, mask)  # Use mask!
                recon_ssim = self.ssim_loss(pred_reconstruction, input_rgb, mask)  # Use mask!
                reconstruction_loss = recon_l1 + (1.0 - recon_ssim)
            except ValueError:
                # If diffuse is missing, we can't reconstruct - skip reconstruction loss
                pass

        losses["Reconstruction"] = reconstruction_loss

        # ===== Diffuse saturation + ring consistency regularizers (optional) =====
        if "diffuse" in prediction:
            diffuse_pred_rgb = prediction["diffuse"][:, :3]
            include_mask = mask if mask is not None else torch.ones_like(diffuse_pred_rgb[:, :1])
            hole_mask, ring_mask = self._hole_and_ring_masks(include_mask)

            # Only compute when there is a non-empty hole
            if self.weight_diffuse_saturation > 0.0 and hole_mask.sum() > 0:
                sat_loss = self._saturation_hinge(diffuse_pred_rgb, hole_mask)
                losses["Saturation"] = sat_loss
            else:
                losses["Saturation"] = torch.tensor(0.0, device=diffuse_pred_rgb.device, dtype=diffuse_pred_rgb.dtype)

            if self.weight_diffuse_ring > 0.0 and (hole_mask.sum() > 0 and ring_mask.sum() > 0):
                ring_loss = self._ring_consistency_loss(diffuse_pred_rgb, hole_mask, ring_mask)
                losses["DiffuseRingConsistency"] = ring_loss
            else:
                losses["DiffuseRingConsistency"] = torch.tensor(0.0, device=diffuse_pred_rgb.device, dtype=diffuse_pred_rgb.dtype)
        else:
            losses["Saturation"] = torch.tensor(0.0)
            losses["DiffuseRingConsistency"] = torch.tensor(0.0)

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
            + self.weight_diffuse_saturation * losses["Saturation"]
            + self.weight_diffuse_ring * losses["DiffuseRingConsistency"]
        )
        
        losses["total"] = total_loss
        return losses


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining student and teacher outputs.
    
    This loss function implements the standard knowledge distillation approach
    where the student model learns from both the ground truth labels and the
    soft predictions of a teacher model.
    """
    
    def __init__(self, alpha=0.7, temperature=4.0, reduction='mean'):
        """
        Args:
            alpha (float): Weight for distillation loss vs hard target loss
            temperature (float): Temperature for softmax scaling
            reduction (str): Reduction method for loss computation
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        
        # Standard cross-entropy loss for hard targets
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        
        # KL divergence loss for distillation
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, targets=None):
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model predictions [B, C, ...]
            teacher_logits: Teacher model predictions [B, C, ...]
            targets: Ground truth targets (optional) [B, ...]
            
        Returns:
            dict: Dictionary containing individual loss components
        """
        losses = {}
        
        # Compute distillation loss (KL divergence between soft predictions)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        distillation_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        losses["distillation"] = distillation_loss
        
        # Compute hard target loss if targets are provided
        if targets is not None:
            hard_loss = self.ce_loss(student_logits, targets)
            losses["hard_target"] = hard_loss
            
            # Combined loss
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        else:
            # Only distillation loss
            total_loss = distillation_loss
        
        losses["total"] = total_loss
        return losses


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level Knowledge Distillation Loss.
    
    This loss function distills knowledge at the feature level rather than
    just the final predictions, which can be more effective for complex models.
    """
    
    def __init__(self, alpha=0.7, temperature=4.0, feature_weights=None):
        """
        Args:
            alpha (float): Weight for distillation loss vs hard target loss
            temperature (float): Temperature for softmax scaling
            feature_weights (list): Weights for different feature layers
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.feature_weights = feature_weights or [1.0]
        
        # Standard cross-entropy loss for hard targets
        self.ce_loss = nn.CrossEntropyLoss()
        
        # MSE loss for feature distillation
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_outputs, teacher_outputs, targets=None):
        """
        Compute feature distillation loss.
        
        Args:
            student_outputs: Dict with student model outputs including features
            teacher_outputs: Dict with teacher model outputs including features
            targets: Ground truth targets (optional)
            
        Returns:
            dict: Dictionary containing individual loss components
        """
        losses = {}
        
        # Feature distillation loss
        feature_loss = 0.0
        if "features" in student_outputs and "features" in teacher_outputs:
            student_features = student_outputs["features"]
            teacher_features = teacher_outputs["features"]
            
            # Handle multiple feature layers
            if isinstance(student_features, (list, tuple)):
                for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
                    weight = self.feature_weights[i] if i < len(self.feature_weights) else 1.0
                    feature_loss += weight * self.mse_loss(s_feat, t_feat)
            else:
                feature_loss = self.mse_loss(student_features, teacher_features)
        
        losses["feature_distillation"] = feature_loss
        
        # Prediction distillation loss
        if "logits" in student_outputs and "logits" in teacher_outputs:
            student_logits = student_outputs["logits"]
            teacher_logits = teacher_outputs["logits"]
            
            student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
            
            pred_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
            losses["prediction_distillation"] = pred_loss
        else:
            pred_loss = 0.0
        
        # Hard target loss if targets are provided
        if targets is not None and "logits" in student_outputs:
            hard_loss = self.ce_loss(student_outputs["logits"], targets)
            losses["hard_target"] = hard_loss
            
            # Combined loss
            total_loss = self.alpha * (feature_loss + pred_loss) + (1 - self.alpha) * hard_loss
        else:
            # Only distillation losses
            total_loss = feature_loss + pred_loss
        
        losses["total"] = total_loss
        return losses

