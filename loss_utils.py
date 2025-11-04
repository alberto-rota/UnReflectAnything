import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # type: ignore
from scipy.ndimage import label  # type: ignore

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
        # mask: [B, 1, H, W] or [B, C, H, W]
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        B, C, H, W = x.shape
        window = self.window.to(device=x.device, dtype=x.dtype).expand(C, 1, -1, -1)

        if mask.shape[1] == 1:
            mask = mask.expand(-1, C, -1, -1)  # [B, C, H, W]

        x_masked = x * mask
        y_masked = y * mask

        mu_x = F.conv2d(x_masked, window, padding=self.window_size // 2, groups=C)
        mu_y = F.conv2d(y_masked, window, padding=self.window_size // 2, groups=C)

        mask_weights = F.conv2d(mask.float(), window, padding=self.window_size // 2, groups=C)

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

        ssim_map_masked = ssim_map * (mask_weights > 0).float()
        return ssim_map_masked.sum() / ((mask_weights > 0).float().sum() + epsilon)


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, mask=None):
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        if mask.shape[1] == 1:
            mask = mask.expand_as(x)

        l1_map = torch.abs(x - y) * mask
        epsilon = 1e-8
        return l1_map.sum() / (mask.sum() + epsilon)


class CharbonnierLoss(nn.Module):
    """Smooth L1: sqrt((x)^2 + eps^2).mean()."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps**2).mean()


class SquaredSoftDiceLoss(nn.Module):
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
    def forward(self, x):
        tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w


def rgb_hsv_saturation(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    maxc, _ = x.max(dim=1, keepdim=True)
    minc, _ = x.min(dim=1, keepdim=True)
    delta = maxc - minc
    return delta / (maxc + eps)


def alpha_composite(components, output_format='rgb'):
    diffuse = components['diffuse']
    B, C, H, W = diffuse.shape

    if C == 1:
        rgb = diffuse.expand(-1, 3, -1, -1)
        alpha = diffuse
        result_rgba = torch.cat([rgb, alpha], dim=1)
    elif C == 3:
        alpha = torch.ones_like(diffuse[:, :1])
        result_rgba = torch.cat([diffuse, alpha], dim=1)
    else:
        result_rgba = diffuse

    for key, comp in components.items():
        if key == 'diffuse':
            continue

        C = comp.shape[1]
        if C == 1:
            rgb = comp.expand(-1, 3, -1, -1)
            alpha = comp
            comp_rgba = torch.cat([rgb, alpha], dim=1)
        elif C == 3:
            alpha = torch.ones_like(comp[:, :1])
            comp_rgba = torch.cat([comp, alpha], dim=1)
        else:
            comp_rgba = comp

        fg_rgb = comp_rgba[:, :3]
        fg_a = comp_rgba[:, 3:4]
        bg_rgb = result_rgba[:, :3]
        bg_a = result_rgba[:, 3:4]

        out_rgb = fg_a * fg_rgb + (1 - fg_a) * bg_rgb
        out_a = fg_a + (1 - fg_a) * bg_a
        result_rgba = torch.cat([out_rgb, out_a], dim=1)

    if output_format.lower() == 'rgba':
        return result_rgba
    else:
        return result_rgba[:, :3]


class HighlightRegressionLoss(nn.Module):
    def __init__(
        self,
        w_l1=1.0,
        use_charbonnier=True,
        w_dice=0.0,
        w_ssim=0.0,
        w_grad=0.0,
        w_tv=0.0,
        dice_smooth=1e-6,
        charbonnier_eps=1e-6,
        clamp_to_unit=True,
        balance_mode: str = "none",
        pos_weight: float = 1.0,
        focal_gamma: float = 0.0,
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
        self.ssim = SSIMLoss()
        self.balance_mode = balance_mode
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma

    def forward(self, pred, target):
        if self.clamp_to_unit:
            pred = torch.clamp(pred, 0.0, 1.0)
            target = torch.clamp(target, 0.0, 1.0)

        loss = 0.0
        if self.w_l1 > 0:
            if self.focal_gamma > 0.0:
                resid = (pred - target).abs()
                focal_w = torch.pow(resid.clamp_min(1e-6), self.focal_gamma)
            else:
                focal_w = 1.0

            if self.balance_mode == "none":
                main_term = self.l_main(pred * focal_w, target * focal_w)
            else:
                if self.balance_mode == "auto":
                    pos_frac = (target >= 0.5).float().mean().clamp_min(1e-6)
                    w_pos = 0.5 / pos_frac
                    w_neg = 0.5 / (1.0 - pos_frac)
                    pixel_w = torch.where(target >= 0.5, w_pos, w_neg)
                elif self.balance_mode == "pos_weight":
                    pixel_w = torch.where(target >= 0.5, self.pos_weight, 1.0)
                else:
                    pixel_w = 1.0

                if isinstance(self.l_main, CharbonnierLoss):
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
#   Additional loss utilities
# -------------------------
def to_single_channel_mask(mask: torch.Tensor) -> torch.Tensor:
    """Ensure [B,1,H,W] binary mask from [B,1,H,W] or [B,C,H,W]."""
    if mask.dim() != 4:
        raise ValueError("mask must be 4D [B,C,H,W]")
    if mask.size(1) == 1:
        return mask
    return (mask.sum(dim=1, keepdim=True) > 0).float()


def safe_mean(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: [B,C,H,W], m: [B,1,H,W]
    m = m.clamp(0.0, 1.0)
    denom = m.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (x * m).sum(dim=(2, 3), keepdim=True) / denom


def safe_var(x: torch.Tensor, m: torch.Tensor, mean: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
    m = m.clamp(0.0, 1.0)
    if mean is None:
        mean = safe_mean(x, m, eps)
    denom = m.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (((x - mean) ** 2) * m).sum(dim=(2, 3), keepdim=True) / denom


def hole_and_ring_masks(include_mask: torch.Tensor, ring_kernel_size: int) -> tuple:
    """From include mask build (hole, ring) via max-pool dilation."""
    base = to_single_channel_mask(include_mask)
    hole = (1.0 - base).clamp(0.0, 1.0)
    if ring_kernel_size <= 1:
        return hole, torch.zeros_like(hole)
    pad = ring_kernel_size // 2
    dilated = F.max_pool2d(hole, kernel_size=ring_kernel_size, stride=1, padding=pad)
    ring = (dilated - hole).clamp(0.0, 1.0)
    return hole, ring


def texture_map(x: torch.Tensor) -> torch.Tensor:
    """Finite-difference gradient magnitude per channel."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-8)


def saturation_hinge(diffuse_rgb: torch.Tensor, hole_mask: torch.Tensor, diffuse_saturation_max: float) -> torch.Tensor:
    sat = rgb_hsv_saturation(diffuse_rgb)
    excess = (sat - diffuse_saturation_max).clamp_min(0.0)
    eps = 1e-8
    return (excess * hole_mask).sum() / (hole_mask.sum() + eps)


def ring_consistency_loss(
    diffuse_rgb: torch.Tensor,
    hole_mask: torch.Tensor,
    ring_mask: torch.Tensor,
    ring_var_weight: float,
    ring_texture_weight: float,
) -> torch.Tensor:
    mean_h = safe_mean(diffuse_rgb, hole_mask)
    mean_r = safe_mean(diffuse_rgb, ring_mask)
    var_h = safe_var(diffuse_rgb, hole_mask, mean_h)
    var_r = safe_var(diffuse_rgb, ring_mask, mean_r)
    color_term = (mean_h - mean_r).abs().mean() + ring_var_weight * (var_h - var_r).abs().mean()

    tex = texture_map(diffuse_rgb)
    mean_tex_h = safe_mean(tex, hole_mask)
    mean_tex_r = safe_mean(tex, ring_mask)
    texture_term = (mean_tex_h - mean_tex_r).abs().mean()
    return color_term + ring_texture_weight * texture_term


def single_to_rgb_highlight(highlight_single: torch.Tensor, highlight_color: torch.Tensor) -> torch.Tensor:
    return highlight_single * highlight_color.view(1, 3, 1, 1)


def reconstruct_image_from_components(
    prediction: dict,
    mask: torch.Tensor = None,
    clamp_reconstruction: bool = True,
) -> torch.Tensor:
    """Compose predicted components to RGB with optional masking and clamping."""
    composite_dict = {}
    mask_3ch = None
    if mask is not None:
        mask_3ch = mask.expand(-1, 3, -1, -1)

    if 'diffuse' not in prediction:
        raise ValueError("Diffuse component is required for reconstruction")
    diffuse = prediction['diffuse']
    if mask_3ch is not None:
        if diffuse.shape[1] == 3:
            diffuse = diffuse * mask_3ch
        elif diffuse.shape[1] == 4:
            diffuse = torch.cat([diffuse[:, :3] * mask_3ch, diffuse[:, 3:4] * mask], dim=1)
    composite_dict['diffuse'] = diffuse

    if 'specular' in prediction:
        specular = prediction['specular']
        if mask_3ch is not None:
            if specular.shape[1] == 3:
                specular = specular * mask_3ch
            elif specular.shape[1] == 4:
                specular = torch.cat([specular[:, :3] * mask_3ch, specular[:, 3:4] * mask], dim=1)
        composite_dict['specular'] = specular

    if 'highlight' in prediction:
        highlight = torch.clamp(prediction['highlight'], 0.0, 1.0)
        if mask is not None:
            highlight = highlight * mask
        composite_dict['highlight'] = highlight

    composed_rgb = alpha_composite(composite_dict, output_format='rgb')
    if clamp_reconstruction:
        composed_rgb = torch.clamp(composed_rgb, 0.0, 1.0)
    return composed_rgb



def saturation_ring_blob_consistency(
    diffuse_rgb: torch.Tensor,
    include_mask: torch.Tensor,
    ring_kernel_size: int,
) -> torch.Tensor:
    """Compare saturation inside each hole blob to its surrounding ring; fallback to global if CC not available.

    Shapes:
    - diffuse_rgb: [B,3,H,W]
    - include_mask: [B,1,H,W] (1 = valid/context)
    Returns a scalar tensor (mean over blobs and batch).
    """
    device = diffuse_rgb.device
    dtype = diffuse_rgb.dtype
    B = diffuse_rgb.size(0)
    sat = rgb_hsv_saturation(diffuse_rgb)  # [B,1,H,W]

    hole_mask, ring_mask_global = hole_and_ring_masks(include_mask, ring_kernel_size)

    total = torch.zeros(1, device=device, dtype=dtype)
    count = torch.zeros(1, device=device, dtype=dtype)

    try:

        structure = np.ones((3, 3), dtype=np.int8)  
        for b in range(B):
            hole_b = hole_mask[b, 0].detach().to("cpu").numpy().astype(np.uint8)
            labels, n = label(hole_b, structure=structure)
            if n == 0:
                continue
            for k in range(1, n + 1):
                blob_np = (labels == k)
                # Build blob mask tensor
                blob = torch.from_numpy(blob_np).to(device=device, dtype=dtype).view(1, 1, *hole_b.shape)
                # Per-blob ring via complement include mask => hole==blob
                include_comp = (1.0 - blob).clamp(0.0, 1.0)
                _, ring_blob = hole_and_ring_masks(include_comp, ring_kernel_size)
                if ring_blob.sum() <= 0:
                    continue
                mean_blob = safe_mean(sat[b:b+1], blob).mean()
                mean_ring = safe_mean(sat[b:b+1], ring_blob).mean()
                total = total + (mean_blob - mean_ring).abs()
                count = count + 1.0
    except Exception:
        # Fallback: global hole vs global ring saturation difference
        mean_hole = safe_mean(sat, hole_mask).mean()
        mean_ring = safe_mean(sat, ring_mask_global).mean()
        total = total + (mean_hole - mean_ring).abs()
        count = count + 1.0

    if count.item() == 0.0:
        return torch.zeros((), device=device, dtype=dtype)
    return (total / count).reshape(())

def _total_variation(x):
    # x: (B,C,H,W)
    tv_h = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    tv_w = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return tv_h + tv_w

def _grad_mag(x):
    # Simple |∇x| using forward differences
    dx = F.pad(x[..., 1:, :] - x[..., :-1, :], (0,0,0,1))
    dy = F.pad(x[..., :, 1:] - x[..., :, :-1], (0,1,0,0))
    return (dx.abs() + dy.abs())

def _pixel_to_patch_mask(m_hw: torch.Tensor, patch: int) -> torch.Tensor:
    """
    Convert (B,1,H,W) pixel mask to boolean (B,N) patch mask using max-pool with stride=patch.
    A patch is masked if ANY pixel inside is masked.
    """
    pm = F.max_pool2d((m_hw > 0.5).float(), kernel_size=patch, stride=patch)
    return pm.flatten(1).bool()  # (B, N)
