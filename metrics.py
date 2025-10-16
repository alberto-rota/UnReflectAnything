import math
from typing import Literal, Optional
import torch
import torch.nn.functional as F

# =========================
# Helpers
# =========================
def _to_float_tensor(x: torch.Tensor, device=None) -> torch.Tensor:
    if device is None:
        device = x.device
    return x.to(device=device, dtype=torch.float32)

def _validate_pair(x: torch.Tensor, y: torch.Tensor):
    if x.shape != y.shape:
        raise ValueError(f"Input shapes must match, got {x.shape} and {y.shape}.")
    if x.dim() != 4:
        raise ValueError(f"Inputs must be BCHW tensors, got dim={x.dim()}.")

def _infer_data_range(x: torch.Tensor) -> float:
    # Heuristic: if max > 1.5 assume [0,255]; else [0,1]
    mx = float(x.detach().max())
    return 255.0 if mx > 1.5 else 1.0

def _prepare_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Return a float32 BCHW mask in {0,1}, broadcasted if needed.
    Accepted shapes for mask: BCHW, B1HW, 1CHW, 11HW, BHW, 1HW, HW
    """
    B, C, H, W = x.shape
    if mask is None:
        return torch.ones((B, 1, H, W), dtype=torch.float32, device=x.device)

    m = mask
    if m.dim() == 2:            # HW
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:          # BHW
        m = m.unsqueeze(1)
    elif m.dim() == 4:          # already BCHW-like
        pass
    else:
        raise ValueError("mask must be 2D, 3D, or 4D (HW, BHW, or BCHW-like)")

    # Broadcast to Bx1xHxW (per-pixel same across channels)
    if m.shape[0] == 1 and B > 1: m = m.expand(B, -1, -1, -1)
    if m.shape[1] != 1:
        # If someone passed per-channel mask, collapse to a single channel via any>0
        m = (m > 0).any(dim=1, keepdim=True).to(m.dtype)
    if m.shape[2] != H or m.shape[3] != W:
        raise ValueError(f"mask spatial size must match inputs: got {m.shape[-2:]}, need {(H,W)}")

    return m.to(device=x.device, dtype=torch.float32).clamp(0, 1)

def _masked_reduce_per_image(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    values: [B, C, H, W] or [B, 1, H, W]
    mask:   [B, 1, H, W] in {0,1}
    Returns per-image averages [B] (NaN where mask.sum()==0).
    """
    # If values has multiple channels, average channels before spatial reduction for MSE;
    # for SSIM we already compute scalar map per pixel or average channels later.
    if values.dim() != 4 or mask.dim() != 4:
        raise ValueError("values and mask must be BCHW")

    wsum = mask.flatten(1).sum(dim=1)  # [B]
    # Broadcast mask to values' shape on channel dim
    if values.size(1) != mask.size(1):
        m = mask.expand(-1, values.size(1), -1, -1)
    else:
        m = mask

    num = (values * m).flatten(1).sum(dim=1)  # [B]
    # Avoid divide-by-zero; return NaN where wsum==0
    out = num / torch.where(wsum > 0, wsum, torch.ones_like(wsum))
    out = torch.where(wsum > 0, out, torch.full_like(out, float("nan")))
    return out

# =========================
# 1) MSE (masked)
# =========================
def mse_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    Mean Squared Error, masked.
    - pred, target: BCHW
    - mask: optional binary mask (see _prepare_mask)
    - reduction='mean' -> scalar; 'none' -> per-image [B]
    """
    _validate_pair(pred, target)
    x = _to_float_tensor(pred)
    y = _to_float_tensor(target, device=x.device)
    m = _prepare_mask(x, mask)

    sqerr = (x - y) ** 2  # [B,C,H,W]
    per_image = _masked_reduce_per_image(sqerr.mean(dim=1, keepdim=True), m)  # average channels first
    return per_image.mean() if reduction == "mean" else per_image

# =========================
# 2) PSNR (masked)
# =========================
def psnr_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: Optional[float] = None,
    reduction: Literal["mean", "none"] = "mean",
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio (dB), computed from masked MSE.
    - data_range: 1.0 for [0,1], 255.0 for [0,255]; inferred if None
    """
    _validate_pair(pred, target)
    x = _to_float_tensor(pred)
    y = _to_float_tensor(target, device=x.device)
    if data_range is None:
        data_range = _infer_data_range(torch.stack([x, y], dim=0))

    m = _prepare_mask(x, mask)
    sqerr = (x - y) ** 2
    mse_per = _masked_reduce_per_image(sqerr.mean(dim=1, keepdim=True), m).clamp_min(eps)  # [B]
    psnr_per = 10.0 * torch.log10((data_range ** 2) / mse_per)  # [B], NaN where mask empty

    return psnr_per.mean() if reduction == "mean" else psnr_per

# =========================
# 3) SSIM (masked)
# =========================
def ssim_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: Optional[float] = None,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
    reduction: Literal["mean", "none"] = "mean",
    use_kornia_if_available: bool = True,
) -> torch.Tensor:
    """
    Structural Similarity Index with masking.
    Strategy: compute per-pixel SSIM map, then masked average over spatial dims.
    If Kornia is available, uses it for the SSIM map (GPU-optimized).
    """
    _validate_pair(pred, target)
    x = _to_float_tensor(pred)
    y = _to_float_tensor(target, device=x.device)
    if data_range is None:
        data_range = _infer_data_range(torch.stack([x, y], dim=0))
    m = _prepare_mask(x, mask)  # [B,1,H,W]

    # ---- Try Kornia first ----
    if use_kornia_if_available:
        try:
            import kornia
            if abs(data_range - 1.0) > 1e-6:
                x_s = x / data_range
                y_s = y / data_range
                max_val = 1.0
            else:
                x_s, y_s, max_val = x, y, 1.0

            # Kornia SSIM map: [B, 1, H, W] if reduction="none"
            ssim_map = kornia.metrics.ssim(
                x_s, y_s, window_size=window_size, max_val=max_val, reduction="none"
            )
            # Some Kornia versions return [B, C, H, W]; average channels if so:
            if ssim_map.size(1) > 1:
                ssim_map = ssim_map.mean(dim=1, keepdim=True)

            per_image = _masked_reduce_per_image(ssim_map, m)  # [B]
            return per_image.mean() if reduction == "mean" else per_image
        except Exception:
            pass  # fall back to torch-only

    # ---- Torch-only SSIM (Gaussian window) ----
    def _gaussian_kernel1d(ks: int, sigma: float, device, dtype):
        coords = torch.arange(ks, device=device, dtype=dtype) - ks // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        g = g / g.sum()
        return g

    def _gaussian_filter(img: torch.Tensor, ks: int, sigma: float) -> torch.Tensor:
        c = img.shape[1]
        g1d = _gaussian_kernel1d(ks, sigma, img.device, img.dtype)
        g2d = torch.outer(g1d, g1d)
        kernel = g2d.view(1, 1, ks, ks).repeat(c, 1, 1, 1)  # [C,1,ks,ks]
        return F.conv2d(img, kernel, padding=ks // 2, groups=c)

    # Normalize to [0,1] internally
    if abs(data_range - 1.0) > 1e-6:
        x_ = x / data_range
        y_ = y / data_range
        L = 1.0
    else:
        x_, y_, L = x, y, 1.0

    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    sigma = 1.5 if window_size == 11 else max(0.5, 0.15 * window_size)

    mu_x = _gaussian_filter(x_, window_size, sigma)
    mu_y = _gaussian_filter(y_, window_size, sigma)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = _gaussian_filter(x_ * x_, window_size, sigma) - mu_x2
    sigma_y2 = _gaussian_filter(y_ * y_, window_size, sigma) - mu_y2
    sigma_xy = _gaussian_filter(x_ * y_, window_size, sigma) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = (num / (den + 1e-12)).mean(dim=1, keepdim=True)  # average channels -> [B,1,H,W]

    per_image = _masked_reduce_per_image(ssim_map, m)  # [B]
    return per_image.mean() if reduction == "mean" else per_image
