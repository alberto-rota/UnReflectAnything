import math
from typing import Literal, Optional
import torch
import torch.nn.functional as F

# ---------------------------
# Helpers
# ---------------------------
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

# ---------------------------
# 1) MSE
# ---------------------------
def mse_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    Mean Squared Error over BxCxHxW.
    - pred, target: BCHW, any dtype/device; converted to float32
    - reduction='mean' returns scalar; 'none' returns per-image tensor [B]
    """
    _validate_pair(pred, target)
    x = _to_float_tensor(pred)
    y = _to_float_tensor(target, device=x.device)
    err = (x - y) ** 2  # [B,C,H,W]
    if reduction == "mean":
        return err.mean()
    elif reduction == "none":
        return err.flatten(1).mean(dim=1)  # [B]
    else:
        raise ValueError("reduction must be 'mean' or 'none'")

# ---------------------------
# 2) PSNR
# ---------------------------
def psnr_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None,
    reduction: Literal["mean", "none"] = "mean",
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio (dB).
    - pred, target: BCHW
    - data_range: e.g., 1.0 for [0,1], 255.0 for [0,255]. If None, inferred.
    - reduction='mean' -> scalar dB, 'none' -> per-image dB [B]
    """
    _validate_pair(pred, target)
    x = _to_float_tensor(pred)
    y = _to_float_tensor(target, device=x.device)
    if data_range is None:
        data_range = _infer_data_range(torch.stack([x, y], dim=0))

    # Per-image MSE
    mse_per = ((x - y) ** 2).flatten(1).mean(dim=1).clamp_min(eps)  # [B]
    psnr_per = 10.0 * torch.log10((data_range ** 2) / mse_per)
    if reduction == "mean":
        return psnr_per.mean()
    elif reduction == "none":
        return psnr_per
    else:
        raise ValueError("reduction must be 'mean' or 'none'")

# ---------------------------
# 3) SSIM
# ---------------------------
def ssim_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
    reduction: Literal["mean", "none"] = "mean",
    use_kornia_if_available: bool = True,
) -> torch.Tensor:
    """
    Structural Similarity Index.
    - pred, target: BCHW, float or uint8; any device (CUDA supported)
    - data_range: 1.0 or 255.0 typical; inferred if None
    - If Kornia is available and use_kornia_if_available=True, uses kornia.metrics.ssim
      (GPU-optimized). Otherwise uses a torch-only implementation.
    - reduction='mean' -> scalar, 'none' -> per-image [B]
    """
    _validate_pair(pred, target)
    x = _to_float_tensor(pred)
    y = _to_float_tensor(target, device=x.device)
    if data_range is None:
        data_range = _infer_data_range(torch.stack([x, y], dim=0))

    if use_kornia_if_available:
        try:
            import kornia
            # Kornia expects inputs in [0,1]; rescale if needed
            if abs(data_range - 1.0) > 1e-6:
                x_s = x / data_range
                y_s = y / data_range
                dr = 1.0
            else:
                x_s, y_s, dr = x, y, 1.0

            # Kornia returns BCHW SSIM map; reduce to per-image
            ssim_map = kornia.metrics.ssim(x_s, y_s, window_size=window_size, max_val=dr, reduction="none")
            per_image = ssim_map.flatten(1).mean(dim=1)  # [B]
            return per_image.mean() if reduction == "mean" else per_image
        except Exception:
            pass  # fall back to torch-only version

    # ---- Torch-only SSIM (Gaussian window) ----
    # Build Gaussian kernel separably
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

    # constants
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    # Gaussian smoothing
    # Common choice: sigma ~ 1.5 for window_size=11
    sigma = 1.5 if window_size == 11 else max(0.5, 0.15 * window_size)
    mu_x = _gaussian_filter(x_, window_size, sigma)
    mu_y = _gaussian_filter(y_, window_size, sigma)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _gaussian_filter(x_ * x_, window_size, sigma) - mu_x2
    sigma_y2 = _gaussian_filter(y_ * y_, window_size, sigma) - mu_y2
    sigma_xy = _gaussian_filter(x_ * y_, window_size, sigma) - mu_xy

    # SSIM map
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-12)  # [B,C,H,W]

    # Reduce over channels and spatial -> per-image
    per_image = ssim_map.mean(dim=(1, 2, 3))  # [B]
    return per_image.mean() if reduction == "mean" else per_image
