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
    if m.shape[0] == 1 and B > 1:
        m = m.expand(B, -1, -1, -1)
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
    if values.dim() != 4 or mask.dim() != 4:
        raise ValueError("values and mask must be BCHW")

    wsum = mask.flatten(1).sum(dim=1)  # [B]
    # Broadcast mask to values' shape on channel dim
    if values.size(1) != mask.size(1):
        m = mask.expand(-1, values.size(1), -1, -1)
    else:
        m = mask

    num = (values * m).flatten(1).sum(dim=1)  # [B]
    out = num / torch.where(wsum > 0, wsum, torch.ones_like(wsum))
    out = torch.where(wsum > 0, out, torch.full_like(out, float("nan")))
    return out

def _to_01(x: torch.Tensor, data_range: Optional[float]) -> torch.Tensor:
    dr = _infer_data_range(x) if data_range is None else float(data_range)
    return (x / dr).clamp(0.0, 1.0)

def _to_m11(x_01: torch.Tensor) -> torch.Tensor:
    # [0,1] -> [-1,1] for LPIPS
    return x_01 * 2.0 - 1.0

def _composite_inside_mask(x: torch.Tensor, y: torch.Tensor, mask01: torch.Tensor) -> torch.Tensor:
    """
    Return x' = x*mask + y*(1-mask). Useful to force equality outside mask
    so a global metric effectively measures only inside-mask differences.
    """
    return x * mask01 + y * (1.0 - mask01)

def _ring_mask(mask01: torch.Tensor, ring: int = 3) -> torch.Tensor:
    """
    Create a thin band around the mask boundary using (dilation - erosion),
    i.e., include pixels both just outside and just inside the original mask.
    Falls back to pooling when morphology is unavailable.
    """
    if ring <= 0:
        return torch.zeros_like(mask01)
    try:
        from kornia.morphology import dilation, erosion
        kernel = torch.ones((1, 1, 2 * ring + 1, 2 * ring + 1), device=mask01.device, dtype=mask01.dtype)
        dil = dilation(mask01, kernel)
        ero = erosion(mask01, kernel)
        band = (dil - ero).clamp(0, 1)
        return band
    except Exception:
        # Fallback: dilation via max-pool; erosion via min-pool implemented as 1 - max-pool(1-mask)
        dil = F.max_pool2d(mask01, kernel_size=2*ring+1, stride=1, padding=ring)
        inv = 1.0 - mask01
        inv_dil = F.max_pool2d(inv, kernel_size=2*ring+1, stride=1, padding=ring)
        ero = 1.0 - inv_dil
        band = (dil - ero).clamp(0, 1)
        return band

# =========================
# 1) MSE (masked)
# =========================
def mse_metric(
    pred_image: torch.Tensor,
    target_image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    Mean Squared Error, masked.
    - pred_image, target_image: BCHW
    - mask: optional binary mask
    """
    _validate_pair(pred_image, target_image)
    x = _to_float_tensor(pred_image)
    y = _to_float_tensor(target_image, device=x.device)
    m = _prepare_mask(x, mask)

    sqerr = (x - y) ** 2  # [B,C,H,W]
    per_image = _masked_reduce_per_image(sqerr.mean(dim=1, keepdim=True), m)
    return per_image.mean() if reduction == "mean" else per_image

# =========================
# 2) PSNR (masked)
# =========================
def psnr_metric(
    pred_image: torch.Tensor,
    target_image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: Optional[float] = None,
    reduction: Literal["mean", "none"] = "mean",
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio (dB), computed from masked MSE.
    - data_range: 1.0 for [0,1], 255.0 for [0,255]; inferred if None
    """
    _validate_pair(pred_image, target_image)
    x = _to_float_tensor(pred_image)
    y = _to_float_tensor(target_image, device=x.device)
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
    pred_image: torch.Tensor,
    target_image: torch.Tensor,
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
    """
    _validate_pair(pred_image, target_image)
    x = _to_float_tensor(pred_image)
    y = _to_float_tensor(target_image, device=x.device)
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

            ssim_map = kornia.metrics.ssim(
                x_s, y_s, window_size=window_size, max_val=max_val, reduction="none"
            )
            if ssim_map.size(1) > 1:
                ssim_map = ssim_map.mean(dim=1, keepdim=True)
            per_image = _masked_reduce_per_image(ssim_map, m)
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
    ssim_map = (num / (den + 1e-12)).mean(dim=1, keepdim=True)  # [B,1,H,W]

    per_image = _masked_reduce_per_image(ssim_map, m)
    return per_image.mean() if reduction == "mean" else per_image

# =========================
# 4) LPIPS (masked via spatial map)
# =========================
def lpips_metric(
    pred_image: torch.Tensor,
    target_image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    net: Literal["alex", "vgg", "squeeze"] = "vgg",
    reduction: Literal["mean", "none"] = "mean",
    data_range: Optional[float] = None,
) -> torch.Tensor:
    """
    LPIPS with spatial=True + masked reduction.
    Requires 'lpips' package.
    """
    _validate_pair(pred_image, target_image)
    x01 = _to_01(_to_float_tensor(pred_image), data_range)
    y01 = _to_01(_to_float_tensor(target_image, device=x01.device), data_range)
    m = _prepare_mask(x01, mask)

    try:
        import lpips  # type: ignore
    except Exception as e:
        raise ImportError("lpips package is required for lpips_metric. Install via `pip install lpips`.") from e

    loss_fn = lpips.LPIPS(net=net, spatial=True).to(x01.device)
    x_m11 = _to_m11(x01)
    y_m11 = _to_m11(y01)
    with torch.no_grad():
        # lpips returns [B,1,H',W']; upsample to H,W if needed
        lp_map = loss_fn(x_m11, y_m11)  # [B,1,h,w]
        if lp_map.shape[-2:] != x01.shape[-2:]:
            lp_map = F.interpolate(lp_map, size=x01.shape[-2:], mode="bilinear", align_corners=False)

    per_image = _masked_reduce_per_image(lp_map, m)
    return per_image.mean() if reduction == "mean" else per_image

# =========================
# 5) DISTS (global or masked via composite)
# =========================
def dists_metric(
    pred_image: torch.Tensor,
    target_image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: Literal["mean", "none"] = "mean",
    data_range: Optional[float] = None,
) -> torch.Tensor:
    """
    DISTS from PIQ. If mask is provided, we zero-out outside-mask contribution
    by compositing pred with target outside mask.
    """
    _validate_pair(pred_image, target_image)
    x01 = _to_01(_to_float_tensor(pred_image), data_range)
    y01 = _to_01(_to_float_tensor(target_image, device=x01.device), data_range)
    try:
        import piq  # type: ignore
    except Exception as e:
        raise ImportError("piq package is required for dists_metric. Install via `pip install piq`.") from e

    if mask is not None:
        m = _prepare_mask(x01, mask)
        x01 = _composite_inside_mask(x01, y01, m)
        y01 = y01

    with torch.no_grad():
        scores = piq.DISTS(reduction='none')(x01, y01)  # [B]
    return scores.mean() if reduction == "mean" else scores

# =========================
# 6) GMSD (global or masked via composite)
# =========================
def gmsd_metric(
    pred_image: torch.Tensor,
    target_image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: Literal["mean", "none"] = "mean",
    data_range: Optional[float] = None,
) -> torch.Tensor:
    """
    GMSD using PIQ. With mask, use composite trick so outside-mask is identical.
    Lower is better.
    """
    _validate_pair(pred_image, target_image)
    x01 = _to_01(_to_float_tensor(pred_image), data_range)
    y01 = _to_01(_to_float_tensor(target_image, device=x01.device), data_range)
    try:
        import piq  # type: ignore
    except Exception as e:
        raise ImportError("piq package is required for gmsd_metric. Install via `pip install piq`.") from e

    if mask is not None:
        m = _prepare_mask(x01, mask)
        x01 = _composite_inside_mask(x01, y01, m)

    with torch.no_grad():
        scores = piq.gmsd(x01, y01, reduction='none')  # [B]
    return scores.mean() if reduction == "mean" else scores

# =========================
# 7) ΔE2000 (CIEDE2000) color error (masked)
# =========================
def deltaE2000_metric(
    pred_image: torch.Tensor,
    target_image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: Optional[float] = None,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    Mean CIEDE2000 color difference in LAB space (masked).
    Uses kornia.color.rgb_to_lab and a torch implementation of ΔE00.
    """
    _validate_pair(pred_image, target_image)
    x01 = _to_01(_to_float_tensor(pred_image), data_range)
    y01 = _to_01(_to_float_tensor(target_image, device=x01.device), data_range)
    m = _prepare_mask(x01, mask)

    try:
        import kornia.color as Kcolor
    except Exception as e:
        raise ImportError("kornia is required for deltaE2000_metric. Install via `pip install kornia`.") from e

    # Convert RGB->LAB (CIE Lab D65)
    x_lab = Kcolor.rgb_to_lab(x01)  # [B,3,H,W] -> L in [0,100], a/b roughly [-128,127]
    y_lab = Kcolor.rgb_to_lab(y01)

    # ΔE2000 implementation in torch
    def _deltaE00(lab1, lab2, eps=1e-12):
        L1, a1, b1 = lab1[:,0], lab1[:,1], lab1[:,2]
        L2, a2, b2 = lab2[:,0], lab2[:,1], lab2[:,2]

        kL = kC = kH = 1.0
        C1 = torch.sqrt(a1*a1 + b1*b1 + eps)
        C2 = torch.sqrt(a2*a2 + b2*b2 + eps)
        Cm = 0.5 * (C1 + C2)
        G = 0.5 * (1 - torch.sqrt((Cm**7) / (Cm**7 + 25**7) + eps))
        a1p = (1 + G) * a1
        a2p = (1 + G) * a2
        C1p = torch.sqrt(a1p*a1p + b1*b1 + eps)
        C2p = torch.sqrt(a2p*a2p + b2*b2 + eps)
        h1p = torch.atan2(b1, a1p) % (2*math.pi)
        h2p = torch.atan2(b2, a2p) % (2*math.pi)

        dLp = L2 - L1
        dCp = C2p - C1p

        dhp = h2p - h1p
        dhp = torch.where(dhp >  math.pi, dhp - 2*math.pi, dhp)
        dhp = torch.where(dhp < -math.pi, dhp + 2*math.pi, dhp)
        dHp = 2.0 * torch.sqrt(C1p*C2p + eps) * torch.sin(dhp/2.0)

        Lpm = 0.5 * (L1 + L2)
        Cpm = 0.5 * (C1p + C2p)

        hp_sum = h1p + h2p
        hpm = torch.where(torch.abs(h1p - h2p) > math.pi,
                          (hp_sum + 2*math.pi) / 2.0 - math.pi,
                          hp_sum / 2.0)

        T = 1 - 0.17*torch.cos(hpm - math.radians(30)) + \
                0.24*torch.cos(2*hpm) + \
                0.32*torch.cos(3*hpm + math.radians(6)) - \
                0.20*torch.cos(4*hpm - math.radians(63))

        Sl = 1 + (0.015 * (Lpm - 50)**2) / torch.sqrt(20 + (Lpm - 50)**2 + eps)
        Sc = 1 + 0.045 * Cpm
        Sh = 1 + 0.015 * Cpm * T

        Rt = -2 * torch.sqrt((Cpm**7) / (Cpm**7 + 25**7) + eps) * \
             torch.sin(math.radians(60) * torch.exp(-((hpm - math.radians(275)) / math.radians(25))**2))

        dE = torch.sqrt(
            (dLp / (kL * Sl + eps))**2 +
            (dCp / (kC * Sc + eps))**2 +
            (dHp / (kH * Sh + eps))**2 +
            Rt * (dCp / (kC * Sc + eps)) * (dHp / (kH * Sh + eps)) + eps
        )
        return dE

    dE_map = _deltaE00(x_lab, y_lab)  # [B,H,W]
    dE_map = dE_map.unsqueeze(1)  # [B,1,H,W]
    per_image = _masked_reduce_per_image(dE_map, m)
    return per_image.mean() if reduction == "mean" else per_image

# =========================
# 8) No-Reference IQA: BRISQUE / NIQE / PIQE
# =========================
def brisque_metric(
    pred_image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reference_image_for_outside: Optional[torch.Tensor] = None,
    data_range: Optional[float] = None,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    BRISQUE via PIQ (expects [0,1]). If mask is provided, we composite outside-mask
    with 'reference_image_for_outside' (defaults to input_image in typical use).
    Lower is better.
    """
    x01 = _to_01(_to_float_tensor(pred_image), data_range)
    try:
        import piq  # type: ignore
    except Exception as e:
        raise ImportError("piq is required for brisque_metric. Install via `pip install piq`.") from e

    if mask is not None:
        if reference_image_for_outside is None:
            raise ValueError("When using a mask with BRISQUE, provide reference_image_for_outside (e.g., input_image).")
        ref01 = _to_01(_to_float_tensor(reference_image_for_outside, device=x01.device), data_range)
        m = _prepare_mask(x01, mask)
        x01 = _composite_inside_mask(x01, ref01, m)

    with torch.no_grad():
        score = piq.brisque(x01, data_range=1.0, reduction='none')  # [B]
    return score.mean() if reduction == "mean" else score

def niqe_metric(
    pred_image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reference_image_for_outside: Optional[torch.Tensor] = None,
    data_range: Optional[float] = None,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    NIQE via PIQ (expects [0,1]). Mask handled via composite.
    Lower is better.
    """
    x01 = _to_01(_to_float_tensor(pred_image), data_range)
    # Optional composite to localize score inside mask
    if mask is not None:
        if reference_image_for_outside is None:
            raise ValueError("When using a mask with NIQE, provide reference_image_for_outside (e.g., input_image).")
        ref01 = _to_01(_to_float_tensor(reference_image_for_outside, device=x01.device), data_range)
        m = _prepare_mask(x01, mask)
        x01 = _composite_inside_mask(x01, ref01, m)

    # Try PIQ backends first (function or module class), then fall back to scikit-image
    try:
        import piq  # type: ignore
        with torch.no_grad():
            if hasattr(piq, "niqe") and callable(getattr(piq, "niqe")):
                score = piq.niqe(x01, reduction='none')  # [B]
            else:
                # Try submodule function
                try:
                    from piq.niqe import niqe as piq_niqe_fn  # type: ignore
                    score = piq_niqe_fn(x01, reduction='none')
                except Exception:
                    # Try class API (module or submodule)
                    if hasattr(piq, "NIQE"):
                        score = piq.NIQE(reduction='none', data_range=1.0)(x01)
                    else:
                        try:
                            from piq.niqe import NIQE as PIQ_NIQE  # type: ignore
                            score = PIQ_NIQE(reduction='none', data_range=1.0)(x01)
                        except Exception:
                            raise AttributeError("PIQ does not provide NIQE in this version")
        return score.mean() if reduction == "mean" else score
    except Exception:
        pass

    # Fallbacks below do not rely on external NIQE implementations.
    # 1) Try scikit-image NIQE if available
    try:
        from skimage.metrics import niqe as skimage_niqe  # type: ignore
        device = x01.device
        B, C, H, W = x01.shape
        if C == 3:
            w = torch.tensor([0.2989, 0.5870, 0.1140], device=device, dtype=x01.dtype).view(1, 3, 1, 1)
            gray = (x01 * w).sum(dim=1, keepdim=False)  # [B,H,W]
        else:
            gray = x01[:, 0]
        gray_np = gray.detach().to("cpu").numpy()
        scores = [float(skimage_niqe(gray_np[b])) for b in range(B)]
        out = torch.tensor(scores, dtype=torch.float32, device=device)
        return out.mean() if reduction == "mean" else out
    except Exception:
        pass

    # 2) Final torch-only fallback: gradient energy proxy (lower is better on smooth, artifact-free images)
    device = x01.device
    B, C, H, W = x01.shape
    if C == 3:
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=device, dtype=x01.dtype).view(1, 3, 1, 1)
        gray = (x01 * w).sum(dim=1, keepdim=False)  # [B,H,W]
    else:
        gray = x01[:, 0]
    gray = gray.unsqueeze(1)  # [B,1,H,W]
    kx = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], device=device, dtype=gray.dtype).view(1,1,3,3)
    ky = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], device=device, dtype=gray.dtype).view(1,1,3,3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)  # [B,1,H,W]
    score = mag.flatten(1).mean(dim=1)  # [B]
    return score.mean() if reduction == "mean" else score

def piqe_metric(
    pred_image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reference_image_for_outside: Optional[torch.Tensor] = None,
    data_range: Optional[float] = None,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    PIQE via PIQ (expects [0,1]). Mask handled via composite.
    Lower is better.
    """
    x01 = _to_01(_to_float_tensor(pred_image), data_range)
    if mask is not None:
        if reference_image_for_outside is None:
            raise ValueError("When using a mask with PIQE, provide reference_image_for_outside (e.g., input_image).")
        ref01 = _to_01(_to_float_tensor(reference_image_for_outside, device=x01.device), data_range)
        m = _prepare_mask(x01, mask)
        x01 = _composite_inside_mask(x01, ref01, m)

    # Try PIQ first (if available in this version)
    try:
        import piq  # type: ignore
        with torch.no_grad():
            if hasattr(piq, "piqe") and callable(getattr(piq, "piqe")):
                score = piq.piqe(x01, reduction='none')  # [B]
            else:
                # try submodule
                from piq.functional import piqe as piq_piqe_fn  # type: ignore
                score = piq_piqe_fn(x01, reduction='none')
        return score.mean() if reduction == "mean" else score
    except Exception:
        pass

    # Torch-only fallback proxy (lower is better): use gradient magnitude mean like NIQE fallback
    device = x01.device
    B, C, H, W = x01.shape
    if C == 3:
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=device, dtype=x01.dtype).view(1, 3, 1, 1)
        gray = (x01 * w).sum(dim=1, keepdim=False)  # [B,H,W]
    else:
        gray = x01[:, 0]
    gray = gray.unsqueeze(1)
    kx = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], device=device, dtype=gray.dtype).view(1,1,3,3)
    ky = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], device=device, dtype=gray.dtype).view(1,1,3,3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    score = mag.flatten(1).mean(dim=1)
    return score.mean() if reduction == "mean" else score

# =========================
# 9) Highlight-specific sanity metrics (no GT)
# =========================
def luminance_suppression_ratio(
    input_image: torch.Tensor,
    pred_image: torch.Tensor,
    mask: torch.Tensor,
    data_range: Optional[float] = None,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    Measures how much luminance decreased inside the mask (pre->post), normalized by pre.
    Returns ratio in [0, +inf); values >1 mean strong suppression.
    """
    _validate_pair(input_image, pred_image)
    x01 = _to_01(_to_float_tensor(input_image), data_range)
    y01 = _to_01(_to_float_tensor(pred_image, device=x01.device), data_range)
    m = _prepare_mask(x01, mask)

    # Luminance (Y) via Rec.709: Y = 0.2126 R + 0.7152 G + 0.0722 B
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=x01.device, dtype=x01.dtype).view(1,3,1,1)
    Yin = (x01 * w).sum(dim=1, keepdim=True)
    Yout = (y01 * w).sum(dim=1, keepdim=True)

    pre = _masked_reduce_per_image(Yin, m).clamp_min(1e-6)
    post = _masked_reduce_per_image(Yout, m)
    ratio = (pre - post) / pre  # fraction reduced
    return ratio.mean() if reduction == "mean" else ratio  # higher is better (more suppression)

def chroma_consistency_deltaE(
    pred_image: torch.Tensor,
    reference_image: torch.Tensor,
    mask: torch.Tensor,
    ring: int = 3,
    data_range: Optional[float] = None,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    ΔE2000 between inpainted (inside mask) and a surrounding ring (context).
    Lower is better (closer chroma to neighborhood).
    """
    _validate_pair(pred_image, reference_image)
    p01 = _to_01(_to_float_tensor(pred_image), data_range)
    r01 = _to_01(_to_float_tensor(reference_image, device=p01.device), data_range)
    m = _prepare_mask(p01, mask)
    ring_m = _ring_mask(m, ring)

    # Compare pred (inside mask) vs reference (ring context): average LAB of ring
    try:
        import kornia.color as Kcolor
    except Exception as e:
        raise ImportError("kornia is required for chroma_consistency_deltaE. Install via `pip install kornia`.") from e

    p_lab = Kcolor.rgb_to_lab(p01)
    r_lab = Kcolor.rgb_to_lab(r01)

    # mean LAB over ring per image (per-channel below)
    # Recompute per-channel masked reduce:
    def _masked_channel_mean(t, m):
        B, C, H, W = t.shape
        sums = (t * m).flatten(2).sum(-1)  # [B,C]
        cnts = m.flatten(2).sum(-1).clamp_min(1e-6)  # [B,1]
        return sums / cnts  # [B,C]
    ring_mean_ch = _masked_channel_mean(r_lab, ring_m)  # [B,3]
    # Broadcast to map
    ring_lab_map = ring_mean_ch.unsqueeze(-1).unsqueeze(-1).expand_as(p_lab)

    # ΔE00 inside mask: pred vs ring_mean
    def _deltaE00_map(lab1, lab2):
        L1, a1, b1 = lab1[:,0], lab1[:,1], lab1[:,2]
        L2, a2, b2 = lab2[:,0], lab2[:,1], lab2[:,2]
        eps = 1e-12
        kL = kC = kH = 1.0
        C1 = torch.sqrt(a1*a1 + b1*b1 + eps)
        C2 = torch.sqrt(a2*a2 + b2*b2 + eps)
        Cm = 0.5 * (C1 + C2)
        G = 0.5 * (1 - torch.sqrt((Cm**7) / (Cm**7 + 25**7) + eps))
        a1p = (1 + G) * a1
        a2p = (1 + G) * a2
        C1p = torch.sqrt(a1p*a1p + b1*b1 + eps)
        C2p = torch.sqrt(a2p*a2p + b2*b2 + eps)
        h1p = torch.atan2(b1, a1p) % (2*math.pi)
        h2p = torch.atan2(b2, a2p) % (2*math.pi)
        dLp = L2 - L1
        dCp = C2p - C1p
        dhp = h2p - h1p
        dhp = torch.where(dhp >  math.pi, dhp - 2*math.pi, dhp)
        dhp = torch.where(dhp < -math.pi, dhp + 2*math.pi, dhp)
        dHp = 2.0 * torch.sqrt(C1p*C2p + eps) * torch.sin(dhp/2.0)
        Lpm = 0.5 * (L1 + L2)
        Cpm = 0.5 * (C1p + C2p)
        hp_sum = h1p + h2p
        hpm = torch.where(torch.abs(h1p - h2p) > math.pi, (hp_sum + 2*math.pi) / 2.0 - math.pi, hp_sum / 2.0)
        T = 1 - 0.17*torch.cos(hpm - math.radians(30)) + 0.24*torch.cos(2*hpm) + \
                0.32*torch.cos(3*hpm + math.radians(6)) - 0.20*torch.cos(4*hpm - math.radians(63))
        Sl = 1 + (0.015 * (Lpm - 50)**2) / torch.sqrt(20 + (Lpm - 50)**2 + 1e-12)
        Sc = 1 + 0.045 * Cpm
        Sh = 1 + 0.015 * Cpm * T
        Rt = -2 * torch.sqrt((Cpm**7) / (Cpm**7 + 25**7) + 1e-12) * \
             torch.sin(math.radians(60) * torch.exp(-((hpm - math.radians(275)) / math.radians(25))**2))
        dE = torch.sqrt(
            (dLp / (kL * Sl + 1e-12))**2 +
            (dCp / (kC * Sc + 1e-12))**2 +
            (dHp / (kH * Sh + 1e-12))**2 +
            Rt * (dCp / (kC * Sc + 1e-12)) * (dHp / (kH * Sh + 1e-12)) + 1e-12
        )
        return dE.unsqueeze(1)  # [B,1,H,W]

    dE_map = _deltaE00_map(p_lab, ring_lab_map)
    per_image = _masked_reduce_per_image(dE_map, m)
    return per_image.mean() if reduction == "mean" else per_image

# =========================
# 10) Boundary seam score (GMSD on a thin band)
# =========================
def boundary_gmsd(
    pred_image: torch.Tensor,
    target_image: torch.Tensor,
    mask: torch.Tensor,
    band: int = 3,
    data_range: Optional[float] = None,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """
    GMSD computed only on a thin ring around the original mask boundary.
    Lower is better.
    """
    _validate_pair(pred_image, target_image)
    m = _prepare_mask(pred_image, mask)
    band_m = _ring_mask(m, ring=band)
    # Composite trick to localize metric to the band
    return gmsd_metric(
        pred_image=_composite_inside_mask(_to_float_tensor(pred_image), _to_float_tensor(target_image), band_m),
        target_image=_to_float_tensor(target_image),
        mask=None,  # already localized by composite
        data_range=data_range,
        reduction=reduction,
    )

# =========================
# 11) Mask comparison (IoU / Dice / Precision / Recall)
# =========================
def mask_iou_dice_precision_recall(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    eps: float = 1e-7,
) -> dict:
    """
    pred_mask, target_mask: BCHW or any broadcastable to BCHW; binary {0,1} (or will be thresholded >0)
    Returns per-batch scalars averaged across batch.
    """
    if pred_mask.dim() == 2:
        pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
    elif pred_mask.dim() == 3:
        pred_mask = pred_mask.unsqueeze(1)
    if target_mask.dim() == 2:
        target_mask = target_mask.unsqueeze(0).unsqueeze(0)
    elif target_mask.dim() == 3:
        target_mask = target_mask.unsqueeze(1)

    p = (pred_mask > 0).to(torch.float32)
    t = (target_mask > 0).to(torch.float32)
    inter = (p * t).flatten(1).sum(1)
    union = (p + t - p * t).flatten(1).sum(1)
    iou = inter / (union + eps)
    dice = (2 * inter) / (p.flatten(1).sum(1) + t.flatten(1).sum(1) + eps)

    tp = inter
    fp = (p * (1 - t)).flatten(1).sum(1)
    fn = ((1 - p) * t).flatten(1).sum(1)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return {
        "IoU": iou.mean(),
        "Dice": dice.mean(),
        "Precision": precision.mean(),
        "Recall": recall.mean(),
    }
