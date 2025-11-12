"""
Loss utilities for UnReflectAnything model.

This module contains:
- Basic loss modules (SSIM, L1, Charbonnier, Dice, Gradient, TV)
- Composite loss modules (HighlightRegressionLoss, SeamLoss)
- Helper functions for color processing, mask operations, and image reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # type: ignore
from scipy.ndimage import label  # type: ignore


# ============================================================================
# BASIC PIXEL-LEVEL LOSSES
# ============================================================================

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss with optional masking.
    
    Computes SSIM between two images using a Gaussian-weighted window.
    Supports masked evaluation where only masked regions contribute to the loss.
    
    Args:
        window_size: Size of the Gaussian window (default: 11)
        sigma: Standard deviation of the Gaussian kernel (default: 1.5)
    
    Forward:
        x: (B, C, H, W) predicted image
        y: (B, C, H, W) target image
        mask: (B, 1, H, W) or (B, C, H, W) optional mask (1 = valid region)
    
    Returns:
        Scalar SSIM value (higher is better, typically used as 1 - SSIM for loss)
    """
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", self._create_window())

    def _create_window(self):
        """Create Gaussian-weighted window for SSIM computation."""
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
    """
    L1 (Mean Absolute Error) loss with optional masking.
    
    Computes mean absolute error between predictions and targets,
    optionally masked to only evaluate on specific regions.
    
    Forward:
        x: (B, C, H, W) predicted image
        y: (B, C, H, W) target image
        mask: (B, 1, H, W) or (B, C, H, W) optional mask (1 = valid region)
    
    Returns:
        Scalar L1 loss value
    """
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
    """
    Charbonnier (smooth L1) loss: sqrt((x - y)^2 + eps^2).
    
    A differentiable approximation of L1 loss that is more robust to outliers
    than L2 loss while remaining smooth everywhere.
    
    Args:
        eps: Small constant for numerical stability (default: 1e-6)
    
    Forward:
        pred: (B, C, H, W) predicted image
        target: (B, C, H, W) target image
    
    Returns:
        Scalar Charbonnier loss value
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps**2).mean()


# ============================================================================
# SEGMENTATION LOSSES
# ============================================================================

class SquaredSoftDiceLoss(nn.Module):
    """
    Squared soft Dice loss for segmentation tasks.
    
    Computes 1 - Dice coefficient, where Dice = (2 * intersection + smooth) / (sum(pred^2) + sum(target^2) + smooth).
    Useful for binary or soft segmentation masks.
    
    Args:
        smooth: Smoothing constant to avoid division by zero (default: 1e-6)
    
    Forward:
        pred: (B, C, H, W) predicted segmentation mask
        target: (B, C, H, W) target segmentation mask
    
    Returns:
        Scalar Dice loss value (0 = perfect match, 1 = no overlap)
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


# ============================================================================
# REGULARIZATION LOSSES
# ============================================================================

class GradientLoss(nn.Module):
    """
    Gradient matching loss.
    
    Penalizes differences in image gradients between prediction and target.
    Encourages smooth transitions and edge preservation.
    
    Forward:
        pred: (B, C, H, W) predicted image
        target: (B, C, H, W) target image
    
    Returns:
        Scalar gradient loss value (sum of horizontal and vertical gradient differences)
    """
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
    """
    Total Variation (TV) loss for smoothness regularization.
    
    Penalizes large differences between adjacent pixels, encouraging
    piecewise-smooth images. Useful for denoising and inpainting.
    
    Forward:
        x: (B, C, H, W) image tensor
    
    Returns:
        Scalar TV loss value (sum of horizontal and vertical variations)
    """
    def forward(self, x):
        tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w


# ============================================================================
# COMPOSITE LOSSES
# ============================================================================

class HighlightRegressionLoss(nn.Module):
    """
    Composite loss for highlight mask regression.
    
    Combines multiple loss terms (L1/Charbonnier, Dice, SSIM, Gradient, TV)
    with optional class balancing and focal weighting for highlight detection.
    
    Args:
        w_l1: Weight for L1/Charbonnier loss (default: 1.0)
        use_charbonnier: If True, use Charbonnier loss; else use L1 (default: True)
        w_dice: Weight for Dice loss (default: 0.0)
        w_ssim: Weight for SSIM loss (default: 0.0)
        w_grad: Weight for gradient loss (default: 0.0)
        w_tv: Weight for TV loss (default: 0.0)
        dice_smooth: Smoothing constant for Dice loss (default: 1e-6)
        charbonnier_eps: Epsilon for Charbonnier loss (default: 1e-6)
        clamp_to_unit: Clamp inputs to [0, 1] before loss computation (default: True)
        balance_mode: Class balancing mode: "none", "auto", or "pos_weight" (default: "none")
        pos_weight: Weight for positive class when balance_mode="pos_weight" (default: 1.0)
        focal_gamma: Focal loss gamma parameter (0 = disabled) (default: 0.0)
    
    Forward:
        pred: (B, 1, H, W) predicted highlight mask
        target: (B, 1, H, W) target highlight mask
    
    Returns:
        Scalar combined loss value
    """
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
        
        # Main L1/Charbonnier term with optional balancing and focal weighting
        if self.w_l1 > 0:
            # Focal weighting: focus on hard examples
            if self.focal_gamma > 0.0:
                resid = (pred - target).abs()
                focal_w = torch.pow(resid.clamp_min(1e-6), self.focal_gamma)
            else:
                focal_w = 1.0

            # Class balancing: handle imbalanced positive/negative pixels
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
        
        # Additional loss terms
        if self.w_dice > 0:
            loss = loss + self.w_dice * self.l_dice(pred, target)
        if self.w_ssim > 0 and self.ssim is not None:
            loss = loss + self.w_ssim * (1.0 - self.ssim(pred, target))
        if self.w_grad > 0:
            loss = loss + self.w_grad * self.l_grad(pred, target)
        if self.w_tv > 0:
            loss = loss + self.w_tv * self.l_tv(pred)
        
        return loss


class SeamLoss(nn.Module):
    """
    Seam loss for boundary consistency in inpainting.
    
    Penalizes discontinuities ONLY along the boundary ring of a hole mask.
    The boundary ring is computed as dilate(mask) - mask, focusing the loss
    on the transition region between inpainted and original content.
    
    Args:
        ring_kernel: Odd kernel size for morphological dilation that defines the ring thickness (default: 7)
        use_charbonnier: If True, uses Charbonnier loss; else uses L1 (default: False)
        eps: Epsilon for numerical stability in Charbonnier loss and denominators (default: 1e-6)
        weight_l1: Weight of the pixel-wise difference term on the seam (default: 1.0)
        weight_grad: Weight of the gradient-consistency term on the seam (0 disables) (default: 0.0)
        reduction: 'mean' or 'sum' over the seam pixels (default: "mean")
    
    Forward:
        pred: (B, C, H, W) predicted RGB (or any channels)
        target: (B, C, H, W) ground-truth RGB
        mask: (B, 1, h, w) or (B, 1, H, W); 1=inpaint region (hole), 0=background.
              If spatial size != pred, it will be bilinearly upsampled to (H,W).
    
    Returns:
        Scalar seam loss value
    
    Notes:
        - The "seam/boundary ring" is computed as dilate(mask) - mask, using max-pooling
          with `ring_kernel`. This focuses the penalty only where discontinuities appear.
        - Set `weight_grad>0` to also match image gradients across the seam (reduces
          patch-grid edges further).
    """
    def __init__(
        self,
        ring_kernel: int = 7,
        use_charbonnier: bool = False,
        eps: float = 1e-6,
        weight_l1: float = 1.0,
        weight_grad: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        assert ring_kernel % 2 == 1, "ring_kernel must be odd."
        assert reduction in ("mean", "sum")
        self.k = ring_kernel
        self.use_charb = use_charbonnier
        self.eps = eps
        self.w_l1 = weight_l1
        self.w_g = weight_grad
        self.reduction = reduction

    @staticmethod
    def _ensure_size(mask, H, W):
        """Ensure mask has spatial size (H, W), upsampling if necessary."""
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
        return mask.clamp_(0.0, 1.0)

    @staticmethod
    def _make_ring(mask, k):
        """
        Compute boundary ring mask via morphological dilation.
        
        Args:
            mask: (B, 1, H, W), 1=inpaint region
            k: Odd kernel size for dilation
        
        Returns:
            ring: (B, 1, H, W), 1 only on a thin boundary around the hole
        """
        pad = k // 2
        dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
        ring = (dil - mask).clamp_(0.0, 1.0)  # 1 only on a thin boundary around the hole
        return ring

    def _pixel_loss(self, diff, ring):
        """
        Compute pixel-wise loss on the seam ring.
        
        Args:
            diff: (B, C, H, W) difference between pred and target
            ring: (B, 1, H, W) boundary ring mask
        
        Returns:
            Scalar pixel loss value
        """
        if self.use_charb:
            loss_map = torch.sqrt(diff * diff + self.eps * self.eps)
        else:
            loss_map = diff.abs()
        loss_map = loss_map * ring  # broadcast over channels
        if self.reduction == "mean":
            denom = (ring.sum() * diff.shape[1]).clamp_min(self.eps)
            return loss_map.sum() / denom
        else:
            return loss_map.sum()

    def _grad_loss(self, pred, target, ring):
        """
        Compute gradient consistency loss on the seam.
        
        Matches gradients across the seam to ensure smooth transitions.
        We compute forward differences; the seam mask is cropped accordingly.
        
        Args:
            pred: (B, C, H, W) predicted image
            target: (B, C, H, W) target image
            ring: (B, 1, H, W) boundary ring mask
        
        Returns:
            Scalar gradient loss value
        """
        # Horizontal gradients (B, C, H, W-1)
        dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
        ring_dx = ring[:, :, :, 1:] * ring[:, :, :, :-1]  # keep seam pixels near boundary
        
        # Vertical gradients (B, C, H-1, W)
        dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
        ring_dy = ring[:, :, 1:, :] * ring[:, :, :-1, :]

        if self.use_charb:
            dx_loss = torch.sqrt((dx_p - dx_t) ** 2 + self.eps * self.eps)
            dy_loss = torch.sqrt((dy_p - dy_t) ** 2 + self.eps * self.eps)
        else:
            dx_loss = (dx_p - dx_t).abs()
            dy_loss = (dy_p - dy_t).abs()

        dx_loss = dx_loss * ring_dx
        dy_loss = dy_loss * ring_dy

        if self.reduction == "mean":
            denom = ((ring_dx.sum() + ring_dy.sum()) * pred.shape[1]).clamp_min(self.eps)
            return (dx_loss.sum() + dy_loss.sum()) / denom
        else:
            return dx_loss.sum() + dy_loss.sum()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape, "pred and target must have same shape (B,C,H,W)"
        B, C, H, W = pred.shape
        
        # Handle various mask input formats
        if mask.dim() == 2:
            mask = mask.view(1, 1, *mask.shape)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        assert mask.shape[0] in (1, B), "mask batch size must be 1 or match pred batch size"
        if mask.shape[0] == 1 and B > 1:
            mask = mask.expand(B, -1, -1, -1)

        mask = self._ensure_size(mask.float(), H, W)  # (B, 1, H, W), float in [0, 1]
        ring = self._make_ring(mask, self.k)  # (B, 1, H, W)

        # If ring is empty (e.g., no holes), return zero
        if (ring.sum() <= 0).item():
            return pred.new_zeros(())

        diff = pred - target
        loss = self.w_l1 * self._pixel_loss(diff, ring)

        if self.w_g > 0.0:
            loss = loss + self.w_g * self._grad_loss(pred, target, ring)

        return loss


class TokenInpaintLoss(nn.Module):
    """
    Token-space inpainting loss for feature-space distillation.
    
    Computes a combination of L1 and cosine distance between predicted and
    teacher token features, only on patches that are masked and supervised.
    This loss encourages the inpainting model to produce features that match
    the ground truth diffuse features in the masked regions.
    
    Args:
        token_feat_alpha: Mixing weight for L1 vs cosine distance:
                         alpha * L1 + (1-alpha) * (1-cosine) (default: 0.5)
    
    Forward:
        tokens_completed: List[L] of (B, N, C) tensors - predicted tokens from inpainting model
        tokens_teacher: List[L] of (B, N, C) tensors - target tokens from ground truth diffuse
        patch_supervision_mask: (B, N) boolean mask - where to supervise
                        (True = NOT supervised, False = supervised)
    
    Returns:
        Scalar loss value (averaged over layers)
    
    Notes:
        - The mask is inverted internally: patch_supervision_mask=True means NOT supervised
        - Features are normalized before computing distances for stable geometry
        - Loss is averaged over all layers
    """
    def __init__(self, token_feat_alpha: float = 0.5):
        super().__init__()
        self.token_feat_alpha = token_feat_alpha
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(
        self,
        tokens_completed: list[torch.Tensor],
        tokens_teacher: list[torch.Tensor],
        patch_supervision_mask: torch.Tensor,  # (B, N) boolean — supervised masked patches (synthetic ∧ ¬dataset)
    ) -> torch.Tensor:

        
        if not isinstance(tokens_completed, (list, tuple)) or not isinstance(
            tokens_teacher, (list, tuple)
        ):
            raise ValueError(
                "tokens_completed and tokens_teacher must be lists of tensors."
            )

        if len(tokens_completed) != len(tokens_teacher):
            raise ValueError(
                "tokens_completed and tokens_teacher must have same length (layers)."
            )

        l_total = 0.0

        # No supervision, return zero loss
        do_we_even_supervise = patch_supervision_mask.any()
        if not do_we_even_supervise:
            return torch.zeros((), device=tokens_teacher[0].device)

        # For each DINO hidden state
        for Tc, Tt in zip(tokens_completed, tokens_teacher):
            # Tc, Tt: (B, N, C)
            # Index only masked-supervised positions
            
            # Index only masked-supervised positions
            idx = patch_supervision_mask.unsqueeze(-1).expand_as(Tc)  # (B, N, C) bool
            Tc_m = Tc[idx].view(-1, Tc.shape[-1])  # (M, C)
            Tt_m = Tt[idx].view(-1, Tt.shape[-1])  # (M, C)
            if Tc_m.numel() == 0:
                continue
            
            # # Normalize features for stable geometry in feature space
            # Tc_m = F.normalize(Tc_m, dim=-1)
            # Tt_m = F.normalize(Tt_m, dim=-1)
            
            # Combined L1 and cosine distance
            l1 = (Tc_m - Tt_m).abs().mean()
            cosd = 1.0 - self.cos_sim(Tc_m, Tt_m).mean()
            l_total = l_total + (
                self.token_feat_alpha * l1 + (1.0 - self.token_feat_alpha) * cosd
            )

        return l_total / max(1, len(tokens_completed))


class DiffuseHighlightPenaltyLoss(nn.Module):
    """
    Loss that explicitly penalizes highlights in the diffuse decoder output.
    
    Detects bright pixels in the diffuse component using brightness/luminance thresholding
    and penalizes them to encourage the diffuse decoder to produce highlight-free
    outputs. Highlights should be handled by the dedicated highlight component.
    
    Args:
        brightness_threshold: Brightness/luminance threshold for highlight detection (default: 0.7)
                             Pixels with brightness/luminance above this are penalized
        use_charbonnier: If True, use Charbonnier loss; else use L1 (default: True)
        charbonnier_eps: Epsilon for Charbonnier loss (default: 1e-6)
        penalty_mode: How to penalize highlights:
                     - "brightness": Penalize brightness/luminance above threshold (default)
                     - "pixel": Penalize RGB pixel values directly
        target_brightness: Target brightness/luminance for penalized pixels (default: threshold)
                         Pixels are pushed toward this value
        use_luminance: If True, use perceptually-weighted luminance (0.299*R + 0.587*G + 0.114*B);
                      if False, use simple mean brightness (default: False)
    
    Forward:
        diffuse_rgb: (B, 3, H, W) predicted diffuse RGB image
        mask: (B, 1, H, W) optional mask (1 = region to evaluate, 0 = ignore)
    
    Returns:
        Scalar loss value
    """
    def __init__(
        self,
        brightness_threshold: float = 0.7,
        use_charbonnier: bool = True,
        charbonnier_eps: float = 1e-6,
        penalty_mode: str = "brightness",
        target_brightness: float = None,
        use_luminance: bool = False,
    ):
        super().__init__()
        self.brightness_threshold = brightness_threshold
        self.use_charbonnier = use_charbonnier
        self.charbonnier_eps = charbonnier_eps
        self.penalty_mode = penalty_mode
        self.use_luminance = use_luminance
        if target_brightness is None:
            target_brightness = brightness_threshold
        self.target_brightness = target_brightness
        
        if penalty_mode not in ("brightness", "pixel"):
            raise ValueError(f"penalty_mode must be 'brightness' or 'pixel', got {penalty_mode}")
        
        # Register luminance weights as buffer for GPU efficiency
        if use_luminance:
            luminance_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer("luminance_weights", luminance_weights)

    def forward(
        self,
        diffuse_rgb: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute penalty loss for highlights in diffuse output.
        
        Args:
            diffuse_rgb: (B, 3, H, W) predicted diffuse RGB image
            mask: (B, 1, H, W) optional mask (1 = region to evaluate)
        
        Returns:
            Scalar loss value
        """
        B, C, H, W = diffuse_rgb.shape
        if C != 3:
            raise ValueError(f"diffuse_rgb must have 3 channels, got {C}")
        
        # Compute brightness/luminance
        if self.use_luminance:
            # Perceptually-weighted luminance: 0.299*R + 0.587*G + 0.114*B
            brightness = (diffuse_rgb * self.luminance_weights).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        else:
            # Simple mean brightness across RGB channels
            brightness = diffuse_rgb.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Identify highlight pixels (brightness/luminance > threshold)
        highlight_mask = (brightness > self.brightness_threshold).float()  # (B, 1, H, W)
        
        # Apply optional spatial mask
        if mask is not None:
            if mask.shape[1] == 1:
                mask = mask.expand(-1, 1, -1, -1)
            highlight_mask = highlight_mask * mask
            if highlight_mask.sum() <= 0:
                return torch.zeros((), device=diffuse_rgb.device, dtype=diffuse_rgb.dtype)
        
        # Compute penalty based on mode
        if self.penalty_mode == "brightness":
            # Penalize brightness above threshold
            excess_brightness = (brightness - self.target_brightness).clamp_min(0.0)
            penalty = excess_brightness * highlight_mask
        else:  # penalty_mode == "pixel"
            # Penalize RGB pixel values directly
            target_rgb = torch.full_like(diffuse_rgb, self.target_brightness)
            diff = (diffuse_rgb - target_rgb) * highlight_mask.expand(-1, 3, -1, -1)
            # Only penalize where pixels are brighter than target
            diff = diff.clamp_min(0.0)
            penalty = diff.mean(dim=1, keepdim=True)  # Average across RGB channels
        
        # Apply Charbonnier or L1 loss
        if self.use_charbonnier:
            loss_map = torch.sqrt(penalty * penalty + self.charbonnier_eps * self.charbonnier_eps)
        else:
            loss_map = penalty.abs()
        
        # Average over highlighted pixels only
        eps = 1e-8
        loss = loss_map.sum() / (highlight_mask.sum() + eps)
        
        return loss


# ============================================================================
# COLOR UTILITY FUNCTIONS
# ============================================================================

def rgb_hsv_saturation(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute HSV saturation from RGB image.
    
    Args:
        x: (B, 3, H, W) RGB image tensor
        eps: Small constant to avoid division by zero (default: 1e-6)
    
    Returns:
        sat: (B, 1, H, W) saturation map
    """
    maxc, _ = x.max(dim=1, keepdim=True)
    minc, _ = x.min(dim=1, keepdim=True)
    delta = maxc - minc
    return delta / (maxc + eps)


def single_to_rgb_highlight(highlight_single: torch.Tensor, highlight_color: torch.Tensor) -> torch.Tensor:
    """
    Convert single-channel highlight mask to RGB highlight.
    
    Args:
        highlight_single: (B, 1, H, W) single-channel highlight mask
        highlight_color: (3,) RGB color for highlights
    
    Returns:
        highlight_rgb: (B, 3, H, W) RGB highlight image
    """
    return highlight_single * highlight_color.view(1, 3, 1, 1)


# ============================================================================
# MASK UTILITY FUNCTIONS
# ============================================================================

def to_single_channel_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Ensure mask is single-channel [B, 1, H, W] format.
    
    Args:
        mask: (B, C, H, W) or (B, 1, H, W) mask tensor
    
    Returns:
        mask: (B, 1, H, W) single-channel mask
    """
    if mask.dim() != 4:
        raise ValueError("mask must be 4D [B,C,H,W]")
    if mask.size(1) == 1:
        return mask
    return (mask.sum(dim=1, keepdim=True) > 0).float()


def safe_mean(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute mean of x over masked region m.
    
    Args:
        x: (B, C, H, W) input tensor
        m: (B, 1, H, W) mask tensor (1 = valid region)
        eps: Small constant to avoid division by zero (default: 1e-8)
    
    Returns:
        mean: (B, C, 1, 1) mean value per channel
    """
    m = m.clamp(0.0, 1.0)
    denom = m.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (x * m).sum(dim=(2, 3), keepdim=True) / denom


def safe_var(x: torch.Tensor, m: torch.Tensor, mean: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute variance of x over masked region m.
    
    Args:
        x: (B, C, H, W) input tensor
        m: (B, 1, H, W) mask tensor (1 = valid region)
        mean: (B, C, 1, 1) precomputed mean (optional)
        eps: Small constant to avoid division by zero (default: 1e-8)
    
    Returns:
        var: (B, C, 1, 1) variance value per channel
    """
    m = m.clamp(0.0, 1.0)
    if mean is None:
        mean = safe_mean(x, m, eps)
    denom = m.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (((x - mean) ** 2) * m).sum(dim=(2, 3), keepdim=True) / denom


def hole_and_ring_masks(include_mask: torch.Tensor, ring_kernel_size: int) -> tuple:
    """
    Build hole and ring masks from include mask via max-pool dilation.
    
    Args:
        include_mask: (B, 1, H, W) mask where 1 = valid/context region
        ring_kernel_size: Kernel size for ring dilation (must be odd)
    
    Returns:
        hole: (B, 1, H, W) hole mask (1 = hole region)
        ring: (B, 1, H, W) ring mask (1 = boundary ring around hole)
    """
    base = to_single_channel_mask(include_mask)
    hole = (1.0 - base).clamp(0.0, 1.0)
    if ring_kernel_size <= 1:
        return hole, torch.zeros_like(hole)
    pad = ring_kernel_size // 2
    dilated = F.max_pool2d(hole, kernel_size=ring_kernel_size, stride=1, padding=pad)
    ring = (dilated - hole).clamp(0.0, 1.0)
    return hole, ring


def texture_map(x: torch.Tensor) -> torch.Tensor:
    """
    Compute finite-difference gradient magnitude per channel.
    
    Args:
        x: (B, C, H, W) input image tensor
    
    Returns:
        grad_mag: (B, C, H, W) gradient magnitude map
    """
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-8)


def saturation_hinge(diffuse_rgb: torch.Tensor, hole_mask: torch.Tensor, diffuse_saturation_max: float) -> torch.Tensor:
    """
    Compute saturation hinge loss: penalize saturation above threshold in hole region.
    
    Args:
        diffuse_rgb: (B, 3, H, W) diffuse RGB image
        hole_mask: (B, 1, H, W) hole mask (1 = hole region)
        diffuse_saturation_max: Maximum allowed saturation value
    
    Returns:
        Scalar saturation hinge loss
    """
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
    """
    Compute ring consistency loss: match mean, variance, and texture between hole and ring.
    
    Args:
        diffuse_rgb: (B, 3, H, W) diffuse RGB image
        hole_mask: (B, 1, H, W) hole mask (1 = hole region)
        ring_mask: (B, 1, H, W) ring mask (1 = boundary ring)
        ring_var_weight: Weight for variance matching term (default: 0.5)
        ring_texture_weight: Weight for texture consistency term (default: 1.0)
    
    Returns:
        Scalar ring consistency loss
    """
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


def saturation_ring_blob_consistency(
    diffuse_rgb: torch.Tensor,
    include_mask: torch.Tensor,
    ring_kernel_size: int,
) -> torch.Tensor:
    """
    Compare saturation inside each hole blob to its surrounding ring.
    
    Uses connected components to identify individual hole blobs and computes
    per-blob saturation consistency. Falls back to global comparison if
    connected components analysis fails.
    
    Args:
        diffuse_rgb: (B, 3, H, W) diffuse RGB image
        include_mask: (B, 1, H, W) mask where 1 = valid/context region
        ring_kernel_size: Kernel size for ring dilation
    
    Returns:
        Scalar tensor (mean over blobs and batch)
    """
    device = diffuse_rgb.device
    dtype = diffuse_rgb.dtype
    B = diffuse_rgb.size(0)
    sat = rgb_hsv_saturation(diffuse_rgb)  # [B, 1, H, W]

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


# ============================================================================
# IMAGE RECONSTRUCTION UTILITY FUNCTIONS
# ============================================================================

def alpha_composite(components, output_format='rgb'):
    """
    Alpha-composite multiple image components (diffuse, specular, highlight).
    
    Args:
        components: Dict with keys like 'diffuse', 'specular', 'highlight'
                   - Each component: (B, C, H, W) where C ∈ {1, 3, 4}
                   - C=1: grayscale, expanded to RGB with alpha=grayscale
                   - C=3: RGB, alpha=1.0
                   - C=4: RGBA
        output_format: 'rgb' or 'rgba' (default: 'rgb')
    
    Returns:
        composed: (B, 3, H, W) or (B, 4, H, W) composited image
    """
    diffuse = components['diffuse']
    B, C, H, W = diffuse.shape

    # Initialize result with diffuse component
    if C == 1:
        rgb = diffuse.expand(-1, 3, -1, -1)
        alpha = diffuse
        result_rgba = torch.cat([rgb, alpha], dim=1)
    elif C == 3:
        alpha = torch.ones_like(diffuse[:, :1])
        result_rgba = torch.cat([diffuse, alpha], dim=1)
    else:
        result_rgba = diffuse

    # Composite additional components
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


def reconstruct_image_from_components(
    prediction: dict,
    mask: torch.Tensor = None,
    clamp_reconstruction: bool = True,
) -> torch.Tensor:
    """
    Compose predicted components (diffuse, specular, highlight) to RGB image.
    
    Args:
        prediction: Dict with keys like 'diffuse', 'specular', 'highlight'
                   - diffuse: (B, C, H, W) where C ∈ {3, 4}
                   - specular: (B, C, H, W) where C ∈ {3, 4} (optional)
                   - highlight: (B, 1, H, W) (optional)
        mask: (B, 1, H, W) optional mask to apply to components before composition
        clamp_reconstruction: If True, clamp output to [0, 1] (default: True)
    
    Returns:
        composed_rgb: (B, 3, H, W) reconstructed RGB image
    """
    composite_dict = {}
    mask_3ch = None
    if mask is not None:
        mask_3ch = mask.expand(-1, 3, -1, -1)

    if 'diffuse' not in prediction:
        raise ValueError("Diffuse component is required for reconstruction")
    
    # Process diffuse component
    diffuse = prediction['diffuse']
    if mask_3ch is not None:
        if diffuse.shape[1] == 3:
            diffuse = diffuse * mask_3ch
        elif diffuse.shape[1] == 4:
            diffuse = torch.cat([diffuse[:, :3] * mask_3ch, diffuse[:, 3:4] * mask], dim=1)
    composite_dict['diffuse'] = diffuse

    # Process specular component (optional)
    if 'specular' in prediction:
        specular = prediction['specular']
        if mask_3ch is not None:
            if specular.shape[1] == 3:
                specular = specular * mask_3ch
            elif specular.shape[1] == 4:
                specular = torch.cat([specular[:, :3] * mask_3ch, specular[:, 3:4] * mask], dim=1)
        composite_dict['specular'] = specular

    # Process highlight component (optional)
    if 'highlight' in prediction:
        highlight = torch.clamp(prediction['highlight'], 0.0, 1.0)
        if mask is not None:
            highlight = highlight * mask
        composite_dict['highlight'] = highlight

    composed_rgb = alpha_composite(composite_dict, output_format='rgb')
    if clamp_reconstruction:
        composed_rgb = torch.clamp(composed_rgb, 0.0, 1.0)
    return composed_rgb


# ============================================================================
# ADDITIONAL HELPER FUNCTIONS
# ============================================================================

def _total_variation(x):
    """
    Compute total variation (TV) of image tensor.
    
    Args:
        x: (B, C, H, W) image tensor
    
    Returns:
        Scalar TV value (sum of horizontal and vertical variations)
    """
    tv_h = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    tv_w = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return tv_h + tv_w


def _grad_mag(x):
    """
    Compute gradient magnitude using forward differences.
    
    Args:
        x: (B, C, H, W) image tensor
    
    Returns:
        grad_mag: (B, C, H, W) gradient magnitude map
    """
    dx = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    dy = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    return (dx.abs() + dy.abs())


def _pixel_to_patch_mask(m_hw: torch.Tensor, patch: int) -> torch.Tensor:
    """
    Convert pixel mask to patch mask using max-pooling.
    
    A patch is marked as masked if ANY pixel inside is masked.
    
    Args:
        m_hw: (B, 1, H, W) pixel mask
        patch: Patch size (kernel size and stride for pooling)
    
    Returns:
        pm: (B, N) boolean patch mask where N = (H//patch) * (W//patch)
    """
    pm = F.max_pool2d((m_hw > 0.5).float(), kernel_size=patch, stride=patch)
    return pm.flatten(1).bool()  # (B, N)
