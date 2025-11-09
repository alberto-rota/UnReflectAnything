import torch
import torch.nn as nn
import torch.nn.functional as F
import math

####
### NAIVE LINEAR TOKEN INPAINTER
####


class _TinyMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hid, dim)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class _TransformerBlk(nn.Module):
    def __init__(self, dim=768, heads=12, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = _TinyMLP(dim, 4.0, drop)

    def forward(self, x, attn_bias=None):
        q = k = v = self.n1(x)
        if attn_bias is None:
            a, _ = self.attn(q, k, v, need_weights=False)
        else:
            # attn_bias: (B, N, N) added to logits; implement via attn_mask per batch
            # Fallback: no bias (PyTorch's MHA doesn't support per-batch logit bias directly)
            a, _ = self.attn(q, k, v, need_weights=False)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x


class TokenInpainter_Naive(nn.Module):
    """
    Completes masked patch tokens from context.
    Input:  T  = (B, N, C) tokens for a selected DINO layer
            pm = (B, N)    boolean mask at patch resolution (True = masked/hole)
    Output: X  = (B, N, C) refined tokens; we will take X at masked positions
    """

    def __init__(
        self,
        dim=768,
        depth=4,
        heads=16,
        drop=0.0,
        use_positional_encoding=True,
        use_final_norm=None,  # Not used in base TokenInpainter, but accepted for compatibility
        use_local_prior=None,  # Not used in base TokenInpainter, but accepted for compatibility
        seed_noise_std=None,  # Not used in base TokenInpainter, but accepted for compatibility
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TransformerBlk(dim, heads, drop) for _ in range(depth)]
        )
        self.out_proj = nn.Linear(dim, dim, bias=True)
        # Learnable mask token to seed missing positions (randomly initialized with truncated normal, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Learnable indicator for masked positions (helps network identify holes)
        self.mask_indicator = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_indicator, std=0.02)
        # Enable fixed 2D sinusoidal positional encodings
        self.use_positional_encoding = use_positional_encoding

    def forward(self, T: torch.Tensor, pm_bool: torch.Tensor):
        # T: [B, N, C], pm_bool: [B, N] (True = hole)
        B, N, C = T.shape
        mask = pm_bool.unsqueeze(-1)  # [B, N, 1]
        mask_tok = self.mask_token  # [1, 1, C]
        # Seed masked positions with learned token, keep context as-is
        T_seed = torch.where(mask, mask_tok.expand(B, N, C), T)

        # Add positional encodings (crucial for spatial reasoning in attention)
        if self.use_positional_encoding:
            hw = int(N**0.5)
            if hw * hw != N:
                hw = int(round(N**0.5))
            pos = self._build_2d_sincos_pos_embed(hw, hw, C, T_seed.device)  # [1,N,C]
            T_seed = T_seed + pos

        # Add explicit masked-position indicator
        T_seed = T_seed + torch.where(
            mask,
            self.mask_indicator.expand(B, N, C),
            torch.zeros_like(T_seed),
        )
        X = T_seed
        for blk in self.blocks:
            X = blk(X)
        # Expected shape after projection: [B, N, C]
        return self.out_proj(X)

    @staticmethod
    def _build_2d_sincos_pos_embed(
        h: int, w: int, dim: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create 2D sinusoidal positional embeddings of shape [1, h*w, dim].
        """
        assert dim % 2 == 0, "positional dim must be even"
        half = dim // 2
        emb_h = TokenInpainter_Naive._build_1d_sincos_embed(
            half, h, device
        )  # [h, half]
        emb_w = TokenInpainter_Naive._build_1d_sincos_embed(
            half, w, device
        )  # [w, half]
        emb_h = emb_h[:, None, :].expand(h, w, half)
        emb_w = emb_w[None, :, :].expand(h, w, half)
        pos = torch.cat([emb_h, emb_w], dim=-1).reshape(1, h * w, dim)
        return pos

    @staticmethod
    def _build_1d_sincos_embed(
        dim: int, length: int, device: torch.device
    ) -> torch.Tensor:
        assert dim % 2 == 0, "1D pos dim must be even"
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(
            1
        )  # [L,1]
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-(torch.log(torch.tensor(10000.0, device=device))))
            / (dim // 2)
        )  # [dim/2]
        angles = positions * div_term  # [L, dim/2]
        emb = torch.empty((length, dim), device=device, dtype=torch.float32)
        emb[:, 0::2] = torch.sin(angles)
        emb[:, 1::2] = torch.cos(angles)
        return emb


####
### TOKEN INPAINTER WITH FIXED LOCAL MEAN PRIOR
####


class _PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class _MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.fc2 = nn.Linear(hid, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _SelfAttn(nn.Module):
    def __init__(self, dim, heads=8, drop=0.0, bias=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=drop, bias=bias, batch_first=True
        )

    def forward(self, x):
        # full self-attention over all tokens; no key padding/masking
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


class _TransformerBlk(nn.Module):
    def __init__(self, dim, heads=8, drop=0.0, mlp_ratio=4.0):
        super().__init__()
        self.attn = _PreNorm(dim, _SelfAttn(dim, heads, drop))
        self.mlp = _PreNorm(dim, _MLP(dim, mlp_ratio, drop))

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# --------- positionals + fixed local mean prior (depthwise) ---------


def _build_1d_sincos_embed(dim: int, length: int, device: torch.device) -> torch.Tensor:
    assert dim % 2 == 0
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)  # [L,1]
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    div = torch.exp(-(math.log(10000.0)) * i / (dim // 2))
    ang = pos * div
    emb = torch.empty((length, dim), device=device, dtype=torch.float32)
    emb[:, 0::2] = torch.sin(ang)
    emb[:, 1::2] = torch.cos(ang)
    return emb


def _build_2d_sincos_pos_embed(
    h: int, w: int, dim: int, device: torch.device
) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    eh = _build_1d_sincos_embed(half, h, device)[:, None, :].expand(h, w, half)
    ew = _build_1d_sincos_embed(half, w, device)[None, :, :].expand(h, w, half)
    return torch.cat([eh, ew], dim=-1).reshape(1, h * w, dim)


def _local_mean_prior(T, pm_bool, H, W, k=3):
    """
    Depthwise box-filter local mean of *visible* neighbors for masked seeds.
    T: [B,N,C], pm_bool: [B,N] (True=masked). Returns [B,N,C].
    """
    B, N, C = T.shape
    x = T.transpose(1, 2).reshape(B, C, H, W)  # B,C,H,W
    vis = (~pm_bool).float().reshape(B, 1, H, W)  # B,1,H,W  (1 = visible)
    pad = k // 2
    kernel = torch.ones(1, 1, k, k, device=T.device, dtype=T.dtype)

    # per-channel numerator via depthwise conv
    num = F.conv2d(x * vis, kernel.expand(C, 1, k, k), padding=pad, groups=C)  # B,C,H,W
    den = F.conv2d(vis, kernel, padding=pad).clamp_min(1e-4)  # B,1,H,W
    den = den.repeat(1, C, 1, 1)  # B,C,H,W

    mean = (num / den).reshape(B, C, N).transpose(1, 2)  # B,N,C
    return mean


class TokenInpainter_Prior(nn.Module):
    """
    Completes masked patch tokens from context.
    Input:  T  = (B, N, C) tokens for a selected DINO layer
            pm = (B, N)    boolean mask at patch resolution (True = masked/hole)
    Output: X  = (B, N, C) refined tokens; downstream will take masked positions.
    """

    def __init__(
        self,
        dim=768,
        depth=4,
        heads=16,
        drop=0.1,
        use_positional_encoding=True,
        use_final_norm=True,
        use_local_prior=True,
        local_prior_weight=0.5,
        seed_noise_std=0.01,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TransformerBlk(dim, heads, drop) for _ in range(depth)]
        )
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.mask_indicator = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_indicator, std=0.02)
        self.use_positional_encoding = use_positional_encoding

        # internal niceties (do NOT change mask semantics)
        self._final_norm = nn.LayerNorm(dim) if use_final_norm else None
        self._use_local_prior = use_local_prior  # only affects masked seeds
        self._seed_noise_std = seed_noise_std  # small train-time noise on masked seeds
        self._local_prior_weight = local_prior_weight

    def forward(self, T: torch.Tensor, pm_bool: torch.Tensor):
        # T: [B,N,C], pm_bool: [B,N] (True = hole)
        B, N, C = T.shape
        device = T.device

        # infer (H,W) like your original; require perfect square
        hw = int(round(N**0.5))
        assert hw * hw == N, (
            "Token count N must be a perfect square (pass flattened grid)."
        )
        H = W = hw

        mask = pm_bool.unsqueeze(-1)  # [B,N,1]

        # (1) seed masked positions with learned token (same as your logic)
        T_seed = torch.where(mask, self.mask_token.expand(B, N, C), T)

        # optional: blend a local mean *only for masked positions* (does not alter visible tokens)
        if self._use_local_prior:
            mean_prior = _local_mean_prior(T, pm_bool, H, W, k=5)
            T_seed = torch.where(
                mask,
                self._local_prior_weight * T_seed
                + (1.0 - self._local_prior_weight) * mean_prior,
                T_seed,
            )

        # tiny stochasticity on masked seeds during training (robustness)
        if self.training and self._seed_noise_std > 0:
            noise = torch.randn_like(T_seed) * self._seed_noise_std
            T_seed = torch.where(mask, T_seed + noise, T_seed)

        # (2) add positionals to all tokens (unchanged) + mask indicator only at masked sites
        if self.use_positional_encoding:
            pos = _build_2d_sincos_pos_embed(H, W, C, device).expand(B, N, C)
            X = T_seed + pos
        else:
            X = T_seed

        X = X + torch.where(
            mask, self.mask_indicator.expand(B, N, C), torch.zeros_like(X)
        )

        # (3) full self-attention across all tokens (NO key padding/masking) — same as your behavior
        for blk in self.blocks:
            X = blk(X)

        # Apply final normalization if enabled
        if self._final_norm is not None:
            X = self._final_norm(X)

        # (4) project ALL tokens (your original returns out_proj(X) for all positions)
        return self.out_proj(X)


class TokenInpainter_Blended(nn.Module):
    """
    Token inpainter with soft boundary blending to eliminate visible seams.

    Applies feathered blending at masked tokens near boundaries:
    - Interior masked tokens: 100% inpainted
    - Boundary masked tokens: smooth blend between inpainted and original
    - Visible tokens: returned as inpainted (downstream will replace with original)

    This maintains the same input/output contract as Naive and Prior versions.
    """

    def __init__(
        self,
        dim=768,
        depth=4,
        heads=16,
        drop=0.1,
        use_positional_encoding=True,
        use_final_norm=True,
        use_local_prior=True,
        seed_noise_std=0.01,
        blend_border_width=3,  # Width of blending zone in patches
        blend_kernel_size=5,  # Kernel size for distance approximation
        local_prior_weight=0.5,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TransformerBlk(dim, heads, drop) for _ in range(depth)]
        )
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.mask_indicator = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_indicator, std=0.02)
        self.use_positional_encoding = use_positional_encoding

        self._final_norm = nn.LayerNorm(dim) if use_final_norm else None
        self._use_local_prior = use_local_prior
        self._seed_noise_std = seed_noise_std
        self._local_prior_weight = local_prior_weight
        # Boundary blending parameters
        self.blend_border_width = blend_border_width
        self.blend_kernel_size = blend_kernel_size

    def forward(self, T: torch.Tensor, pm_bool: torch.Tensor):
        """
        Args:
            T: [B, N, C] - input tokens
            pm_bool: [B, N] - boolean mask (True = masked/hole)

        Returns:
            [B, N, C] - refined tokens at all positions
        """
        B, N, C = T.shape
        device = T.device

        hw = int(round(N**0.5))
        assert hw * hw == N, "Token count N must be a perfect square"
        H = W = hw

        mask = pm_bool.unsqueeze(-1)  # [B, N, 1]

        # === STEP 1: Seed masked positions ===
        T_seed = torch.where(mask, self.mask_token.expand(B, N, C), T)

        if self._use_local_prior:
            mean_prior = _local_mean_prior(T, pm_bool, H, W, k=5)  # [B, N, C]
            T_seed = torch.where(
                mask,
                self._local_prior_weight * T_seed
                + (1.0 - self._local_prior_weight) * mean_prior,
                T_seed,
            )


        if self.training and self._seed_noise_std > 0:
            noise = torch.randn_like(T_seed) * self._seed_noise_std  # [B, N, C]
            T_seed = torch.where(mask, T_seed + noise, T_seed)

        # === STEP 2: Add positionals + mask indicator ===
        if self.use_positional_encoding:
            pos = _build_2d_sincos_pos_embed(H, W, C, device)  # [1, N, C]
            pos = pos.expand(B, N, C)
            X = T_seed + pos
        else:
            X = T_seed

        X = X + torch.where(
            mask, self.mask_indicator.expand(B, N, C), torch.zeros_like(X)
        )

        # === STEP 3: Transformer processing ===
        for blk in self.blocks:
            X = blk(X)

        if self._final_norm is not None:
            X = self._final_norm(X)

        X_inpainted = self.out_proj(X)  # [B, N, C]

        # === STEP 4: Soft boundary blending ===
        # Compute blend weights for masked tokens based on distance from boundary
        blend_weights = self._compute_blend_weights(pm_bool, H, W)  # [B, N, 1]

        # Apply blending only at masked positions
        # blend_weights: 0.0 = use original T, 1.0 = use inpainted X
        X_blended = blend_weights * X_inpainted + (1.0 - blend_weights) * T  # [B, N, C]

        # At masked positions: use blended result
        # At visible positions: use inpainted (downstream will replace with original T)
        X_final = torch.where(mask, X_blended, X_inpainted)  # [B, N, C]

        return X_final

    def _compute_blend_weights(
        self, pm_bool: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        Compute soft blending weights for masked tokens based on distance from boundary.

        Interior masked tokens → weight = 1.0 (use inpainted)
        Boundary masked tokens → weight < 1.0 (blend with original)
        Visible tokens → weight irrelevant (will use X_inpainted regardless)

        Args:
            pm_bool: [B, N] - boolean mask (True = masked)
            H, W: spatial dimensions

        Returns:
            [B, N, 1] - blend weights in [0, 1]
        """
        B, N = pm_bool.shape
        device = pm_bool.device

        # Reshape to spatial: [B, 1, H, W]
        mask_spatial = pm_bool.float().reshape(B, 1, H, W)

        # Compute distance from each masked pixel to nearest boundary
        distance = self._distance_from_boundary(mask_spatial, H, W)  # [B, 1, H, W]

        # Convert distance to blend weight using smooth transition
        # distance = 0 (at boundary) → weight ≈ 0.0 (use original)
        # distance ≥ blend_border_width (interior) → weight = 1.0 (use inpainted)
        max_dist = float(self.blend_border_width)

        # Smooth sigmoid-like transition
        # weight = smoothstep(distance / max_dist)
        t = (distance / max_dist).clamp(0.0, 1.0)  # [B, 1, H, W]
        weights = t * t * (3.0 - 2.0 * t)  # Smoothstep interpolation

        # Reshape back: [B, N, 1]
        weights = weights.reshape(B, N, 1)

        return weights

    def _distance_from_boundary(
        self, mask: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        Approximate distance (in patches) from each masked pixel to mask boundary.
        Uses iterative erosion to compute distance transform.

        Args:
            mask: [B, 1, H, W] - float mask (1.0 = masked)

        Returns:
            [B, 1, H, W] - distance map (0 at boundary, max at interior)
        """
        B = mask.shape[0]
        device = mask.device

        # Initialize distance map
        distance = torch.zeros_like(mask)  # [B, 1, H, W]
        current_mask = mask.clone()

        k = self.blend_kernel_size
        pad = k // 2

        # Erosion kernel: all ones
        kernel = torch.ones(1, 1, k, k, device=device, dtype=mask.dtype)

        max_iter = self.blend_border_width + 1

        for d in range(max_iter):
            # Pixels in current_mask but not in eroded mask are at distance d
            if d > 0:
                # Erode: convolve and threshold (keep only fully surrounded pixels)
                neighbor_sum = F.conv2d(
                    current_mask, kernel, padding=pad
                )  # [B, 1, H, W]
                eroded = (
                    neighbor_sum >= k * k - 0.5
                ).float()  # All neighbors must be masked

                # Pixels lost in erosion are at distance d from boundary
                boundary_ring = (current_mask > 0.5) & (eroded < 0.5)  # [B, 1, H, W]
                distance = torch.where(
                    boundary_ring, torch.full_like(distance, float(d)), distance
                )

                current_mask = eroded

                # Early exit if mask fully eroded
                if current_mask.sum() == 0:
                    break

        # Remaining interior pixels get maximum distance
        interior = current_mask > 0.5
        distance = torch.where(
            interior, torch.full_like(distance, float(max_iter)), distance
        )

        return distance


# ---------- helpers: positional encodings ----------
def _build_1d_sincos_embed(dim: int, length: int, device: torch.device) -> torch.Tensor:
    assert dim % 2 == 0
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)  # [L,1]
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    div = torch.exp(-(math.log(10000.0)) * i / (dim // 2))
    ang = pos * div
    emb = torch.empty((length, dim), device=device, dtype=torch.float32)
    emb[:, 0::2] = torch.sin(ang)
    emb[:, 1::2] = torch.cos(ang)
    return emb


def _build_2d_sincos_pos_embed(
    h: int, w: int, dim: int, device: torch.device
) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    eh = _build_1d_sincos_embed(half, h, device)[:, None, :].expand(h, w, half)
    ew = _build_1d_sincos_embed(half, w, device)[None, :, :].expand(h, w, half)
    return torch.cat([eh, ew], dim=-1).reshape(1, h * w, dim)


# ---------- soft local mean prior (visibility-weighted) ----------
def _soft_local_mean_prior(
    T: torch.Tensor, m_soft: torch.Tensor, H: int, W: int, k: int = 5
) -> torch.Tensor:
    """
    Local mean of *visible* neighbors with soft visibility weights (1 - m).
    T:      [B,N,C]
    m_soft: [B,N] in [0,1]
    Return: [B,N,C]
    """
    B, N, C = T.shape
    x = T.transpose(1, 2).reshape(B, C, H, W)  # B,C,H,W
    vis = (1.0 - m_soft).reshape(B, 1, H, W)  # B,1,H,W (visibility in [0,1])

    pad = k // 2
    kernel = torch.ones(1, 1, k, k, device=T.device, dtype=T.dtype)

    # depthwise numerator = sum(vis * x) in kxk window
    num = F.conv2d(x * vis, kernel.expand(C, 1, k, k), padding=pad, groups=C)  # B,C,H,W
    den = F.conv2d(vis, kernel, padding=pad).clamp_min(1e-6)  # B,1,H,W
    den = den.repeat(1, C, 1, 1)

    mean = (num / den).reshape(B, C, N).transpose(1, 2)  # B,N,C
    return mean


# ---------- blocks with visibility-biased self-attn and gated residuals ----------
class _PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class _MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hid, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class _SelfAttnSoft(nn.Module):
    """
    Visibility-biased self-attention:
    - queries = normed x
    - keys/values = normed x * w_vis, where w_vis = (1 - m_soft)
    This steers masked queries to attend more to visible tokens.
    """

    def __init__(self, dim, heads=8, drop=0.0, bias=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            dim, heads, dropout=drop, bias=bias, batch_first=True
        )

    def forward(self, x, w_vis: torch.Tensor):
        # x: [B,N,C], w_vis: [B,N,1] in [0,1]
        q = x
        kv = x * w_vis  # downweight masked tokens as keys/values
        out, _ = self.mha(q, kv, kv, need_weights=False)
        return out


class _TransformerBlkSoft(nn.Module):
    def __init__(self, dim, heads=8, drop=0.0, mlp_ratio=4.0):
        super().__init__()
        self.attn = _PreNorm(dim, _SelfAttnSoft(dim, heads, drop))
        self.mlp = _PreNorm(dim, _MLP(dim, mlp_ratio, drop))

    def forward(self, x, gate: torch.Tensor, w_vis: torch.Tensor):
        """
        x:    [B,N,C]
        gate: [B,N,1]  — residual strength per token in [0,1] (≈ m_soft)
        w_vis:[B,N,1]  — visibility for keys/values = (1 - m_soft)
        """
        x = x + gate * self.attn(x, w_vis=w_vis)
        x = x + gate * self.mlp(x)
        return x


# ---------- SOFT-GUIDED TOKEN INPAINTER ----------
class TokenInpainter_Soft(nn.Module):
    """
    Soft-mask guided token inpainter.
    Input:
        T       : (B, N, C) tokens
        pm_soft : (B, N) soft mask in [0,1] (0=keep original, 1=inpaint fully)
    Output:
        (B, N, C) refined tokens
    Behavior:
        - Minimal edits where pm_soft≈0
        - Full inpaint where pm_soft≈1
        - Boundary naturally blended by soft gating
    """

    def __init__(
        self,
        dim=768,
        depth=4,
        heads=16,
        drop=0.1,
        use_positional_encoding=True,
        use_final_norm=True,
        use_local_prior=True,
        seed_noise_std=0.01,
        prior_kernel=5,
        seed_mix_alpha=0.5,  # mix between mask_token and local prior for seeds
        use_smoothstep_blend=True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TransformerBlkSoft(dim, heads, drop) for _ in range(depth)]
        )
        self.out_proj = nn.Linear(dim, dim, bias=True)

        # learned tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_indicator = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.mask_indicator, std=0.02)

        self.use_positional_encoding = use_positional_encoding
        self._final_norm = nn.LayerNorm(dim) if use_final_norm else None
        self._use_local_prior = use_local_prior
        self._seed_noise_std = seed_noise_std
        self._prior_kernel = prior_kernel
        self._alpha = seed_mix_alpha
        self._use_smoothstep = use_smoothstep_blend

    @staticmethod
    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        # x in [0,1]
        return x * x * (3.0 - 2.0 * x)

    def forward(self, T: torch.Tensor, pm_soft: torch.Tensor):
        """
        T:        [B, N, C]
        pm_soft:  [B, N]  (float, 0..1). If boolean is passed, it will be cast to float.
        """
        B, N, C = T.shape
        device = T.device

        m = pm_soft.to(dtype=T.dtype).clamp(0.0, 1.0)  # [B,N]
        gate = m.unsqueeze(-1)  # [B,N,1]
        w_vis = (1.0 - m).unsqueeze(-1)  # [B,N,1]

        hw = int(round(N**0.5))
        assert hw * hw == N, "Token count N must be a perfect square (flattened grid)."
        H = W = hw

        # ---- Seed with soft mix: original vs (mask_token ⊕ local_prior) according to m ----
        seed_target = self.mask_token.expand(B, N, C)
        if self._use_local_prior:
            prior = _soft_local_mean_prior(T, m, H, W, k=self._prior_kernel)  # [B,N,C]
            seed_target = self._alpha * seed_target + (1.0 - self._alpha) * prior

        T_seed = (1.0 - gate) * T + gate * seed_target  # [B,N,C]

        # tiny train-time noise only where masked (softly)
        if self.training and self._seed_noise_std > 0:
            noise = torch.randn_like(T_seed) * self._seed_noise_std
            T_seed = T_seed + gate * noise

        # positionals and mask indicator (scaled by soft mask)
        if self.use_positional_encoding:
            pos = _build_2d_sincos_pos_embed(H, W, C, device).expand(B, N, C)
            X = T_seed + pos
        else:
            X = T_seed
        X = X + gate * self.mask_indicator.expand(B, N, C)

        # visibility-biased attention + gated residuals
        for blk in self.blocks:
            X = blk(X, gate=gate, w_vis=w_vis)

        if self._final_norm is not None:
            X = self._final_norm(X)

        X_hat = self.out_proj(X)  # [B,N,C]

        # ---- Soft output blending (small edits where mask≈0) ----
        blend_w = self._smoothstep(gate) if self._use_smoothstep else gate  # [B,N,1]
        X_final = blend_w * X_hat + (1.0 - blend_w) * T
        return X_final
