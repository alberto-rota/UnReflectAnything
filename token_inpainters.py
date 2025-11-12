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

    def forward(self, T: torch.Tensor, patch_inpaint_mask: torch.Tensor):
        # T: [B, N, C], patch_inpaint_mask: [B, N] (True = hole)
        B, N, C = T.shape
        mask = patch_inpaint_mask.unsqueeze(-1)  # [B, N, 1]
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


def _local_mean_prior(T, patch_inpaint_mask, H, W, k=3):
    """
    Depthwise box-filter local mean of *visible* neighbors for masked seeds.
    T: [B,N,C], patch_inpaint_mask: [B,N] (True=masked). Returns [B,N,C].
    """
    B, N, C = T.shape
    x = T.transpose(1, 2).reshape(B, C, H, W)  # B,C,H,W
    vis = torch.logical_not(patch_inpaint_mask).float().reshape(B, 1, H, W)  # B,1,H,W  (1 = visible)
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
        local_prior_kernel=15,
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
        self._prior_kernel = local_prior_kernel
    def forward(self, T: torch.Tensor, patch_inpaint_mask: torch.Tensor):
        # T: [B,N,C], patch_inpaint_mask: [B,N] (True = hole)
        B, N, C = T.shape
        device = T.device

        # infer (H,W) like your original; require perfect square
        hw = int(round(N**0.5))
        assert hw * hw == N, (
            "Token count N must be a perfect square (pass flattened grid)."
        )
        H = W = hw

        mask = patch_inpaint_mask.unsqueeze(-1)  # [B,N,1]

        # Seed masked positions with learned token on the positions indicated by the mask
        T_seed = torch.where(mask, self.mask_token.expand(B, N, C), T)

        # Blend the local mean prior on the masked positions (does not alter visible tokens)
        if self._use_local_prior:
            mean_prior = _local_mean_prior(T, patch_inpaint_mask, H, W, k=self._prior_kernel)
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

        # Add positionals to all tokens 
        if self.use_positional_encoding:
            pos = _build_2d_sincos_pos_embed(H, W, C, device).expand(B, N, C)
            X = T_seed + pos
        else:
            X = T_seed
            
        # Addmask indicator only at masked sites
        X = X + torch.where(
            mask, self.mask_indicator.expand(B, N, C), torch.zeros_like(X)
        )

        # full self-attention across all tokens (NO key padding/masking) — same as your behavior
        for blk in self.blocks:
            X = blk(X)

        # Apply final normalization if enabled
        if self._final_norm is not None:
            X = self._final_norm(X)

        # project ALL tokens
        return self.out_proj(X)
