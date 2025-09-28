# sd_decomposer.py
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import wrappers/helpers from your models.py
from models import DINOv3, _build


# -----------------------------
# 1) DINO → SD Token Adapter
# -----------------------------
class DINOToSDTokenAdapter(nn.Module):
    """
    Projects a LIST of DINOv3 hidden states (selected layers) to Stable Diffusion's
    cross-attention context (SD 1.x: dim=768, length=77 by default).

    Input:  hidden_states: List[Tensor], each [B, N_i, C_dino]
    Output: encoder_hidden_states: [B, ctx_len, ctx_dim]
    """
    def __init__(
        self,
        dino_dim: int,
        ctx_dim: int = 768,
        ctx_len: int = 77,
        n_layers_proj: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
        reduce_mode: str = "learned_pool",   # "learned_pool" | "mean_pool" | "concat_trunc"
        layer_fusion: str = "weighted_sum",  # "weighted_sum" | "concat_project"
    ):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.ctx_len = ctx_len
        self.reduce_mode = reduce_mode
        self.layer_fusion = layer_fusion
        self._dino_dim = dino_dim

        # Lazy because number of selected layers can vary
        self._initialized = False
        self._n_layers_seen = 0
        self.per_layer_proj = nn.ModuleList()
        self.per_layer_ln = nn.ModuleList()
        self.concat_proj: Optional[nn.Linear] = None

        # Tiny transformer to refine fused tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=ctx_dim, nhead=n_heads, dim_feedforward=ctx_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.refiner = nn.TransformerEncoder(enc_layer, num_layers=n_layers_proj)

        # Learned pooling queries (for reduce_mode="learned_pool")
        if self.reduce_mode == "learned_pool":
            self.pool_tokens = nn.Parameter(torch.randn(1, ctx_len, ctx_dim))
            nn.init.trunc_normal_(self.pool_tokens, std=0.02)
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=ctx_dim, num_heads=n_heads, batch_first=True, dropout=dropout
            )

        # Layer weights for weighted_sum
        self.layer_weights: Optional[nn.Parameter] = None

    def _lazy_init(self, n_layers: int, device: torch.device, dtype: torch.dtype):
        if self._initialized:
            return
        self._n_layers_seen = n_layers
        for _ in range(n_layers):
            ln_layer = nn.LayerNorm(self._dino_dim)
            proj_layer = nn.Linear(self._dino_dim, self.ctx_dim)
            # Move layers to the correct device and dtype
            ln_layer = ln_layer.to(device=device, dtype=dtype)
            proj_layer = proj_layer.to(device=device, dtype=dtype)
            self.per_layer_ln.append(ln_layer)
            self.per_layer_proj.append(proj_layer)
        if self.layer_fusion == "concat_project":
            self.concat_proj = nn.Linear(n_layers * self.ctx_dim, self.ctx_dim)
            self.concat_proj = self.concat_proj.to(device=device, dtype=dtype)
        if self.layer_weights is None and self.layer_fusion == "weighted_sum":
            self.layer_weights = nn.Parameter(torch.zeros(n_layers, device=device, dtype=dtype))
        self._initialized = True

    def _ensure_device_consistency(self, device: torch.device, dtype: torch.dtype):
        """Ensure all components are on the correct device and dtype."""
        # Move pool_tokens and pool_attn if they exist
        if hasattr(self, 'pool_tokens'):
            self.pool_tokens.data = self.pool_tokens.data.to(device=device, dtype=dtype)
        if hasattr(self, 'pool_attn'):
            self.pool_attn = self.pool_attn.to(device=device, dtype=dtype)
        
        # Move refiner transformer
        self.refiner = self.refiner.to(device=device, dtype=dtype)

    def _align_length(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        # Make every [B, N, C] have the same N (choose minimum for memory friendliness)
        N_min = min(t.shape[1] for t in tensors)
        aligned = []
        for t in tensors:
            if t.shape[1] == N_min:
                aligned.append(t)
            else:
                B, N_old, C = t.shape
                idx = torch.linspace(0, N_old - 1, N_min, device=t.device)
                idx_long = idx.round().long().clamp(0, N_old - 1)
                aligned.append(t.index_select(dim=1, index=idx_long))
        return aligned

    def _fuse(self, per_layer_tokens: List[torch.Tensor]) -> torch.Tensor:
        # All [B, N, C] with same N and C=ctx_dim
        if self.layer_fusion == "weighted_sum":
            alpha = torch.softmax(self.layer_weights, dim=0)  # [L]
            stacked = torch.stack(per_layer_tokens, dim=0)    # [L, B, N, C]
            return (alpha[:, None, None, None] * stacked).sum(dim=0)
        elif self.layer_fusion == "concat_project":
            x = torch.cat(per_layer_tokens, dim=-1)           # [B, N, L*C]
            return self.concat_proj(x)                        # [B, N, C]
        else:
            raise ValueError(self.layer_fusion)

    def _reduce_len(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C] → [B, ctx_len, C]
        B, N, C = x.shape
        if N == self.ctx_len:
            return x
        if self.reduce_mode == "mean_pool":
            idx = torch.linspace(0, N, steps=self.ctx_len + 1, device=x.device, dtype=torch.long)
            idx[-1] = N
            chunks = []
            for i in range(self.ctx_len):
                s, e = idx[i].item(), idx[i + 1].item()
                if e <= s:
                    chunks.append(x[:, s:s+1].mean(dim=1, keepdim=True))
                else:
                    chunks.append(x[:, s:e].mean(dim=1, keepdim=True))
            return torch.cat(chunks, dim=1)
        if self.reduce_mode == "concat_trunc":
            if N >= self.ctx_len:
                return x[:, : self.ctx_len, :]
            pad = self.ctx_len - N
            return torch.cat([x, x[:, -1:, :].repeat(1, pad, 1)], dim=1)
        if self.reduce_mode == "learned_pool":
            q = self.pool_tokens.expand(B, -1, -1)
            out, _ = self.pool_attn(q, x, x, need_weights=False)
            return out
        raise ValueError(self.reduce_mode)

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # Normalize device/dtype
        model_dev = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        self._lazy_init(len(hidden_states), model_dev, model_dtype)
        self._ensure_device_consistency(model_dev, model_dtype)

        # Per-layer: LN + Linear to ctx_dim
        projected = []
        for i, h in enumerate(hidden_states):
            h = h.to(device=model_dev, dtype=model_dtype)
            h = self.per_layer_ln[i](h)
            h = self.per_layer_proj[i](h)
            projected.append(h)

        # Align length, fuse, refine, reduce
        projected = self._align_length(projected)
        fused = self._fuse(projected)
        fused = self.refiner(fused)
        fused = self._reduce_len(fused)
        return fused  # [B, ctx_len, ctx_dim]


# -----------------------------
# 2) Component heads (pixel space)
# -----------------------------
class ComponentHead(nn.Module):
    """
    Lightweight CNN head that maps a decoded RGB (and optionally recon features)
    to a component image (e.g., specular, diffuse, highlight).
    """
    def __init__(self, in_ch=3, out_ch=3, hidden=64, use_bn=False):
        super().__init__()
        Norm = nn.BatchNorm2d if use_bn else nn.Identity
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            Norm(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, padding=1),
            Norm(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, padding=1),
            Norm(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, out_ch, 1),
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------
# 3) StableDiffusionDecomposer (SD as learned decoder)
# ---------------------------------------------------
class StableDiffusionDecomposer(nn.Module):
    """
    Use SD's VAE+UNet as a decoder conditioned on DINOv3 features (no text),
    and predict specular/diffuse/highlight from the reconstructed RGB.

    Returns (both train and inference):
      {
        "specular":  (B,3,H,W),
        "diffuse":   (B,3,H,W),
        "highlight": (B,1,H,W) or (B,3,H,W) depending on init,
        "recon":     (B,3,H,W),
        "loss":      scalar (during training)
      }
    """
    def __init__(
        self,
        dinov3: Union[DINOv3, Dict],
        sd_vae,                      # diffusers AutoencoderKL
        sd_unet,                     # diffusers UNet2DConditionModel
        scheduler,                   # diffusers scheduler (e.g., DDPMScheduler)
        token_adapter: Optional[DINOToSDTokenAdapter] = None,
        adapter_cfg: Optional[Dict] = None,
        image_size: int = 512,
        latent_scale: float = 0.18215,   # SD 1.x default
        freeze_vae: bool = True,
        freeze_unet: bool = True,
        unfreeze_unet_attn_qkv: bool = False,
        # Component heads
        highlight_out_ch: int = 1,       # set 3 if you want RGB highlight
        use_bn_heads: bool = False,
        enforce_sum_to_recon: bool = True,  # if True, diffuse := clamp(recon - specular)
        # Inference settings for reconstruction in forward()
        num_inference_steps_components: int = 15,  # small loop to get recon
        eta: float = 0.0,  # only used with DDIM schedulers, ignored for DDPM
    ):
        super().__init__()

        # ---- DINO ----
        self.dino: DINOv3 = _build(dinov3, DINOv3)
        self.image_size = image_size
        self.latent_scale = latent_scale

        # ---- SD ----
        self.vae = sd_vae
        self.unet = sd_unet
        self.scheduler = scheduler

        # Freezing
        if freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False
        if freeze_unet:
            for p in self.unet.parameters():
                p.requires_grad = False
        if unfreeze_unet_attn_qkv:
            for name, module in self.unet.named_modules():
                if "attn2" in name:  # cross-attention blocks
                    for pname, p in module.named_parameters(recurse=False):
                        if any(k in pname for k in ("to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj")):
                            p.requires_grad = True

        # ---- Adapter ----
        if token_adapter is None:
            adapter_cfg = adapter_cfg or {}
            self.adapter = DINOToSDTokenAdapter(
                dino_dim=self.dino.feature_dim,
                **adapter_cfg
            )
        else:
            self.adapter = token_adapter

        # ---- Component heads (pixel space) ----
        self.head_spec = ComponentHead(in_ch=3, out_ch=3, hidden=64, use_bn=use_bn_heads)
        self.head_high = ComponentHead(in_ch=3, out_ch=highlight_out_ch, hidden=64, use_bn=use_bn_heads)
        self.enforce_sum_to_recon = enforce_sum_to_recon

        # Inference sampling for recon inside forward() (to return components)
        self.num_inference_steps_components = num_inference_steps_components
        self.eta = eta

    # ---------- device helpers ----------
    def _dev(self) -> torch.device:
        return next(self.parameters()).device

    def _dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    # ---------- SD latent encode/decode ----------
    @torch.no_grad()
    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W] in [0,1] on the SAME DEVICE as the module
        returns: z0 [B,4,64,64] scaled by latent_scale
        """
        x = x.to(device=self._dev(), dtype=self._dtype())
        x_ = (x * 2.0) - 1.0
        posterior = self.vae.encode(x_).latent_dist
        z = posterior.sample()
        return z * self.latent_scale

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(device=self._dev(), dtype=self._dtype())
        z_ = z / self.latent_scale
        x = self.vae.decode(z_).sample
        x = (x + 1.0) / 2.0
        return x.clamp(0, 1)

    # ---------- DINO → SD context ----------
    def _make_context(self, x_rgb: torch.Tensor) -> torch.Tensor:
        # Everything on the module device/dtype
        x_rgb = x_rgb.to(device=self._dev(), dtype=self._dtype())
        with torch.no_grad():
            x_proc = self.dino.preprocess_image(x_rgb)               # already sent to input device in your wrapper
            x_proc = x_proc.to(device=self._dev(), dtype=self._dtype())
            dino_out = self.dino(x_proc)
        if "selected_hidden_states" in dino_out:
            hidden_states: List[torch.Tensor] = dino_out["selected_hidden_states"]
        elif "last_hidden_state" in dino_out:
            hidden_states = [dino_out["last_hidden_state"]]
        else:
            raise KeyError("DINOv3 wrapper must return 'selected_hidden_states' or 'last_hidden_state'.")
        # Adapter ensures device/dtype internally
        context = self.adapter(hidden_states)                        # [B, L, ctx_dim]
        return context.to(device=self._dev(), dtype=self._dtype())

    # ---------- core diffusion loss (train) ----------
    def diffusion_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard latent diffusion MSE on noise prediction.
        """
        B = x.size(0)
        z0 = self.encode_latent(x)                                   # [B,4,64,64]
        noise = torch.randn_like(z0, device=self._dev(), dtype=self._dtype())
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (B,),
            device=self._dev(), dtype=torch.long
        )
        zt = self.scheduler.add_noise(z0, noise, timesteps)
        context = self._make_context(x)

        eps_hat = self.unet(zt, timesteps, encoder_hidden_states=context).sample
        return F.mse_loss(eps_hat, noise)

    # ---------- small sampler to get recon for heads ----------
    @torch.no_grad()
    def _get_recon_via_sampling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Small DDIM-like loop to get a reconstruction x_hat used by component heads.
        Deterministic-ish and runs entirely on the module device/dtype.
        
        Compatible with both DDPM and DDIM schedulers:
        - DDIM schedulers: uses eta parameter for deterministic sampling
        - DDPM schedulers: ignores eta parameter (stochastic sampling)
        """
        x = x.to(device=self._dev(), dtype=self._dtype())
        z0 = self.encode_latent(x)
        context = self._make_context(x)

        self.scheduler.set_timesteps(self.num_inference_steps_components, device=self._dev())
        # Start near z0 (simple one-step noise forward)
        t0 = self.scheduler.timesteps[0]
        alpha_prod = self.scheduler.alphas_cumprod[t0]
        noise = torch.randn_like(z0, device=self._dev(), dtype=self._dtype())
        z = torch.sqrt(alpha_prod) * z0 + torch.sqrt(1 - alpha_prod) * noise

        for t in self.scheduler.timesteps:
            model_output = self.unet(z, t, encoder_hidden_states=context).sample
            # Check if scheduler supports eta parameter (DDIM vs DDPM)
            scheduler_name = self.scheduler.__class__.__name__
            if 'DDIM' in scheduler_name:
                step_out = self.scheduler.step(model_output, t, z, eta=self.eta)
            else:
                step_out = self.scheduler.step(model_output, t, z)
            z = step_out.prev_sample

        x_hat = self.decode_latent(z)
        return x_hat

    # ---------- forward ----------
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Train-time forward:
          - computes diffusion loss
          - also returns component predictions (using a quick sampler to get recon)

        Returns:
          {
            "specular":  (B,3,H,W),
            "diffuse":   (B,3,H,W),
            "highlight": (B,1 or 3,H,W),
            "recon":     (B,3,H,W),
            "loss":      scalar
          }
        """
        x = batch["rgb"]
        # 1) diffusion MSE loss
        loss = self.diffusion_loss(x)

        # 2) get a reconstruction to feed component heads
        with torch.no_grad():
            recon = self._get_recon_via_sampling(x)                  # [B,3,H,W] on module device/dtype

        # 3) component heads (trainable)
        spec = self.head_spec(recon)                                 # [B,3,H,W]
        high = self.head_high(recon)                                 # [B,1 or 3,H,W]

        if self.enforce_sum_to_recon:
            diff = (recon - spec).clamp(0.0, 1.0)
        else:
            # Free-form diffuse head (optional): uncomment to add a separate head
            # diff = self.head_diff(recon)
            # For simplicity keep derived diffuse:
            diff = (recon - spec).clamp(0.0, 1.0)

        return {
            "specular": spec,
            "diffuse": diff,
            "highlight": high,
            "recon": recon,
            "loss": loss,
        }

    # ---------- inference helper ----------
    @torch.no_grad()
    def decompose(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Inference-only decomposition (no training loss). Fully on a single device.
        """
        recon = self._get_recon_via_sampling(x)
        spec = self.head_spec(recon)
        high = self.head_high(recon)
        if self.enforce_sum_to_recon:
            diff = (recon - spec).clamp(0.0, 1.0)
        else:
            diff = (recon - spec).clamp(0.0, 1.0)
        return {
            "specular": spec,
            "diffuse": diff,
            "highlight": high,
            "recon": recon,
        }

    # ---------- utilities ----------
    def freeze_all_but_adapter_and_heads(self):
        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.unet.parameters():
            p.requires_grad = False
        for p in self.adapter.parameters():
            p.requires_grad = True
        for p in self.head_spec.parameters():
            p.requires_grad = True
        for p in self.head_high.parameters():
            p.requires_grad = True

    def unfreeze_unet_cross_attention_qkv(self):
        for name, module in self.unet.named_modules():
            if "attn2" in name:
                for pname, p in module.named_parameters(recurse=False):
                    if any(k in pname for k in ("to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj")):
                        p.requires_grad = True
