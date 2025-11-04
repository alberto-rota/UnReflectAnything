from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoImageProcessor
from models import _build, DINOv3, DPT_Decoder

# ---------- 1) POL preprocessing ----------
class PolarizationPreprocess(nn.Module):
    """
    Input:
      aolp: (B,1,H,W) in radians, typically [0, pi)
      dolp: (B,1,H,W) in [0,1]
    Output:
      (B,3,H,W) = [cos(2*AoLP), sin(2*AoLP), DoLP]
    """

    def __init__(self, dinov3_model_name: str, height: int, width: int):
        super().__init__()
        self.prep_fn = AutoImageProcessor.from_pretrained(dinov3_model_name)
        self.prep_fn.do_normalize = False
        self.prep_fn.do_rescale = False
        self.prep_fn.size = {"height": height, "width": width}

    def forward(self, aolp: torch.Tensor, dolp: torch.Tensor) -> torch.Tensor:
        # aolp = self.prep_fn(images=aolp, return_tensors="pt")["pixel_values"]
        # dolp = self.prep_fn(images=dolp, return_tensors="pt")["pixel_values"]
        cos2 = torch.cos(2.0 * aolp)
        sin2 = torch.sin(2.0 * aolp)
        cos2sin2dolp = torch.cat([cos2, sin2, dolp.clamp(0, 1)], dim=1)
        cropped_pol = self.prep_fn(images=cos2sin2dolp, return_tensors="pt")[
            "pixel_values"
        ]
        return cropped_pol



# ---------- 2) Tiny ViT-style POL encoder ----------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):  # x: (B,C,H,W)
        x = self.proj(x)  # (B,embed_dim,H/P,W/P)
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C) with N = Hp*Wp
        return x, (Hp, Wp)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc2(self.drop(self.act(self.fc1(x))))
        return self.drop(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=12, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=4.0, drop=drop)

    def forward(self, x):
        # Self-attention
        x = (
            x
            + self.attn(
                self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False
            )[0]
        )
        x = x + self.mlp(self.norm2(x))
        return x
class POLViTEncoder(nn.Module):
    """
    ViT-like encoder for POL features; match DINOv3 hidden dim and patch size.

    Args:
        config: Dict containing configuration parameters:
            - in_ch: int, input channels (default: 3 for cos2θ, sin2θ, DoLP)
            - embed_dim: int, embedding dimension (default: 768)
            - depth: int, number of transformer blocks (default: 4)
            - n_heads: int, number of attention heads (default: 12)
            - patch_size: int, patch size for embedding (default: 16)
            - drop: float, dropout rate (default: 0.0)

    Returns:
        tokens: (B, N, embed_dim)  # N = (H/P)*(W/P)
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        # Default configuration
        default_config = {
            "in_ch": 3,
            "embed_dim": 768,
            "depth": 4,
            "n_heads": 12,
            "patch_size": 16,
            "drop": 0.0,
        }

        self.config = {**default_config, **(config or {})}

        self.patch = PatchEmbed(
            self.config["in_ch"], self.config["embed_dim"], self.config["patch_size"]
        )
        self.pos = None  # initialized at first forward pass
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.config["embed_dim"],
                    self.config["n_heads"],
                    self.config["drop"],
                )
                for _ in range(self.config["depth"])
            ]
        )
        self.norm = nn.LayerNorm(self.config["embed_dim"])

    def forward(self, x):  # x: (B,3,H,W)
        x, (Hp, Wp) = self.patch(x)  # (B,N,C)
        # learnable 2D pos embedding (initialized on first run to match N)
        N = x.shape[1]
        if (self.pos is None) or (self.pos.shape[1] != N):
            self.pos = nn.Parameter(torch.zeros(1, N, x.shape[2], device=x.device))
            nn.init.trunc_normal_(self.pos, std=0.02)
        x = x + self.pos
        pol_hidden_states = []
        for blk in self.blocks:
            x = blk(x)
            pol_hidden_states.append(self.norm(x))
        return pol_hidden_states  # (B,N,C)


# ---------- 3) Cross-attention block ----------
class CrossAttentionBlock(nn.Module):
    """
    One cross-attention layer with residual + MLP.
    Q from x_q (e.g., RGB tokens), K/V from x_kv (e.g., POL tokens).
    Shapes:
      x_q:  (B, Nq, C)
      x_kv: (B, Nk, C)
    Returns:
      (B, Nq, C) fused features.
    """

    def __init__(self, dim=768, n_heads=12, drop=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=drop, batch_first=True
        )
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=4.0, drop=drop)

    def forward(self, x_q, x_kv, attn_mask=None):
        q = self.norm_q(x_q)
        kv = self.norm_kv(x_kv)
        x, _ = self.cross_attn(q, kv, kv, attn_mask=attn_mask, need_weights=False)
        x = x_q + x  # residual after attention
        x = x + self.mlp(self.norm_out(x))  # residual after MLP
        return x



class RGBPOLCrossFuse(nn.Module):
    """
    One-way (RGB<-POL) or two-way (bi-directional) fusion.
    If bi_directional=True, returns concat([RGB_fused, POL_fused]) projected back to dim.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        default_config = {
            "embed_dim": 768,
            "n_heads": 12,
            "dropout": 0.0,
            "bi_directional": False,
        }
        self.config = {**default_config, **(config or {})}
        self.rgb_from_pol = CrossAttentionBlock(
            self.config["embed_dim"], self.config["n_heads"], self.config["dropout"]
        )
        self.bi = self.config["bi_directional"]
        if self.config["bi_directional"]:
            self.pol_from_rgb = CrossAttentionBlock(
                self.config["embed_dim"], self.config["n_heads"], self.config["dropout"]
            )
            self.proj = nn.Linear(
                2 * self.config["embed_dim"], self.config["embed_dim"]
            )

    def forward(self, rgb_tokens, pol_tokens, attn_mask=None):
        rgb_fused = self.rgb_from_pol(
            rgb_tokens, pol_tokens, attn_mask
        )  # Q=RGB, K/V=POL
        if not self.bi:
            return rgb_fused
        pol_fused = self.pol_from_rgb(
            pol_tokens, rgb_tokens, attn_mask
        )  # Q=POL, K/V=RGB
        fused = torch.cat([rgb_fused, pol_fused], dim=-1)
        return self.proj(fused)  # (B, N_rgb, dim) if N_rgb == N_pol; else up to you




# ------------------ top-level model ------------------
 

class RGBPOLDecomposer(nn.Module):
    """
    RGB + POL decomposition with cross-attention and flexible DPT decoders.

    Inputs (forward):
      batch["rgb"] : (B,3,H,W) in [0,1]
      batch["AoP"] : (B,1,H,W) radians
      batch["DoP"] : (B,1,H,W) in [0,1]

    Returns:
      {
        "decoder_name": (B,C,H,W) for each configured decoder
        "tokens": {
           "rgb": (B,N,C),
           "pol": (B,N,C),
           "cross": (B,N,C)
        }
      }
    """

    def __init__(
        self,
        # 1) RGB encoder (DINOv3) — instance or config dict
        dinov3,
        # 2) POL encoder — instance or configs (preprocess is created inside if not passed)
        pol_encoder=None,  # POLViTEncoder instance or dict
        pol_preprocess=None,  # PolarizationPreprocess instance or dict
        pol_cross_attn=None,  # RGBPOLCrossFuse instance or dict
        # 3) Flexible decoders — dict of decoder_name -> DPT_Decoder config/instance
        decoders=None,  # Dict[str, DPT_Decoder instance or dict config]
        # Legacy support for backward compatibility (will be deprecated)
        spec_decoder=None,  # DPT_Decoder instance or dict
        diffuse_decoder=None,  # DPT_Decoder instance or dict
        highlight_decoder=None,  # DPT_Decoder instance or dict
        # Optional: if your DINO wrapper needs these hints
        patch_size: int = 16,
    ):
        super().__init__()

        # ---- RGB (DINOv3) ----
        # Accept either an instance or a DINOv3(**cfg) dict
        self.dinov3 = _build(dinov3, DINOv3)

        self.image_size = self.dinov3.config["image_size"]
        self.patch_size = patch_size
        self.embed_dim = self.dinov3.feature_dim

        # ---- POL branch ----
        # Preprocess (AoLP, DoLP) → [cos2θ, sin2θ, DoLP]
        if pol_preprocess is None:
            self.pol_pre = PolarizationPreprocess(
                self.dinov3.config["model_name"], self.image_size, self.image_size
            )
        else:
            self.pol_pre = _build(pol_preprocess, PolarizationPreprocess)

        # Encoder (ViT-like), align dim/patch with DINO
        if pol_encoder is None:
            self.pol_enc = POLViTEncoder(
                in_ch=3,
                embed_dim=self.embed_dim,
                depth=4,
                n_heads=12,
                patch_size=patch_size,
            )
        else:
            self.pol_enc = _build(pol_encoder, POLViTEncoder)

        # Cross-attention: Q=RGB, K/V=POL
        if pol_cross_attn is None:
            self.cross = nn.ModuleList(
                [
                    RGBPOLCrossFuse(
                        embed_dim=self.embed_dim,
                        n_heads=12,
                        dropout=0.1,
                        bi_directional=False,
                    )
                    for _ in range(4)
                ]
            )
        else:
            self.cross = nn.ModuleList(
                [_build(pol_cross_attn, RGBPOLCrossFuse) for _ in range(4)]
            )

        # ---- Decoders (DPT_Decoder) ----
        # Handle flexible decoder configuration with legacy support
        def build_dpt(dec):
            """Build DPT decoder from config or instance."""
            if isinstance(dec, DPT_Decoder):
                return dec
            if isinstance(dec, dict):
                # DPT_Decoder takes a single config dict
                config = {
                    "feature_dim": self.embed_dim,
                    **dec,
                }
                return DPT_Decoder(config)
            raise TypeError("Decoder must be DPT_Decoder instance or dict.")

        # Use flexible decoders if provided, otherwise fall back to legacy format
        if decoders is not None:
            self.decoder_names = list(decoders.keys())
            self.decoders = nn.ModuleDict()
            for decoder_name, decoder_config in decoders.items():
                self.decoders[decoder_name] = build_dpt(decoder_config)
        else:
            # Legacy support - create decoders from individual parameters
            legacy_decoders = {}
            
            # Specular decoder
            if spec_decoder is not None:
                legacy_decoders["specular"] = spec_decoder
            else:
                legacy_decoders["specular"] = {
                    "use_bn": True, 
                    "readout_type": "project",
                    "output_channels": 3,
                }
            
            # Diffuse decoder  
            if diffuse_decoder is not None:
                legacy_decoders["diffuse"] = diffuse_decoder
            else:
                legacy_decoders["diffuse"] = {
                    "use_bn": True,
                    "readout_type": "project", 
                    "output_channels": 3,
                }
                
            # Highlight decoder
            if highlight_decoder is not None:
                legacy_decoders["highlight"] = highlight_decoder
            else:
                legacy_decoders["highlight"] = {
                    "use_bn": True,
                    "readout_type": "project",
                    "output_channels": 1,
                }
            
            self.decoder_names = list(legacy_decoders.keys())
            self.decoders = nn.ModuleDict()
            for decoder_name, decoder_config in legacy_decoders.items():
                self.decoders[decoder_name] = build_dpt(decoder_config)

    def _rgb_tokens(self, rgb_preproc):
        """Extract DINOv3 tokens and infer (Hp, Wp) if wrapper doesn’t return them."""
        with torch.no_grad():
            out = self.dinov3(rgb_preproc)
        tokens = out.get("last_hidden_state", out.get("tokens"))
        if tokens is None:
            raise KeyError(
                # "DINOv3 wrapper must return 'last_hidden_state' or 'tokens'."
            )
        Hp = self.image_size // self.patch_size
        Wp = self.image_size // self.patch_size
        return tokens, (Hp, Wp)

    def forward(self, batch):
        # 1) RGB → DINO tokens
        rgb_in = self.dinov3.preprocess_image(batch["rgb"])
        rgb_tokens = self.dinov3(rgb_in)["selected_hidden_states"]

        # 2) POL → preprocess → POL tokens
        pol_in = self.pol_pre(batch["AoP"], batch["DoP"])  # (B,3,H,W)
        pol_tokens = self.pol_enc(pol_in)  # (B,N,C)
        # 3) CROSS (Q=RGB, K/V=POL)
        cross_tokens = []
        for i in range(4):
            cross_tokens.append(self.cross[i](rgb_tokens[i], pol_tokens[i]))

        # 6) Decode with flexible decoder heads
        outputs = {}
        for decoder_name in self.decoder_names:
            decoder_output = self.decoders[decoder_name](cross_tokens)
            outputs[decoder_name] = decoder_output

        # Optional: Add tokens for debugging/analysis
        # outputs.update({
        #     "rgb_tokens": rgb_tokens,
        #     "pol_tokens": pol_tokens,
        #     "cross_tokens": cross_tokens,
        # })

        return outputs
    
