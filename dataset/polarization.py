import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class PolarizationProcessor:
    """
    Utilities to load and process polarization inputs into a unified dictionary of tensors.

    All returned tensors are torch tensors with channel-first convention where applicable.
    Shapes:
    - RGB-like planes: [3, H, W]
    - Scalar maps (S0, S1, S2, S3, DoLP, AoP, etc.): [1, H, W]
    - Stokes stack: [1, 3, H, W]
    """

    def __init__(
        self,
        rho_s: float,
        eps: float,
        dolp_min_intensity: float,
        dolp_min_value: float,
        # ---- new knobs (reasonable defaults) ----
        use_linear_rgb: bool = True,
        guided_r: int = 12,
        guided_eps: float = 1e-3,
        tau_dolp: float = 0.28,
        tau_I: float = 0.50,
        tau_sat: float = 0.20,
        k_soft: float = 10.0,
        noise_sigma: float | None = None,   # if None, estimated crudely from S1/S2
    ) -> None:
        self.rho_s = rho_s
        self.eps = eps
        self.dolp_min_intensity = dolp_min_intensity
        self.dolp_min_value = dolp_min_value

        self.use_linear_rgb = use_linear_rgb
        self.guided_r = guided_r
        self.guided_eps = guided_eps
        self.tau_dolp = tau_dolp
        self.tau_I = tau_I
        self.tau_sat = tau_sat
        self.k_soft = k_soft
        self.noise_sigma = noise_sigma

    # ---------- helpers ----------
    @staticmethod
    def _to_luminance_standard(rgb_hwc: torch.Tensor) -> torch.Tensor:
        # rgb_hwc: [H, W, 3] in [0,1]
        return (
            0.2126 * rgb_hwc[..., 0]
            + 0.7152 * rgb_hwc[..., 1]
            + 0.0722 * rgb_hwc[..., 2]
        )

    @staticmethod
    def _to_luminance_equal(rgb_hwc: torch.Tensor) -> torch.Tensor:
        return (
            0.3333 * rgb_hwc[..., 0]
            + 0.3333 * rgb_hwc[..., 1]
            + 0.3333 * rgb_hwc[..., 2]
        )

    @staticmethod
    def _rgb_to_linear(rgb_hwc: torch.Tensor) -> torch.Tensor:
        """Inverse sRGB gamma. rgb in [0,1]."""
        a = 0.055
        low = rgb_hwc <= 0.04045
        out = torch.empty_like(rgb_hwc)
        out[low] = rgb_hwc[low] / 12.92
        out[~low] = ((rgb_hwc[~low] + a) / (1 + a)) ** 2.4
        return out.clamp_(0, 1)

    @staticmethod
    def _normalize01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        xmin = x.amin(dim=(-2, -1), keepdim=True)
        xmax = x.amax(dim=(-2, -1), keepdim=True)
        return (x - xmin) / (xmax - xmin + eps)

    @staticmethod
    def _rgb_saturation(rgb_chw: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """rgb [3,H,W] -> saturation [1,H,W]"""
        mx = rgb_chw.max(dim=0, keepdim=True).values
        mn = rgb_chw.min(dim=0, keepdim=True).values
        v = mx
        s = torch.where(v > eps, (v - mn) / (v + eps), torch.zeros_like(v))
        return s

    def _box_filter(self, x: torch.Tensor, r: int) -> torch.Tensor:
        """
        Fast separable box filter via cumulative sums.
        x: [B,C,H,W], returns same shape. Window = (2r+1)^2
        """
        if r <= 0:
            return x

        B, C, H, W = x.shape
        k = 2 * r + 1
        area = float(k * k)

        # --- vertical pass (pad H only) ---
        x_pad = F.pad(x, (0, 0, r, r), mode='reflect')          # [B,C,H+2r,W]
        cs = torch.cumsum(x_pad, dim=2)                         # [B,C,H+2r,W]
        cs = torch.cat([torch.zeros_like(cs[:, :, :1, :]), cs], dim=2)  # prefix zero

        # window sum over height -> [B,C,H,W]
        vsum = cs[:, :, k:(k + H), :] - cs[:, :, :H, :]

        # --- horizontal pass (pad W only) ---
        vsum_pad = F.pad(vsum, (r, r, 0, 0), mode='reflect')    # [B,C,H,W+2r]
        cs2 = torch.cumsum(vsum_pad, dim=3)                     # [B,C,H,W+2r]
        cs2 = torch.cat([torch.zeros_like(cs2[:, :, :, :1]), cs2], dim=3)

        # window sum over width -> [B,C,H,W]
        hsum = cs2[:, :, :, k:(k + W)] - cs2[:, :, :, :W]

        return hsum / area



    def _fast_guided_filter(self, I_chw: torch.Tensor, p_hw: torch.Tensor) -> torch.Tensor:
        """
        I_chw: [3,H,W] guide in [0,1]
        p_hw:  [H,W]   src  in [0,1]
        return: [H,W]
        """
        I = I_chw.unsqueeze(0)          # [1,3,H,W]
        p = p_hw.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        mean_I = self._box_filter(x=I, r=self.guided_r)
        mean_p = self._box_filter(p, self.guided_r)
        mean_Ip = self._box_filter(I * p, self.guided_r)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self._box_filter(I * I, self.guided_r)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.guided_eps)
        b = mean_p - (a * mean_I).sum(dim=1, keepdim=True)

        mean_a = self._box_filter(a, self.guided_r)
        mean_b = self._box_filter(b, self.guided_r)
        q = (mean_a * I).sum(dim=1, keepdim=True) + mean_b
        return q.squeeze(0).squeeze(0).clamp_(0, 1)

    def _finalize_common(
        self,
        I0_rgb: torch.Tensor,
        I45_rgb: torch.Tensor,
        I90_rgb: torch.Tensor,
        I135_rgb: torch.Tensor,
        S0: torch.Tensor,
        S1: torch.Tensor,
        S2: torch.Tensor,
        f_spec: torch.Tensor,
        AoP: torch.Tensor,
        DoLP: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        S3 = torch.zeros_like(S0)
        return {
            "I0": I0_rgb.permute(2, 0, 1),
            "I45": I45_rgb.permute(2, 0, 1),
            "I90": I90_rgb.permute(2, 0, 1),
            "I135": I135_rgb.permute(2, 0, 1),
            "S0": S0.unsqueeze(0),
            "S1": S1.unsqueeze(0),
            "S2": S2.unsqueeze(0),
            "S3": S3.unsqueeze(0),
            "stokes": torch.cat([S0, S1, S2], dim=0).unsqueeze(0),
            "intensity": S0.unsqueeze(0),
            "DoLP": DoLP.unsqueeze(0),
            "AoP": AoP.unsqueeze(0),
            "AoLP": AoP.unsqueeze(0),
            "DoP": DoLP.unsqueeze(0),
            "DoCP": torch.zeros_like(DoLP).unsqueeze(0),
            "ellipticity_angle": torch.zeros_like(DoLP).unsqueeze(0),
            "f_spec": f_spec.unsqueeze(0),
        }

    def _mask_dolp(self, S0: torch.Tensor, DoLP: torch.Tensor) -> torch.Tensor:
        valid_mask = (S0 >= self.dolp_min_intensity).float()
        DoLP = DoLP * valid_mask
        return torch.where(DoLP < self.dolp_min_value, torch.zeros_like(DoLP), DoLP)

    # ---------- core specular estimator (used by all loaders) ----------
    def _compute_spec_from_tiles(
        self,
        I0_rgb: torch.Tensor, I45_rgb: torch.Tensor,
        I90_rgb: torch.Tensor, I135_rgb: torch.Tensor,
        use_equal_luma: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inputs are HxWx3 (float [0,1]).
        Returns: S0, S1, S2, f_spec  (all HxW floats)
        """
        # Linearize if desired
        if self.use_linear_rgb:
            I0_rgb   = self._rgb_to_linear(I0_rgb)
            I45_rgb  = self._rgb_to_linear(I45_rgb)
            I90_rgb  = self._rgb_to_linear(I90_rgb)
            I135_rgb = self._rgb_to_linear(I135_rgb)

        # Luminance
        toY = self._to_luminance_equal if use_equal_luma else self._to_luminance_standard
        I0   = toY(I0_rgb)
        I45  = toY(I45_rgb)
        I90  = toY(I90_rgb)
        I135 = toY(I135_rgb)

        # Robust Stokes (average both pairs for S0)
        S0a = I0 + I90
        S0b = I45 + I135
        S0  = 0.5 * (S0a + S0b)
        S1  = I0 - I90
        S2  = I45 - I135

        # DoLP with de-biasing
        mag2 = S1.square() + S2.square()
        if self.noise_sigma is None:
            # crude noise estimate from local variation
            blurS1 = F.avg_pool2d(S1.unsqueeze(0).unsqueeze(0), 7, stride=1, padding=3).squeeze()
            blurS2 = F.avg_pool2d(S2.unsqueeze(0).unsqueeze(0), 7, stride=1, padding=3).squeeze()
            sigma2 = (blurS1.std() + blurS2.std()) / 2
            sigma2 = (sigma2 + 1e-6) ** 2
        else:
            sigma2 = torch.as_tensor(self.noise_sigma**2, dtype=S0.dtype, device=S0.device)

        DoLP_raw = (mag2.sqrt() / torch.clamp(S0, min=self.eps)).clamp(0, 1)
        DoLP_db  = ((mag2 - sigma2).clamp_min(0).sqrt() / torch.clamp(S0, min=self.eps)).clamp(0, 1)

        # Edge-aware smoothing guided by the average RGB of the four tiles
        guide_rgb = torch.stack([I0_rgb, I45_rgb, I90_rgb, I135_rgb], dim=0).mean(0)  # HxWx3
        guide_chw = guide_rgb.permute(2, 0, 1).contiguous()
        
        dolp_smooth = self._fast_guided_filter(guide_chw, DoLP_db)

        # Additional cues: intensity & saturation
        I_norm = self._normalize01(S0)
        sat = self._rgb_saturation(guide_chw)  # [1,H,W]

        # Soft fusion (product of sigmoids)
        sig = torch.sigmoid
        f_spec = (
            sig(self.k_soft * (dolp_smooth - self.tau_dolp)) *
            sig(self.k_soft * (I_norm - self.tau_I)) *
            sig(self.k_soft * ((1.0 - sat.squeeze(0)) - self.tau_sat))
        ).clamp(0, 1)

        DoLP_final = self._mask_dolp(S0, dolp_smooth)  # keep your masking behavior
        f_spec = self._mask_dolp(S0, f_spec)

        return S0, S1, S2, DoLP_final, f_spec

    # ---------- loaders ----------
    def load_single_file_clock(self, pol_path: str) -> Dict[str, torch.Tensor]:
        pol_img = Image.open(pol_path).convert("RGB")
        pol_rgb = torch.from_numpy(np.asarray(pol_img, dtype=np.float32)) / 255.0

        H, W, _ = pol_rgb.shape
        hh, hw = H // 2, W // 2

        I0_rgb   = pol_rgb[:hh, :hw, :]
        I45_rgb  = pol_rgb[:hh, hw:, :]
        I90_rgb  = pol_rgb[hh:, hw:, :]
        I135_rgb = pol_rgb[hh:, :hw, :]

        S0, S1, S2, DoLP, f_spec = self._compute_spec_from_tiles(
            I0_rgb, I45_rgb, I90_rgb, I135_rgb, use_equal_luma=False
        )
        AoP = 0.5 * torch.atan2(S2, S1)

        return self._finalize_common(I0_rgb, I45_rgb, I90_rgb, I135_rgb, S0, S1, S2, f_spec, AoP, DoLP)

    def load_separate_files(self, pol_base_path: str, pol_ext: str) -> Dict[str, torch.Tensor]:
        pol_paths = {
            "000": f"{pol_base_path}_000{pol_ext}",
            "045": f"{pol_base_path}_045{pol_ext}",
            "090": f"{pol_base_path}_090{pol_ext}",
            "135": f"{pol_base_path}_135{pol_ext}",
        }

        pol_images: Dict[str, torch.Tensor] = {}
        for angle, path in pol_paths.items():
            img = Image.open(path).convert("RGB")
            pol_images[angle] = torch.from_numpy(np.asarray(img, dtype=np.float32)) / 255.0

        I0_rgb   = pol_images["000"]
        I45_rgb  = pol_images["045"]
        I90_rgb  = pol_images["090"]
        I135_rgb = pol_images["135"]

        S0, S1, S2, DoLP, f_spec = self._compute_spec_from_tiles(
            I0_rgb, I45_rgb, I90_rgb, I135_rgb, use_equal_luma=False
        )
        AoP = 0.5 * torch.atan2(S2, S1)

        return self._finalize_common(I0_rgb, I45_rgb, I90_rgb, I135_rgb, S0, S1, S2, f_spec, AoP, DoLP)

    def load_separate_stokes(self, pol_base_path: str) -> Dict[str, torch.Tensor]:
        stokes_paths = {
            "S0": f"{pol_base_path}_S0.npy",
            "S1": f"{pol_base_path}_S1.npy",
            "S2": f"{pol_base_path}_S2.npy",
        }

        stokes_data: Dict[str, torch.Tensor] = {}
        for name, path in stokes_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Stokes file not found: {path}")
            arr = np.load(path)
            stokes_data[name] = torch.from_numpy(arr.astype(np.float32))

        S0 = stokes_data["S0"].mean(-1)
        S1 = stokes_data["S1"].mean(-1)
        S2 = stokes_data["S2"].mean(-1)

        mag2 = S1**2 + S2**2
        sigma2 = torch.as_tensor(
            (self.noise_sigma or 0.0) ** 2, dtype=S0.dtype, device=S0.device
        )
        DoLP_db = ((mag2 - sigma2).clamp_min(0).sqrt() / torch.clamp(S0, min=self.eps)).clamp(0, 1)

        # Build a guidance RGB from S0 (grayscale) for smoothing
        guide_rgb = torch.stack([S0, S0, S0], dim=0)  # [3,H,W]
        DoLP = self._fast_guided_filter(guide_rgb, DoLP_db)

        # f_spec fusion with intensity only (no color available here)
        I_norm = self._normalize01(S0)
        sig = torch.sigmoid
        f_spec = (sig(self.k_soft * (DoLP - self.tau_dolp)) * sig(self.k_soft * (I_norm - self.tau_I))).clamp(0, 1)

        DoLP = self._mask_dolp(S0, DoLP)
        f_spec = self._mask_dolp(S0, f_spec)

        AoP = 0.5 * torch.atan2(S2, S1)

        I0 = (S0 + S1) / 2.0
        I90 = (S0 - S1) / 2.0
        I0_rgb = torch.stack([I0, I0, I0], dim=-1)
        I45_rgb = torch.zeros_like(I0_rgb)
        I90_rgb = torch.stack([I90, I90, I90], dim=-1)
        I135_rgb = torch.zeros_like(I0_rgb)

        return self._finalize_common(I0_rgb, I45_rgb, I90_rgb, I135_rgb, S0, S1, S2, f_spec, AoP, DoLP)

    def load_single_file_topdown(self, pol_path: str) -> Dict[str, torch.Tensor]:
        pol_img = Image.open(pol_path).convert("RGB")
        pol_rgb = torch.from_numpy(np.asarray(pol_img, dtype=np.float32)) / 255.0

        H, W, _ = pol_rgb.shape
        nh = H // 4

        I0_rgb   = pol_rgb[0 * nh:1 * nh, :, :]
        I45_rgb  = pol_rgb[1 * nh:2 * nh, :, :]
        I90_rgb  = pol_rgb[2 * nh:3 * nh, :, :]
        I135_rgb = pol_rgb[3 * nh:4 * nh, :, :]

        S0, S1, S2, DoLP, f_spec = self._compute_spec_from_tiles(
            I0_rgb, I45_rgb, I90_rgb, I135_rgb, use_equal_luma=False
        )
        AoP = 0.5 * torch.atan2(S2, S1)

        return self._finalize_common(I0_rgb, I45_rgb, I90_rgb, I135_rgb, S0, S1, S2, f_spec, AoP, DoLP)

    def load(
        self,
        pol_path: str,
        polarization_format: str,
        pol_ext: str,
    ) -> Dict[str, torch.Tensor]:
        if polarization_format == "single_file_clock":
            return self.load_single_file_clock(pol_path)
        if polarization_format == "separate_files":
            return self.load_separate_files(pol_path, pol_ext)
        if polarization_format == "separate_files_stokes":
            return self.load_separate_stokes(pol_path)
        if polarization_format == "single_file_topdown":
            return self.load_single_file_topdown(pol_path)
        raise ValueError(f"Unknown polarization format: {polarization_format}")
