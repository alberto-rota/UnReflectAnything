import os
from typing import Dict, Tuple

import numpy as np
import torch
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
    ) -> None:
        self.rho_s = rho_s
        self.eps = eps
        self.dolp_min_intensity = dolp_min_intensity
        self.dolp_min_value = dolp_min_value

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
        # rgb_hwc: [H, W, 3] in [0,1]
        return (
            0.3333 * rgb_hwc[..., 0]
            + 0.3333 * rgb_hwc[..., 1]
            + 0.3333 * rgb_hwc[..., 2]
        )

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

    def load_single_file_clock(self, pol_path: str) -> Dict[str, torch.Tensor]:
        pol_img = Image.open(pol_path).convert("RGB")
        pol_rgb = torch.from_numpy(np.asarray(pol_img, dtype=np.float32)) / 255.0

        H, W, _ = pol_rgb.shape
        hh, hw = H // 2, W // 2

        I0_rgb = pol_rgb[:hh, :hw, :]
        I45_rgb = pol_rgb[:hh, hw:, :]
        I90_rgb = pol_rgb[hh:, hw:, :]
        I135_rgb = pol_rgb[hh:, :hw, :]

        I0 = self._to_luminance_equal(I0_rgb)
        I45 = self._to_luminance_equal(I45_rgb)
        I90 = self._to_luminance_equal(I90_rgb)
        I135 = self._to_luminance_equal(I135_rgb)

        S0 = I0 + I90
        S1 = I0 - I90
        S2 = I45 - I135

        DoLP = torch.clamp(torch.sqrt(S1**2 + S2**2) / torch.clamp(S0, min=self.eps), 0.0, 1.0)
        AoP = 0.5 * torch.atan2(S2, S1)
        DoLP = self._mask_dolp(S0, DoLP)

        # Preserve original behavior: final f_spec assignment uses S0 clamped
        f_spec = torch.clamp(S0, 0.0, 1.0)

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

        I0_rgb = pol_images["000"]
        I45_rgb = pol_images["045"]
        I90_rgb = pol_images["090"]
        I135_rgb = pol_images["135"]

        I0 = self._to_luminance_standard(I0_rgb)
        I45 = self._to_luminance_standard(I45_rgb)
        I90 = self._to_luminance_standard(I90_rgb)
        I135 = self._to_luminance_standard(I135_rgb)

        S0 = I0 + I90
        S1 = I0 - I90
        S2 = I45 - I135

        R = torch.sqrt(S1**2 + S2**2)
        DoLP = torch.clamp(R / torch.clamp(S0, min=self.eps), 0.0, 1.0)
        AoP = 0.5 * torch.atan2(S2, S1)
        DoLP = self._mask_dolp(S0, DoLP)

        # Preserve original behavior: last assignment in original code
        f_spec = torch.clamp(S0, 0.0, 1.0) * S0

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

        R = torch.sqrt(S1**2 + S2**2)
        DoLP = torch.clamp(R / torch.clamp(S0, min=self.eps), 0.0, 1.0)
        AoP = 0.5 * torch.atan2(S2, S1)
        DoLP = self._mask_dolp(S0, DoLP)

        # f_spec final behavior per original code
        f_spec = torch.clamp(S0, 0.0, 1.0)

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

        I0_rgb = pol_rgb[0 * nh:1 * nh, :, :]
        I45_rgb = pol_rgb[1 * nh:2 * nh, :, :]
        I90_rgb = pol_rgb[2 * nh:3 * nh, :, :]
        I135_rgb = pol_rgb[3 * nh:4 * nh, :, :]

        I0 = self._to_luminance_standard(I0_rgb)
        I45 = self._to_luminance_standard(I45_rgb)
        I90 = self._to_luminance_standard(I90_rgb)
        I135 = self._to_luminance_standard(I135_rgb)

        S0 = I0 + I90
        S1 = I0 - I90
        S2 = I45 - I135

        R = torch.sqrt(S1**2 + S2**2)
        DoLP = torch.clamp(R / torch.clamp(S0, min=self.eps), 0.0, 1.0)
        AoP = 0.5 * torch.atan2(S2, S1)
        DoLP = self._mask_dolp(S0, DoLP)

        # f_spec final behavior per original code
        f_spec = torch.clamp(S0, 0.0, 1.0)

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


