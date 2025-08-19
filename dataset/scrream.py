import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, List
from torch.utils.data import Dataset, DataLoader

try:
    import polanalyser as pa
    POLANALYSER_AVAILABLE = True
except ImportError:
    POLANALYSER_AVAILABLE = False
    print("Warning: polanalyser not available. Using manual implementation.")


class SCRREAM(Dataset):
    """
    Dataset for loading RGB and polarization images with computed features.

    Expected folder structure:
    root_dir/
    ├── scene1/
    │   ├── rgb/  └── *.png
    │   ├── pol/  └── *.png
    │   └── intrinsics.txt
    ├── scene2/
    │   ├── rgb/  └── *.png
    │   ├── pol/  └── *.png
    │   └── intrinsics.txt
    ...
    """

    def __init__(
        self,
        root_dir: str,
        rho_s: float = 0.6,
        eps: float = 1e-8,
        rgb_ext: str = ".png",
        pol_ext: str = ".png",
        transform=None,
        # --- NEW: smoothing / robustness controls ---
        smooth_specular: bool = True,
        smooth_cfg: Optional[Dict] = None,
        scene_names: Optional[List[str]] = None,
    ):
        """
        Args:
            rho_s: Assumed specular DoLP (dielectrics).
            eps:   Small epsilon to avoid division by zero.
            smooth_specular: Enable edge-aware upsample + smoothing of f_spec.
            smooth_cfg: Dict of smoothing params (see defaults below).
            scene_names: Optional list of scene names to load. If provided and not empty,
                        only scenes with these names will be loaded.
        """
        self.root_dir = root_dir
        self.rho_s = rho_s
        self.eps = eps
        self.rgb_ext = rgb_ext
        self.pol_ext = pol_ext
        self.transform = transform
        self.scene_names = scene_names

        self.smooth_specular = smooth_specular
        self.smooth_cfg = smooth_cfg or {
            # Half-res Stokes prefilter (in pixels)
            "stokes_gauss_sigma": 0.0,
            # DoLP robustness
            "dolp_min_intensity": 0.05,   # mask DoLP when S0 < this (0..1)
            "dolp_min_value": 0.04,       # set DoLP<min to 0
            # Full-res edge-aware refinement
            "guided_radius": 8,
            "guided_eps": 1e-3,
            # Fallback joint-bilateral if guided not available
            "bilateral_d": 9,
            "bilateral_sigma_color": 0.1,
            "bilateral_sigma_space": 8,
            # Optional TV denoise on full-res f_spec
            "use_tv": False,
            "tv_weight": 0.05,
            "tv_iters": 30,
        }

        self.scene_pairs = self._find_scene_pairs()

    # ----------------------- small helpers -----------------------

    @staticmethod
    def _to_luminance(rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to luminance. Input: [...,3] in [0,1]."""
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        # return 1/3 * rgb[..., 0] + 1/3 * rgb[..., 1] + 1/3 * rgb[..., 2]

    def _gaussian_blur_np(self, img: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return img
        k = int(2 * round(3.0 * sigma) + 1)
        return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)

    def _guided_filter_gray(self, guide_rgb: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
        """Prefer OpenCV ximgproc guided filter; fallback to two bilateral passes."""
        try:
            import cv2.ximgproc as xip
            gf = xip.createGuidedFilter(guide_rgb.astype(np.float32), radius, eps)
            return gf.filter(src.astype(np.float32))
        except Exception:
            d = self.smooth_cfg["bilateral_d"]
            sigmaColor = float(self.smooth_cfg["bilateral_sigma_color"] * 255.0)
            sigmaSpace = float(self.smooth_cfg["bilateral_sigma_space"])
            out = cv2.bilateralFilter(src.astype(np.float32), d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
            out = cv2.bilateralFilter(out, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
            return out

    def _tv_denoise_np(self, img: np.ndarray, weight: float, iters: int) -> np.ndarray:
        """Simple ROF-TV denoise (on [0,1])."""
        u = img.astype(np.float32).copy()
        px = np.zeros_like(u)
        py = np.zeros_like(u)
        tau = 0.125
        lam = float(weight)
        for _ in range(int(iters)):
            ux = np.roll(u, -1, axis=1) - u
            uy = np.roll(u, -1, axis=0) - u
            px = (px + tau * ux) / (1.0 + tau * np.maximum(1e-12, np.abs(ux)))
            py = (py + tau * uy) / (1.0 + tau * np.maximum(1e-12, np.abs(uy)))
            divp = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
            u = np.clip((img + lam * divp), 0.0, 1.0)
        return np.clip(u, 0.0, 1.0)

    def _edge_aware_upsample(self, f_half: torch.Tensor, rgb_full_hw3: torch.Tensor) -> torch.Tensor:
        """
        Upsample half-res f_spec to full-res, refine with guided/joint-bilateral filtering.
        f_half: [Hh, Wh] torch in [0,1]; rgb_full_hw3: [H, W, 3] torch in [0,1]
        """
        fh = f_half.detach().cpu().numpy()
        guide = rgb_full_hw3.detach().cpu().numpy()

        # first, smooth resize
        H, W, _ = guide.shape
        f_up = cv2.resize(fh, (W, H), interpolation=cv2.INTER_CUBIC)

        # guided refinement
        f_up = self._guided_filter_gray(
            guide_rgb=guide,
            src=f_up,
            radius=int(self.smooth_cfg["guided_radius"]),
            eps=float(self.smooth_cfg["guided_eps"]),
        )

        # optional TV
        if self.smooth_cfg.get("use_tv", False):
            f_up = self._tv_denoise_np(
                f_up, weight=float(self.smooth_cfg["tv_weight"]), iters=int(self.smooth_cfg["tv_iters"])
            )

        return torch.from_numpy(np.clip(f_up, 0.0, 1.0))

    # --------------------- dataset plumbing ----------------------

    def _find_scene_pairs(self) -> List[Tuple[str, str, str]]:
        scene_pairs = []
        for scene_name in os.listdir(self.root_dir):
            # Filter by scene_names if provided and not empty
            if self.scene_names is not None and len(self.scene_names) > 0:
                if scene_name not in self.scene_names:
                    continue
            
            scene_path = os.path.join(self.root_dir, scene_name)
            if not os.path.isdir(scene_path):
                continue

            rgb_dir = os.path.join(scene_path, "rgb")
            pol_dir = os.path.join(scene_path, "pol")
            intrinsics_path = os.path.join(scene_path, "intrinsics.txt")
            if not (os.path.exists(rgb_dir) and os.path.exists(pol_dir) and os.path.exists(intrinsics_path)):
                continue

            rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(self.rgb_ext)]
            pol_files = [f for f in os.listdir(pol_dir) if f.endswith(self.pol_ext)]
            for rgb_file in rgb_files:
                pol_file = rgb_file.replace(self.rgb_ext, self.pol_ext)
                if pol_file in pol_files:
                    scene_pairs.append((os.path.join(rgb_dir, rgb_file),
                                        os.path.join(pol_dir, pol_file),
                                        intrinsics_path))
        return scene_pairs

    def __len__(self) -> int:
        return len(self.scene_pairs)
    
    def get_loaded_scenes(self) -> List[str]:
        """Get list of scene names that are actually loaded in the dataset."""
        loaded_scenes = set()
        for rgb_path, _, _ in self.scene_pairs:
            # Extract scene name from path: root_dir/scene_name/rgb/file.png
            scene_name = os.path.basename(os.path.dirname(os.path.dirname(rgb_path)))
            loaded_scenes.add(scene_name)
        return sorted(list(loaded_scenes))

    def _load_intrinsics(self, intrinsics_path: str) -> torch.Tensor:
        try:
            K = np.loadtxt(intrinsics_path).reshape(3, 3).astype(np.float32)
            return torch.from_numpy(K)
        except Exception as e:
            print(f"Warning: Could not load intrinsics from {intrinsics_path}: {e}")
            return torch.eye(3, dtype=torch.float32)

    # -------------------- polarization processing --------------------

    def _load_and_process_polarization(self, pol_path: str) -> Dict[str, torch.Tensor]:
        if POLANALYSER_AVAILABLE:
            return self._load_and_process_polarization_polanalyser(pol_path)
        else:
            return self._load_and_process_polarization_manual(pol_path)

    def _load_and_process_polarization_polanalyser(self, pol_path: str) -> Dict[str, torch.Tensor]:
        # Load composite RGB (0..1), split quadrants
        pol_rgb = np.asarray(Image.open(pol_path).convert("RGB"), dtype=np.float32) / 255.0
        H, W, _ = pol_rgb.shape
        hh, hw = H // 2, W // 2
        I0_rgb   = pol_rgb[0:hh,    0:hw,    :]
        I45_rgb  = pol_rgb[0:hh,    hw:W,    :]
        I90_rgb  = pol_rgb[hh:H,    hw:W,    :]
        I135_rgb = pol_rgb[hh:H,    0:hw,    :]

        # to grayscale (luminance) for Stokes
        I0  = cv2.cvtColor((I0_rgb  * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        I45 = cv2.cvtColor((I45_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        I90 = cv2.cvtColor((I90_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        I135= cv2.cvtColor((I135_rgb* 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # Stokes (half-res)
        ang = np.deg2rad([0, 45, 90, 135])
        stokes3 = pa.calcStokes([I0, I45, I90, I135], ang)   # Hh x Hw x 3
        img_s0, img_s1, img_s2 = cv2.split(stokes3)

        # --- NEW: Stokes prefilter ---
        sg = float(self.smooth_cfg["stokes_gauss_sigma"])
        if sg > 0:
            img_s0 = self._gaussian_blur_np(img_s0, sg)
            img_s1 = self._gaussian_blur_np(img_s1, sg)
            img_s2 = self._gaussian_blur_np(img_s2, sg)

        # DoLP/AoLP from smoothed Stokes
        stokes3_smooth = cv2.merge([img_s0, img_s1, img_s2])
        img_intensity = pa.cvtStokesToIntensity(stokes3_smooth)
        img_dolp      = pa.cvtStokesToDoLP(stokes3_smooth)
        img_aolp      = pa.cvtStokesToAoLP(stokes3_smooth)

        # --- NEW: mask DoLP in low-intensity or tiny values ---
        min_I = float(self.smooth_cfg["dolp_min_intensity"])
        min_d = float(self.smooth_cfg["dolp_min_value"])
        valid = (img_s0 >= min_I).astype(np.float32)
        img_dolp = img_dolp * valid
        img_dolp[img_dolp < min_d] = 0.0

        # 4-component Stokes (S3=0) for extra pol params
        img_s3 = np.zeros_like(img_s0, dtype=np.float32)
        stokes4 = np.stack([img_s0, img_s1, img_s2, img_s3], axis=2)
        img_dop = pa.cvtStokesToDoP(stokes4)
        img_ell = pa.cvtStokesToEllipticityAngle(stokes4)
        img_docp= pa.cvtStokesToDoCP(stokes4)

        # Specular fraction (half-res)
        f_spec = np.clip(img_dolp / max(self.rho_s, 1e-6), 0.0, 1.0)

        # Pack tensors (CxHxW)
        out = {
            'I0':   torch.from_numpy(I0_rgb).permute(2, 0, 1).float(),
            'I45':  torch.from_numpy(I45_rgb).permute(2, 0, 1).float(),
            'I90':  torch.from_numpy(I90_rgb).permute(2, 0, 1).float(),
            'I135': torch.from_numpy(I135_rgb).permute(2, 0, 1).float(),
            'S0':   torch.from_numpy(img_s0)[None].float(),
            'S1':   torch.from_numpy(img_s1)[None].float(),
            'S2':   torch.from_numpy(img_s2)[None].float(),
            'S3':   torch.from_numpy(img_s3)[None].float(),
            'intensity': torch.from_numpy(img_intensity)[None].float(),
            'DoLP':      torch.from_numpy(img_dolp)[None].float(),
            'AoP':       torch.from_numpy(img_aolp)[None].float(),
            'AoLP':      torch.from_numpy(img_aolp)[None].float(),
            'DoP':       torch.from_numpy(img_dop)[None].float(),
            'DoCP':      torch.from_numpy(img_docp)[None].float(),
            'ellipticity_angle': torch.from_numpy(img_ell)[None].float(),
            'f_spec':    torch.from_numpy(f_spec)[None].float(),
        }
        return out

    def _load_and_process_polarization_manual(self, pol_path: str) -> Dict[str, torch.Tensor]:
        # Load composite RGB (0..1) as torch
        pol_rgb = torch.from_numpy(np.asarray(Image.open(pol_path).convert("RGB"), dtype=np.float32)) / 255.0
        H, W, _ = pol_rgb.shape
        hh, hw = H // 2, W // 2
        I0_rgb   = pol_rgb[0:hh,   0:hw,   :]
        I45_rgb  = pol_rgb[0:hh,   hw:W,   :]
        I90_rgb  = pol_rgb[hh:H,   hw:W,   :]
        I135_rgb = pol_rgb[hh:H,   0:hw,   :]

        # luminance for Stokes
        I0   = self._to_luminance(I0_rgb)
        I45  = self._to_luminance(I45_rgb)
        I90  = self._to_luminance(I90_rgb)
        I135 = self._to_luminance(I135_rgb)

        # Stokes (half-res)
        S0 = I0 + I90
        S1 = I0 - I90
        S2 = I45 - I135

        # --- NEW: Stokes prefilter (via numpy/OpenCV for simplicity) ---
        sg = float(self.smooth_cfg["stokes_gauss_sigma"])
        if sg > 0:
            S0_np = self._gaussian_blur_np(S0.cpu().numpy(), sg)
            S1_np = self._gaussian_blur_np(S1.cpu().numpy(), sg)
            S2_np = self._gaussian_blur_np(S2.cpu().numpy(), sg)
            S0, S1, S2 = torch.from_numpy(S0_np), torch.from_numpy(S1_np), torch.from_numpy(S2_np)

        # DoLP / AoP from smoothed Stokes
        R = torch.sqrt(S1 ** 2 + S2 ** 2)
        DoLP = torch.clamp(R / torch.clamp(S0, min=self.eps), 0.0, 1.0)
        AoP  = 0.5 * torch.atan2(S2, S1)

        # --- NEW: mask DoLP in low-intensity / tiny values ---
        min_I = float(self.smooth_cfg["dolp_min_intensity"])
        min_d = float(self.smooth_cfg["dolp_min_value"])
        valid = (S0 >= min_I).float()
        DoLP = DoLP * valid
        DoLP = torch.where(DoLP < min_d, torch.zeros_like(DoLP), DoLP)

        # Consistency extras
        S3 = torch.zeros_like(S0)
        DoP = DoLP.clone()
        DoCP = torch.zeros_like(DoLP)
        ellipticity_angle = torch.zeros_like(DoLP)

        # Specular fraction (half-res)
        f_spec = torch.clamp(DoLP / max(self.rho_s, 1e-6), 0.0, 1.0)

        # Pack CxHxW tensors
        out = {
            'I0':   I0_rgb.permute(2, 0, 1).float(),
            'I45':  I45_rgb.permute(2, 0, 1).float(),
            'I90':  I90_rgb.permute(2, 0, 1).float(),
            'I135': I135_rgb.permute(2, 0, 1).float(),
            'S0':   S0[None].float(),
            'S1':   S1[None].float(),
            'S2':   S2[None].float(),
            'S3':   S3[None].float(),
            'intensity': S0[None].float(),
            'DoLP': DoLP[None].float(),
            'AoP':  AoP[None].float(),
            'AoLP': AoP[None].float(),
            'DoP':  DoP[None].float(),
            'DoCP': DoCP[None].float(),
            'ellipticity_angle': ellipticity_angle[None].float(),
            'f_spec': f_spec[None].float(),
        }
        return out

    # -------------------- RGB + separation (full-res) --------------------

    def _process_rgb_scene(self, rgb_path: str, f_spec_half: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Load full-res RGB and compute specular/diffuse using edge-aware
        upsampled soft specular map.
        """
        rgb = torch.from_numpy(
            np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
        ) / 255.0  # H x W x 3

        # edge-aware upsample of f_spec to full res
        if self.smooth_specular:
            f_full = self._edge_aware_upsample(f_spec_half, rgb)  # H x W
        else:
            H, W, _ = rgb.shape
            f_full = torch.from_numpy(
                cv2.resize(f_spec_half.cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
            )
        f_full = f_full.clamp(0, 1)

        # compute components (no "*3")
        rgb_chw = rgb.permute(2, 0, 1)  # 3 x H x W
        f1 = f_full.unsqueeze(0)        # 1 x H x W
        I_spec = (f1 * rgb_chw).clamp(0, 1)
        I_diff = (rgb_chw - I_spec).clamp(0, 1)

        return {
            'rgb': rgb_chw.float(),
            'specular': I_spec.float(),   # 3 x H x W
            'diffuse': I_diff.float(),    # 3 x H x W
        }

    # ----------------------------- I/O -----------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with polarization (half-res), RGB (full-res),
        specular/diffuse (full-res), and intrinsics.
        """
        rgb_path, pol_path, intrinsics_path = self.scene_pairs[idx]
        intrinsics = self._load_intrinsics(intrinsics_path)

        pol_data = self._load_and_process_polarization(pol_path)
        # pol_data['f_spec'] is [1, Hh, Wh]
        rgb_data = self._process_rgb_scene(rgb_path, pol_data['f_spec'].squeeze(0))

        sample = {**pol_data, **rgb_data, 'intrinsics': intrinsics}
        if self.transform:
            sample = self.transform(sample)
        return sample

    # -------------------- optional visualizations --------------------

    def get_polarization_visualizations(self, idx: int, save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        if not POLANALYSER_AVAILABLE:
            raise ImportError("polanalyser is required for visualizations")

        sample = self[idx]
        dolp = sample['DoLP'].squeeze(0).cpu().numpy()
        aolp = sample['AoLP'].squeeze(0).cpu().numpy()
        dop  = sample.get('DoP', sample['DoLP']).squeeze(0).cpu().numpy()
        docp = sample.get('DoCP', torch.zeros_like(sample['DoLP'])).squeeze(0).cpu().numpy()
        ellipt = sample.get('ellipticity_angle', torch.zeros_like(sample['DoLP'])).squeeze(0).cpu().numpy()

        vis = {
            'dolp_vis': pa.applyColorToDoLP(dolp),
            'aolp_vis': pa.applyColorToAoLP(aolp),
            'aolp_light_vis': pa.applyColorToAoLP(aolp, saturation=dolp, value=1.0),
            'aolp_dark_vis':  pa.applyColorToAoLP(aolp, saturation=dolp, value=0.3),
            'dop_vis': pa.applyColorToDoP(dop),
            'top_vis': pa.applyColorToToP(ellipt, dop),
            'cop_vis': pa.applyColorToCoP(ellipt),
            'docp_vis': pa.applyColorToDoCP(docp),
        }

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            for name, img in vis.items():
                cv2.imwrite(os.path.join(save_path, f"{name}.png"),
                            cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        return vis


# ---------------- convenience: dataloader ----------------

def create_dataloader(
    root_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> DataLoader:
    ds = SCRREAM(root_dir, **dataset_kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
