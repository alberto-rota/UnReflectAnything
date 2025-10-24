import random
import time
from collections import defaultdict
from functools import wraps
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Note: transformers imports removed as they are not used
from moge.model.v2 import MoGeModel


# ---------------------------
# helper: smoothstep
def _smoothstep(x, e0, e1):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


@torch.no_grad()
def add_geometric_roughness_torch(
    normals: torch.Tensor,                         # [B,3,H,W], in [-1,1] or [0,1]
    # --- blob controls ---
    n_blobs: int = 16,                             # number of blobs
    avg_blob_size: float = 0.10,                   # avg diameter (fraction of min(H,W) or pixels)
    size_unit: str = "fraction",
    size_spread: float = 0.6,                      # lognormal spread; >0 => many small blobs
    elongation_bias: float = 0.6,                  # 0: circular, 1: very elongated on avg
    falloff_mean: float = 10,                    # mean softness (0=hard edge, 0.5=soft halo)
    falloff_jitter: float = 10,                   # variation of falloff per blob
    edge_wobble: float = 0.6,                      # amplitude of border perturbation (0..1)
    warp_scale: int = 20,                          # spatial scale (px) of border perturbation
    min_separation: float = 0.06,                  # keep centers apart (fraction of min(H,W))
    # --- micro-geometry controls (unchanged) ---
    wavelength_px: float = 12.0,
    wavelength_jitter: float = 0.5,
    orientation_anisotropy: float = 0.4,
    octaves: int = 2,
    roughness_strength: float = 10,              # average angular deviation (radians)
    # misc
    seed = None,
    return_mask: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None]:

    B, C, H, W = normals.shape
    assert C == 3
    dev = normals.device

    g = torch.Generator(device=dev)
    # if seed is None:
    g.manual_seed(random.randint(0, 1000000))
    # else:
        # g.manual_seed(seed)
    # normalize input to [-1,1]
    n = normals.clone()
    if n.min() >= 0:
        n = n * 2 - 1.0
    n = F.normalize(n, dim=1)

    # base grids
    yy, xx = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij"
    )

    # fbm-like noise field for warping (edge perturbations)
    def _smooth_noise(scale=28):
        h0, w0 = max(2, H // scale), max(2, W // scale)
        base = torch.randn(1, 2, h0, w0, generator=g, device=dev)  # 2 channels -> 2D warp
        field = F.interpolate(base, size=(H, W), mode="bicubic", align_corners=False)
        return field  # [1,2,H,W]

    warp_field = _smooth_noise(scale=max(6, warp_scale))  # shared statistics
    warp_field = warp_field / (warp_field.std(dim=(-2, -1), keepdim=True) + 1e-8)  # normalize

    # ---------- soft, perturbed super-ellipse blobs ----------
    def make_soft_blob_mask() -> torch.Tensor:
        mask = torch.zeros(1, H, W, device=dev)
        d_mean = (avg_blob_size * min(H, W)) if size_unit == "fraction" else float(avg_blob_size)
        d_mean = max(4.0, d_mean)
        min_sep_px = max(4.0, min_separation * min(H, W))

        centers = []
        tries = 0
        while len(centers) < n_blobs and tries < 6000:
            tries += 1
            cx = torch.empty((), device=dev).uniform_(0, W, generator=g).item()
            cy = torch.empty((), device=dev).uniform_(0, H, generator=g).item()
            if all(((cx-px)**2 + (cy-py)**2)**0.5 >= min_sep_px for px,py in centers):
                centers.append((cx, cy))

        for (cx, cy) in centers:
            # sample size from lognormal (many small, some large)
            ln_sigma = size_spread * 0.5  # softer control
            s = torch.exp(torch.empty((), device=dev).normal_(0, ln_sigma, generator=g))  # ~lognormal
            D = d_mean * s
            # elongation: sample axis ratio biased to small values
            r = torch.clamp(1.0 - torch.empty((), device=dev).uniform_(0, elongation_bias, generator=g), 0.15, 1.0)
            a = D * 0.5                      # major semi-axis
            b = D * 0.5 * r                  # minor semi-axis
            theta = torch.empty((), device=dev).uniform_(0, 3.1416, generator=g)
            p = torch.empty((), device=dev).uniform_(1.8, 3.2, generator=g)  # super-ellipse exponent

            # coordinate warp for irregularity
            wob_amp = edge_wobble * 0.35 * D  # scale by size so small blobs aren't shredded
            X = xx + wob_amp * warp_field[:, 0] - cx
            Y = yy + wob_amp * warp_field[:, 1] - cy

            ct, st = torch.cos(theta), torch.sin(theta)
            xr =  ct * X + st * Y
            yr = -st * X + ct * Y

            # super-ellipse implicit metric f= (|xr/a|^p + |yr/b|^p)
            f = torch.pow(torch.abs(xr) / (a + 1e-8), p) + torch.pow(torch.abs(yr) / (b + 1e-8), p)

            # soft falloff via smoothstep around f=1 with randomized width
            soft = (falloff_mean * (1.0 + torch.empty((), device=dev).uniform_(-falloff_jitter, falloff_jitter, generator=g))).clamp(0.05, 0.7)
            edge0 = 1.0 - soft   # inside value to start softening
            edge1 = 1.0 + soft   # outside value where it goes to 0
            blob = (1.0 - _smoothstep(f, edge0, edge1)).clamp(0, 1)  # 1 at core -> 0 outside
            blob = blob.unsqueeze(0)

            mask = (mask + blob).clamp(0, 1)

        # mild blur to merge micro-holes but keep shapes
        mask = F.gaussian_blur(mask, (5, 5), sigma=(1.2, 1.2)) if hasattr(F, "gaussian_blur") else mask
        return mask  # [1,H,W]

    mask = torch.cat([make_soft_blob_mask() for _ in range(B)], dim=0)  # [B,1,H,W]

    # ---------- micro-geometry (same as before) ----------
    # band-limited Gabor-like height
    yyn, xxn = torch.meshgrid(
        torch.linspace(-1, 1, H, device=dev),
        torch.linspace(-1, 1, W, device=dev),
        indexing="ij",
    )
    height = torch.zeros(B, 1, H, W, device=dev)
    for o in range(octaves):
        lam = wavelength_px * (0.5 ** o)
        base_freq = 1.0 / max(lam, 1.0)
        dom = torch.empty(B, device=dev).uniform_(0, 3.1416, generator=g)
        theta = dom + torch.empty(B, device=dev).uniform_(-3.1416*orientation_anisotropy*0.25,
                                                          3.1416*orientation_anisotropy*0.25, generator=g)
        freq = base_freq * torch.clamp(1.0 + torch.empty(B, device=dev).uniform_(-wavelength_jitter, wavelength_jitter, generator=g), 0.25, 4.0)
        phase = torch.empty(B, device=dev).uniform_(0, 6.2832, generator=g)
        for b in range(B):
            ct, st = torch.cos(theta[b]), torch.sin(theta[b])
            u = ct * xxn + st * yyn
            height[b:b+1] += torch.sin(2*3.1416*freq[b] * u + phase[b]) * (1.0 / (2**o))
    height = height - height.mean(dim=(-2,-1), keepdim=True)
    height = F.gaussian_blur(height, (5,5), sigma=(1.0,1.0)) if hasattr(F,"gaussian_blur") else height
    height = height * mask

    # slopes
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=dev, dtype=torch.float32).view(1,1,3,3)/8.0
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], device=dev, dtype=torch.float32).view(1,1,3,3)/8.0
    dhdx = F.conv2d(height, kx, padding=1)
    dhdy = F.conv2d(height, ky, padding=1)

    # TBN from n
    ref = torch.tensor([0.0,0.0,1.0], device=dev).view(1,3,1,1).expand(B,-1,H,W)
    parallel = (torch.abs((n*ref).sum(1,keepdim=True)) > 0.99)
    ref = torch.where(parallel, torch.tensor([0.0,1.0,0.0], device=dev).view(1,3,1,1), ref)
    t = F.normalize(torch.cross(ref, n, dim=1), dim=1)
    b = F.normalize(torch.cross(n, t, dim=1), dim=1)

    # unscaled perturbation from slopes
    offset = (-dhdx) * t + (-dhdy) * b

    # scale so average deviation ≈ roughness_strength (radians)
    test = F.normalize(n + offset, dim=1)
    cosang = (n * test).sum(1).clamp(-1, 1)
    mean_angle = torch.acos(cosang).mean().detach().item() + 1e-8
    scale = roughness_strength / mean_angle
    n_pert = F.normalize(n + offset * scale, dim=1)

    # blend by soft mask
    noisy = F.normalize(torch.lerp(n, n_pert, mask), dim=1)
    return (noisy, mask) if return_mask else (noisy, None)


def time_module(module_name):
    """
    Decorator to time method execution when timing is enabled.

    Args:
        module_name: String identifier for the module being timed

    Usage:
        @time_module("depth_estimation")
        def compute_depth(self, image):
            # method implementation
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, "_timing_enabled", False):
                return func(self, *args, **kwargs)

            # Ensure GPU operations are complete before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            try:
                result = func(self, *args, **kwargs)
            finally:
                # Ensure GPU operations are complete after timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                elapsed = (end_time - start_time) * 1000  # Convert to milliseconds
                self.timing_results[module_name].append(elapsed)

            return result

        return wrapper

    return decorator


class PolarHighlighter(nn.Module):
    def __init__(
        self,
        geometry_model_name="Ruicheng/moge-2-vits-normal",  # Changed parameter name
        height=852,
        width=1096,
        enable_timing=False,
        n_rel=1.5,
        F0=0.4,
    ):
        super().__init__()
        # Load MoGe model for joint depth and normal estimation
        self.geometry_model = MoGeModel.from_pretrained(geometry_model_name)
        self.geometry_model.eval()  # Set to eval mode
        self.height = height
        self.width = width
        self.resizer = transforms.Resize((height, width))

        # Material properties
        self.n_rel = n_rel
        self.F0 = F0

        # Timing functionality
        self._timing_enabled = enable_timing
        self.timing_results = defaultdict(list)

    def enable_timing_mode(self, enabled=True):
        """Enable or disable timing mode."""
        self._timing_enabled = enabled

    def reset_timing_stats(self):
        """Clear all timing statistics."""
        self.timing_results.clear()

    def get_timing_stats(self, detailed=False):
        """
        Get timing statistics for all modules.

        Args:
            detailed: If True, return all measurements. If False, return summary stats.

        Returns:
            dict: Timing statistics in milliseconds
        """
        if not self.timing_results:
            return {}

        if detailed:
            return dict(self.timing_results)

        # Return summary statistics
        stats = {}
        for module_name, times in self.timing_results.items():
            times_array = np.array(times)
            stats[module_name] = {
                "mean_ms": float(np.mean(times_array)),
                "std_ms": float(np.std(times_array)),
                "min_ms": float(np.min(times_array)),
                "max_ms": float(np.max(times_array)),
                "total_ms": float(np.sum(times_array)),
                "count": len(times),
            }
        return stats

    def print_timing_stats(self):
        """Print a formatted timing report."""
        stats = self.get_timing_stats()
        if not stats:
            print("No timing data available. Enable timing mode first.")
            return

        print("\n" + "=" * 60)
        print("PolarHighlighter Timing Report")
        print("=" * 60)
        print(f"{'Module':<25} {'Mean (ms)':<10} {'Std (ms)':<10} {'Count':<8}")
        print("-" * 60)

        total_time = 0
        for module_name, module_stats in sorted(stats.items()):
            mean_time = module_stats["mean_ms"]
            std_time = module_stats["std_ms"]
            count = module_stats["count"]
            total_time += module_stats["total_ms"]

            print(f"{module_name:<25} {mean_time:<10.2f} {std_time:<10.2f} {count:<8}")

        print("-" * 60)
        print(f"{'Total time':<25} {total_time:<10.2f} ms")
        print("=" * 60)

    def make_pixel_grid(self, B, H, W, device):
        """Return homogeneous pixel grid [B,3,H,W] with x,y in pixels, 1's row."""
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        ones = torch.ones_like(xs, dtype=torch.float32)
        pix = torch.stack([xs.float(), ys.float(), ones], dim=0)  # [3,H,W]
        return pix.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,3,H,W]

    def backproject_depth(self, depth, K):
        """
        Backproject to 3D camera coordinates.
        depth: [B,1,H,W] (meters)
        K:     [B,3,3] intrinsics
        Returns P_cam: [B,3,H,W]
        """
        B, _, H, W = depth.shape
        grid = self.make_pixel_grid(B, H, W, depth.device)  # [B,3,H,W]
        # Sanitize intrinsics and use pseudo-inverse for numerical stability
        K = torch.nan_to_num(K, nan=0.0, posinf=1e6, neginf=-1e6)
        Kinv = torch.linalg.pinv(K)  # [B,3,3]
        rays = (Kinv @ grid.flatten(2)).view(B, 3, H, W)  # [B,3,H,W]
        # Camera looks along +Z; 3D point = ray * depth
        P = rays * depth  # [B,3,H,W]
        return P

    def normalize_vector(self, v, eps=1e-8):
        """Normalize vectors along dim=1"""
        return v / (v.norm(dim=1, keepdim=True).clamp_min(eps))







    def _gaussian_kernel1d(self, sigma: float, dtype, device):
        radius = max(int(3.0 * sigma + 0.5), 1)
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        w = torch.exp(-(x * x) / (2.0 * sigma * sigma))
        w = w / (w.sum() + 1e-12)
        return w.view(1, 1, -1)

    def gaussian_blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Separable Gaussian blur for [B,C,H,W]. If sigma<=0, returns x.
        """
        if sigma is None or sigma <= 0:
            return x
        B, C, H, W = x.shape
        k1d = self._gaussian_kernel1d(float(sigma), x.dtype, x.device)  # [1,1,K]
        # Horizontal
        pad = (k1d.shape[-1] // 2)
        w_h = k1d.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode='reflect'), w_h, groups=C)
        # Vertical
        w_v = k1d.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode='reflect'), w_v, groups=C)
        return x

    @time_module("depth_estimation")
    @time_module("geometry_estimation")
    def compute_geometry(self, image):
        """
        Compute depth and normals from RGB image using MoGe.

        Args:
            image: [B,3,H,W] RGB image (0-1 normalized)

        Returns:
            depth: [B,1,H,W] depth map
            normals: [B,3,H,W] surface normals
        """
        with torch.no_grad():
            mogeout = self.geometry_model.infer(image)  # image: [B,3,H,W]

        # Extract depth [B,H,W] and normals [B,H,W,3]
        depth = mogeout["depth"]   # [B,H,W]
        normals = mogeout["normal"]  # [B,H,W,3]
        intrinsics = mogeout["intrinsics"].clone()
        intrinsics[:, 0, 2] = image.shape[3] / 2  # cx (width/2)
        intrinsics[:, 1, 2] = image.shape[2] / 2  # cy (height/2)
        intrinsics[:, :2, :2] = intrinsics[:, :2, :2] * 200
        # Sanitize potential NaN/Inf from the geometry model
        depth = torch.nan_to_num(depth, nan=0.0, posinf=1e6, neginf=-1e6)
        normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
        intrinsics = torch.nan_to_num(intrinsics, nan=0.0, posinf=1e6, neginf=-1e6)

        # Resize to target dimensions if needed
        if depth.shape[-2:] != (self.height, self.width):
            depth = self.resizer(depth.unsqueeze(1)).squeeze(1)  # [B,H,W]
            # Resize normals - reshape for resize then back
            normals = normals.permute(0, 3, 1, 2)  # [B,3,H,W]
            normals = self.resizer(normals)  # [B,3,H,W]
            normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
            normals = F.normalize(
                normals, p=2, dim=1, eps=1e-6
            )  # Re-normalize after resize
        else:
            normals = normals.permute(0, 3, 1, 2)  # [B,3,H,W]
            normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
            normals = F.normalize(normals, p=2, dim=1, eps=1e-6)

        # Add channel dimension to depth
        depth = depth.unsqueeze(1)  # [B,1,H,W]

        return depth, -normals, intrinsics

    def sample_light_source(
        self,
        dist_to_camera,
        left_right_angle=None,
        above_below_angle=None,
        batch_size=1,
        device="cuda",
    ):
        """
        Sample random light source positions in 3D space using spherical coordinates with intuitive angle controls.
        Camera coordinate system: Z points forward, Y points down, X points right.

        Spherical coordinates: (r, θ, φ) where:
        - r = distance from camera
        - θ = azimuth angle (left-right rotation around Y axis)
        - φ = elevation angle (above-below rotation from horizontal plane)

        Args:
            dist_to_camera: tuple (min_dist, max_dist) - range of distances from camera in meters
            left_right_angle: tuple (min_left, max_right) - horizontal angle range in degrees
                             Negative values = left of camera, positive = right of camera
                             If None, defaults to (-180, 180) degrees (full horizontal range)
            above_below_angle: tuple (min_above, max_below) - vertical angle range in degrees
                              Negative values = above camera, positive = below camera
                              If None, defaults to (-90, 90) degrees (full vertical range)
            batch_size: number of light positions to generate
            device: torch device for the output tensor

        Returns:
            positions: Light source positions in camera space [batch_size,3]
        """
        # Set default angle ranges if not provided
        if left_right_angle is None:
            left_right_angle = (-180, 180)  # Full horizontal range
        if above_below_angle is None:
            above_below_angle = (-90, 90)  # Full vertical range

        # Unpack ranges
        min_dist, max_dist = dist_to_camera
        min_left, max_right = left_right_angle
        min_above, max_below = above_below_angle

        # Sample random values within ranges
        # Distance: uniform sampling
        dist = (
            torch.rand(batch_size, device=device) * (max_dist - min_dist) + min_dist
        )  # [B]

        # Azimuth (left-right): uniform sampling in degrees, then convert to radians
        az_deg = (
            torch.rand(batch_size, device=device) * (max_right - min_left)
            + min_left
            - 90
        )  # [B]
        az_rad = az_deg * (np.pi / 180.0)  # [B]

        # Elevation (above-below): uniform sampling in degrees, then convert to radians
        elev_deg = (
            torch.rand(batch_size, device=device) * (max_below - min_above)
            + min_above
            - 90
        )  # [B]
        elev_rad = elev_deg * (np.pi / 180.0)  # [B]

        # Convert spherical to Cartesian coordinates
        # Camera coordinate system: Z forward, Y down, X right
        x = dist * torch.cos(elev_rad) * torch.sin(az_rad)  # [B] left-right
        y = -dist * torch.sin(
            elev_rad
        )  # [B] above-below (negative for upward elevation)
        z = dist * torch.cos(elev_rad) * torch.cos(az_rad)  # [B] behind-front

        # Stack into position tensor
        positions = torch.stack([x, y, z], dim=-1)  # [B,3]

        return positions

    def compute_fresnel_gamma(self, theta, n_rel=1.5, eps=1e-8):
        """
        Intrinsic DoLP of specular for unpolarized incident light at incidence theta.
        gamma = |Rs - Rp| / (Rs + Rp), with Snell.

        Args:
            theta: [B,1,H,W] incidence angle in radians
            n_rel: relative refractive index n2/n1
            eps: small value for numerical stability

        Returns:
            gamma: [B,1,H,W] degree of linear polarization in [0,1]
        """
        n1 = 1.0
        n2 = n_rel
        sin_t = (n1 / n2) * torch.sin(theta).clamp(-1 + 1e-7, 1 - 1e-7)
        theta_t = torch.asin(sin_t)
        cos_t = torch.cos(theta)
        cos_tt = torch.cos(theta_t)

        # Fresnel reflectances for s (perp) and p (parallel)
        Rs_num = n2 * cos_t - n1 * cos_tt
        Rs_den = n2 * cos_t + n1 * cos_tt
        Rp_num = n1 * cos_t - n2 * cos_tt
        Rp_den = n1 * cos_t + n2 * cos_tt

        Rs = (Rs_num / (Rs_den.clamp_min(eps))) ** 2
        Rp = (Rp_num / (Rp_den.clamp_min(eps))) ** 2

        gamma = (Rs - Rp).abs() / (Rs + Rp + eps)
        return gamma.clamp(0.0, 1.0)

    def schlick_fresnel(self, cos_theta, F0=0.04):
        """
        Scalar Schlick Fresnel for intensity modulation (approximate).
        cos_theta: [B,1,H,W]
        F0: fresnel reflectance at normal incidence
        """
        return F0 + (1 - F0) * (1 - cos_theta).clamp(0, 1) ** 5

    def aop_from_geometry(self, v, l):
        """
        Compute angle of linear polarization (AoLP) for specular:
        polarization dir is perpendicular to the plane of incidence ~ k = v x l

        Args:
            v: [B,3,H,W] surface->camera unit vectors
            l: [B,3,H,W] surface->light unit vectors

        Returns:
            phi: [B,1,H,W] AoLP in radians, reference x-axis is camera +X
        """
        k = torch.cross(v, l, dim=1)  # [B,3,H,W]
        k = self.normalize_vector(k)  # ensure unit
        ex = k[:, 0:1]  # x-component [B,1,H,W]
        ey = k[:, 1:2]  # y-component [B,1,H,W]
        phi = torch.atan2(ey, ex)  # [-pi,pi] [B,1,H,W]
        return phi

    @time_module("viewing_lighting_geometry")
    def compute_viewing_lighting_geometry(self, depth, normals, K, light_pos):
        """
        Compute viewing and lighting geometry for specular reflection.
        Physics: v = P/|P| (view direction), l = (L-P)/|L-P| (light direction), where P are 3D surface points.
        The geometry determines the reflection conditions via the surface normal n.

        Args:
            depth: [B,1,H,W] depth map in meters
            normals: [B,3,H,W] surface normals
            light_pos: [B,3] light positions
            K: [B,3,3] camera intrinsics

        Returns:
            tuple: (v, l, n, nl, nv, light_pos) - view dirs, light dirs, normals, dot products, light positions
        """
        B, _, H, W = depth.shape

        # Reconstruct 3D points and view direction
        P = self.backproject_depth(depth, K)  # [B,3,H,W]
        v = self.normalize_vector(P)  # [B,3,H,W] surface->camera direction

        # Sample random light position and compute light direction
        L = light_pos.view(B, 3, 1, 1)  # [B,3,1,1]
        l = self.normalize_vector(L - P)  # [B,3,H,W] surface->light direction

        # Normalize surface normals
        n = self.normalize_vector(normals)  # [B,3,H,W]

        # Compute dot products for lighting calculations
        nl = (n * l).sum(1, keepdim=True).clamp_min(0.0)  # cos(theta_l) [B,1,H,W]
        nv = (n * v).sum(1, keepdim=True).clamp_min(0.0)  # cos(theta_v) [B,1,H,W]

        return v, l, n, nl, nv, light_pos, P

    @time_module("blinn_phong_specular")
    def compute_blinn_phong_specular(
        self,
        v,
        l,
        n,
        nv,
        surface_roughness,
        intensity,
        F0,
    ):
        """
        Compute Blinn-Phong specular lobe with Schlick Fresnel approximation.
        Physics: I_spec = intensity * F(θ) * (n·h)^α, where h = (v+l)/|v+l| is the half-vector.
        Fresnel term F(θ) ≈ F0 + (1-F0)(1-cosθ)^5 modulates reflection strength.

        Args:
            v: [B,3,H,W] view directions
            l: [B,3,H,W] light directions
            n: [B,3,H,W] surface normals
            nv: [B,1,H,W] n·v dot product
            surface_roughness: specular exponent α
            intensity: specular strength
            F0: Fresnel reflectance at normal incidence

        Returns:
            H: [B,1,H,W] highlight luminance
        """
        # Compute half-vector
        h = self.normalize_vector(l + v)  # [B,3,H,W] half-vector
        
        # Compute n·h
        nh = (n * h).sum(1, keepdim=True).clamp(0.0, 1.0)

        # Allow surface_roughness to be scalar or broadcastable tensor
        if torch.is_tensor(surface_roughness):
            if surface_roughness.ndim == 1:
                # [B] -> [B,1,1,1]
                surface_roughness = surface_roughness.view(
                    surface_roughness.shape[0], 1, 1, 1
                )
            elif surface_roughness.ndim == 2:
                # [B,1] -> [B,1,1,1]
                surface_roughness = surface_roughness.view(
                    surface_roughness.shape[0], 1, 1, 1
                )
            # else: assume broadcastable to [B,1,H,W]
        spec_lobe = nh**surface_roughness  # [B,1,H,W]

        # Apply Schlick Fresnel approximation
        F = self.schlick_fresnel(nv, F0=F0)  # [B,1,H,W]
        H = intensity * F * spec_lobe  # [B,1,H,W]
        return H

    @time_module("polarization_parameters")
    def compute_polarization_parameters(self, v, l, nl, n_rel):
        """
        Compute polarization parameters from Fresnel reflection theory.
        Physics: DoLP γ = |Rs-Rp|/(Rs+Rp) from Fresnel equations, AoLP φ from plane of incidence.
        The plane of incidence is spanned by incident and reflected rays, perpendicular to v×l.

        Args:
            v: [B,3,H,W] view directions
            l: [B,3,H,W] light directions
            nl: [B,1,H,W] n·l dot product
            n_rel: relative refractive index

        Returns:
            tuple: (gamma_spec, phi_spec) - degree and angle of linear polarization
        """
        # Compute incidence angle and intrinsic DoLP from Fresnel theory
        theta = torch.acos(nl.clamp(-1 + 1e-7, 1 - 1e-7))  # [B,1,H,W]
        gamma_spec = self.compute_fresnel_gamma(theta, n_rel=n_rel)  # [B,1,H,W]

        # Compute AoLP from geometry (plane of incidence)
        phi_spec = self.aop_from_geometry(v, l)  # [B,1,H,W]

        return gamma_spec, phi_spec

    @time_module("highlight_stokes_vector")
    def compute_highlight_stokes_vector(self, H, gamma_spec, phi_spec):
        """
        Construct Stokes vector for specular highlight.
        Physics: S = [S0, S1, S2] where S1 = γS0cos(2φ), S2 = γS0sin(2φ).
        This represents partially linearly polarized light with DoLP γ and AoLP φ.

        Args:
            H: [B,1,H,W] highlight intensity (S0)
            gamma_spec: [B,1,H,W] degree of linear polarization
            phi_spec: [B,1,H,W] angle of linear polarization

        Returns:
            tuple: (S0_H, S1_H, S2_H) - highlight Stokes components
        """
        S0_H = H  # total highlight intensity [B,1,H,W]
        c2p = torch.cos(2.0 * phi_spec)  # [B,1,H,W]
        s2p = torch.sin(2.0 * phi_spec)  # [B,1,H,W]
        S1_H = gamma_spec * S0_H * c2p  # [B,1,H,W]
        S2_H = gamma_spec * S0_H * s2p  # [B,1,H,W]

        return S0_H, S1_H, S2_H

    def update_stokes_with_highlight(self, stokes, S0_H, S1_H, S2_H):
        """
        Add highlight contribution to existing Stokes parameters.
        Physics: Incoherent addition S_total = S_scene + S_highlight for independent sources.
        Each Stokes component adds linearly for incoherent superposition.

        Args:
            stokes: [B,3,H,W] input Stokes parameters (S0,S1,S2)
            S0_H, S1_H, S2_H: [B,1,H,W] highlight Stokes components

        Returns:
            S_new: [B,3,H,W] updated Stokes parameters
        """
        S0, S1, S2 = stokes[:, 0:1], stokes[:, 1:2], stokes[:, 2:3]  # [B,1,H,W] each
        S0_new = S0 + S0_H  # [B,1,H,W]
        S1_new = S1 + S1_H  # [B,1,H,W]
        S2_new = S2 + S2_H  # [B,1,H,W]
        S_new = torch.cat([S0_new, S1_new, S2_new], dim=1)  # [B,3,H,W]

        return S_new

    def update_rgb_with_highlight(self, rgb, H, intensity=1.0):
        """
        Add highlight contribution to existing RGB image.
        Physics: Incoherent addition R_total = R_scene + R_highlight for independent sources.
        Each RGB channel adds linearly for incoherent superposition.
        Values are clamped to [0,1] to prevent overflow in standard RGB representation.
        """
        # H is [B,1,H,W], expand to match RGB channels [B,3,H,W]
        H_rgb = H.expand(-1, 3, -1, -1)  # [B,3,H,W]
        R = rgb + intensity * H_rgb  # [B,3,H,W] - additive highlight contribution
        R = torch.clamp(R, 0.0, 1.0)  # Clamp to valid RGB range [0,1]
        return R

    @time_module("highlight_synthesis")
    def synthesize_highlight_with_stokes(
        self,
        rgb_lin,
        stokes,
        depth,
        normals,
        K,
        light_pos=None,
        surface_roughness=64.0,
        intensity=1.0,
        clamp_H=True,
    ):
        """
        Synthesize specular highlights and build the corresponding Stokes vector.

        Args:
            rgb_lin: [B,3,H,W] linear RGB (0-1)
            stokes: [B,3,H,W] input Stokes (S0,S1,S2); only used for shape/context
            depth: [B,1,H,W] metric depth (meters)
            normals: [B,3,H,W] unit surface normals
            K: [B,3,3] camera intrinsics
            light_pos: Optional [B,3] light positions (camera coordinates); if None, sampled
            surface_roughness: Base Blinn-Phong exponent α (higher = sharper highlight)
            intensity: Scalar specular scale
            clamp_H: If True, normalizes each image's highlight to [0,1]

        Returns:
            H: [B,1,H,W] highlight luminance
            H_stokes: [B,3,H,W] highlight Stokes (S0,S1,S2)
            H_aop: [B,1,H,W] angle of polarization (radians)
            H_dop: [B,1,H,W] degree of linear polarization [0,1]
            light_pos: [B,3] light positions used
            pcloud: [B,3,H,W] reconstructed 3D points
            l: [B,3,H,W] light directions
            v: [B,3,H,W] view directions
        """
        B, _, H, W = rgb_lin.shape
        device = rgb_lin.device
        # 0) Sample random light positions (one per image)
        if light_pos is None:
            light_pos = self.sample_light_source(
                dist_to_camera=(0.3, 1),  # 0.3m to 1m from camera
                left_right_angle=(-110, 110),  # 110° left to 110° right
                above_below_angle=(-90, 90),  # 90° above to 90° below
                batch_size=B,
                device=device,
            )  # [B,3]

        # 1) Compute viewing and lighting geometry
        v, l, n, nl, nv, light_pos, pcloud = self.compute_viewing_lighting_geometry(
            depth, normals, K, light_pos
        )

        # 2) Compute Blinn-Phong specular lobe with Fresnel modulation
        H = self.compute_blinn_phong_specular(
            v,
            l,
            n,
            nv,
            surface_roughness,
            intensity,
            self.F0,
        )
        if clamp_H:
            H = H / (
                H.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            )  # normalize to [0,1]

        # 4) Compute polarization parameters from Fresnel theory
        H_dop, H_aop = self.compute_polarization_parameters(v, l, nl, self.n_rel)

        # 5) Build highlight Stokes vector
        S0_H, S1_H, S2_H = self.compute_highlight_stokes_vector(H, H_dop, H_aop)
        H_stokes = torch.cat([S0_H, S1_H, S2_H], dim=1)

        return H, H_stokes, H_aop, H_dop, light_pos, pcloud, l, v

    # @time_module("forward_pass")
    def forward(
        self,
        rgb,
        pol=None,
        light_pos=None,
        intrinsic="compute",
        surface_roughness=80.0,
        intensity=10.0,
        # Geometric roughness parameters
        n_blobs=16,
        avg_blob_size=0.10,
        size_unit="fraction",
        size_spread=0.6,
        elongation_bias=0.6,
        falloff_mean=10,
        falloff_jitter=10,
        edge_wobble=0.6,
        warp_scale=20,
        min_separation=0.06,
        wavelength_px=12.0,
        wavelength_jitter=0.5,
        orientation_anisotropy=0.4,
        octaves=2,
        roughness_strength=10,
        seed=0,
    ):
        """
        Forward pass for polar highlight synthesis with geometric roughness.

        Args:
            rgb: [B,3,H,W] RGB image (0-1 normalized)
            pol: Optional [B,3,H,W] Stokes (S0,S1,S2); if provided, returns stokes_highlighted
            light_pos: Optional [B,3] light positions; sampled if None
            intrinsic: [B,3,3] intrinsics or "compute" to use MoGe intrinsics
            surface_roughness: Base Blinn-Phong exponent α (higher = sharper highlight)
            intensity: Specular strength multiplier
            # Geometric roughness parameters
            n_blobs: number of blobs
            avg_blob_size: avg diameter (fraction of min(H,W) or pixels)
            size_unit: "fraction" or "pixels"
            size_spread: lognormal spread; >0 => many small blobs
            elongation_bias: 0: circular, 1: very elongated on avg
            falloff_mean: mean softness (0=hard edge, 0.5=soft halo)
            falloff_jitter: variation of falloff per blob
            edge_wobble: amplitude of border perturbation (0..1)
            warp_scale: spatial scale (px) of border perturbation
            min_separation: keep centers apart (fraction of min(H,W))
            wavelength_px: wavelength in pixels
            wavelength_jitter: variation in wavelength
            orientation_anisotropy: anisotropy factor
            octaves: number of octaves for micro-geometry
            roughness_strength: average angular deviation (radians)
            seed: random seed

        Returns:
            dict with keys:
                'highlight': [B,1,H,W]
                'rgb_highlighted': [B,3,H,W]
                'stokes_highlight': [B,3,H,W]
                'stokes_highlighted': Optional [B,3,H,W] if pol provided
                'depth': [B,1,H,W]
                'normals': [B,3,H,W]
                'H_dop': [B,1,H,W]
                'H_aop': [B,1,H,W]
                'intrinsic': [B,3,3]
                'light_pos': [B,3]
                'pcloud': [B,3,H,W]
                'light_dir': [B,3,H,W]
                'view_dir': [B,3,H,W]
        """
        # Ensure tensors are on GPU
        device = rgb.device
        rgb = rgb.to(device)
        if pol is not None:
            pol = pol.to(device)

        depth, normals, moge_intrinsics = self.compute_geometry(
            rgb
        )  # [B,1,H,W], [B,3,H,W]

        # Apply geometric roughness to normals
        normals, _ = add_geometric_roughness_torch(
            normals,
            n_blobs=n_blobs,
            avg_blob_size=avg_blob_size,
            size_unit=size_unit,
            size_spread=size_spread,
            elongation_bias=elongation_bias,
            falloff_mean=falloff_mean,
            falloff_jitter=falloff_jitter,
            edge_wobble=edge_wobble,
            warp_scale=warp_scale,
            min_separation=min_separation,
            wavelength_px=wavelength_px,
            wavelength_jitter=wavelength_jitter,
            orientation_anisotropy=orientation_anisotropy,
            octaves=octaves,
            roughness_strength=roughness_strength,
            seed=seed,
            return_mask=False,
        )

        if intrinsic == "compute":
            intrinsic = moge_intrinsics.to(device)
        else:
            intrinsic = intrinsic.to(device)

        # 3) Synthesize highlights and update Stokes parameters
        # Use dummy pol if not provided for the synthesis function
        pol_input = pol if pol is not None else torch.zeros_like(rgb)
        H, H_stokes, H_aop, H_dop, light_pos_random, pcloud, light_dir, view_dir = (
            self.synthesize_highlight_with_stokes(
                rgb,
                pol_input,
                depth,
                normals,
                intrinsic,
                light_pos=light_pos,
                surface_roughness=surface_roughness,
                intensity=intensity,
            )
        )

        # 4) Update scene Stokes parameters with highlight contribution (only if pol provided)
        result = {
            "highlight": H,
            "rgb_highlighted": self.update_rgb_with_highlight(
                rgb, H, intensity=intensity
            ),
            "stokes_highlight": H_stokes,
            "depth": depth,
            "normals": normals,
            "H_dop": H_dop,
            "H_aop": H_aop,
            "intrinsic": intrinsic,
            "light_pos": light_pos_random if light_pos is None else light_pos,
            "pcloud": pcloud,
            "light_dir": light_dir,
            "view_dir": view_dir,
        }

        # Only compute stokes_highlighted if pol was provided
        if pol is not None:
            S0_H, S1_H, S2_H = H_stokes[:, 0:1], H_stokes[:, 1:2], H_stokes[:, 2:3]
            stokes_updated = self.update_stokes_with_highlight(pol, S0_H, S1_H, S2_H)
            result["stokes_highlighted"] = stokes_updated
        else:
            result["stokes_highlighted"] = None

        return result


def get_soft_highlight_map(
    rgb_image: torch.Tensor, threshold: float = 0.7
) -> torch.Tensor:
    """
    Create a soft map of highlights from an RGB image.

    Args:
        rgb_image: Input RGB image tensor of shape [B, 3, H, W]
        threshold: Threshold value for highlight detection (default: 0.7)

    Returns:
        Soft highlight map tensor of shape [B, 1, H, W]
    """
    # Create highlight mask by averaging across color channels (dim=1)
    is_highlight = (
        rgb_image.mean(dim=1, keepdim=True) >= threshold
    )  # Shape: [B, 1, H, W]

    # Create soft highlights by subtracting threshold and masking non-highlights
    soft_highlights = rgb_image - threshold
    soft_highlights[torch.logical_not(is_highlight).repeat(1, 3, 1, 1)] = 0.0

    scaler = 1 / (1 - threshold)
    # Return the mean across color channels to get single-channel highlight map
    return soft_highlights.mean(dim=1, keepdim=True) * scaler  # Shape: [B, 1, H, W]
