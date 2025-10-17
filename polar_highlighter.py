import time
from collections import defaultdict
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Note: transformers imports removed as they are not used
from moge.model.v2 import MoGeModel


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
        geometry_model_name="Ruicheng/moge-2-vitb-normal",  # Changed parameter name
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

    def generate_fractal_noise(
        self, B, H, W, device, dtype, octaves=4, persistence=0.5
    ):
        """
        Generate batched, smooth multi-octave noise [B,1,H,W] on the given device.
        Uses bilinear-upsampled Gaussian noise at progressively finer scales.

        Args:
            B: batch size
            H, W: spatial size
            device: torch device
            dtype: tensor dtype
            octaves: number of frequency bands
            persistence: amplitude decay per octave (in (0,1])

        Returns:
            noise: tensor in [0,1], shape [B,1,H,W]
        """
        noise_accum = None
        amplitude = 1.0
        total_amp = 0.0
        for o in range(octaves):
            # Coarse resolution decreases with octave depth (start from H/8, W/8)
            div = 2 ** (o + 3)
            h_low = max(1, H // div)
            w_low = max(1, W // div)
            low = torch.randn(B, 1, h_low, w_low, device=device, dtype=dtype)
            up = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
            if noise_accum is None:
                noise_accum = amplitude * up
            else:
                noise_accum = noise_accum + amplitude * up
            total_amp += amplitude
            amplitude *= persistence

        # Normalize to [0,1]
        noise = noise_accum / max(total_amp, 1e-6)
        # Per-sample min-max normalization for stable range
        minv = noise.amin(dim=(2, 3), keepdim=True)
        maxv = noise.amax(dim=(2, 3), keepdim=True)
        noise = (noise - minv) / (maxv - minv + 1e-6)
        return noise

    def generate_worley_noise(self, B, H, W, device, dtype, cells=32):
        """
        Batched 2D Worley (cellular) noise in [0,1], shape [B,1,H,W].
        Uses a single random feature point per cell and 3x3 neighbor search (with wrap).
        """
        # Pixel coordinates normalized to cell grid space
        ys = torch.linspace(0, cells, steps=H, device=device, dtype=dtype)
        xs = torch.linspace(0, cells, steps=W, device=device, dtype=dtype)
        v, u = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]
        u = u.unsqueeze(0).expand(B, -1, -1)  # [B,H,W]
        v = v.unsqueeze(0).expand(B, -1, -1)  # [B,H,W]

        # Integer cell indices
        cell_x0 = torch.floor(u).to(torch.long).clamp(min=0, max=cells - 1)  # [B,H,W]
        cell_y0 = torch.floor(v).to(torch.long).clamp(min=0, max=cells - 1)  # [B,H,W]

        # Random feature offsets per cell
        feats = torch.rand(B, cells, cells, 2, device=device, dtype=dtype)  # in [0,1)

        # 3x3 neighbor offsets
        d = torch.tensor([-1, 0, 1], device=device)
        dx, dy = torch.meshgrid(d, d, indexing="ij")  # [3,3]
        dx = dx.reshape(9, 1, 1)
        dy = dy.reshape(9, 1, 1)

        # Broadcast indices for neighbor cells with wrap-around
        x_idx = (cell_x0.unsqueeze(0) + dx) % cells  # [9,B,H,W]
        y_idx = (cell_y0.unsqueeze(0) + dy) % cells  # [9,B,H,W]

        # Batch indices for advanced indexing
        b_idx = torch.arange(B, device=device).view(1, B, 1, 1).expand(9, B, H, W)

        # Gather feature offsets for neighbors
        neighbor_feats = feats[b_idx, y_idx, x_idx]  # [9,B,H,W,2]

        # Neighbor feature absolute positions in grid space
        fx = x_idx.to(dtype) + neighbor_feats[..., 0]  # [9,B,H,W]
        fy = y_idx.to(dtype) + neighbor_feats[..., 1]  # [9,B,H,W]

        # Pixel positions
        uu = u.unsqueeze(0)  # [1,B,H,W] -> [9,B,H,W] by broadcast
        vv = v.unsqueeze(0)

        # Distances to neighbor features (Euclidean)
        du = uu - fx
        dv = vv - fy
        dist2 = du * du + dv * dv  # [9,B,H,W]
        dmin = dist2.min(dim=0).values.sqrt()  # [B,H,W]

        # Normalize to [0,1] approximately (sqrt(2) is max in cell space)
        noise = (dmin / (2.0**0.5)).clamp(0.0, 1.0)
        return noise.unsqueeze(1)  # [B,1,H,W]

    def generate_noise_map(
        self,
        B,
        H,
        W,
        device,
        dtype,
        noise_type="fbm",
        octaves=4,
        persistence=0.5,
        cells=32,
    ):
        """
        Unified noise generator.
        noise_type: 'fbm' | 'worley'
        """
        if noise_type == "worley":
            return self.generate_worley_noise(B, H, W, device, dtype, cells=cells)
        # default fbm (fractal value noise)
        return self.generate_fractal_noise(
            B, H, W, device, dtype, octaves=octaves, persistence=persistence
        )

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
        depth = mogeout["depth"]  # [B,H,W]
        normals = mogeout["normal"]  # [B,H,W,3]
        intrinsics = mogeout["intrinsics"].clone()
        intrinsics[:, :2] = mogeout["intrinsics"][:, :2] * 1000
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
        noise=None,
        noise_type="fbm",
        noise_octaves=4,
        noise_persistence=0.5,
        worley_cells=32,
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
            noise: optional noise in [0,1]; if provided, adds spatial jitter to falloff
            noise_octaves: number of noise octaves when jittering
            noise_persistence: amplitude falloff across octaves

        Returns:
            H: [B,1,H,W] highlight luminance
        """
        # Compute half-vector
        h = self.normalize_vector(l + v)  # [B,3,H,W] half-vector
        # Optional noise-driven microfacet normal perturbation to break smoothness
        n_eff = n
        if noise is not None:
            if not torch.is_tensor(noise):
                noise_t = torch.tensor(noise, device=n.device, dtype=n.dtype)
            else:
                noise_t = noise.to(device=n.device, dtype=n.dtype)
            noise_t = torch.clamp(noise_t, 0.0, 1.0)
            if (noise_t > 0).any():
                B, _, H, W = v.shape
                # Tangent random direction per-pixel
                r = torch.randn_like(n)
                r = r - (r * n).sum(1, keepdim=True) * n  # project to tangent
                r = self.normalize_vector(r.clamp(min=-1e6, max=1e6))
                # Blend original normal with tangent noise; beta scales with noise
                beta = 0.8 * noise_t  # stronger effect
                if beta.ndim == 1:
                    beta = beta.view(B, 1, 1, 1)
                n_eff = self.normalize_vector((1.0 - beta) * n + beta * r)

        # Compute n·h with possibly perturbed normals
        nh = (n_eff * h).sum(1, keepdim=True).clamp(0.0, 1.0)
        # Optional noise-driven spatial jitter to reduce smoothness of falloff
        if noise is not None:
            if not torch.is_tensor(noise):
                noise_t = torch.tensor(noise, device=nh.device, dtype=nh.dtype)
            else:
                noise_t = noise.to(device=nh.device, dtype=nh.dtype)
            noise_t = torch.clamp(noise_t, 0.0, 1.0)
            if (noise_t > 0).any():
                B, _, H, W = nh.shape
                noise_map = self.generate_noise_map(
                    B,
                    H,
                    W,
                    nh.device,
                    nh.dtype,
                    noise_type=noise_type,
                    octaves=noise_octaves,
                    persistence=noise_persistence,
                    cells=worley_cells,
                )
                # Map noise in [0,1] to multiplicative jitter around 1.0
                # amplitude scales with noise (max +/-100%)
                jitter_ampl = 1.0 * noise_t
                if jitter_ampl.ndim == 1:
                    jitter_ampl = jitter_ampl.view(B, 1, 1, 1)
                nh = nh * (1.0 + (noise_map - 0.5) * 2.0 * jitter_ampl)
                nh = nh.clamp(0.0, 1.0)

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
        noise=0.0,
        noise_type="fbm",
        noise_octaves=4,
        noise_persistence=0.5,
        worley_cells=32,
    ):
        """
        Synthesize specular highlights and build the corresponding Stokes vector.

        Noise controls (very important):
        - noise: Amplitude in [0,1] that controls two effects simultaneously. Values are
          internally clamped to [0,1]. It can be a scalar, [B], [B,1], [B,1,1,1] or
          a per-pixel map [B,1,H,W]. The same value(s) are used for both:
          1) Microfacet normal perturbation (tangent-space randomization).
          2) Spatial jitter of the (n·h)^α falloff via a procedural noise map.
        - noise_type: Selects the procedural noise used for effect (2):
            'fbm' (default): fractal value noise. Controlled by noise_octaves and
            noise_persistence. worley_cells is ignored.
            'worley': cellular (Voronoi) noise. Controlled by worley_cells only.
            Any other value is treated as 'fbm'. Case-insensitive.
        - noise_octaves: Integer >=1, number of frequency bands for 'fbm'. Ignored for 'worley'.
        - noise_persistence: Float in (0,1], amplitude decay per octave for 'fbm'. Ignored for 'worley'.
        - worley_cells: Integer >=1, number of grid cells per side for 'worley'. Ignored for 'fbm'.

        Args:
            rgb_lin: [B,3,H,W] linear RGB (0-1)
            stokes: [B,3,H,W] input Stokes (S0,S1,S2); only used for shape/context
            depth: [B,1,H,W] metric depth (meters)
            normals: [B,3,H,W] unit surface normals
            K: [B,3,3] camera intrinsics
            light_pos: Optional [B,3] light positions (camera coordinates); if None, sampled
            surface_roughness: Base Blinn-Phong exponent α (higher = sharper peak)
            intensity: Scalar specular scale
            clamp_H: If True, normalizes each image's highlight to [0,1]
            noise: See noise controls above
            noise_type: 'fbm' | 'worley' (see above)
            noise_octaves: See noise controls above (fbm only)
            noise_persistence: See noise controls above (fbm only)
            worley_cells: See noise controls above (worley only)

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

        # 2) Compute effective surface_roughness from noise
        # noise in [0,1]: 0 -> base surface_roughness (sharp), 1 -> min_surface_roughness (broad)
        if not torch.is_tensor(noise):
            noise_t = torch.tensor(noise, device=device, dtype=rgb_lin.dtype)
        else:
            noise_t = noise.to(device=device, dtype=rgb_lin.dtype)
        noise_t = torch.clamp(noise_t, 0.0, 1.0)
        min_surface_roughness = torch.as_tensor(1.0, device=device, dtype=rgb_lin.dtype)
        base_surface_roughness = torch.as_tensor(
            surface_roughness, device=device, dtype=rgb_lin.dtype
        )
        # Quadratic falloff provides perceptual smoothness control
        surface_roughness_eff = min_surface_roughness + (
            base_surface_roughness - min_surface_roughness
        ) * ((1.0 - noise_t) ** 2)
        # 3) Compute Blinn-Phong specular lobe with Fresnel modulation
        H = self.compute_blinn_phong_specular(
            v,
            l,
            n,
            nv,
            surface_roughness_eff,
            intensity,
            self.F0,
            noise=noise,
            noise_type=noise_type,
            noise_octaves=noise_octaves,
            noise_persistence=noise_persistence,
            worley_cells=worley_cells,
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
        noise=0.0,
        noise_type="fbm",
        noise_octaves=4,
        noise_persistence=0.5,
        worley_cells=32,
    ):
        """
        Forward pass for polar highlight synthesis with explicit noise controls.

        Noise system overview and how to control it clearly:
        - noise: Single knob in [0,1] (clamped) that scales BOTH
          1) microfacet normal perturbation, and
          2) spatial jitter of the specular falloff via a procedural noise map.
          0.0 = perfectly smooth mirror-like lobe (uses base surface_roughness),
          1.0 = very rough, strong perturbation and strong spatial variation.
          Accepts scalar, [B], [B,1], [B,1,1,1] or [B,1,H,W].
        - noise_type: Selects the procedural noise for effect (2):
            'fbm' (default): fractal Brownian motion/value noise.
                Controlled by noise_octaves (>=1) and noise_persistence (0,1].
                worley_cells is ignored in this mode.
            'worley': cellular/Voronoi noise.
                Controlled by worley_cells (>=1). noise_octaves and noise_persistence are ignored.
            Any other value is treated as 'fbm' (case-insensitive).
        - noise_octaves: Number of bands for 'fbm'. Ignored for 'worley'.
        - noise_persistence: Amplitude decay per octave for 'fbm'. Ignored for 'worley'.
        - worley_cells: Grid resolution for 'worley'. Ignored for 'fbm'.

        Args:
            rgb: [B,3,H,W] RGB image (0-1 normalized)
            pol: Optional [B,3,H,W] Stokes (S0,S1,S2); if provided, returns stokes_highlighted
            light_pos: Optional [B,3] light positions; sampled if None
            intrinsic: [B,3,3] intrinsics or "compute" to use MoGe intrinsics
            surface_roughness: Base Blinn-Phong exponent α (higher = sharper highlight)
            intensity: Specular strength multiplier
            noise: See Noise system overview above (amplitude and optional map)
            noise_type: 'fbm' | 'worley' (procedural map for spatial jitter)
            noise_octaves: FBM-only bands (>=1); ignored for 'worley'
            noise_persistence: FBM-only amplitude decay (0,1]; ignored for 'worley'
            worley_cells: Worley-only grid cells per side (>=1); ignored for 'fbm'

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
                noise=noise,
                noise_type=noise_type,
                noise_octaves=noise_octaves,
                noise_persistence=noise_persistence,
                worley_cells=worley_cells,
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
