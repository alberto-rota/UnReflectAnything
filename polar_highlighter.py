import time
from collections import defaultdict
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import (
    AutoImageProcessor,
    DPTForDepthEstimation,
)
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
    ):
        super().__init__()
        # Load MoGe model for joint depth and normal estimation
        self.geometry_model = MoGeModel.from_pretrained(geometry_model_name)
        self.geometry_model.eval()  # Set to eval mode
        self.height = height
        self.width = width
        self.resizer = transforms.Resize((height, width))

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
        Kinv = torch.inverse(K)  # [B,3,3]
        rays = (Kinv @ grid.flatten(2)).view(B, 3, H, W)  # [B,3,H,W]
        # Camera looks along +Z; 3D point = ray * depth
        P = rays * depth  # [B,3,H,W]
        return P

    def normalize_vector(self, v, eps=1e-8):
        """Normalize vectors along dim=1"""
        return v / (v.norm(dim=1, keepdim=True).clamp_min(eps))

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
    
        # Resize to target dimensions if needed
        if depth.shape[-2:] != (self.height, self.width):
            depth = self.resizer(depth.unsqueeze(1)).squeeze(1)  # [B,H,W]
            # Resize normals - reshape for resize then back
            normals = normals.permute(0, 3, 1, 2)  # [B,3,H,W]
            normals = self.resizer(normals)  # [B,3,H,W]
            normals = F.normalize(normals, p=2, dim=1, eps=1e-6)  # Re-normalize after resize
        else:
            normals = normals.permute(0, 3, 1, 2)  # [B,3,H,W]
    
        # Add channel dimension to depth
        depth = depth.unsqueeze(1)  # [B,1,H,W]
    
        return depth, -normals

    def sample_light_source(
        self, dist_to_camera, azimuth, elevation, batch_size=1, device="cuda"
    ):
        """
        Sample random light source positions in 3D space using spherical coordinates.
        Camera coordinate system: Z points forward, Y points down, X points right.

        Args:
            dist_to_camera: tuple (min_dist, max_dist) - range of signed distances from camera in meters
            azimuth: tuple (min_az, max_az) - horizontal angle range in degrees
            elevation: tuple (min_elev, max_elev) - vertical angle range in degrees
            batch_size: number of light positions to generate
            device: torch device for the output tensor

        Returns:
            positions: Light source positions in camera space [batch_size,3]
        """
        # Unpack ranges
        min_dist, max_dist = dist_to_camera
        min_az, max_az = azimuth
        min_elev, max_elev = elevation

        # Sample random values within ranges
        # Distance: uniform sampling (can be negative for behind camera)
        dist = (
            torch.rand(batch_size, device=device) * (max_dist - min_dist) + min_dist
        )  # [B]

        # Azimuth: uniform sampling in degrees, then convert to radians
        az_deg = (
            torch.rand(batch_size, device=device) * (max_az - min_az) + min_az
        )  # [B]
        az_rad = az_deg * (np.pi / 180.0)  # [B]

        # Elevation: uniform sampling in degrees, then convert to radians
        elev_deg = (
            torch.rand(batch_size, device=device) * (max_elev - min_elev) + min_elev
        )  # [B]
        elev_rad = elev_deg * (np.pi / 180.0)  # [B]

        # Convert spherical to Cartesian coordinates
        x = dist * torch.cos(elev_rad) * torch.sin(az_rad)  # [B]
        y = -dist * torch.sin(elev_rad)  # [B] negative for upward elevation
        z = dist * torch.cos(elev_rad) * torch.cos(az_rad)  # [B] signed distance

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
    def compute_blinn_phong_specular(self, v, l, n, nv, shininess, ks, F0):
        """
        Compute Blinn-Phong specular lobe with Schlick Fresnel approximation.
        Physics: I_spec = ks * F(θ) * (n·h)^α, where h = (v+l)/|v+l| is the half-vector.
        Fresnel term F(θ) ≈ F0 + (1-F0)(1-cosθ)^5 modulates reflection strength.

        Args:
            v: [B,3,H,W] view directions
            l: [B,3,H,W] light directions
            n: [B,3,H,W] surface normals
            nv: [B,1,H,W] n·v dot product
            shininess: specular exponent α
            ks: specular strength
            F0: Fresnel reflectance at normal incidence

        Returns:
            H: [B,1,H,W] highlight luminance
        """
        # Compute half-vector and specular lobe
        h = self.normalize_vector(l + v)  # [B,3,H,W] half-vector
        nh = (n * h).sum(1, keepdim=True).clamp_min(0.0)  # [B,1,H,W]
        spec_lobe = nh**shininess  # [B,1,H,W]

        # Apply Schlick Fresnel approximation
        F = self.schlick_fresnel(nv, F0=F0)  # [B,1,H,W]
        H = ks * F * spec_lobe  # [B,1,H,W]

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

    def update_rgb_with_highlight(self, rgb, H):
        """
        Add highlight contribution to existing RGB image.
        Physics: Incoherent addition R_total = R_scene + R_highlight for independent sources.
        Each RGB channel adds linearly for incoherent superposition.
        Values are clamped to [0,1] to prevent overflow in standard RGB representation.
        """
        # H is [B,1,H,W], expand to match RGB channels [B,3,H,W]
        H_rgb = H.expand(-1, 3, -1, -1)  # [B,3,H,W]
        R = rgb + H_rgb  # [B,3,H,W] - additive highlight contribution
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
        shininess=64.0,
        ks=1.0,
        n_rel=1.5,
        F0=0.04,
        clamp_H=True,
    ):
        """
        Synthesize specular highlights and update Stokes parameters.

        Args:
            rgb_lin: [B,3,H,W] linear RGB
            stokes: [B,3,H,W] input Stokes (S0,S1,S2)
            depth: [B,1,H,W] depth map in meters
            normals: [B,3,H,W] surface normals
            K: [B,3,3] camera intrinsics
            shininess: specular exponent
            ks: specular strength
            n_rel: relative refractive index
            F0: Fresnel reflectance at normal incidence
            clamp_H: whether to normalize highlight intensity

        Returns:
            H: [B,1,H,W] highlight luminance
            S_new: [B,3,H,W] updated Stokes after adding highlight
            H_dop: [B,1,H,W] intrinsic specular DoLP
            H_aop: [B,1,H,W] AoLP (radians)
            phi_spec: [B,1,H,W] AoLP (radians)
            light_pos: [B,3] sampled light positions (camera coords)
        """
        B, _, H, W = rgb_lin.shape
        device = rgb_lin.device
        # 0) Sample random light positions (one per image)
        light_pos = self.sample_light_source(
            (-100, -10), (-110, 110), (-90,90), batch_size=B, device=device
        )  # [B,3]

        # 1) Compute viewing and lighting geometry
        v, l, n, nl, nv, light_pos, pcloud = self.compute_viewing_lighting_geometry(
            depth, normals, K, light_pos
        )

        # 2) Compute Blinn-Phong specular lobe with Fresnel modulation
        H = self.compute_blinn_phong_specular(v, l, n, nv, shininess, ks, F0)
        if clamp_H:
            H = H / (
                H.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            )  # normalize to [0,1]

        # 3) Compute polarization parameters from Fresnel theory
        H_dop, H_aop = self.compute_polarization_parameters(v, l, nl, n_rel)

        # 4) Build highlight Stokes vector
        S0_H, S1_H, S2_H = self.compute_highlight_stokes_vector(H, H_dop, H_aop)
        H_stokes = torch.cat([S0_H, S1_H, S2_H], dim=1)

        return H, H_stokes, H_aop, H_dop, light_pos, pcloud, l, v

    # @time_module("forward_pass")
    def forward(self, rgb, pol=None, light_pos=None, intrinsic=None, shininess=80.0, ks=10.0, n_rel=1.5, F0=0.4, return_light_pos=False):
        """
        Forward pass for polar highlight synthesis.

        Args:
            rgb: [B,3,H,W] RGB image (0-1 normalized)
            pol: [B,3,H,W] input polarization Stokes parameters (S0,S1,S2), optional
            intrinsic: [B,3,3] camera intrinsic matrix
            shininess: specular exponent
            ks: specular strength   
            n_rel: relative refractive index
            F0: Fresnel reflectance at normal incidence

        Returns:
            dict with keys:
                'highlight': [B,1,H,W] synthesized highlight
                'stokes_updated': [B,3,H,W] updated Stokes parameters (if pol provided)
                'depth': [B,1,H,W] estimated depth
                'normals': [B,3,H,W] surface normals
                'gamma': [B,1,H,W] degree of linear polarization
                'aop': [B,1,H,W] angle of polarization
                'light_pos': [B,3] light positions
        """
        # Ensure tensors are on GPU
        device = rgb.device
        rgb = rgb.to(device)
        if pol is not None:
            pol = pol.to(device)
        if intrinsic is not None:
            intrinsic = intrinsic.to(device)

        depth, normals = self.compute_geometry(rgb)  # [B,1,H,W], [B,3,H,W]


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
                shininess=shininess,
                ks=ks,
                n_rel=n_rel,
                F0=F0,
            )
        )

        # 4) Update scene Stokes parameters with highlight contribution (only if pol provided)
        result = {
            "highlight": H,
            "rgb_highlighted": self.update_rgb_with_highlight(rgb, H),
            "stokes_highlight": H_stokes,
            "depth": depth,
            "normals": normals,
            "H_dop": H_dop,
            "H_aop": H_aop,
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

def get_soft_highlight_map(rgb_image: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
    """
    Create a soft map of highlights from an RGB image.
    
    Args:
        rgb_image: Input RGB image tensor of shape [B, 3, H, W]
        threshold: Threshold value for highlight detection (default: 0.7)
    
    Returns:
        Soft highlight map tensor of shape [B, 1, H, W]
    """
    # Create highlight mask by averaging across color channels (dim=1)
    is_highlight = (rgb_image.mean(dim=1, keepdim=True) >= threshold)  # Shape: [B, 1, H, W]
    
    # Create soft highlights by subtracting threshold and masking non-highlights
    soft_highlights = rgb_image - threshold
    soft_highlights[torch.logical_not(is_highlight).repeat(1, 3, 1, 1)] = 0.0
    
    scaler = 1/(1-threshold)
    # Return the mean across color channels to get single-channel highlight map
    return soft_highlights.mean(dim=1, keepdim=True) * scaler  # Shape: [B, 1, H, W]

class PolarHighlighter_DEPRECATED(nn.Module):
    def __init__(
        self,
        depth_estimator="Intel/dpt-large",
        height=852,
        width=1096,
        enable_timing=False,
    ):
        super().__init__()
        self.depth_image_processor = AutoImageProcessor.from_pretrained(depth_estimator,use_fast=True)
        self.depth_image_processor.do_rescale = False
        self.depth_model = DPTForDepthEstimation.from_pretrained(depth_estimator)
        self.height = height
        self.width = width
        self.resizer = transforms.Resize((height, width))

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
        Kinv = torch.inverse(K)  # [B,3,3]
        rays = (Kinv @ grid.flatten(2)).view(B, 3, H, W)  # [B,3,H,W]
        # Camera looks along +Z; 3D point = ray * depth
        P = rays * depth  # [B,3,H,W]
        return P

    def normalize_vector(self, v, eps=1e-8):
        """Normalize vectors along dim=1"""
        return v / (v.norm(dim=1, keepdim=True).clamp_min(eps))

    @time_module("depth_estimation")
    def compute_depth(self, image):
        """
        Compute depth map from RGB image.
        image: [B,3,H,W] normalized RGB image (0-1)
        Returns: [B,1,H,W] depth map
        """
        inputs = self.depth_image_processor(images=image, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.depth_model(**inputs)

        # interpolate to original size
        resized_outputs = self.resizer(outputs["predicted_depth"])
        inverted_depth = 1 / resized_outputs.clamp(min=1e-6) * 300 + 50
        # visualize the prediction
        return inverted_depth.unsqueeze(1)  # [B,1,H,W]

    @time_module("normal_computation")
    def compute_normals(self, depth, intrinsics):
        """
        Compute surface normals from metric depth map.

        Args:
            depth: Batched depth map of shape [B,1,H,W]
            intrinsics: Batched camera intrinsics of shape [B,3,3]

        Returns:
            normals: Surface normal map of shape [B,3,H,W]
        """
        B, _, H, W = depth.shape
        device = depth.device

        # Create pixel grid (H, W, 2)
        y, x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing="ij",
        )

        # Stack to get pixel coordinates (H, W, 3) with homogeneous coord
        pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # [H,W,3]
        pixels = pixels.unsqueeze(0).expand(B, -1, -1, -1)  # [B,H,W,3]

        # Reshape for batch matrix multiplication
        pixels_flat = pixels.reshape(B, H * W, 3).transpose(1, 2)  # [B,3,H*W]

        # Compute inverse intrinsics
        intrinsics_inv = torch.inverse(intrinsics)  # [B,3,3]

        # Backproject to normalized camera coordinates
        rays = torch.bmm(intrinsics_inv, pixels_flat)  # [B,3,H*W]
        rays = rays.transpose(1, 2).reshape(B, H, W, 3)  # [B,H,W,3]

        # Multiply by depth to get 3D points
        depth_squeezed = depth.squeeze(1).unsqueeze(-1)  # [B,H,W,1]
        points = rays * depth_squeezed  # [B,H,W,3]

        # Rearrange to [B,3,H,W] for easier gradient computation
        points = points.permute(0, 3, 1, 2)  # [B,3,H,W]

        # Compute gradients using finite differences
        # Pad the points to handle boundaries
        points_padded = F.pad(points, (1, 1, 1, 1), mode="replicate")  # [B,3,H+2,W+2]

        # Compute differences in x direction (right - left)
        dx = points_padded[:, :, 1:-1, 2:] - points_padded[:, :, 1:-1, :-2]  # [B,3,H,W]
        dx = dx / 2.0  # Central difference

        # Compute differences in y direction (bottom - top)
        dy = points_padded[:, :, 2:, 1:-1] - points_padded[:, :, :-2, 1:-1]  # [B,3,H,W]
        dy = dy / 2.0  # Central difference

        # Compute cross product to get normals
        # Normal = dx × dy
        normals = torch.cross(
            dx.permute(0, 2, 3, 1), dy.permute(0, 2, 3, 1), dim=-1
        )  # [B,H,W,3]

        # Permute back to [B,3,H,W]
        normals = normals.permute(0, 3, 1, 2)  # [B,3,H,W]

        # Normalize the normals
        normals = F.normalize(normals, p=2, dim=1, eps=1e-6)  # [B,3,H,W]

        # Handle invalid depths (set normals to [0,0,-1] for invalid pixels)
        valid_mask = (depth > 0).float()  # [B,1,H,W]
        default_normal = torch.tensor([0.0, 0.0, -1.0], device=device).reshape(
            1, 3, 1, 1
        )
        normals = normals * valid_mask + default_normal * (1 - valid_mask)  # [B,3,H,W]

        return normals

    def sample_light_source(
        self, dist_to_camera, azimuth, elevation, batch_size=1, device="cuda"
    ):
        """
        Sample random light source positions in 3D space using spherical coordinates.
        Camera coordinate system: Z points forward, Y points down, X points right.

        Args:
            dist_to_camera: tuple (min_dist, max_dist) - range of signed distances from camera in meters
            azimuth: tuple (min_az, max_az) - horizontal angle range in degrees
            elevation: tuple (min_elev, max_elev) - vertical angle range in degrees
            batch_size: number of light positions to generate
            device: torch device for the output tensor

        Returns:
            positions: Light source positions in camera space [batch_size,3]
        """
        # Unpack ranges
        min_dist, max_dist = dist_to_camera
        min_az, max_az = azimuth
        min_elev, max_elev = elevation

        # Sample random values within ranges
        # Distance: uniform sampling (can be negative for behind camera)
        dist = (
            torch.rand(batch_size, device=device) * (max_dist - min_dist) + min_dist
        )  # [B]

        # Azimuth: uniform sampling in degrees, then convert to radians
        az_deg = (
            torch.rand(batch_size, device=device) * (max_az - min_az) + min_az
        )  # [B]
        az_rad = az_deg * (np.pi / 180.0)  # [B]

        # Elevation: uniform sampling in degrees, then convert to radians
        elev_deg = (
            torch.rand(batch_size, device=device) * (max_elev - min_elev) + min_elev
        )  # [B]
        elev_rad = elev_deg * (np.pi / 180.0)  # [B]

        # Convert spherical to Cartesian coordinates
        x = dist * torch.cos(elev_rad) * torch.sin(az_rad)  # [B]
        y = -dist * torch.sin(elev_rad)  # [B] negative for upward elevation
        z = dist * torch.cos(elev_rad) * torch.cos(az_rad)  # [B] signed distance

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
    def compute_blinn_phong_specular(self, v, l, n, nv, shininess, ks, F0):
        """
        Compute Blinn-Phong specular lobe with Schlick Fresnel approximation.
        Physics: I_spec = ks * F(θ) * (n·h)^α, where h = (v+l)/|v+l| is the half-vector.
        Fresnel term F(θ) ≈ F0 + (1-F0)(1-cosθ)^5 modulates reflection strength.

        Args:
            v: [B,3,H,W] view directions
            l: [B,3,H,W] light directions
            n: [B,3,H,W] surface normals
            nv: [B,1,H,W] n·v dot product
            shininess: specular exponent α
            ks: specular strength
            F0: Fresnel reflectance at normal incidence

        Returns:
            H: [B,1,H,W] highlight luminance
        """
        # Compute half-vector and specular lobe
        h = self.normalize_vector(l + v)  # [B,3,H,W] half-vector
        nh = (n * h).sum(1, keepdim=True).clamp_min(0.0)  # [B,1,H,W]
        spec_lobe = nh**shininess  # [B,1,H,W]

        # Apply Schlick Fresnel approximation
        F = self.schlick_fresnel(nv, F0=F0)  # [B,1,H,W]
        H = ks * F * spec_lobe  # [B,1,H,W]

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

    def update_rgb_with_highlight(self, rgb, H):
        """
        Add highlight contribution to existing RGB image.
        Physics: Incoherent addition R_total = R_scene + R_highlight for independent sources.
        Each RGB channel adds linearly for incoherent superposition.
        Values are clamped to [0,1] to prevent overflow in standard RGB representation.
        """
        # H is [B,1,H,W], expand to match RGB channels [B,3,H,W]
        H_rgb = H.expand(-1, 3, -1, -1)  # [B,3,H,W]
        R = rgb + H_rgb  # [B,3,H,W] - additive highlight contribution
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
        shininess=64.0,
        ks=1.0,
        n_rel=1.5,
        F0=0.04,
        clamp_H=True,
    ):
        """
        Synthesize specular highlights and update Stokes parameters.

        Args:
            rgb_lin: [B,3,H,W] linear RGB
            stokes: [B,3,H,W] input Stokes (S0,S1,S2)
            depth: [B,1,H,W] depth map in meters
            normals: [B,3,H,W] surface normals
            K: [B,3,3] camera intrinsics
            shininess: specular exponent
            ks: specular strength
            n_rel: relative refractive index
            F0: Fresnel reflectance at normal incidence
            clamp_H: whether to normalize highlight intensity

        Returns:
            H: [B,1,H,W] highlight luminance
            S_new: [B,3,H,W] updated Stokes after adding highlight
            H_dop: [B,1,H,W] intrinsic specular DoLP
            H_aop: [B,1,H,W] AoLP (radians)
            phi_spec: [B,1,H,W] AoLP (radians)
            light_pos: [B,3] sampled light positions (camera coords)
        """
        B, _, H, W = rgb_lin.shape
        device = rgb_lin.device
        # 0) Sample random light positions (one per image)
        light_pos = self.sample_light_source(
            (-100, -10), (-110, 110), (-90,90), batch_size=B, device=device
        )  # [B,3]

        # 1) Compute viewing and lighting geometry
        v, l, n, nl, nv, light_pos, pcloud = self.compute_viewing_lighting_geometry(
            depth, normals, K, light_pos
        )

        # 2) Compute Blinn-Phong specular lobe with Fresnel modulation
        H = self.compute_blinn_phong_specular(v, l, n, nv, shininess, ks, F0)
        if clamp_H:
            H = H / (
                H.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            )  # normalize to [0,1]

        # 3) Compute polarization parameters from Fresnel theory
        H_dop, H_aop = self.compute_polarization_parameters(v, l, nl, n_rel)

        # 4) Build highlight Stokes vector
        S0_H, S1_H, S2_H = self.compute_highlight_stokes_vector(H, H_dop, H_aop)
        H_stokes = torch.cat([S0_H, S1_H, S2_H], dim=1)

        return H, H_stokes, H_aop, H_dop, light_pos, pcloud, l, v

    # @time_module("forward_pass")
    def forward(self, rgb, pol=None, intrinsic=None, shininess=80.0, ks=10.0, n_rel=1.5, F0=0.4):
        """
        Forward pass for polar highlight synthesis.

        Args:
            rgb: [B,3,H,W] RGB image (0-1 normalized)
            pol: [B,3,H,W] input polarization Stokes parameters (S0,S1,S2), optional
            intrinsic: [B,3,3] camera intrinsic matrix
            shininess: specular exponent
            ks: specular strength   
            n_rel: relative refractive index
            F0: Fresnel reflectance at normal incidence

        Returns:
            dict with keys:
                'highlight': [B,1,H,W] synthesized highlight
                'stokes_updated': [B,3,H,W] updated Stokes parameters (if pol provided)
                'depth': [B,1,H,W] estimated depth
                'normals': [B,3,H,W] surface normals
                'gamma': [B,1,H,W] degree of linear polarization
                'aop': [B,1,H,W] angle of polarization
                'light_pos': [B,3] light positions
        """
        # Ensure tensors are on GPU
        device = rgb.device
        rgb = rgb.to(device)
        if pol is not None:
            pol = pol.to(device)
        if intrinsic is not None:
            intrinsic = intrinsic.to(device)

        # 1) Estimate depth from RGB
        depth = self.compute_depth(rgb)  # [B,1,H,W]

        # 2) Compute surface normals
        normals = self.compute_normals(depth, intrinsic)  # [B,3,H,W]

        # 3) Synthesize highlights and update Stokes parameters
        # Use dummy pol if not provided for the synthesis function
        pol_input = pol if pol is not None else torch.zeros_like(rgb)
        H, H_stokes, H_aop, H_dop, light_pos, pcloud, light_dir, view_dir = (
            self.synthesize_highlight_with_stokes(
                rgb,
                pol_input,
                depth,
                normals,
                intrinsic,
                shininess=shininess,
                ks=ks,
                n_rel=n_rel,
                F0=F0,
            )
        )

        # 4) Update scene Stokes parameters with highlight contribution (only if pol provided)
        result = {
            "highlight": H,
            "rgb_highlighted": self.update_rgb_with_highlight(rgb, H),
            "stokes_highlight": H_stokes,
            "depth": depth,
            "normals": normals,
            "H_dop": H_dop,
            "H_aop": H_aop,
            "light_pos": light_pos,
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

def get_soft_highlight_map(rgb_image: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
    """
    Create a soft map of highlights from an RGB image.
    
    Args:
        rgb_image: Input RGB image tensor of shape [B, 3, H, W]
        threshold: Threshold value for highlight detection (default: 0.7)
    
    Returns:
        Soft highlight map tensor of shape [B, 1, H, W]
    """
    # Create highlight mask by averaging across color channels (dim=1)
    is_highlight = (rgb_image.mean(dim=1, keepdim=True) >= threshold)  # Shape: [B, 1, H, W]
    
    # Create soft highlights by subtracting threshold and masking non-highlights
    soft_highlights = rgb_image - threshold
    soft_highlights[torch.logical_not(is_highlight).repeat(1, 3, 1, 1)] = 0.0
    
    scaler = 1/(1-threshold)
    # Return the mean across color channels to get single-channel highlight map
    return soft_highlights.mean(dim=1, keepdim=True) * scaler  # Shape: [B, 1, H, W]