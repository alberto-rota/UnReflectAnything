import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation


class PolarHighlighter(nn.Module):
    def __init__(
        self, depth_estimator="Intel/zoedepth-nyu-kitti", height=852, width=1096
    ):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(depth_estimator)
        self.model = ZoeDepthForDepthEstimation.from_pretrained(depth_estimator)
        self.height = height
        self.width = width
        self.resizer = transforms.Resize((height, width))

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

    def compute_depth(self, image):
        """
        Compute depth map from RGB image.
        image: [B,3,H,W] normalized RGB image (0-1)
        Returns: [B,1,H,W] depth map
        """
        inputs = self.image_processor(images=image * 255, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        # interpolate to original size
        resized_outputs = self.resizer(outputs["predicted_depth"])

        # visualize the prediction
        return resized_outputs.unsqueeze(1)  # [B,1,H,W]

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

    def sample_random_light(
        self, B, radius_range=(0.5, 2.0), z_range=(0.5, 2.0), device="cuda"
    ):
        """
        Sample point lights in camera coords (x,y within sphere shell, z forward).
        Returns L: [B,3] (camera space).
        """
        rmin, rmax = radius_range
        # sample x,y from a disk, z from z_range
        theta = torch.rand(B, device=device) * 2 * math.pi
        r = torch.sqrt(torch.rand(B, device=device)) * rmax
        r = (r - 0) / (rmax - 0) * (rmax - rmin) + rmin  # push away from center
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        z = torch.rand(B, device=device) * (z_range[1] - z_range[0]) + z_range[0]
        L = torch.stack([x, y, z], dim=-1)  # [B,3]
        return L

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

    def synthesize_highlight_and_update_stokes(
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
            gamma_spec: [B,1,H,W] intrinsic specular DoLP
            phi_spec: [B,1,H,W] AoLP (radians)
            light_pos: [B,3] sampled light positions (camera coords)
        """

        B, _, H, W = depth.shape
        device = depth.device

        # 1) Reconstruct 3D points and view/light directions
        P = self.backproject_depth(depth, K)  # [B,3,H,W]
        # Surface->camera direction (propagation toward camera): v
        v = self.normalize_vector(P)  # [B,3,H,W]  (camera at origin)

        # Sample a random camera-space light position
        light_pos = self.sample_random_light(B, device=device)  # [B,3]
        L = light_pos.view(B, 3, 1, 1)  # [B,3,1,1]
        l = self.normalize_vector(L - P)  # [B,3,H,W] (surface->light)

        n = self.normalize_vector(normals)  # [B,3,H,W]
        nl = (n * l).sum(1, keepdim=True).clamp_min(0.0)  # cos(theta_l) [B,1,H,W]
        nv = (n * v).sum(1, keepdim=True).clamp_min(0.0)  # cos(theta_v) [B,1,H,W]

        # 2) Specular lobe (Blinn-Phong + Schlick Fresnel as energy proxy)
        h = self.normalize_vector(l + v)  # [B,3,H,W]
        nh = (n * h).sum(1, keepdim=True).clamp_min(0.0)  # [B,1,H,W]
        spec_lobe = nh**shininess  # [B,1,H,W]
        F = self.schlick_fresnel(nv, F0=F0)  # modulate by Fresnel approx [B,1,H,W]
        H = ks * F * spec_lobe  # highlight luminance proxy [B,1,H,W]
        if clamp_H:
            H = H / (
                H.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            )  # normalize to [0,1]

        # 3) Intrinsic specular DoLP gamma from Fresnel at incidence
        # Incidence angle between l and n:
        theta = torch.acos(nl.clamp(-1 + 1e-7, 1 - 1e-7))  # [B,1,H,W]
        gamma_spec = self.compute_fresnel_gamma(theta, n_rel=n_rel)  # [B,1,H,W]

        # 4) AoLP from geometry (plane of incidence spanned by v & l)
        phi_spec = self.aop_from_geometry(v, l)  # [B,1,H,W]

        # 5) Build highlight Stokes
        S0_H = H  # total highlight intensity [B,1,H,W]
        c2p = torch.cos(2.0 * phi_spec)  # [B,1,H,W]
        s2p = torch.sin(2.0 * phi_spec)  # [B,1,H,W]
        S1_H = gamma_spec * S0_H * c2p  # [B,1,H,W]
        S2_H = gamma_spec * S0_H * s2p  # [B,1,H,W]

        # 6) Update Stokes parameters
        S0, S1, S2 = stokes[:, 0:1], stokes[:, 1:2], stokes[:, 2:3]  # [B,1,H,W] each
        S0_new = S0 + S0_H  # [B,1,H,W]
        S1_new = S1 + S1_H  # [B,1,H,W]
        S2_new = S2 + S2_H  # [B,1,H,W]
        S_new = torch.cat([S0_new, S1_new, S2_new], dim=1)  # [B,3,H,W]

        return H, S_new, gamma_spec, phi_spec, light_pos

    def forward(self, rgb, pol, intrinsic, shininess=80.0, ks=10.0, n_rel=1.5, F0=0.4):
        """
        Forward pass for polar highlight synthesis.

        Args:
            rgb: [B,3,H,W] RGB image (0-1 normalized)
            pol: [B,3,H,W] input polarization Stokes parameters (S0,S1,S2)
            intrinsic: [B,3,3] camera intrinsic matrix
            shininess: specular exponent
            ks: specular strength
            n_rel: relative refractive index
            F0: Fresnel reflectance at normal incidence

        Returns:
            dict with keys:
                'highlight': [B,1,H,W] synthesized highlight
                'stokes_updated': [B,3,H,W] updated Stokes parameters
                'depth': [B,1,H,W] estimated depth
                'normals': [B,3,H,W] surface normals
                'gamma': [B,1,H,W] degree of linear polarization
                'aop': [B,1,H,W] angle of polarization
                'light_pos': [B,3] light positions
        """
        # Ensure tensors are on GPU
        device = rgb.device
        rgb = rgb.to(device)
        pol = pol.to(device)
        intrinsic = intrinsic.to(device)

        # 1) Estimate depth from RGB
        depth = self.compute_depth(rgb)  # [B,1,H,W]

        # 2) Compute surface normals
        normals = self.compute_normals(depth, intrinsic)  # [B,3,H,W]

        # 3) Synthesize highlights and update Stokes parameters
        highlight, stokes_updated, gamma_spec, phi_spec, light_pos = (
            self.synthesize_highlight_and_update_stokes(
                rgb,
                pol,
                depth,
                normals,
                intrinsic,
                shininess=shininess,
                ks=ks,
                n_rel=n_rel,
                F0=F0,
            )
        )

        return {
            "highlight": highlight,
            "stokes_updated": stokes_updated,
            "depth": depth,
            "normals": normals,
            "gamma": gamma_spec,
            "aop": phi_spec,
            "light_pos": light_pos,
        }
