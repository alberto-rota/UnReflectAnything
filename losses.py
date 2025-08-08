# -------------------------------------------------------------------------------------------------#

""" Copyright (c) 2024 Asensus Surgical """

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from rich.tree import Tree
from rich import print as rprint


# -------------------------------------------------------------------------------------------
# BASE LOSS FUNCTIONS
# These loss functions are the lowest leveels one and take in inputs directly. They are not
# combinations of other losses
# -------------------------------------------------------------------------------------------
class TripletLoss(nn.Module):
    def __init__(self, margin=1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def dist(self, A, B):
        cosine_sim = nn.functional.cosine_similarity(A, B)
        return torch.sqrt(torch.clamp(2 - 2 * cosine_sim, min=0))

    def forward(self, A, P, N):
        return torch.mean(
            torch.clamp(self.margin + self.dist(A, P) - self.dist(A, N), min=0)
        )


class EpipolarLoss(nn.Module):
    """PyTorch module implementing robust symmetric epipolar distance loss.

    This loss computes the robust symmetric epipolar distance between corresponding points
    in two images given a predicted fundamental matrix.
    """

    def __init__(self, gamma=0.5):
        """Initialize the loss module.

        Args:
            gamma (float): Robust parameter for clamping the loss. Defaults to 0.5.
        """
        super().__init__()
        self.gamma = gamma

    def _symmetric_epipolar_distance(self, pts1, pts2, F):
        """Compute symmetric epipolar distance.

        Args:
            pts1 (torch.Tensor): Points in first image (B, N, 3)
            pts2 (torch.Tensor): Points in second image (B, N, 3)
            F (torch.Tensor): Fundamental matrix (B, 3, 3)

        Returns:
            torch.Tensor: Symmetric epipolar distance (B, N)
        """

        # Normalize points to [-1, 1]
        pts1 = pts1 / 192 - 1
        pts2 = pts2 / 192 - 1
        # COnvert points to homogeneous coordinates
        pts1 = torch.cat([pts1, torch.ones_like(pts1[:, :, :1])], dim=2)  # (B, N, 3)
        pts2 = torch.cat([pts2, torch.ones_like(pts2[:, :, :1])], dim=2)  # (B, N, 3)

        # Compute epipolar lines
        line1 = torch.bmm(pts1, F)  # (B, N, 3)
        line2 = torch.bmm(pts2, F.permute(0, 2, 1))  # (B, N, 3)

        # Compute scalar product
        scalar_product = (pts2 * line1).sum(2)  # (B, N)

        # Compute normalized distance
        norm_term = 1 / line1[:, :, :2].norm(2, 2) + 1 / line2[:, :, :2].norm(  # (B, N)
            2, 2
        )  # (B, N)

        return scalar_product.abs() * norm_term

    def forward(self, F_pred, pts1, pts2, gamma=None):
        """Forward pass computing the loss.

        Args:
            F_pred (torch.Tensor): Predicted fundamental matrix (B, 3, 3)
            pts1 (torch.Tensor): Points in first image (B, N, 3)
            pts2 (torch.Tensor): Points in second image (B, N, 3)
            gamma (float, optional): Override default gamma value. Defaults to None.

        Returns:
            torch.Tensor: Mean robust symmetric epipolar distance
        """
        if gamma is None:
            gamma = self.gamma

        # Compute symmetric epipolar distance
        sed = self._symmetric_epipolar_distance(pts1, pts2, F_pred)

        # Apply robust clamping
        loss = torch.clamp(sed, max=gamma)

        # Return mean loss
        return loss.mean()


class InliersLoss(nn.Module):
    def __init__(self, threshold=0.75):
        super(InliersLoss, self).__init__()
        self.threshold = threshold
        self.eps = 1e-12

    def forward(self, F_pred, F_gt, pts1, pts2):
        """
        Compute F1 score based on inlier classification.

        pts1: Tensor of shape (B, N, 2) - pixel coordinates in image 1
        pts2: Tensor of shape (B, N, 2) - pixel coordinates in image 2
        F_pred: Tensor of shape (B, 3, 3) - predicted fundamental matrix
        F_gt: Tensor of shape (B, 3, 3) - ground truth fundamental matrix
        """
        # Convert pixel coordinates to homogeneous coordinates
        batch_size, num_pts, _ = pts1.size()
        hom_pts1 = torch.cat(
            [pts1, torch.ones(batch_size, num_pts, 1, device=pts1.device)], dim=2
        )  # (B, N, 3)
        hom_pts2 = torch.cat(
            [pts2, torch.ones(batch_size, num_pts, 1, device=pts2.device)], dim=2
        )  # (B, N, 3)

        def epipolar_error(hom_pts1, hom_pts2, F):
            """Compute symmetric epipolar error for a batch."""
            Ft_pts2 = torch.bmm(
                F.transpose(1, 2), hom_pts2.transpose(1, 2)
            )  # (B, 3, N)
            F_pts1 = torch.bmm(F, hom_pts1.transpose(1, 2))  # (B, 3, N)

            res = 1 / (Ft_pts2[:, :2, :].norm(dim=1) + self.eps)  # (B, N)
            res += 1 / (F_pts1[:, :2, :].norm(dim=1) + self.eps)  # (B, N)
            res *= torch.abs(
                torch.sum(
                    hom_pts2 * torch.bmm(F, hom_pts1.transpose(1, 2)).transpose(1, 2),
                    dim=2,
                )
            )  # (B, N)
            return res

        # Compute epipolar errors
        est_res = epipolar_error(hom_pts1, hom_pts2, F_pred)  # (B, N)
        gt_res = epipolar_error(hom_pts1, hom_pts2, F_gt)  # (B, N)

        # Determine inliers
        est_inliers = est_res < self.threshold  # (B, N)
        gt_inliers = gt_res < self.threshold  # (B, N)
        true_positives = est_inliers & gt_inliers  # (B, N)

        gt_inlier_count = gt_inliers.float().sum(dim=1)  # (B,)
        est_inlier_count = est_inliers.float().sum(dim=1)  # (B,)
        true_positive_count = true_positives.float().sum(dim=1)  # (B,)

        # Precision and recall
        precision = true_positive_count / (est_inlier_count + self.eps)  # (B,)
        recall = true_positive_count / (gt_inlier_count + self.eps)  # (B,)

        # F1 score
        f1_score = 2 * precision * recall / (precision + recall + self.eps)  # (B,)

        return 1 - f1_score.mean()


class SoftRankLoss(nn.Module):

    def __init__(self):
        super(SoftRankLoss, self).__init__()

    def forward(self, F, delta=1e-6):
        # Ensure F is positive definite by computing F^T F
        F_t_F = torch.bmm(F.transpose(1, 2), F)

        # Add delta * I to ensure numerical stability
        batch_size, n, _ = F_t_F.shape
        identity = delta * torch.eye(n, device=F.device).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        F_t_F_stable = F_t_F + identity

        # Compute log-det loss
        log_det = torch.linalg.slogdet(F_t_F_stable).logabsdet
        loss = torch.sum(log_det)

        return loss


class SSIMLoss(nn.Module):
    """
    ------------------------------------------------------------------------------------------------
    Copyright of this class fully belongs to Shuwei Shao @ https://github.com/ShuweiShao/AF-SfMLearner
    -------------------------------------------------------------------------------------------------
    Layer to compute the SSIMLoss loss between a pair of images
    """

    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, target: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        x = self.refl(target)
        y = self.refl(warped)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1).mean()

    def __str__(self):
        return "SSIMLoss()"


class L1Loss(nn.Module):
    """L1 Loss."""

    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, target: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        """Compute the L1 loss.

        Args:
            target (torch.Tensor): Target image.
            warped (torch.Tensor): Warped image.
            depthmap (torch.Tensor, optional): Depth map. Defaults to None.

        Returns:
            torch.Tensor: L1 loss.
        """
        l1_loss = torch.mean(torch.abs(target - warped))
        return l1_loss

    def __str__(self):
        return "L1Loss()"


class EASLoss(nn.Module):
    """Edge-Aware Smoothness (EAS) Loss."""

    def __init__(self, alpha: int = 1):
        super(EASLoss, self).__init__()
        self.alpha = alpha

    def forward(self, warped: torch.Tensor, depthmap: torch.Tensor) -> torch.Tensor:
        """Compute the EAS loss.

        Args:
            target (torch.Tensor): Target image.
            warped (torch.Tensor): Warped image.
            depthmap (torch.Tensor): Depth map.

        Returns:
            torch.Tensor: EAS loss.
        """
        # Calculate horizontal and vertical gradients of the disparity map
        # Calculate horizontal and vertical gradients of the image
        disparity_gradients_x = torch.abs(depthmap[:, :, :-1] - depthmap[:, :, 1:])
        disparity_gradients_y = torch.abs(depthmap[:, :-1, :] - depthmap[:, 1:, :])

        image_gradients_x = torch.mean(
            torch.abs(warped[:, :, :, :-1] - warped[:, :, :, 1:]), dim=1
        )  # Resulting shape: Bx1x448x447
        image_gradients_y = torch.mean(
            torch.abs(warped[:, :, :-1, :] - warped[:, :, 1:, :]), dim=1
        )  # Resulting shape: Bx1x447x448
        # Adjust disparity gradients to match the shape of image gradients
        disparity_gradients_x = disparity_gradients_x[:, :-1, :]  # Shape: Bx447x447
        disparity_gradients_y = disparity_gradients_y[:, :, :-1]  # Shape: Bx447x447

        # Match the dimensions for smoothness loss calculation
        image_gradients_x = image_gradients_x[:, :-1, :]  # Shape: Bx1x447x447
        image_gradients_y = image_gradients_y[:, :, :-1]  # Shape: Bx1x447x447

        # Calculate the smoothness loss
        smoothness_loss = disparity_gradients_x * torch.exp(
            -image_gradients_x
        ) + disparity_gradients_y * torch.exp(-image_gradients_y)

        # Calculate the average loss
        smoothness_loss = torch.mean(smoothness_loss)

        # Scale the loss by alpha
        smoothness_loss = self.alpha * smoothness_loss

        return smoothness_loss

    def __str__(self):
        return f"EASLoss(alpha={self.alpha})"


class ScaleLoss(nn.Module):
    """Scale Loss."""

    def __init__(self):
        super(ScaleLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pose_gt: torch.Tensor, pose_pred: torch.Tensor) -> torch.Tensor:
        """Compute the scale loss.

        Args:
            pose_gt (torch.Tensor): Ground truth pose.
            pose_pred (torch.Tensor): Predicted pose.
            source (torch.Tensor): Source image.
            target (torch.Tensor): Target image.

        Returns:
            torch.Tensor: Scale loss.
        """
        # Calculate the translation scale
        return self.mse(pose_gt, pose_pred)

    def __str__(self):
        return "ScaleLoss()"


# -------------------------------------------------------------------------------------------
# COMBINATION LOSS FUNCTIONS
# These loss functions are linear combinations of other losses
# -------------------------------------------------------------------------------------------
@dataclass
class LossComponent:
    """
    Dataclass to store information about a loss component and its required parameters
    """

    name: str
    loss_fn: nn.Module
    weight: float
    required_params: Set[str]
    decay_rate: Optional[float] = None

    @property
    def current_weight(self) -> float:
        """Get the current weight after decay"""
        return self.weight


class WeightedCombinationLoss(nn.Module):
    """
    A loss module that combines multiple loss functions with weights.
    """

    def __init__(
        self,
        components: List[Tuple[str, nn.Module, float, Set[str]]],
        decay_config: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the WeightedCombinationLoss module.
        """
        super(WeightedCombinationLoss, self).__init__()

        # Normalize weights to sum to 1
        total_weight = sum(weight for _, _, weight, _ in components)
        decay_config = decay_config or {}

        # Create loss components
        self.components = [
            LossComponent(
                name=name,
                loss_fn=loss_fn,
                weight=weight,  # / total_weight,
                required_params=required_params,
                decay_rate=decay_config.get(name),
            )
            for name, loss_fn, weight, required_params in components
        ]

        # Store all required parameters
        self.all_required_params = set().union(
            *(comp.required_params for comp in self.components)
        )

        self.step_count = 0

    def forward(self, **kwargs):
        """
        Compute the combined loss from all components.
        """
        # Validate that all required parameters are provided
        missing_params = self.all_required_params - set(kwargs.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        total_loss = 0.0

        for component in self.components:
            # Extract only the arguments needed for this specific loss function
            fn_args = {k: kwargs[k] for k in component.required_params}

            loss = component.loss_fn(**fn_args)
            total_loss += loss * component.current_weight

        return total_loss

    def get_dict(self, prepend_tonames="", **kwargs):
        """
        Get detailed breafkdown of all loss components.
        """
        # Validate parameters first
        missing_params = self.all_required_params - set(kwargs.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        results = {}
        total_loss = 0.0

        for component in self.components:
            component_loss = 0.0
            currentcomponentname = f"{prepend_tonames}{component.name}"
            fn_args = {k: kwargs[k] for k in component.required_params}

            # Handle nested loss functions that have their own get_dict
            if hasattr(component.loss_fn, "get_dict"):
                sub_losses = component.loss_fn.get_dict(
                    prepend_tonames=f"{prepend_tonames}{component.name}/", **fn_args
                )
                results.update(sub_losses)
                for c in component.loss_fn.components:
                    fn_args = {k: kwargs[k] for k in c.required_params}
                    loss = c.loss_fn(**fn_args)
                    component_loss += loss * c.current_weight
                results[f"{prepend_tonames}{component.name}"] = component_loss.item()

            else:
                loss = component.loss_fn(**fn_args)
                loss_value = loss.item()
                results[f"{prepend_tonames}{component.name}"] = loss_value

        return results

    def get_weights(self, prepend_tonames=""):
        """
        Recursively get a dictionary of weights for all components.

        Args:
            prepend_tonames: String to prepend to all loss names in the output dictionary

        Returns:
            Dict[str, float]: Dictionary mapping loss component names to their weights
        """
        weights = {}
        for component in self.components:
            component_name = f"{prepend_tonames}{component.name}"
            if isinstance(component.loss_fn, WeightedCombinationLoss):
                # Recurse into nested WeightedCombinationLoss
                sub_weights = component.loss_fn.get_weights(
                    prepend_tonames=f"{component_name}_"
                )
                weights.update(sub_weights)
            weights[component_name] = component.current_weight
        return weights

    def step(self):
        # print("Stepping", self.__class__.__name__)
        """Update step count and decay weights if configured, recursively stepping child components."""
        self.step_count += 1
        for component in self.components:
            # Decay the weight if a decay rate is specified
            if component.decay_rate:
                component.weight *= np.exp(-component.decay_rate * self.step_count)
            # Recursively call step on nested WeightedCombinationLoss
            if hasattr(component.loss_fn, "step"):
                component.loss_fn.step()

    def __str__(self):
        """Create a hierarchical string representation of the loss structure"""
        components_str = []
        for comp in self.components:
            weight_str = f"{comp.current_weight:.3f}"
            if comp.decay_rate:
                weight_str += f" (decaying @ {comp.decay_rate:.2e})"

            loss_str = str(comp.loss_fn).replace("\n", "\n    ")

            params_str = f"params={sorted(comp.required_params)}"
            components_str.append(
                f"    {weight_str} * {loss_str} [{comp.name}]\n    └─ {params_str}"
            )

        return f"{self.__class__.__name__}(\n" + "\n".join(components_str) + "\n)"

    def rich_print(self, parent_tree=None):
        """Print a rich tree visualization of the loss structure"""
        if parent_tree is None:
            tree = Tree(f"{self.__class__.__name__}")
        else:
            tree = parent_tree.add(f"{self.__class__.__name__}")

        for comp in self.components:
            weight_str = f"{comp.current_weight:.3f}"
            if comp.decay_rate:
                weight_str += f" (decay: {comp.decay_rate:.2e})"

            branch = tree.add(
                f"[blue]{comp.name}[/blue]([cyan]{','.join(comp.required_params)}[/cyan]) × [yellow]{weight_str}[/yellow]"
            )

            # Add parameters branch

            # Add loss function branch
            if hasattr(comp.loss_fn, "rich_print"):
                comp.loss_fn.rich_print(branch)
            else:
                loss_str = str(comp.loss_fn).replace("\n", "\n    ")
                # branch.add(f"[green]{loss_str}[/green]")

        if parent_tree is None:
            rprint(tree)


class MONO3D_Loss(WeightedCombinationLoss):
    """
    A flexible loss module that combines multiple loss functions with weights.
    Supports weight decay and detailed loss reporting.
    """

    def __init__(
        self,
        components: List[Tuple[str, nn.Module, float, Set[str]]],
        decay_config: Optional[Dict[str, float]] = None,
    ):
        super(MONO3D_Loss, self).__init__(components, decay_config)

    def get_dict(self, prepend_tonames="", **kwargs):
        results = super().get_dict(prepend_tonames, **kwargs)
        results[f"{prepend_tonames}Loss"] = super().forward(**kwargs).item()
        return results

    def __str__(self):
        return super().__str__()
