# -------------------------------------------------------------------------------------------------#

""" Copyright (c) 2024 Asensus Surgical """

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

import matplotlib.pyplot as plt
import numpy as np
import torch


def scatterpptk(cloud: torch.Tensor, rgb_vec: torch.Tensor = None) -> None:
    """
    Displays a 3D point cloud using the pptk viewer.

    Args:
        cloud (torch.Tensor): Input point cloud.
        rgb_vec (torch.Tensor, optional): RGB color information for the point cloud. Defaults to None.
    """
    rgbcloud = pptk.viewer(cloud[:3, :].cpu().detach()().permute(1, 0))
    if rgb_vec is not None:
        rgbcloud.attributes(rgb_vec.permute(1, 0))
    rgbcloud.set(show_axis=True)


def scattero3d(cloud: torch.Tensor, rgb_vec: torch.Tensor = None) -> None:
    """
    Displays a 3D point cloud using the Open3D viewer.

    Args:
        cloud (torch.Tensor): Input point cloud.
        rgb_vec (torch.Tensor, optional): RGB color information for the point cloud. Defaults to None.
    """


def Tplot(ax: plt.Axes, T: np.ndarray) -> None:
    """
    Plots a transformation matrix as arrows in a 3D plot.

    Args:
        ax (plt.Axes): A matplotlib 3D axis.
        T (np.ndarray): 4x4 transformation matrix.
    """
    X, Y, Z = T[:3, 0], T[:3, 1], T[:3, 2]
    x, y, z = T[0, -1], T[1, -1], T[2, -1]

    ax.quiver(x, y, z, X[0], X[1], X[2], color="r", normalize=True)
    ax.quiver(x, y, z, Y[0], Y[1], Y[2], color="g", normalize=True)
    ax.quiver(x, y, z, Z[0], Z[1], Z[2], color="b", normalize=True)


def fig3d() -> plt.Axes:
    """
    Creates a 3D plot figure.

    Returns:
        plt.Axes: 3D plot axis.
    """
    return plt.figure().add_subplot(projection="3d")


def set_axes_equal(ax: plt.Axes) -> None:
    """
    Makes axes of a 3D plot have equal scale.

    Args:
        ax (plt.Axes): A matplotlib 3D axis.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
