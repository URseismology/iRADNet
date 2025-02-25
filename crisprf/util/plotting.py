import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def plot_radon3d(radon3d_init: np.ndarray, radon3d: np.ndarray, ilow: int, ihigh: int):
    pass
    # now radon3d_init and radon3d are both (nfft, np, nq) .complex128
    # we want to plot the difference between these two matrices
    # we can use np.linalg.norm to calculate the difference
    # and then plot the difference


def plot_radon2d(radon2d_init: torch.Tensor, radon2d: torch.Tensor):
    # use surface plot to plot (np, nq)
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(121, projection="3d")
    plot_surface(torch.abs(radon2d_init) - torch.abs(radon2d), ax)
    ax.set_title("Difference in magnitude")

    ax: Axes3D = fig.add_subplot(122, projection="3d")
    plot_surface(torch.angle(radon2d_init) - torch.angle(radon2d), ax)
    ax.set_title("Difference in phase")
    return fig


def plot_surface(Z: torch.Tensor, ax: Axes3D):
    assert ax is not None and isinstance(ax, Axes3D)

    if Z is None:
        m, n = 10, 15  # example dimensions
        Z = np.random.randn(m, n)  # Example 2D data array
    else:
        Z = Z.detach().cpu().numpy()
        m, n = Z.shape

    # Create X and Y meshgrid from the shape of Z
    X, Y = np.meshgrid(np.arange(n), np.arange(m))  # Meshgrid for plotting

    # Plot the surface
    ax.plot_surface(
        X, Y, Z, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3
    )

    ax.contourf(X, Y, Z, zdir="z", offset=-100, cmap="coolwarm")
    ax.contourf(X, Y, Z, zdir="x", offset=-40, cmap="coolwarm")
    ax.contourf(X, Y, Z, zdir="y", offset=40, cmap="coolwarm")

    # Labels and title
    ax.set(
        xlim=(0, n),
        ylim=(0, m),
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
    )
