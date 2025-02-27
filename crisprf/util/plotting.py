import torch

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D, axes3d

import os


def plot_radon3d(radon3d_init: torch.Tensor, radon3d: torch.Tensor, **kwargs):
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(121, projection="3d")
    diff = (
        (torch.abs(radon3d_init) - torch.abs(radon3d)) / torch.abs(radon3d_init) * 100
    )
    plot_outliers(torch.min(diff, dim=0).values, ax)
    # plot_surface(torch.median(diff, dim=0).values, ax, edgecolor="gray")
    # plot_surface(torch.max(diff, dim=0).values, ax, edgecolor="red")
    ax.set_title("Diff in relative magnitude %")

    ax: Axes3D = fig.add_subplot(122, projection="3d")
    diff = torch.angle(radon3d_init) - torch.angle(radon3d)
    plot_surface(torch.min(diff, dim=0).values, ax, edgecolor="royalblue")
    # plot_surface(torch.median(diff, dim=0).values, ax, edgecolor="gray")
    plot_surface(torch.max(diff, dim=0).values, ax, edgecolor="red")
    ax.set_title("Diff in phase")
    return fig


def plot_radon2d(radon2d_init: torch.Tensor, radon2d: torch.Tensor):
    # use surface plot to plot (nP, nQ)
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(121, projection="3d")
    diff = (
        (torch.abs(radon2d_init) - torch.abs(radon2d)) / torch.abs(radon2d_init) * 100
    )
    plot_surface(diff, ax)
    ax.set_title("Diff in relative magnitude %")

    ax: Axes3D = fig.add_subplot(122, projection="3d")
    diff = torch.angle(radon2d_init) - torch.angle(radon2d)
    plot_surface(diff, ax)
    ax.set_title("Diff in phase")
    return fig


def plot_surface(Z: torch.Tensor, ax: Axes3D, **kwargs):
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
        X,
        Y,
        Z,
        **(
            dict(
                edgecolor="royalblue",
                lw=0.5,
                rstride=8,
                cstride=8,
                alpha=0.3,
            )
            | kwargs
        ),
    )

    # ax.contourf(X, Y, Z, zdir="z", offset=-1, cmap="coolwarm")
    # ax.contourf(X, Y, Z, zdir="x", offset=-m - 10, cmap="coolwarm")
    # ax.contourf(X, Y, Z, zdir="y", offset=n + 10, cmap="coolwarm")

    # Labels and title
    ax.set(
        xlim=(0, n - 1),
        ylim=(0, m - 1),
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
    )


def plot_outliers(radon3d_init: torch.Tensor, radon3d: torch.Tensor):
    fig = plt.figure(figsize=(12, 5))
    nFFT, nP, nQ = radon3d_init.shape

    ax: Axes3D = fig.add_subplot(121, projection="3d")
    diff_mag = (1 - torch.abs(radon3d) / torch.abs(radon3d_init)) * 100
    diff_mag = diff_mag.detach().cpu().numpy()
    outlier_mag_map = np.abs(diff_mag) > 0.1
    for x, y, z in zip(*np.where(outlier_mag_map)):
        cax = ax.scatter(
            y, z, diff_mag[x, y, z], s=4, c=x / nFFT, vmin=0, vmax=1, alpha=0.5
        )
    ax.set(
        xlim=(0, nP - 1),
        ylim=(0, nQ - 1),
        xlabel="nP",
        ylabel="nQ",
        zlabel="mag %",
    )
    plt.colorbar(cax, ax=ax)

    ax: Axes3D = fig.add_subplot(122, projection="3d")
    diff_phase = torch.angle(radon3d_init) - torch.angle(radon3d)
    diff_phase = diff_phase.detach().cpu().numpy()
    outlier_phase_map = np.abs(diff_phase) > 0.1
    for x, y, z in zip(*np.where(outlier_phase_map)):
        cax = ax.scatter(y, z, diff_phase[x, y, z], s=4, c=x / nFFT, vmin=0, vmax=1)
    ax.set(
        xlim=(0, nP - 1),
        ylim=(0, nQ - 1),
        xlabel="nP",
        ylabel="nQ",
        zlabel="phase (rad)",
    )
    plt.colorbar(cax, ax=ax)
    fig.savefig("tmp/outliers.png")
    plt.close(fig)


def plot_sample(
    prefix_scope: tuple[str] = ("x", "y"), save_path: str = "fig/example.png", **kwargs
):
    xy_keys = sorted(
        list(
            filter(lambda k: any(k.startswith(p) for p in prefix_scope), kwargs.keys())
        )
    )
    n_plots = len(xy_keys)

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots), sharex=True)
    axes: list[plt.Axes] = [axes] if n_plots == 1 else axes.ravel()

    for ax, k in zip(axes, xy_keys):
        if "x" in k:
            plot_x(data=kwargs[k], ax=ax, **kwargs)
        else:
            plot_y(data=kwargs[k], ax=ax, **kwargs)
        # set key as ax title, e.g. "y_hat"
        ax.set_title(k)

    # axes[-1].locator_params(axis="x", nbins=10)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, pad_inches=0)
    plt.close(fig)


def plot_x(data: torch.Tensor, ax: plt.Axes, t: torch.Tensor, q: torch.Tensor, **_):
    # plot sparse codes x (T, Q)
    nT = t.numel()
    if data.shape[-1] != nT:
        data = data.T

    sns.heatmap(
        data.detach().cpu(),
        ax=ax,
        xticklabels=list(map(lambda x: f"{x:.0f}", t.detach().cpu().numpy())),
        yticklabels=list(map(lambda x: f"{x:.0f}", q.detach().cpu().numpy())),
        linewidths=0,
        center=0,
        cmap="vlag",
    )
    ax.locator_params(axis="both", nbins=10)
    ax.set_ylabel("q (s/km)")


def plot_y(data: torch.Tensor, ax: plt.Axes, t: torch.Tensor, rayP: torch.Tensor, **_):
    # plot sparse codes y (T, P)
    nT = t.numel()
    if data.shape[-1] != nT:
        data = data.T

    sns.heatmap(
        data.detach().cpu(),
        ax=ax,
        xticklabels=list(map(lambda x: f"{x:.0f}", t.detach().cpu().numpy())),
        yticklabels=list(map(lambda x: f"{x:.3f}", rayP.detach().cpu().numpy())),
        linewidths=0,
        center=0,
        cmap="vlag",
    )
    ax.locator_params(axis="both", nbins=10)
    ax.set_ylabel("Ray parameter (deg)")
