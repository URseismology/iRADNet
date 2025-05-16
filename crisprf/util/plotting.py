import torch

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

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
    prefix_incl: list[str] = ["x", "y"],
    prefix_excl: list[str] = [],
    key_incl: list[str] = [],
    key_excl: list[str] = ["y_hat"],
    save_path: str = "fig/example.png",
    **kwargs,
):
    def _key_filter(k: str) -> bool:
        # include if any prefix_incl is a prefix of k
        # exclude if any prefix_excl is a prefix of k
        if k in key_incl:
            return True
        if k in key_excl:
            return False

        return any(k.startswith(p) for p in prefix_incl) and not any(
            k.startswith(p) for p in prefix_excl
        )

    xy_keys = sorted(list(filter(_key_filter, kwargs.keys())))
    n_plots = len(xy_keys)

    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots), sharex=True)
    axes: list[plt.Axes] = [axes] if n_plots == 1 else axes.ravel()

    for i_ax, (ax, k) in enumerate(zip(axes, xy_keys)):
        if "x" in k:
            plot_x(data=kwargs[k], ax=ax, **kwargs)
        else:
            plot_wiggle(data=kwargs[k], ax=ax, **kwargs)
        if len(xy_keys) > 1:
            ax.text(
                -0.1,
                1.1 if "x" in k else 1,
                f"({chr(97 + i_ax)})",
                transform=ax.transAxes,
                size=16,
            )
        # set key as ax title, e.g. "y_hat"
        ax.set_title(k)

    axes[-1].locator_params(axis="x", nbins=10)
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
        center=0,
        cmap="seismic",
        cbar_kws={
            "orientation": "horizontal",
            "aspect": 40,
            "pad": 0.05,
            "location": "top",
        },
    )
    ax.set_yticks(
        np.linspace(0, q.numel(), 5),
        [f"{i:.0f}" for i in np.linspace(q.min().item(), q.max().item(), 5)],
    )
    ax.grid()
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
        cmap="seismic",
        cbar_kws={"orientation": "horizontal", "aspect": 40, "pad": 0.05},
    )
    ax.grid()
    ax.locator_params(axis="y", nbins=5)
    ax.set_ylabel("Ray parameter (deg)")


def plot_wiggle(
    data: torch.Tensor, ax: plt.Axes, t: torch.Tensor, rayP: torch.Tensor, **_
):
    nT = t.numel()
    if data.shape[-1] != nT:
        data = data.T

    # y is a (nP, nT) tensor
    data = data / data.abs().max()
    data = data.detach().cpu().numpy()
    p_min = rayP.min().item()
    p_max = rayP.max().item()
    nP = rayP.numel()

    x_rng = np.arange(nT)
    ax.grid()
    for i in reversed(range(data.shape[0])):
        _data = data[i] + i
        ax.plot(x_rng, _data, color="black", lw=1, rasterized=True)
        ax.fill_between(
            x_rng, _data, i, where=(data[i] > 0), facecolor="#f00", rasterized=True
        )
        ax.fill_between(
            x_rng, _data, i, where=(data[i] < 0), facecolor="#00f", rasterized=True
        )

    ax.set_yticks(np.linspace(0, nP, 5), np.linspace(p_min, p_max, 5))
    ax.set_ylabel("Ray parameter (deg)")
