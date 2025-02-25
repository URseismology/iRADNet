import torch
from scipy.io import loadmat

import seaborn as sns
from matplotlib import pyplot as plt

import os.path as osp
from typing import TypedDict

from .constants import AUTO_DEVICE, TIME_DTYPE

EXAMPLE = "data/sample.mat"


class RFData(TypedDict):
    data_id: str
    q: torch.Tensor  # (Q,) q range
    rayP: torch.Tensor  # (P,) ray parameters
    t: torch.Tensor  # (T,) time dimension
    x: torch.Tensor  # (T, Q) sparse codes
    y: torch.Tensor  # (T, P) signal
    y_hat: torch.Tensor | None  # (T, P) signal after filtering


def nextpow2(x: int) -> int:
    if x == 0:
        return 0
    return 2 ** (x - 1).bit_length()


class RFDataShape:
    def __init__(self, nt: int, np: int, nq: int, nfft: int, dt: float):
        self.nt = nt
        self.np = np
        self.nq = nq
        self.nfft = nfft
        self.dt = dt

    @staticmethod
    def from_sample(
        y: torch.Tensor, q: torch.Tensor, t: torch.Tensor, **_
    ) -> "RFDataShape":
        nt, np = y.shape
        nq = q.numel()
        nfft = 2 * nextpow2(nt)
        dt = (t[1] - t[0]).item()
        return RFDataShape(nt=nt, np=np, nq=nq, nfft=nfft, dt=dt)

    @staticmethod
    def peek(sample: RFData):
        return {k: v.shape for k, v in sample.items() if type(v) is torch.Tensor}

    @staticmethod
    def verify(sample: RFData) -> None:
        assert len(sample["t"].shape) == 1
        assert len(sample["rayP"].shape) == 1
        assert len(sample["q"].shape) == 1

        nt = sample["t"].numel()
        np = sample["rayP"].numel()
        nq = sample["q"].numel()

        assert sample["y"].shape == (nt, np)
        assert sample["x"].shape == (nt, nq)
        if sample["y_hat"] is not None:
            assert sample["y_hat"].shape == sample["y"].shape


def retrieve_single_xy(
    path: str = EXAMPLE, device: torch.device = AUTO_DEVICE
) -> RFData:
    # key translations, for 1d data and 2d data
    v1d_translation = {
        "rayP": "rayP",
        "taus": "t",  # time dimension
        "qs": "q",
    }
    v2d_translation = {
        "Min": "x",  # sparse codes
        "tx": "y",  # signal
        "tx_filt": "y_hat",  # signal after filtering
    }
    data = loadmat(path)
    nt = data["taus"].size

    return (
        {
            "data_id": osp.basename(path),
        }
        | {
            k2: torch.tensor(data[k1], device=device, dtype=TIME_DTYPE).squeeze()
            for k1, k2 in v1d_translation.items()
        }
        | {
            # (.., nt) -> (nt, ..)
            k2: (
                torch.tensor(data[k1], device=device, dtype=TIME_DTYPE)
                if data[k1].shape[0] == nt
                else torch.tensor(data[k1], device=device, dtype=TIME_DTYPE).T
            )
            for k1, k2 in v2d_translation.items()
            if k1 in data
        }
    )


def plot_sample(
    prefix_scope: tuple[str] = ("x", "y"), save_path: str = "fig/example.png", **kwargs
):
    xy_keys = sorted(list(filter(lambda k: k[0] in prefix_scope, kwargs.keys())))
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
    plt.savefig(save_path, pad_inches=0)


def plot_x(data: torch.Tensor, ax: plt.Axes, t: torch.Tensor, q: torch.Tensor, **_):
    # plot sparse codes x (T, Q)
    nt = t.numel()
    if data.shape[-1] != nt:
        data = data.T

    sns.heatmap(
        data.detach().cpu(),
        ax=ax,
        xticklabels=list(map(lambda x: f"{x:.0f}", t.detach().cpu().numpy())),
        yticklabels=list(map(lambda x: f"{x:.0f}", q.detach().cpu().numpy())),
        linewidths=0,
        center=0,
    )
    ax.locator_params(axis="both", nbins=10)
    ax.set_ylabel("q (s/km)")


def plot_y(data: torch.Tensor, ax: plt.Axes, t: torch.Tensor, rayP: torch.Tensor, **_):
    # plot sparse codes y (T, P)
    nt = t.numel()
    if data.shape[-1] != nt:
        data = data.T

    sns.heatmap(
        data.detach().cpu(),
        ax=ax,
        xticklabels=list(map(lambda x: f"{x:.0f}", t.detach().cpu().numpy())),
        yticklabels=list(map(lambda x: f"{x:.3f}", rayP.detach().cpu().numpy())),
        linewidths=0,
        center=0,
    )
    ax.locator_params(axis="both", nbins=10)
    ax.set_ylabel("Ray parameter (deg)")
