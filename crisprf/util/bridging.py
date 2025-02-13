import torch
from scipy.io import loadmat

import seaborn as sns
from matplotlib import pyplot as plt

import os.path as osp
from typing import TypedDict

from .constants import FREQ_DTYPE, TIME_DTYPE

EXAMPLE = "/home/wmeng/crisprf/data/Ps_RF_syn1.mat"


class RFData(TypedDict):
    data_id: str
    q: torch.Tensor  # (Q,) q range
    rayP: torch.Tensor  # (P,) ray parameters
    t: torch.Tensor  # (T,) time dimension
    x: torch.Tensor  # (T, Q) sparse codes
    y: torch.Tensor  # (T, P) signal


def retrieve_single_xy(
    path: str = EXAMPLE, device: torch.device = torch.device("cpu")
) -> RFData:
    key_translation = {
        "rayP": "rayP",
        "taus": "t",  # time dimension
    }
    xy_translation = {
        "Min_2": "x",  # sparse codes
        "tx": "y",  # signal
    }
    data = loadmat(path)

    return (
        {
            k2: torch.tensor(data[k1], device=device, dtype=TIME_DTYPE).squeeze()
            for k1, k2 in key_translation.items()
        }
        | {
            # (.., nt) -> (nt, ..)
            k2: torch.tensor(data[k1], device=device, dtype=TIME_DTYPE).T
            for k1, k2 in xy_translation.items()
        }
        | {
            "q": torch.linspace(-1000, 1000, 200, device=device, dtype=TIME_DTYPE),
            "data_id": osp.basename(path),
        }
    )


def peek(**kwargs):
    return {k: v.shape for k, v in kwargs.items() if type(v) is torch.Tensor}


def plot_sample(sample: RFData):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, dpi=100)
    sns.heatmap(
        sample["y"].T,
        ax=ax0,
        yticklabels=list(map(lambda x: f"{x:.3f}", sample["rayP"].numpy())),
        linewidths=0,
        center=0,
    )
    sns.heatmap(
        sample["x"].T,
        ax=ax1,
        xticklabels=list(map(lambda x: f"{x:.0f}", sample["t"].numpy())),
        yticklabels=list(map(lambda x: f"{x:.0f}", sample["q"].numpy())),
        linewidths=0,
        center=0,
    )
    ax0.locator_params(axis="y", nbins=10)
    ax0.set_ylabel("Ray parameter (km)")
    ax0.set_title("(a) Seismic traces")
    ax1.locator_params(axis="both", nbins=10)
    ax1.set_ylabel("q (s/km)")
    ax1.set_xlabel("Time (s)")
    ax1.set_title("(b) Radon transform")
    plt.tight_layout()
    plt.savefig("fig/example.pdf", pad_inches=0)


if __name__ == "__main__":
    from pprint import pprint

    data = retrieve_single_xy()
    pprint(peek(**data))
    plot_sample(data)
    # heatmap(rng=[-1, 1], **data)
    # heatmap_one_plot(data["x"], data["y"], rng=[-1, 1])
    # for k in ["x", "y"]:
    #     fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    #     sns.heatmap(data[k], cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    #     plt.tight_layout()
    #     plt.savefig(f"fig/{k}.png")
    #     plt.clf()
