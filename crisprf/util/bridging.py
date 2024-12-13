import torch
import numpy as np
from scipy.io import loadmat
from typing import TypedDict
from matplotlib import pyplot as plt
import seaborn as sns


EXAMPLE = "/home/wmeng/crisprf/data/Sp_RF_syn1.mat"


class RFData(TypedDict):
    tshift: torch.Tensor  # (1, 1)
    rayP: torch.Tensor  # (1, 38)
    t: torch.Tensor  # (1, 1000), time dimension
    x: torch.Tensor  # (200, 1000), sparse codes
    y: torch.Tensor  # (38, 1000), signal
    q: torch.Tensor


def retrieve_single_xy(path: str = EXAMPLE) -> RFData:
    key_translation = {
        "tshift": "tshift",
        "rayP": "rayP",
        "taus": "t",  # time dimension
        "Min_2": "x",  # sparse codes
        "tx": "y",  # signal
    }
    data: RFData = loadmat(path)

    return {
        key_translation[k]: torch.tensor(v).squeeze()
        for k, v in data.items()
        if k in key_translation
    } | {"q": torch.linspace(-1000, 1000, 200)}


def peek(**kwargs):
    return {k: v.shape for k, v in kwargs.items() if type(v) is torch.Tensor}


def heatmap_one_plot(*args: np.ndarray, rng=None):
    N = len(args)
    fig, axes = plt.subplots(1, N, figsize=(8 * N, 5), dpi=300)
    axes = axes if N > 1 else [axes]
    for v, ax in zip(args, axes):
        sns.heatmap(
            v,
            cmap="coolwarm",
            ax=ax,
            vmin=rng[0] if rng else None,
            vmax=rng[1] if rng else None,
        )
    plt.savefig("fig/heatmap.png")


def heatmap(rng=None, **kwargs: np.ndarray):
    for k, v in kwargs.items():
        if len(v.shape) <= 2:
            continue
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        sns.heatmap(
            v,
            cmap="coolwarm",
            ax=ax,
            vmin=rng[0] if rng else None,
            vmax=rng[1] if rng else None,
        )
        plt.tight_layout()
        plt.savefig(f"fig/{k}.png")
        plt.clf()


if __name__ == "__main__":
    from pprint import pprint

    data = retrieve_single_xy("data/Ps_RF_syn1.mat")

    pprint(peek(**data))
    heatmap(rng=[-1, 1], **data)
    heatmap_one_plot(data["x"], data["y"], rng=[-1, 1])
    # for k in ["x", "y"]:
    #     fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    #     sns.heatmap(data[k], cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    #     plt.tight_layout()
    #     plt.savefig(f"fig/{k}.png")
    #     plt.clf()
