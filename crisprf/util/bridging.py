import torch
import numpy as np
from scipy.io import loadmat
from typing import TypedDict
from matplotlib import pyplot as plt
import seaborn as sns
import os.path as osp


EXAMPLE = "/home/wmeng/crisprf/data/Sp_RF_syn1.mat"


class RFData(TypedDict):
    data_id: str
    tshift: torch.Tensor  # ()
    rayP: torch.Tensor  # (np,) ray parameters
    t: torch.Tensor  # (nt,) time dimension
    x: torch.Tensor  # (nt, nq) sparse codes
    y: torch.Tensor  # (nt, np) signal
    q: torch.Tensor  # (nq,) q range


def retrieve_single_xy(
    path: str = EXAMPLE, device: torch.device = torch.device("cpu")
) -> RFData:
    key_translation = {
        "tshift": "tshift",
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
            k2: torch.tensor(data[k1], device=device).squeeze()
            for k1, k2 in key_translation.items()
        }
        | {
            # (.., nt) -> (nt, ..)
            k2: torch.tensor(data[k1], device=device).T
            for k1, k2 in xy_translation.items()
        }
        | {
            "q": torch.linspace(-1000, 1000, 200, device=device),
            "data_id": osp.basename(path),
        }
    )


def peek(**kwargs):
    return {k: v.shape for k, v in kwargs.items() if type(v) is torch.Tensor}


def heatmap_one_plot(*args: np.ndarray):
    N = len(args)
    fig, axes = plt.subplots(1, N, figsize=(8 * N, 6), dpi=300)
    axes = axes if N > 1 else [axes]
    for v, ax in zip(args, axes):
        sns.heatmap(
            v,
            cmap="coolwarm",
            ax=ax,
            center=0,
        )
    plt.savefig("fig/heatmap.png")


def heatmap(rng=None, **kwargs: np.ndarray):
    for k, v in kwargs.items():
        if len(v.shape) != 2:
            continue
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        sns.heatmap(
            v,
            cmap="coolwarm",
            ax=ax,
            center=0,
        )
        plt.tight_layout()
        plt.savefig(f"fig/{k}.png")
        plt.clf()


if __name__ == "__main__":
    from pprint import pprint

    data = retrieve_single_xy("data/Ps_RF_syn1.mat")

    pprint(peek(**data))
    # heatmap(rng=[-1, 1], **data)
    # heatmap_one_plot(data["x"], data["y"], rng=[-1, 1])
    # for k in ["x", "y"]:
    #     fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    #     sns.heatmap(data[k], cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    #     plt.tight_layout()
    #     plt.savefig(f"fig/{k}.png")
    #     plt.clf()
