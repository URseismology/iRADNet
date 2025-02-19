import torch
from scipy.io import loadmat

import seaborn as sns
from matplotlib import pyplot as plt

import os.path as osp
from typing import TypedDict

from .constants import FREQ_DTYPE, TIME_DTYPE

EXAMPLE = "data/sample.mat"


class RFData(TypedDict):
    data_id: str
    q: torch.Tensor  # (Q,) q range
    rayP: torch.Tensor  # (P,) ray parameters
    t: torch.Tensor  # (T,) time dimension
    x: torch.Tensor  # (T, Q) sparse codes
    y: torch.Tensor  # (T, P) signal
    y_hat: torch.Tensor | None  # (T, P) signal after filtering


class RFDataUtils:
    @staticmethod
    def peek(sample: RFData):
        return {k: v.shape for k, v in sample.items() if type(v) is torch.Tensor}

    @staticmethod
    def verify(sample: RFData):
        _, T, P, Q = RFDataUtils.get_shapes(sample["y"], sample["q"])

        # Y is (T, P)
        assert sample["y"].shape == (T, P)
        # X is (T, Q)
        assert sample["x"].shape == (T, Q)

        # y_hat ~ y same shape
        if sample["y_hat"] is not None:
            assert sample["y_hat"].shape == sample["y"].shape

    @staticmethod
    def get_shapes(y: torch.Tensor, q: torch.Tensor) -> tuple[int, int, int, int]:
        nt, np = y.shape
        nq = q.numel()
        nfft = 2 * RFDataUtils.nextpow2(nt)
        return nfft, nt, np, nq

    @staticmethod
    def nextpow2(x: int):
        if x == 0:
            return 0
        return 2 ** (x - 1).bit_length()


def retrieve_single_xy(
    path: str = EXAMPLE, device: torch.device = torch.device("cpu")
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
                torch.tensor(data[k1], device=device, dtype=TIME_DTYPE).reshape(nt, -1)
                if k1 in data
                else None
            )
            for k1, k2 in v2d_translation.items()
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
