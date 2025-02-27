import torch
from scipy.io import loadmat

from typing import TypedDict

from .constants import AUTO_DEVICE, TIME_DTYPE


class RFData(TypedDict):
    q: torch.Tensor  # (Q,) q range
    rayP: torch.Tensor  # (P,) ray parameters
    t: torch.Tensor  # (T,) time dimension

    x: torch.Tensor  # (T, Q) sparse codes
    y: torch.Tensor  # (T, P) signal

    y_noise: torch.Tensor | None  # (T, P) signal with noise
    x_hat: torch.Tensor | None  # (T, Q) sparse codes after filtering
    y_hat: torch.Tensor | None  # (T, P) signal after filtering


def _nextpow2(x: int) -> int:
    if x == 0:
        return 0
    return 2 ** (x - 1).bit_length()


class RFDataShape:
    def __init__(self, nT: int, nP: int, nQ: int, nFFT: int, dT: float):
        self.nT = nT
        self.nP = nP
        self.nQ = nQ
        self.nFFT = nFFT
        self.dT = dT

    @staticmethod
    def from_sample(
        y: torch.Tensor, q: torch.Tensor, t: torch.Tensor, **_
    ) -> "RFDataShape":
        nT, nP = y.shape
        nQ = q.numel()
        nFFT = 2 * _nextpow2(nT)
        dT = (t[1] - t[0]).item()
        return RFDataShape(nT=nT, nP=nP, nQ=nQ, nFFT=nFFT, dT=dT)

    @staticmethod
    def peek(sample: RFData):
        return {k: v.shape for k, v in sample.items() if type(v) is torch.Tensor}

    @staticmethod
    def verify(sample: RFData) -> None:
        assert len(sample["t"].shape) == 1
        assert len(sample["rayP"].shape) == 1
        assert len(sample["q"].shape) == 1

        nT = sample["t"].numel()
        nP = sample["rayP"].numel()
        nQ = sample["q"].numel()

        assert sample["y"].shape == (nT, nP)
        assert sample["x"].shape == (nT, nQ)
        if sample["y_hat"] is not None:
            assert sample["y_hat"].shape == sample["y"].shape


def retrieve_single_xy(
    path: str = "data/sample.mat", device: torch.device = AUTO_DEVICE
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
    nT = data["taus"].size

    return {
        k2: torch.tensor(data[k1], device=device, dtype=TIME_DTYPE).squeeze()
        for k1, k2 in v1d_translation.items()
    } | {
        # (.., nT) -> (nT, ..)
        k2: (
            torch.tensor(data[k1], device=device, dtype=TIME_DTYPE)
            if data[k1].shape[0] == nT
            else torch.tensor(data[k1], device=device, dtype=TIME_DTYPE).T
        )
        for k1, k2 in v2d_translation.items()
        if k1 in data
    }
