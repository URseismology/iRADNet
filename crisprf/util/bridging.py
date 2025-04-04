import torch
from scipy.io import loadmat

import numpy as np

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
        nT = t.numel()
        dT = (t[1] - t[0]).item()

        nFFT = 2 * _nextpow2(nT)
        nQ = q.numel()

        if y.shape[0] == nT:
            nP = y.shape[1]
        elif y.shape[1] == nT:
            nP = y.shape[0]
        else:
            raise ValueError(f"nT {nT} not in y.shape {y.shape}")

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

    def get_freq_bounds(
        self, freq_bounds: tuple[float, float] | None
    ) -> tuple[int, int]:
        # determine a [ilow, ihigh) frequency range, bounded by
        # [1, nFFT // 2) <-> (nFFT // 2, nFFT - 1] such that it's symmetric
        # e.g. [1, 32) <-> (32, 63], so ilow = 1, ihigh = 32
        if freq_bounds is None:
            return 1, self.nFFT // 2

        ilow = max(int(freq_bounds[0] * self.dT * self.nFFT), 1)
        ihigh = min(int(freq_bounds[1] * self.dT * self.nFFT), self.nFFT // 2)
        return ilow, ihigh

    def __repr__(self):
        return f"RFDataShape(nT={self.nT}, nP={self.nP}, nQ={self.nQ}, nFFT={self.nFFT}, dT={self.dT})"

    def __str__(self):
        return repr(self)


def retrieve_single_xy(*paths: str, device: torch.device = AUTO_DEVICE) -> RFData:
    # key translations, for 1d data and 2d data
    param_key_translate = {
        "rayP": "rayP",
        "taus": "t",  # time dimension
        "time": "t",
        "qs": "q",
    }
    xy_key_translate = {
        "tx": "y",  # signal
        "radRF": "y",
        "tx_filt": "y_hat",  # signal after filtering
        "Min": "x",  # sparse codes
        "best_m_out": "x",
    }

    # load data from all paths as a single datapoint
    # warning: same key may be overwritten, last one wins
    # p1 contains {1: 1}, p2 contains {1: 2} -> {1: 2}
    data = {k: v for path in paths for k, v in loadmat(path).items()}
    if "bin" in data:
        assert data["bin"].shape[-1] == 2
        data["rayP"] = data["bin"][:, 1]
        data["qs"] = np.linspace(-3000, 3000, 400)

    param = {
        k2: torch.tensor(data[k1], device=device, dtype=TIME_DTYPE).squeeze()
        for k1, k2 in param_key_translate.items()
        if k1 in data
    }
    nT = param["t"].numel()

    # (.., nT) -> (nT, ..)
    xy = {
        k2: (
            torch.tensor(data[k1], device=device, dtype=TIME_DTYPE)
            if data[k1].shape[0] == nT
            else torch.tensor(data[k1], device=device, dtype=TIME_DTYPE).T
        )
        for k1, k2 in xy_key_translate.items()
        if k1 in data
    }

    # join two dict together -> a single datapoint
    return param | xy
