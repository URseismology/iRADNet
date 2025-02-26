import torch

from time import time_ns
from typing import Generator

from ..util.bridging import RFData, RFDataShape
from ..util.constants import AUTO_DEVICE, FREQ_DTYPE
from .FISTA import fista
from .radon3d import init_radon3d_mat


def get_x0(
    y_freq: torch.Tensor,
    radon3d: torch.Tensor,
    ilow: int,
    ihigh: int,
    mu: float,
    shapes: RFDataShape,
) -> torch.Tensor:
    nFFT = shapes.nFFT
    nQ = shapes.nQ
    nT = shapes.nT

    x0_freq = torch.zeros((nFFT, nQ), device=y_freq.device, dtype=FREQ_DTYPE)
    for ifreq in range(ilow, ihigh):
        # (nP, nQ)
        radon2d = radon3d[ifreq, :, :]
        # (nP, nQ).T @ (nP, ) -> (nQ, )
        B = radon2d.T.conj() @ y_freq[ifreq, :].conj()
        # (nP, nQ).T @ (nP, nP) @ (nP, nQ) -> (nQ, nQ)
        A = radon2d.T.conj() @ radon2d

        # (A + alpha * I) x = B, solve for x (nQ, )
        x_i: torch.Tensor = torch.linalg.solve(
            A + mu * torch.eye(nQ, dtype=FREQ_DTYPE, device=y_freq.device), B
        )

        x0_freq[ifreq, :] = x_i.conj()
        x0_freq[nFFT - ifreq, :] = x_i

    x0_freq[nFFT // 2, :] = 0
    # (nFFT, nQ) -> (nT, nP)
    return torch.real(torch.fft.ifft(x0_freq, dim=0)[:nT, :])


def sparse_inverse_radon_fista(
    sample: RFData,
    alphas: tuple[float, float],
    n_layers: int = 10,
    freq_bounds: tuple[float, float] = None,
    device: torch.device = AUTO_DEVICE,
    ista_fn: callable = fista,
) -> Generator[tuple[torch.Tensor, int], None, None]:
    """
    sparse inverse radon transform with FISTA for SRTFISTA reconstruction

    Parameters
    ----------
    sample : RFData
        seismic traces, ray parameters, q range
    alphas : tuple[float, float]
        regularization parameter for lambda and mu
    n_layers : int
        max number of iterations
    freq_bounds : tuple[float, float]
        freq. low cut-off and high cut-off
    device : torch.device, optional
        device to run the computation, by default AUTO_DEVICE
    ista_fn : callable, optional
        iterative shrinkage thresholding algorithm, by default fista

    Returns
    -------
    torch.Tensor
        reconstructed radon image
    """
    shapes = RFDataShape.from_sample(**sample)
    nFFT = shapes.nFFT
    dT = shapes.dT

    # init signal in frequency domain
    y_freq: torch.Tensor = torch.fft.fft(sample["y"], nFFT, dim=0)

    # init radon transform matrix
    radon3d = init_radon3d_mat(
        q=sample["q"], rayP=sample["rayP"], shapes=shapes, device=device
    )

    # determine a [ilow, ihigh) frequency range, bounded by
    # [1, nFFT // 2) <-> (nFFT // 2, nFFT - 1] such that it's symmetric
    # e.g. [1, 32) <-> (32, 63], so ilow = 1, ihigh = 32
    if freq_bounds is not None:
        ilow = max(int(freq_bounds[0] * dT * nFFT), 1)
        ihigh = min(int(freq_bounds[1] * dT * nFFT), nFFT // 2)
    else:
        ilow, ihigh = 1, nFFT // 2

    x0 = get_x0(
        y_freq=y_freq,
        radon3d=radon3d,
        ilow=ilow,
        ihigh=ihigh,
        mu=alphas[1],
        shapes=shapes,
    )

    # Perform FISTA
    return ista_fn(
        x0=x0,
        y_freq=y_freq,
        radon3d=radon3d,
        shapes=shapes,
        ilow=ilow,
        ihigh=ihigh,
        n_layers=n_layers,
        lambd=alphas[0],
    )


if __name__ == "__main__":
    pass
