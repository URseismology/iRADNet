import torch

from time import time_ns
from typing import Generator

from crisprf.util.constants import FREQ_DTYPE, TIME_DTYPE

from .FISTA import fista
from .radon3d import get_shapes, init_radon3d_mat


def get_x0(
    y_freq: torch.Tensor,
    radon_mat: torch.Tensor,
    ilow: int,
    ihigh: int,
    nt: int,
    mu: float,
):
    nfft, np, nq = radon_mat.shape

    x0_freq = torch.zeros((nfft, nq), device=y_freq.device, dtype=FREQ_DTYPE)
    for ifreq in range(ilow, ihigh):
        # (np, nq)
        L = radon_mat[ifreq, :, :]
        # (np, nq).T @ (np, ) -> (nq, )
        B = L.T @ y_freq[ifreq, :]
        # (np, nq).T @ (np, np) @ (np, nq) -> (nq, nq)
        A = L.T @ L
        assert A.dtype == FREQ_DTYPE
        assert B.dtype == FREQ_DTYPE

        # (A + alpha * I) x = B, solve for x (nq, )
        xi: torch.Tensor = torch.linalg.solve(
            A + mu * torch.eye(nq, dtype=FREQ_DTYPE, device=y_freq.device), B
        )

        assert xi.dtype == FREQ_DTYPE
        x0_freq[ifreq, :] = xi
        x0_freq[nfft - ifreq, :] = xi.conj()

    x0_freq[nfft // 2, :] = 0
    # (nfft, nq) -> (nt, np)
    return torch.real(torch.fft.ifft(x0_freq, dim=0)[:nt, :])


def sparse_inverse_radon_fista(
    data_id: str,
    y: torch.Tensor,
    rayP: torch.Tensor,
    q: torch.Tensor,
    dt: float,
    freq_bounds: tuple[float, float],
    alphas: tuple[float, float],
    n_layers: int,
    device: torch.device = torch.device("cpu"),
    ista_fn: callable = fista,
    **_,
) -> Generator[tuple[torch.Tensor, int], None, None]:
    """
    sparse inverse radon transform with FISTA for SRTFISTA reconstruction

    Parameters
    ----------
    y : torch.Tensor
        seismic traces
    rayP : torch.Tensor
        ray parameters
    q : torch.Tensor
        q range in 1d
    dt : float
        sampling delta time t1-t0, in seconds
    freq_bounds : tuple[float, float]
        freq. low cut-off and high cut-off
    alphas : tuple[float, float]
        regularization parameter for L1 and L2
    n_layers : int
        max number of iterations
    device : torch.device, optional
        device to run the computation, by default torch.device("cpu")

    Returns
    -------
    torch.Tensor
        reconstructed radon image
    """
    nfft, nt, np, nq = get_shapes(y, q, rayP)

    # init signal in frequency domain
    y_freq: torch.Tensor = torch.fft.fft(y, nfft, dim=0)
    assert y_freq.shape == (nfft, np)

    # init radon transform matrix
    radon_mat = init_radon3d_mat(y=y, q=q, rayP=rayP, dt=dt)

    # determine a [ilow, ihigh) frequency range, bounded by
    # [1, nfft // 2) <-> (nfft // 2, nfft - 1] such that it's symmetric
    # e.g. [1, 32) <-> (32, 63], so ilow = 1, ihigh = 32
    ilow = max(int(freq_bounds[0] * dt * nfft), 1)
    ihigh = min(int(freq_bounds[1] * dt * nfft), nfft // 2)

    x0 = get_x0(
        y_freq=y_freq, radon_mat=radon_mat, ilow=ilow, ihigh=ihigh, nt=nt, mu=alphas[1]
    )

    # Perform FISTA
    start = time_ns()
    for x in ista_fn(
        x0=x0,
        y_freq=y_freq,
        L=radon_mat,
        nt=nt,
        ilow=ilow,
        ihigh=ihigh,
        n_layers=n_layers,
        lambd=alphas[0],
    ):
        yield x, time_ns() - start


if __name__ == "__main__":
    pass
