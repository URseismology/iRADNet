import torch
from .FISTA import fista
from math import floor, ceil, log2
from typing import Generator
from time import time_ns
import os.path as osp


def get_shapes(
    y: torch.Tensor,
    q: torch.Tensor,
    rayP: torch.Tensor = None,
) -> tuple[int, int, int, int, torch.Tensor]:
    nt, np = y.shape
    nfft = 2 ** (ceil(log2(nt)) + 1)
    nq = q.shape[0]

    if rayP is not None:
        assert rayP.shape == (np,), f"{rayP.shape} != ({np},)"

    return nfft, nt, np, nq


def get_x0(
    y_freq: torch.Tensor,
    radon_mat: torch.Tensor,
    ilow: int,
    ihigh: int,
    nt: int,
    mu: float,
):
    nfft, np, nq = radon_mat.shape

    x0_freq = torch.zeros((nfft, nq), device=y_freq.device, dtype=torch.complex128)
    for ifreq in range(ilow, ihigh):
        # (np, nq)
        L = radon_mat[ifreq, :, :]
        # (np, nq).T @ (np, ) -> (nq, )
        B = L.T @ y_freq[ifreq, :]
        # (np, nq).T @ (np, np) @ (np, nq) -> (nq, nq)
        A = L.T @ L
        assert A.dtype == torch.complex128
        assert B.dtype == torch.complex128

        # (A + alpha * I) x = B, solve for x (nq, )
        xi: torch.Tensor = torch.linalg.solve(
            A + mu * torch.eye(nq, dtype=torch.complex128, device=y_freq.device), B
        )

        assert xi.dtype == torch.complex128
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

    cache_path = f"cache/{ista_fn.__name__}.{data_id}.pt"
    if osp.exists(cache_path):
        radon_mat = torch.load(cache_path)
    else:
        # # init radon transform matrix
        radon_mat = torch.zeros((nfft, np, nq), device=device, dtype=torch.complex128)
        ifreq_f = (
            2.0
            * torch.pi
            * torch.arange(nfft, device=device, dtype=torch.float64)
            / nfft
            / dt
        )
        # (nfft) @ (np) @ (nq) = (nfft, np, nq)
        radon_mat[:, :, :] = torch.exp(
            1j * torch.einsum("f,p,q->fpq", ifreq_f, rayP**2, q).to(torch.complex128)
        )
        torch.save(radon_mat, cache_path)

    # determine a [ilow, ihigh) frequency range, bounded by
    # [1, nfft // 2] <-> [nfft // 2, nfft - 1] such that it's symmetric
    ilow = max(floor(freq_bounds[0] * dt * nfft), 1)
    ihigh = min(floor(freq_bounds[1] * dt * nfft), nfft // 2)

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
