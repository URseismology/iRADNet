import torch
import torch.nn.functional as F
from FISTA import fista
from LISTA_CP import lista
from math import floor, ceil, log2
from typing import Generator


def sparse_inverse_radon_fista(
    y: torch.Tensor,
    rayP: torch.Tensor,
    q: torch.Tensor,
    seconds: float,
    freq_bounds: tuple[float, float],
    alphas: tuple[float, float],
    maxiter: int,
    device: torch.device = torch.device("cpu"),
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
    seconds : float
        sampling in seconds
    freq_bounds : tuple[float, float]
        freq. low cut-off and high cut-off
    alphas : tuple[float, float]
        regularization parameter for L1 and L2
    maxiter : int
        max number of iterations
    device : torch.device, optional
        device to run the computation, by default torch.device("cpu")

    Returns
    -------
    torch.Tensor
        reconstructed radon image
    """
    # Initialization
    nt, np = y.shape
    nq = q.shape[0]

    # d = d.to(device)
    # rayP = rayP.to(device)
    # q = q.to(device)

    assert rayP.shape == (np,), f"{rayP.shape} != ({np},)"
    assert q.shape == (nq,), f"{q.shape} != ({nq},)"

    # find 2^(k+1) such that 2^k >= nt
    nfft = 2 ** (ceil(log2(nt)) + 1)

    # Initialize model matrices in frequency domain
    y_freq: torch.Tensor = torch.fft.fft(y, nfft, dim=0)
    assert y_freq.shape == (nfft, np)

    # Initialize kernel matrix (3D) in frequency domain
    kernels = torch.zeros((nfft, np, nq), device=device, dtype=torch.complex128)

    for ifreq in range(nfft):
        f = 2 * torch.pi * ifreq / nfft / seconds
        # (np, 1) @ (1, nq) = (np, nq)
        kernels[ifreq, :, :] = torch.exp(
            1j * f * (rayP.unsqueeze(1) ** 2) @ q.unsqueeze(0).to(torch.complex128)
        )

    # Perform projection on the data
    ilow = max(floor(freq_bounds[0] * seconds * nfft) + 1, 2)
    ihigh = min(floor(freq_bounds[1] * seconds * nfft), nfft // 2) + 2

    M0 = torch.zeros((nfft, nq), device=device, dtype=torch.complex128)
    for ifreq in range(ilow, ihigh):
        L = kernels[ifreq, :, :]  # (np, nq)
        y = y_freq[ifreq, :]  # (np, )
        B = L.T @ y  # (nq, )
        A = L.T @ L  # (nq, nq)

        # (A + alpha * I) x = B, solve for x (nq, )
        x: torch.Tensor = torch.linalg.solve(
            A + alphas[1] * torch.eye(nq, device=device), B
        )

        M0[ifreq, :] = x
        M0[nfft + 1 - ifreq, :] = x.conj()

    M0[nfft // 2 + 1, :] = 0
    # (nfft, nq) -> (nfft, nq) -> (nt, np)
    m0 = torch.real(torch.fft.ifft(M0, dim=0).squeeze(0))[:nt, :]

    # Perform FISTA
    start = time_ns()
    method = lista
    print(method.__name__)
    for m in method(
        x0=m0,
        y_freq=y_freq,
        D=kernels,
        nt=nt,
        ilow=ilow,
        ihigh=ihigh,
        maxiter=maxiter,
        lambd=alphas[0],
    ):
        yield m, time_ns() - start


if __name__ == "__main__":
    from crisprf.util.bridging import retrieve_single_xy, heatmap
    from tqdm import tqdm
    import os.path as osp
    from time import time_ns

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = retrieve_single_xy("data/Ps_RF_syn1.mat")

    nt = 5000
    dt = 0.02

    _alpha0_range = torch.linspace(0.8, 3.7, 30)
    alpha1 = 1

    y = data["y"].T.to(DEVICE)
    rayP = data["rayP"].to(DEVICE)
    q = data["q"].to(DEVICE)
    for alpha0 in [1.6]:
        exp_name = f"y_hat_{alpha0:.4f}_{alpha1:.4f}"
        # if osp.exists(f"log/{exp_name}.pt"):
        #     continue

        for x_hat, elapsed in sparse_inverse_radon_fista(
            y=y,
            rayP=rayP,
            q=q,
            seconds=nt * dt,
            freq_bounds=(0, 1 / 2 / dt),
            alphas=(1, 0.2),
            maxiter=10,
            device=DEVICE,
        ):
            # get number of none-zero elements
            x_hat = x_hat.detach().cpu().T
            mse = F.mse_loss(data["x"], x_hat)
            mse_0 = F.mse_loss(data["x"], torch.zeros_like(data["x"]))
            print(
                f"{torch.count_nonzero(x_hat):2e} {mse.item():6e}/{mse_0.item():6e}={(mse/mse_0).item():4e},{elapsed}"
            )
            # torch.save(x_hat, f"log/{exp_name}.pt")
            # heatmap(rng=[-1, 1], **{exp_name: x_hat})
