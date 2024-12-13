import torch
import torch.nn.functional as F
from math import ceil, floor, log2, sqrt
from tqdm import tqdm
import os.path as osp


def radon3d_forward(LLL: torch.Tensor, M: torch.Tensor, nt: int, ilow: int, ihigh: int):
    nfft, np, _ = LLL.shape
    freq_ret = torch.zeros((nfft, np), device=LLL.device, dtype=torch.complex128)

    for ifreq in range(ilow, ihigh):
        # (np, nq) @ (nq, ) = (np, )
        y = LLL[ifreq, :, :] @ M[ifreq, :]

        freq_ret[ifreq, :] = y
        freq_ret[nfft + 1 - ifreq, :] = torch.conj(y)

    freq_ret[nfft // 2 + 1, :] = 0
    time_ret = torch.real(torch.fft.ifft(freq_ret, dim=0))[:nt, :]

    return freq_ret, time_ret


def radon3d_forward_adjoint(
    LLL: torch.Tensor, M: torch.Tensor, nt: int, ilow: int, ihigh: int
):
    nfft, _, nq = LLL.shape
    freq_ret = torch.zeros((nfft, nq), device=LLL.device, dtype=torch.complex128)

    for ifreq in range(ilow, min(ihigh, nfft)):
        # (nq, np) @ (np, ) = (nq, )
        y = LLL[ifreq, :, :].T @ M[ifreq, :]

        freq_ret[ifreq, :] = y
        freq_ret[nfft + 1 - ifreq, :] = torch.conj(y)

    freq_ret[nfft // 2 + 1, :] = 0
    time_ret = torch.real(torch.fft.ifft(freq_ret, dim=0))[:nt, :]

    return freq_ret, time_ret


def cal_step_size(kernels: torch.Tensor, nt: int, ilow: int, ihigh: int):
    nfft, np, nq = kernels.shape
    b_k = torch.randn((nt, nq), device=kernels.device, dtype=torch.float64)
    B_k = torch.fft.fft(b_k, nfft, dim=0)

    for _ in range(2):
        tmp, _ = radon3d_forward(kernels, B_k, nt, ilow, ihigh)
        _, b_k1 = radon3d_forward_adjoint(kernels, tmp, nt, ilow, ihigh)
        b_k = b_k1 / (1e-10 + torch.linalg.vector_norm(b_k1))
        B_k = torch.fft.fft(b_k, nfft, dim=0)

    B_k_tmp, _ = radon3d_forward(kernels, B_k, nt, ilow, ihigh)
    _, b_k_tmp = radon3d_forward_adjoint(kernels, B_k_tmp, nt, ilow, ihigh)
    L = torch.sum(b_k * b_k_tmp) / sum(b_k**2)
    return 1 / L * 0.9


def fista(
    m0: torch.Tensor,
    y_freq: torch.Tensor,
    kernels: torch.Tensor,
    nt: int,
    ilow: int,
    ihigh: int,
    maxiter: int,
    lamdb: float,
):
    nfft, _, _ = kernels.shape

    with torch.no_grad():
        m = m0
        s = m
        q_t = 1
        step_size = cal_step_size(kernels, nt, ilow, ihigh)

        for _ in range(maxiter):
            # Z-update
            tmp, _ = radon3d_forward(
                kernels, torch.fft.fft(torch.real(s), nfft, dim=0), nt, ilow, ihigh
            )
            _, approx = radon3d_forward_adjoint(kernels, tmp - y_freq, nt, ilow, ihigh)
            z = s - step_size * approx
            # M-update
            m_prev = m
            m = torch.where(torch.abs(z) > step_size * lamdb, z, 0)
            # Q-update
            q_new = 0.5 * (1 + sqrt(1 + 4 * (q_t**2)))
            # S-update
            s = m + (q_t - 1) / q_new * (m - m_prev)
            q_t = q_new
        return m


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
) -> torch.Tensor:
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
    m = fista(
        m0=m0,
        y_freq=y_freq,
        kernels=kernels,
        nt=nt,
        ilow=ilow,
        ihigh=ihigh,
        maxiter=maxiter,
        lamdb=alphas[0],
    )
    return m


if __name__ == "__main__":
    from crisprf.util.bridging import retrieve_single_xy, heatmap

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = retrieve_single_xy("data/Ps_RF_syn1.mat")

    nt = 5000
    dt = 0.02

    _alpha0_range = torch.linspace(0.8, 3.7, 30)
    alpha1 = 1

    y = data["y"].T.to(DEVICE)
    rayP = data["rayP"].to(DEVICE)
    q = data["q"].to(DEVICE)
    for alpha0 in tqdm(_alpha0_range):
        exp_name = f"y_hat_{alpha0:.4f}_{alpha1:.4f}"
        if osp.exists(f"log/{exp_name}.pt"):
            continue

        y_hat = (
            sparse_inverse_radon_fista(
                y=y,
                rayP=rayP,
                q=q,
                seconds=nt * dt,
                freq_bounds=(0, 1 / 2 / dt),
                alphas=(1, 0.2),
                maxiter=20,
                device=DEVICE,
            )
            .T.detach()
            .cpu()
        )
        torch.save(y_hat, f"log/{exp_name}.pt")
        heatmap(rng=[-1, 1], **{exp_name: y_hat})
