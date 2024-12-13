import torch
from math import sqrt


def radon3d_forward(
    kernels: torch.Tensor, M: torch.Tensor, nt: int, ilow: int, ihigh: int
):
    nfft, np, _ = kernels.shape
    freq_ret = torch.zeros((nfft, np), device=kernels.device, dtype=torch.complex128)

    for ifreq in range(ilow, ihigh):
        # (np, nq) @ (nq, ) = (np, )
        y = kernels[ifreq, :, :] @ M[ifreq, :]

        freq_ret[ifreq, :] = y
        freq_ret[nfft + 1 - ifreq, :] = torch.conj(y)

    freq_ret[nfft // 2 + 1, :] = 0
    time_ret = torch.real(torch.fft.ifft(freq_ret, dim=0))[:nt, :]

    return freq_ret, time_ret


def radon3d_forward_adjoint(
    kernels: torch.Tensor, M: torch.Tensor, nt: int, ilow: int, ihigh: int
):
    nfft, _, nq = kernels.shape
    freq_ret = torch.zeros((nfft, nq), device=kernels.device, dtype=torch.complex128)

    for ifreq in range(ilow, ihigh):
        # (nq, np) @ (np, ) = (nq, )
        y = kernels[ifreq, :, :].T @ M[ifreq, :]

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
            yield m.detach().cpu().T
        # return m
