import torch


def radon3d_forward(
    x: torch.Tensor, L: torch.Tensor, nt: int, ilow: int, ihigh: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A operator for 3D radon transform :math:`A* = F^-1 @ L @ F`

    Parameters
    ----------
    x : torch.Tensor
        sparse code
    L : torch.Tensor
        3D matrix (nfft, np, nq)
    nt : int
        number of time samples
    ilow : int
        low frequency index
    ihigh : int
        high frequency index

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        frequency domain (complex128) and time domain (float64) as Ax
    """
    nfft, np, _ = L.shape
    freq_ret = torch.zeros((nfft, np), device=L.device, dtype=torch.complex128)

    # (slice, np, nq) @ (slice, nq, ) = (slice, np, )
    freq_ret[ilow:ihigh, :] = torch.einsum(
        "bpq,bq->bp", L[ilow:ihigh, :, :], x[ilow:ihigh, :]
    )

    # symmetrically fill the rest of the frequency domain
    # by freq_ret[i] = freq_ret[nfft + 1 - i].conj()
    freq_ret[nfft + 1 - ihigh : nfft + 1 - ilow, :] = (
        freq_ret[ilow:ihigh, :].conj().flip(0)
    )

    # implicit, since init to 0
    # freq_ret[nfft // 2 + 1, :] = 0

    time_ret = torch.real(torch.fft.ifft(freq_ret, dim=0))[:nt, :]

    return freq_ret, time_ret


def radon3d_forward_adjoint(
    x: torch.Tensor, L: torch.Tensor, nt: int, ilow: int, ihigh: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A* adjoint operator for 3D radon transform :math:`A* = F^-1 @ L^T @ F`

    Parameters
    ----------
    x : torch.Tensor
        sparse code
    L : torch.Tensor
        3D matrix (nfft, np, nq)
    nt : int
        number of time samples
    ilow : int
        low frequency index
    ihigh : int
        high frequency index

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        frequency domain (complex128) and time domain (float64) as A*x
    """
    nfft, _, nq = L.shape
    freq_ret = torch.zeros((nfft, nq), device=L.device, dtype=torch.complex128)

    # (slice, np, nq) @ (slice, np, ) = (slice, nq, )
    freq_ret[ilow:ihigh, :] = torch.einsum(
        "bpq,bp->bq", L[ilow:ihigh, :, :], x[ilow:ihigh, :]
    )

    # symmetrically fill the rest of the frequency domain
    # by freq_ret[i] = freq_ret[nfft + 1 - i].conj()
    freq_ret[nfft + 1 - ihigh : nfft + 1 - ilow, :] = (
        freq_ret[ilow:ihigh, :].conj().flip(0)
    )

    # implicit, since init to 0
    # freq_ret[nfft // 2 + 1, :] = 0

    time_ret = torch.real(torch.fft.ifft(freq_ret, dim=0))[:nt, :]

    return freq_ret, time_ret


def cal_lipschitz(L: torch.Tensor, nt: int, ilow: int, ihigh: int):
    nfft, np, nq = L.shape
    b_k = torch.randn((nt, nq), device=L.device, dtype=torch.float64)
    B_k = torch.fft.fft(b_k, nfft, dim=0)

    for _ in range(2):
        B_tmp, _ = radon3d_forward(B_k, L=L, nt=nt, ilow=ilow, ihigh=ihigh)
        _, b_tmp = radon3d_forward_adjoint(B_tmp, L=L, nt=nt, ilow=ilow, ihigh=ihigh)
        b_k = b_tmp / (1e-10 + torch.linalg.vector_norm(b_tmp, ord=2))
        B_k = torch.fft.fft(b_k, nfft, dim=0)

    B_kp, _ = radon3d_forward(B_k, L=L, nt=nt, ilow=ilow, ihigh=ihigh)
    _, b_kp = radon3d_forward_adjoint(B_kp, L=L, nt=nt, ilow=ilow, ihigh=ihigh)
    lipschitz = torch.sum(b_k * b_kp) / torch.sum(b_k**2)
    return lipschitz


def cal_step_size(L: torch.Tensor, nt: int, ilow: int, ihigh: int, alpha: float = 0.9):
    lipschitz = cal_lipschitz(L, nt, ilow, ihigh)
    return alpha / lipschitz
