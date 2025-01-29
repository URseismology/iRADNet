import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from typing import Generator


def radon3d_forward(
    x: torch.Tensor, L: torch.Tensor, nt: int, ilow: int, ihigh: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """function for A operator for 3D radon transform

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
        frequency domain (complex128) and time domain (float64) adjoint operator
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
    """function for A* adjoint operator for 3D radon transform

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
        frequency domain (complex128) and time domain (float64) adjoint operator
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
    L = torch.sum(b_k * b_kp) / torch.sum(b_k**2)
    return L


class SRTLISTA(nn.Module):

    def __init__(self, L: torch.Tensor, maxiter: int, lambd: float, **kwargs):
        super().__init__()
        assert L is not None
        # L: (nfft, np, nq)
        self.nfft, self.np, self.nq = L.shape

        self.L = L
        self.maxiter = maxiter

        self.W1 = nn.Parameter(L.clone(), requires_grad=True)
        self.W2 = nn.Parameter(L.clone(), requires_grad=True)
        assert self.W1.device == L.device

        # etas and gammas are for each iteration, and we have `maxiter` iterations
        self.gammas = nn.Parameter(
            torch.ones(maxiter, device=L.device), requires_grad=True
        )
        self.etas = nn.Parameter(
            torch.ones(maxiter, device=L.device), requires_grad=True
        )

        L = cal_lipschitz(L=L, **kwargs)
        self.gammas.data *= 0.9 / L
        self.etas.data *= 0.9 / L
        self.reinit_num = 0
        self.lambd = lambd

    def _shrink(self, x: torch.Tensor, eta: torch.Tensor, lambd: float):
        # assert eta.numel() == 1

        return eta * F.softshrink(x / eta, lambd)

    def forward(
        self, x0: torch.Tensor, y: torch.Tensor, **kwargs
    ) -> Generator[torch.Tensor, None, None]:
        # obtain x0
        x = x0
        s = x
        q_t = 1

        # obtain x1 based on x0,
        # and so on x2, x3, ..., xT
        for i in range(self.maxiter):
            _As, _ = radon3d_forward(
                x=torch.fft.fft(torch.real(s), self.nfft, dim=0), L=self.W1, **kwargs
            )
            _, _AAs = radon3d_forward_adjoint(_As - y, L=self.W2, **kwargs)

            # shrinking
            x_prev = x
            x = self._shrink(s - self.gammas[i] * _AAs, self.etas[i], self.lambd)

            q_t1 = (1 + sqrt(1 + 4 * q_t**2)) / 2
            s = x + (q_t - 1) / q_t1 * (x - x_prev)
            q_t = q_t1

            # next do update on _AAx to get x
            yield x

    pass


def lista(
    x0: torch.Tensor,
    y_freq: torch.Tensor,
    L: torch.Tensor,
    nt: int,
    ilow: int,
    ihigh: int,
    maxiter: int,
    lambd: float,
):
    """_summary_

    Parameters
    ----------
    x0 : torch.Tensor
        inital sparse code guess
    y_freq : torch.Tensor
        _description_
    L : torch.Tensor
        _description_
    nt : int
        _description_
    ilow : int
        freq low index
    ihigh : int
        freq high index
    maxiter : int
        max iteration of ISTA
    lambd : float
        _description_

    Yields
    ------
    torch.Tensor
        :code:`x1` :code:`xT` sparse code after each iteration of ISTA
    """
    lista_model = SRTLISTA(
        L=L, maxiter=maxiter, lambd=lambd, nt=nt, ilow=ilow, ihigh=ihigh
    )

    return lista_model(x0, y_freq, nt=nt, ilow=ilow, ihigh=ihigh)
