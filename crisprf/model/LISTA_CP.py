import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from typing import Generator
from .radon3d import radon3d_forward, radon3d_forward_adjoint, cal_lipschitz


class SRT_LISTA_CP(nn.Module):

    def __init__(
        self,
        L: torch.Tensor,
        n_layers: int,
        lambd: float,
        device: torch.device = torch.device("cuda:0"),
        **kwargs
    ):
        super().__init__()
        assert L is not None
        # L: (nfft, np, nq)
        self.nfft, self.np, self.nq = L.shape

        self.L = L
        self.n_layers = n_layers

        self.W1 = nn.Parameter(L.clone(), requires_grad=True)
        self.W2 = nn.Parameter(L.clone(), requires_grad=True)

        # each eta/gamma for each iteration; we have `n_layers`
        L = cal_lipschitz(L=L, **kwargs)
        self.gammas = nn.ParameterList(
            [
                nn.Parameter(torch.ones(1, device=device) * 0.9 / L)
                for _ in range(n_layers)
            ]
        )
        self.etas = nn.ParameterList(
            [
                nn.Parameter(torch.ones(1, device=device) * 0.9 / L)
                for _ in range(n_layers)
            ]
        )

        self.reinit_num = 0
        self.lambd = lambd

    def _shrink(self, x: torch.Tensor, eta: torch.Tensor, lambd: float):
        # assert eta.numel() == 1

        return eta * F.softshrink(x / eta, lambd)

    def forward(
        self, x0: torch.Tensor, y: torch.Tensor, **kwargs
    ) -> Generator[torch.Tensor, None, None]:
        # x0: (nt, nq)
        # y: (nfft, np)
        x = x0
        s = x
        q_t = 1

        # obtain x1 based on x0,
        # and so on x2, x3, ..., xT
        for i in range(self.n_layers):
            # _As (nfft, np), _AAs (nt, nq)
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
    n_layers: int,
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
    n_layers : int
        max iteration of ISTA
    lambd : float
        _description_

    Yields
    ------
    torch.Tensor
        :code:`x1` :code:`xT` sparse code after each iteration of ISTA
    """
    lista_model = SRT_LISTA_CP(
        L=L, n_layers=n_layers, lambd=lambd, nt=nt, ilow=ilow, ihigh=ihigh
    )

    return lista_model(x0, y_freq, nt=nt, ilow=ilow, ihigh=ihigh)
