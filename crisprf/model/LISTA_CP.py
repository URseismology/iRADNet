import torch
import torch.nn as nn

from math import sqrt
from typing import Generator

from ..util import AUTO_DEVICE, RFDataShape, shrink_soft
from .LISTA_base import LISTA_base
from .radon3d import (
    cal_lipschitz,
    freq2time,
    radon3d_forward,
    radon3d_forward_adjoint,
    time2freq,
)


class SRT_LISTA_CP(LISTA_base):

    def __init__(
        self,
        radon3d: torch.Tensor,
        n_layers: int,
        shapes: RFDataShape,
        shared_theta: bool = False,
        shared_weight: bool = True,
        freq_index_bounds: tuple[int, int] = None,
        alpha: float = 0.9,
        device: torch.device = AUTO_DEVICE,
    ):
        super().__init__(
            radon3d=radon3d,
            n_layers=n_layers,
            shapes=shapes,
            shared_theta=False,
            shared_weight=True,
            freq_index_bounds=freq_index_bounds,
            alpha=alpha,
            device=device,
        )

        # \Theta = {W1, W2} \cup {gamma(k), eta(k)}^K_{k=1}
        self.W1 = nn.Parameter(radon3d.clone()).to(device)
        self.W2 = nn.Parameter(radon3d.clone()).to(device)

        # each eta/gamma for each iteration; we have `n_layers`
        self.lip = cal_lipschitz(
            radon3d=radon3d, nT=shapes.nT, ilow=self.ilow, ihigh=self.ihigh
        )

    def forward(
        self, x0: torch.Tensor, y_freq: torch.Tensor
    ) -> Generator[torch.Tensor, None, None]:
        # x0: (nT, nQ)
        # y: (nFFT, nP)
        x = x0
        z = x
        q_t = 1

        # obtain x1 based on x0,
        # and so on x2, x3, ..., xT
        for k in range(self.n_layers):
            # y_tilde = A(x)
            y_tilde_freq = radon3d_forward(
                x_freq=time2freq(z, self.shapes.nFFT),
                radon3d=self.W1,
                ilow=self.ilow,
                ihigh=self.ihigh,
            )
            # x_tilde = F^-1 L*(y_tilde - y_freq)
            x_tilde_freq = radon3d_forward_adjoint(
                y_tilde_freq - y_freq, radon3d=self.W2, ilow=self.ilow, ihigh=self.ihigh
            )
            x_tilde = freq2time(x_tilde_freq, nT=self.shapes.nT)

            # shrinking
            x_prev = x
            x = shrink_soft(
                z - self.get_gamma(k) * x_tilde / self.lip, self.get_eta(k) / self.lip
            )

            # q update
            q_t1 = (1 + sqrt(1 + 4 * q_t**2)) / 2
            # z update
            z = x + (q_t - 1) / q_t1 * (x - x_prev)
            q_t = q_t1

            # next do update on _AAx to get x
            yield x


def lista(
    x0: torch.Tensor,
    y_freq: torch.Tensor,
    radon3d: torch.Tensor,
    shapes: RFDataShape,
    ilow: int,
    ihigh: int,
    n_layers: int,
    lambd: None = None,
):
    """_summary_

    Parameters
    ----------
    x0 : torch.Tensor
        inital sparse code guess
    y_freq : torch.Tensor
        signal in fourier domain
    radon3d : torch.Tensor
        radon3d tensor (nFFT, nP, nQ)
    shapes : RFDataShape
        shape information, e.g. nFFT, nT
    ilow : int
        freq low index
    ihigh : int
        freq high index
    n_layers : int
        max iteration of ISTA
    lambd : None
        FISTA argument, only to retain same parameters

    Yields
    ------
    torch.Tensor
        :code:`x1` :code:`xT` sparse code after each iteration of ISTA
    """
    assert lambd is None

    lista_model = SRT_LISTA_CP(
        radon3d=radon3d,
        n_layers=n_layers,
        shapes=shapes,
        freq_index_bounds=(ilow, ihigh),
    )

    return lista_model(x0, y_freq)
