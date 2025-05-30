import torch
import torch.nn as nn

from ..util import AUTO_DEVICE, RFDataShape, shrink_soft
from .LISTA_base import LISTA_base
from .radon3d import (
    cal_lipschitz,
    freq2time,
    radon3d_forward,
    radon3d_forward_adjoint,
    time2freq,
)


class SRT_AdaLISTA(LISTA_base):

    def __init__(
        self,
        radon3d: torch.Tensor,
        n_layers: int,
        shapes: RFDataShape,
        shared_theta: bool = False,
        shared_weight: bool = True,
        freq_index_bounds: tuple[int, int] = None,
        alpha: float = 1.0,
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

        # \Theta = {W1, W2, gamma_k, theta_k}
        self.W1 = nn.Parameter(
            torch.eye(self.shapes.nQ).unsqueeze(0).expand(self.shapes.nFFT, -1, -1)
        ).to(device)
        self.W2 = nn.Parameter(
            torch.eye(self.shapes.nP).unsqueeze(0).expand(self.shapes.nFFT, -1, -1)
        ).to(device)

        self.lip = cal_lipschitz(
            radon3d=radon3d, ilow=self.ilow, ihigh=self.ihigh, shapes=shapes
        )

    def forward(self, x0: torch.Tensor, y_freq: torch.Tensor):
        # x0: (nT, nQ)
        # y: (nFFT, nP)
        x = x0
        yield x0

        for k in range(self.n_layers):
            L1 = torch.einsum("...fpq,fqq->...fpq", self.radon3d, self.W1)
            L2 = torch.einsum("...fpq,fpp->...fpq", self.radon3d, self.W2)
            x_freq = time2freq(x, nFFT=self.shapes.nFFT)

            y_tilde_freq = radon3d_forward(
                x_freq=x_freq,
                radon3d=L1,
                ilow=self.ilow,
                ihigh=self.ihigh,
                out_y=torch.zeros_like(y_freq),
                nFFT=self.shapes.nFFT,
            )
            x_tilde_freq = radon3d_forward_adjoint(
                y_freq=y_tilde_freq - y_freq,
                radon3d=L2,
                ilow=self.ilow,
                ihigh=self.ihigh,
                out_x=torch.zeros_like(x_freq),
                nFFT=self.shapes.nFFT,
            )
            x_tilde = freq2time(x_tilde_freq, nT=self.shapes.nT)
            x = shrink_soft(
                x - self.get_gamma(k) * x_tilde / self.lip, self.get_eta(k) / self.lip
            )

            yield x
