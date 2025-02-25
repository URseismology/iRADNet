import torch
import torch.nn.functional as F

from ..util.bridging import RFDataShape
from .radon3d import (
    cal_lipschitz,
    freq2time,
    radon3d_forward,
    radon3d_forward_adjoint,
    time2freq,
)


def fista(
    x0: torch.Tensor,
    y_freq: torch.Tensor,
    radon3d: torch.Tensor,
    shapes: RFDataShape,
    ilow: int,
    ihigh: int,
    n_layers: int,
    lambd: float,
    alpha: float = 0.9,
):
    with torch.no_grad():
        x = x0
        z = x
        q_t = torch.ones(1, device=x.device)
        step_size = alpha / cal_lipschitz(
            radon3d=radon3d, nT=shapes.nT, ilow=ilow, ihigh=ihigh
        )

        for _ in range(n_layers):
            # z update
            # y_hat = A(x)
            y_tilde_freq = radon3d_forward(
                time2freq(z, shapes.nFFT),
                radon3d=radon3d,
                ilow=ilow,
                ihigh=ihigh,
            )
            # x_tilde = F^-1 L*(y_tilde_freq - y_freq)
            x_tilde_freq = radon3d_forward_adjoint(
                y_tilde_freq - y_freq, radon3d=radon3d, ilow=ilow, ihigh=ihigh
            )
            x_tilde = freq2time(x_tilde_freq, shapes.nT)
            # threshold
            x_prev = x
            x = F.softshrink(z - (step_size * x_tilde), step_size * lambd)
            # q update
            q_new = (1 + torch.sqrt(1 + 4 * q_t**2)) / 2
            # s update
            z = x + (q_t - 1) / q_new * (x - x_prev)
            q_t = q_new
            yield x
        # return m
