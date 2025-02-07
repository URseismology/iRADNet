import torch
import torch.nn.functional as F
from math import sqrt
from .radon3d import radon3d_forward, radon3d_forward_adjoint, cal_step_size


def fista(
    x0: torch.Tensor,
    y_freq: torch.Tensor,
    L: torch.Tensor,
    nt: int,
    ilow: int,
    ihigh: int,
    n_layers: int,
    lambd: float,
):
    nfft, _, _ = L.shape

    fixed_params = dict(L=L, nt=nt, ilow=ilow, ihigh=ihigh)
    with torch.no_grad():
        x = x0
        z = x
        q_t = torch.ones(1, device=x.device)
        step_size = cal_step_size(**fixed_params)

        for _ in range(n_layers):
            # z update
            # y_hat = A(x)
            y_hat_freq, _ = radon3d_forward(
                torch.fft.fft(torch.real(z), nfft, dim=0), **fixed_params
            )
            # x_hat = A*(y_hat - y_freq)
            _, x_hat = radon3d_forward_adjoint(y_hat_freq - y_freq, **fixed_params)
            # threshold
            x_prev = x
            x = F.softshrink(z - step_size * x_hat, step_size * lambd)
            # q update
            q_new = (1 + torch.sqrt(1 + 4 * q_t**2)) / 2
            # s update
            z = x + (q_t - 1) / q_new * (x - x_prev)
            q_t = q_new
            yield x
        # return m
