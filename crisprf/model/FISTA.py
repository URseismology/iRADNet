import torch
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
        q_t = 1
        step_size = cal_step_size(**fixed_params)

        for _ in range(n_layers):
            # z update
            _As, _ = radon3d_forward(
                torch.fft.fft(torch.real(z), nfft, dim=0), **fixed_params
            )
            # now _AAs <=> A*(Ax - y_freq)
            _, _AAs = radon3d_forward_adjoint(_As - y_freq, **fixed_params)
            # threshold
            x_prev = x
            x = torch.where(
                torch.abs(z - step_size * _AAs) > step_size * lambd,
                z - step_size * _AAs,
                0,
            )
            # q update
            q_new = 0.5 * (1 + sqrt(1 + 4 * (q_t**2)))
            # s update
            z = x + (q_t - 1) / q_new * (x - x_prev)
            q_t = q_new
            yield x
        # return m
