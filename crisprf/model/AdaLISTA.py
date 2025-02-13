import torch
import torch.nn as nn

from .radon3d import (
    cal_lipschitz,
    freq2time,
    radon3d_forward,
    radon3d_forward_adjoint,
    shrink,
)


class SRT_AdaLISTA(nn.Module):

    def __init__(
        self,
        L: torch.Tensor,
        n_layers: int,
        device: torch.device = torch.device("cuda:0"),
        **kwargs,
    ):
        super().__init__()
        # L: (nfft, np, nq)
        self.L = L
        self.nfft, self.np, self.nq = L.shape

        self.n_layers = n_layers

        # \Theta = {W1, W2, gamma_k, theta_k}
        self.W1 = nn.Parameter(torch.eye(self.np))
        self.W2 = nn.Parameter(torch.eye(self.nq))

        self.gammas = nn.ParameterList(
            [nn.Parameter(torch.ones(1, device=device)) for _ in range(n_layers)]
        )
        self.thetas = nn.ParameterList(
            [nn.Parameter(torch.ones(1, device=device)) for _ in range(n_layers)]
        )

    def forward(self, x0: torch.Tensor, y, nt, ilow, ihigh):
        # x0: (nt, nq)
        # y: (nfft, np)
        x = torch.zeros_like(x0)

        for i in range(self.n_layers):
            L1 = torch.einsum("bpq,pp->bpq", self.L, self.W1)
            L2 = torch.einsum("bpq,qq->bpq", self.L, self.W2)

            y_tilde_freq = radon3d_forward(
                x_freq=torch.fft.fft(torch.real(x), self.nfft, dim=0),
                L=L1,
                ilow=ilow,
                ihigh=ihigh,
            )
            x_tilde_freq = radon3d_forward_adjoint(
                y_tilde_freq - y, L=L2, ilow=ilow, ihigh=ihigh
            )
            x_tilde = freq2time(x_tilde_freq, nt)
            x = shrink(x - self.gammas[i] * x_tilde, self.thetas[i])

            yield x


def adalista(
    x0: torch.Tensor,
    y_freq: torch.Tensor,
    L: torch.Tensor,
    nt: int,
    ilow: int,
    ihigh: int,
    n_layers: int,
    device: torch.device = torch.device("cpu"),
    **_,
):
    model = SRT_AdaLISTA()
    with torch.no_grad():
        pass
