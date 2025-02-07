import torch
import torch.nn.functional as F
import torch.fft as fft


def radon3d_forward(
    x_freq: torch.Tensor, L: torch.Tensor, nt: int, ilow: int, ihigh: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A operator for 3D radon transform :math:`A = F^-1 @ L @ F`
    Forward linear and parabolic Radon transform in frequency domain

    Parameters
    ----------
    x_freq : torch.Tensor
        code in freq domain (nfft, nq)
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
        :math:`\hat{Y}_f, \hat{Y} = A(x)` in frequency domain (complex128) and time domain (float64)
    """
    nfft, np, _ = L.shape
    y_hat_freq = torch.zeros((nfft, np), device=L.device, dtype=torch.complex128)

    # (slice, np, nq) @ (slice, nq, ) = (slice, np, )
    y_hat_freq[ilow:ihigh, :] = torch.einsum(
        "bpq,bq->bp", L[ilow:ihigh, :, :], x_freq[ilow:ihigh, :]
    )

    # symmetrically fill the rest of the frequency domain, center -> 0
    y_hat_freq[nfft - ihigh : nfft - ilow, :] = y_hat_freq[ilow:ihigh, :].conj().flip(0)
    y_hat_freq[nfft // 2, :] = 0
    y_hat = torch.real(fft.ifft(y_hat_freq, dim=0)[:nt, :])

    return y_hat_freq, y_hat


def radon3d_forward_adjoint(
    y_freq: torch.Tensor, L: torch.Tensor, nt: int, ilow: int, ihigh: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A* adjoint operator for 3D radon transform :math:`A* = F^-1 @ L^T @ F`

    Parameters
    ----------
    y_freq : torch.Tensor
        image in freq domain (nfft, np)
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
        :math:`\hat{X}_f, \hat{X} = A*(y)` in frequency domain (complex128) and time domain (float64)
    """
    nfft, _, nq = L.shape
    x_hat_freq = torch.zeros((nfft, nq), device=L.device, dtype=torch.complex128)

    # (slice, np, nq) @ (slice, np, ) = (slice, nq, )
    x_hat_freq[ilow:ihigh, :] = torch.einsum(
        "bpq,bp->bq", L[ilow:ihigh, :, :], y_freq[ilow:ihigh, :]
    )

    # symmetrically fill the rest of the frequency domain
    x_hat_freq[nfft - ihigh : nfft - ilow, :] = x_hat_freq[ilow:ihigh, :].conj().flip(0)
    x_hat_freq[nfft // 2, :] = 0
    x_hat = torch.real(torch.fft.ifft(x_hat_freq, dim=0)[:nt, :])

    return x_hat_freq, x_hat


def cal_lipschitz(L: torch.Tensor, nt: int, ilow: int, ihigh: int):
    # estimate the max eigenvalue of A* A
    nfft, np, nq = L.shape
    x = torch.rand((nt, nq), device=L.device, dtype=torch.float64)
    x_freq = torch.fft.fft(torch.real(x), nfft, dim=0)

    for _ in range(2):
        y_freq, _ = radon3d_forward(x_freq, L=L, nt=nt, ilow=ilow, ihigh=ihigh)
        _, x = radon3d_forward_adjoint(y_freq, L=L, nt=nt, ilow=ilow, ihigh=ihigh)
        x = x / torch.linalg.norm(x, ord=2)
        x_freq = torch.fft.fft(torch.real(x), nfft, dim=0)

    y_freq, _ = radon3d_forward(x_freq, L=L, nt=nt, ilow=ilow, ihigh=ihigh)
    _, x1 = radon3d_forward_adjoint(y_freq, L=L, nt=nt, ilow=ilow, ihigh=ihigh)
    lipschitz = torch.sum(x * x1) / torch.sum(x**2)
    return lipschitz.item()


def cal_step_size(L: torch.Tensor, nt: int, ilow: int, ihigh: int, alpha: float = 0.9):
    lipschitz = cal_lipschitz(L, nt, ilow, ihigh)
    print(f"Lipschitz={lipschitz}; step={alpha / lipschitz}")
    return alpha / lipschitz


def shrink(x: torch.Tensor, eta: torch.Tensor, lambd: float = 1.0):
    # assert eta.numel() == 1

    return eta * F.softshrink(x / eta, lambd)
