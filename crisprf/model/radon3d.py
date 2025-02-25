import torch
import torch.fft as fft

from crisprf.util.bridging import RFDataShape
from crisprf.util.constants import AUTO_DEVICE, FREQ_DTYPE, TIME_DTYPE


def init_radon3d_mat(
    q: torch.Tensor,
    rayP: torch.Tensor,
    shapes: RFDataShape,
    N: int = 2,
    device: torch.device = AUTO_DEVICE,
) -> torch.Tensor:
    """
    Initializes the 3D Radon transform matrix.

    Parameters
    ----------
    q : torch.Tensor
        q range (nq,)
    rayP : torch.Tensor
        ray parameters (np,)
    shapes : RFDataShape
        A shape recording object, containing nfft, np, nq, dt
    N : int, optional
        power of rayP, by default 2
    device : torch.device, optional
        device to run the computation, by default AUTO_DEVICE

    Returns
    -------
    torch.Tensor
        3D Radon transform matrix (nfft, np, nq)
    """
    nfft, np, nq = shapes.nfft, shapes.np, shapes.nq
    dt = shapes.dt

    # init radon transform matrix
    radon3d = torch.zeros((nfft, np, nq), device=device, dtype=FREQ_DTYPE)
    ifreq_f = (
        2.0 * torch.pi * torch.arange(nfft, device=device, dtype=TIME_DTYPE) / nfft / dt
    )
    # (nfft) @ (np) @ (nq) = (nfft, np, nq)
    radon3d[:, :, :] = torch.exp(
        1j
        * torch.einsum(
            "f,p,q->fpq", ifreq_f, torch.pow(rayP.to(device), N), q.to(device)
        ).to(FREQ_DTYPE)
    )
    return radon3d


def radon3d_forward(
    x_freq: torch.Tensor, radon3d: torch.Tensor, ilow: int, ihigh: int
) -> torch.Tensor:
    """3D radon transform :math:`L` frequency domain :math:`\\tilde{y}_f = L(x_f)`

    Parameters
    ----------
    x_freq : torch.Tensor
        code in freq domain (nfft, nq)
    radon3d : torch.Tensor
        3D matrix (nfft, np, nq)
    ilow : int
        low frequency index
    ihigh : int
        high frequency index

    Returns
    -------
    torch.Tensor
        :math:`\\tilde{y}_f = L(x_f)` in frequency domain
    """
    nfft, np, nq = radon3d.shape
    assert x_freq.shape == (nfft, nq)
    y_freq = torch.zeros((nfft, np), device=radon3d.device, dtype=FREQ_DTYPE)

    # (slice, np, nq) @ (slice, nq, ) = (slice, np, )
    y_slice = torch.einsum(
        "bpq,bq->bp", radon3d[ilow:ihigh, :, :], x_freq[ilow:ihigh, :].conj()
    )

    # symmetrically fill the rest of the frequency domain, center -> 0
    y_freq[ilow:ihigh, :] = y_slice.conj_physical()
    y_freq[nfft + 1 - ihigh : nfft + 1 - ilow, :] = y_slice.flip(0)
    y_freq[nfft // 2, :] = 0.0

    return y_freq


def radon3d_forward_adjoint(
    y_freq: torch.Tensor, radon3d: torch.Tensor, ilow: int, ihigh: int
) -> torch.Tensor:
    """3D adjoint radon transform :math:`L*` frequency domain :math:`\tilde{x}_f = L*(y_f)`

    Parameters
    ----------
    y_freq : torch.Tensor
        image in freq domain (nfft, np)
    radon3d : torch.Tensor
        3D matrix (nfft, np, nq)
    ilow : int
        low frequency index
    ihigh : int
        high frequency index

    Returns
    -------
    torch.Tensor
        :math:`\tilde{x}_f = L*(y_f)` in frequency domain
    """
    nfft, np, nq = radon3d.shape
    assert y_freq.shape == (nfft, np)
    x_freq = torch.zeros((nfft, nq), device=radon3d.device, dtype=FREQ_DTYPE)

    # (slice, np, nq) @ (slice, np, ) = (slice, nq, )
    x_slice = torch.einsum(
        "bpq,bp->bq", radon3d[ilow:ihigh, :, :].conj(), y_freq[ilow:ihigh, :].conj()
    )

    # symmetrically fill the rest of the frequency domain
    x_freq[ilow:ihigh, :] = x_slice.conj_physical()
    x_freq[nfft + 1 - ihigh : nfft + 1 - ilow, :] = x_slice.flip(0)
    x_freq[nfft // 2, :] = 0.0

    return x_freq


def freq2time(inp_freq: torch.Tensor, nt: int) -> torch.Tensor:
    return torch.real(fft.ifft(inp_freq, dim=0)[:nt, :])


def time2freq(inp: torch.Tensor, nfft: int) -> torch.Tensor:
    return fft.fft(torch.real(inp), n=nfft, dim=0)


def cal_lipschitz(radon3d: torch.Tensor, nt: int, ilow: int, ihigh: int):
    # estimate the max eigenvalue of A* A
    nfft, np, nq = radon3d.shape
    x = torch.rand((nt, nq), device=radon3d.device, dtype=TIME_DTYPE)
    x_freq = torch.fft.fft(x, nfft, dim=0)

    for _ in range(2):
        y_freq = radon3d_forward(x_freq, radon3d=radon3d, ilow=ilow, ihigh=ihigh)
        x_freq = radon3d_forward_adjoint(
            y_freq, radon3d=radon3d, ilow=ilow, ihigh=ihigh
        )
        x = freq2time(x_freq, nt)
        x = x / torch.linalg.norm(x, ord=2)
        x_freq = torch.fft.fft(x, nfft, dim=0)

    y_freq = radon3d_forward(x_freq, radon3d=radon3d, ilow=ilow, ihigh=ihigh)
    x1_freq = radon3d_forward_adjoint(y_freq, radon3d=radon3d, ilow=ilow, ihigh=ihigh)
    x1 = freq2time(x1_freq, nt)
    lipschitz = torch.sum(x * x1) / torch.sum(x**2)
    return lipschitz.item()
