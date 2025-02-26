import torch
import torch.fft as fft

from crisprf.util import AUTO_DEVICE, FREQ_DTYPE, TIME_DTYPE, RFDataShape


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
        q range (nQ,)
    rayP : torch.Tensor
        ray parameters (nP,)
    shapes : RFDataShape
        A shape recording object, containing nFFT, nP, nQ, dT
    N : int, optional
        power of rayP, by default 2
    device : torch.device, optional
        device to run the computation, by default AUTO_DEVICE

    Returns
    -------
    torch.Tensor
        3D Radon transform matrix (nFFT, nP, nQ)
    """
    nFFT, nP, nQ = shapes.nFFT, shapes.nP, shapes.nQ
    dT = shapes.dT

    # init radon transform matrix
    radon3d = torch.zeros((nFFT, nP, nQ), device=device, dtype=FREQ_DTYPE)
    ifreq_f = (
        2.0 * torch.pi * torch.arange(nFFT, device=device, dtype=TIME_DTYPE) / nFFT / dT
    )
    # (nFFT) @ (nP) @ (nQ) = (nFFT, nP, nQ)
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
        code in freq domain (nFFT, nQ)
    radon3d : torch.Tensor
        3D matrix (nFFT, nP, nQ)
    ilow : int
        low frequency index
    ihigh : int
        high frequency index

    Returns
    -------
    torch.Tensor
        :math:`\\tilde{y}_f = L(x_f)` in frequency domain
    """
    nFFT, nP, nQ = radon3d.shape
    assert x_freq.shape == (nFFT, nQ)
    y_freq = torch.zeros((nFFT, nP), device=radon3d.device, dtype=FREQ_DTYPE)

    # (slice, nP, nQ) @ (slice, nQ, ) = (slice, nP, )
    y_slice = torch.einsum(
        "bpq,bq->bp", radon3d[ilow:ihigh, :, :], x_freq[ilow:ihigh, :].conj()
    )

    # symmetrically fill the rest of the frequency domain, center -> 0
    y_freq[ilow:ihigh, :] = y_slice.conj_physical()
    y_freq[nFFT + 1 - ihigh : nFFT + 1 - ilow, :] = y_slice.flip(0)
    y_freq[nFFT // 2, :] = 0.0

    return y_freq


def radon3d_forward_adjoint(
    y_freq: torch.Tensor, radon3d: torch.Tensor, ilow: int, ihigh: int
) -> torch.Tensor:
    """3D adjoint radon transform :math:`L*` frequency domain :math:`\tilde{x}_f = L*(y_f)`

    Parameters
    ----------
    y_freq : torch.Tensor
        image in freq domain (nFFT, nP)
    radon3d : torch.Tensor
        3D matrix (nFFT, nP, nQ)
    ilow : int
        low frequency index
    ihigh : int
        high frequency index

    Returns
    -------
    torch.Tensor
        :math:`\tilde{x}_f = L*(y_f)` in frequency domain
    """
    nFFT, nP, nQ = radon3d.shape
    assert y_freq.shape == (nFFT, nP)
    x_freq = torch.zeros((nFFT, nQ), device=radon3d.device, dtype=FREQ_DTYPE)

    # (slice, nP, nQ) @ (slice, nP, ) = (slice, nQ, )
    x_slice = torch.einsum(
        "bpq,bp->bq", radon3d[ilow:ihigh, :, :].conj(), y_freq[ilow:ihigh, :].conj()
    )

    # symmetrically fill the rest of the frequency domain
    x_freq[ilow:ihigh, :] = x_slice.conj_physical()
    x_freq[nFFT + 1 - ihigh : nFFT + 1 - ilow, :] = x_slice.flip(0)
    x_freq[nFFT // 2, :] = 0.0

    return x_freq


def freq2time(inp_freq: torch.Tensor, nT: int) -> torch.Tensor:
    return torch.real(fft.ifft(inp_freq, dim=0)[:nT, :])


def time2freq(inp: torch.Tensor, nFFT: int) -> torch.Tensor:
    return fft.fft(torch.real(inp), n=nFFT, dim=0)


def cal_lipschitz(radon3d: torch.Tensor, nT: int, ilow: int, ihigh: int):
    # estimate the max eigenvalue of A* A
    nFFT, nP, nQ = radon3d.shape
    x = torch.rand((nT, nQ), device=radon3d.device, dtype=TIME_DTYPE)
    x_freq = torch.fft.fft(x, nFFT, dim=0)

    for _ in range(2):
        y_freq = radon3d_forward(x_freq, radon3d=radon3d, ilow=ilow, ihigh=ihigh)
        x_freq = radon3d_forward_adjoint(
            y_freq, radon3d=radon3d, ilow=ilow, ihigh=ihigh
        )
        x = freq2time(x_freq, nT)
        x = x / torch.linalg.norm(x, ord=2)
        x_freq = torch.fft.fft(x, nFFT, dim=0)

    y_freq = radon3d_forward(x_freq, radon3d=radon3d, ilow=ilow, ihigh=ihigh)
    x1_freq = radon3d_forward_adjoint(y_freq, radon3d=radon3d, ilow=ilow, ihigh=ihigh)
    x1 = freq2time(x1_freq, nT)
    lipschitz = torch.sum(x * x1) / torch.sum(x**2)
    return lipschitz.item()
