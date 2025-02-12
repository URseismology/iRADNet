import torch
import torch.nn.functional as F
import torch.fft as fft
from crisprf.util.constants import FREQ_DTYPE, TIME_DTYPE, nextpow2


def get_shapes(
    y: torch.Tensor, q: torch.Tensor, rayP: torch.Tensor = None, **_
) -> tuple[int, int, int, int, torch.Tensor]:
    nt, np = y.shape
    nfft = 2 * nextpow2(nt)
    nq = q.shape[0]

    if rayP is not None:
        assert rayP.shape == (np,), f"{rayP.shape} != ({np},)"

    return nfft, nt, np, nq


def init_radon3d_mat(
    y: torch.Tensor, q: torch.Tensor, rayP: torch.Tensor, dt: float, N: int = 2, **_
) -> torch.Tensor:
    """
    Initializes the 3D Radon transform matrix.

    Parameters
    ----------
    y : torch.Tensor
        image in freq domain (nt, np)
    q : torch.Tensor
        q range (nq,)
    rayP : torch.Tensor
        ray parameters (np,)
    dt : float
        sampling interval, in seconds

    Returns
    -------
    torch.Tensor
        3D Radon transform matrix (nfft, np, nq)
    """
    nfft, nt, np, nq = get_shapes(y=y, q=q, rayP=rayP)
    # init radon transform matrix
    radon3d = torch.zeros((nfft, np, nq), device=y.device, dtype=FREQ_DTYPE)
    ifreq_f = (
        2.0
        * torch.pi
        * torch.arange(nfft, device=y.device, dtype=TIME_DTYPE)
        / nfft
        / dt
    )
    # (nfft) @ (np) @ (nq) = (nfft, np, nq)
    radon3d[:, :, :] = torch.exp(
        1j * torch.einsum("f,p,q->fpq", ifreq_f, torch.pow(rayP, N), q).to(FREQ_DTYPE)
    )
    return radon3d


def radon3d_forward(
    x_freq: torch.Tensor, L: torch.Tensor, ilow: int, ihigh: int
) -> torch.Tensor:
    """3D radon transform :math:`L` frequency domain :math:`\tilde{y}_f = L(x_f)`

    Parameters
    ----------
    x_freq : torch.Tensor
        code in freq domain (nfft, nq)
    L : torch.Tensor
        3D matrix (nfft, np, nq)
    ilow : int
        low frequency index
    ihigh : int
        high frequency index

    Returns
    -------
    torch.Tensor
        :math:`\tilde{y}_f = L(x_f)` in frequency domain
    """
    nfft, np, _ = L.shape
    y_freq = torch.zeros((nfft, np), device=L.device, dtype=FREQ_DTYPE)

    # (slice, np, nq) @ (slice, nq, ) = (slice, np, )
    y_freq[ilow:ihigh, :] = torch.einsum(
        "bpq,bq->bp", L[ilow:ihigh, :, :], x_freq[ilow:ihigh, :]
    )

    # symmetrically fill the rest of the frequency domain, center -> 0
    y_freq[nfft + 1 - ihigh : nfft + 1 - ilow, :] = (
        y_freq[ilow:ihigh, :].conj().resolve_conj().flip(0)
    )
    y_freq[nfft // 2, :] = 0

    return y_freq


def radon3d_forward_adjoint(
    y_freq: torch.Tensor, L: torch.Tensor, ilow: int, ihigh: int
) -> torch.Tensor:
    """3D adjoint radon transform :math:`L*` frequency domain :math:`\tilde{x}_f = L*(y_f)`

    Parameters
    ----------
    y_freq : torch.Tensor
        image in freq domain (nfft, np)
    L : torch.Tensor
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
    nfft, _, nq = L.shape
    x_freq = torch.zeros((nfft, nq), device=L.device, dtype=FREQ_DTYPE)

    # (slice, np, nq) @ (slice, np, ) = (slice, nq, )
    x_freq[ilow:ihigh, :] = torch.einsum(
        "bpq,bp->bq", L[ilow:ihigh, :, :], y_freq[ilow:ihigh, :]
    )

    # symmetrically fill the rest of the frequency domain
    x_freq[nfft + 1 - ihigh : nfft + 1 - ilow, :] = (
        x_freq[ilow:ihigh, :].conj().resolve_conj().flip(0)
    )
    x_freq[nfft // 2, :] = 0

    return x_freq


def freq2time(inp_freq: torch.Tensor, nt: int) -> torch.Tensor:
    return torch.real(fft.ifft(inp_freq, dim=0)[:nt, :])


def time2freq(inp: torch.Tensor, nfft: int) -> torch.Tensor:
    return fft.fft(torch.real(inp), n=nfft, dim=0)


def cal_lipschitz(L: torch.Tensor, nt: int, ilow: int, ihigh: int):
    # estimate the max eigenvalue of A* A
    nfft, np, nq = L.shape
    x = torch.rand((nt, nq), device=L.device, dtype=TIME_DTYPE)
    x_freq = torch.fft.fft(x, nfft, dim=0)

    for _ in range(2):
        y_freq = radon3d_forward(x_freq, L=L, ilow=ilow, ihigh=ihigh)
        x_freq = radon3d_forward_adjoint(y_freq, L=L, ilow=ilow, ihigh=ihigh)
        x = freq2time(x_freq, nt)
        x = x / torch.linalg.norm(x, ord=2)
        x_freq = torch.fft.fft(x, nfft, dim=0)

    y_freq = radon3d_forward(x_freq, L=L, ilow=ilow, ihigh=ihigh)
    x1_freq = radon3d_forward_adjoint(y_freq, L=L, ilow=ilow, ihigh=ihigh)
    x1 = freq2time(x1_freq, nt)
    lipschitz = torch.sum(x * x1) / torch.sum(x**2)
    return lipschitz.item()


def cal_step_size(L: torch.Tensor, nt: int, ilow: int, ihigh: int, alpha: float = 1.0):
    lipschitz = cal_lipschitz(L, nt, ilow, ihigh)
    print(f"Lipschitz={lipschitz}; step={alpha / lipschitz}")
    return alpha / lipschitz


def shrink(x: torch.Tensor, eta: torch.Tensor, lambd: float = 1.0):
    # assert eta.numel() == 1

    return eta * F.softshrink(x / eta, lambd)
