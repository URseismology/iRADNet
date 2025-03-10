import torch
import torch.nn.functional as F

from math import sqrt
from typing import Generator

from ..util import (
    AUTO_DEVICE,
    FREQ_DTYPE,
    RFData,
    RFDataShape,
    SRTDataset,
    eval_metrics,
)
from .radon3d import (
    cal_lipschitz,
    freq2time,
    init_radon3d_mat,
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
    alpha: float = 1.0,
):
    static_kwargs = dict(
        radon3d=radon3d,
        ilow=ilow,
        ihigh=ihigh,
    )
    with torch.no_grad():
        x = x0
        z = x0
        q_t = 1
        step_size = alpha / cal_lipschitz(nT=shapes.nT, **static_kwargs)

        for _ in range(n_layers):
            # z update
            # y_hat = A(x)
            x_freq = time2freq(z, shapes.nFFT)
            y_tilde_freq = radon3d_forward(
                x_freq=x_freq,
                out_y=torch.zeros_like(y_freq),
                nFFT=shapes.nFFT,
                **static_kwargs,
            )
            # x_tilde = F^-1 L*(y_tilde_freq - y_freq)
            x_tilde_freq = radon3d_forward_adjoint(
                y_freq=y_tilde_freq - y_freq,
                out_x=torch.zeros_like(x_freq),
                nFFT=shapes.nFFT,
                **static_kwargs,
            )
            x_tilde = freq2time(x_tilde_freq, shapes.nT)
            # threshold
            x_prev = x
            x = F.softshrink(z - (step_size * x_tilde), step_size * lambd)
            # q update
            q_new = (1 + sqrt(1 + 4 * q_t**2)) / 2
            # s update
            z = x + (q_t - 1) / q_new * (x - x_prev)
            q_t = q_new
            yield x


def get_x0(
    y_freq: torch.Tensor,
    radon3d: torch.Tensor,
    ilow: int,
    ihigh: int,
    mu: float,
    shapes: RFDataShape,
) -> torch.Tensor:
    nFFT = shapes.nFFT
    nQ = shapes.nQ
    nT = shapes.nT

    x0_freq = torch.zeros((nFFT, nQ), device=y_freq.device, dtype=FREQ_DTYPE)
    for ifreq in range(ilow, ihigh):
        # (nP, nQ)
        radon2d = radon3d[ifreq]
        # (nP, nQ).T @ (nP, ) -> (nQ, )
        B = radon2d.T.conj() @ y_freq[ifreq].conj()
        # (nP, nQ).T @ (nP, nP) @ (nP, nQ) -> (nQ, nQ)
        A = radon2d.T.conj() @ radon2d

        # (A + alpha * I) x = B, solve for x (nQ, )
        x_i: torch.Tensor = torch.linalg.solve(
            A + mu * torch.eye(nQ, dtype=FREQ_DTYPE, device=y_freq.device), B
        )

        x0_freq[ifreq] = x_i.conj()
        x0_freq[nFFT - ifreq] = x_i

    x0_freq[nFFT // 2] = 0

    return freq2time(x0_freq, nT=nT)


def sparse_inverse_radon_fista(
    alphas: tuple[float, float],
    snr: float | None = None,
    n_layers: int = 10,
    freq_bounds: tuple[float, float] = None,
    device: torch.device = AUTO_DEVICE,
) -> Generator[RFData, None, None]:
    """
    sparse inverse radon transform with FISTA for SRTFISTA reconstruction

    Parameters
    ----------
    alphas : tuple[float, float]
        regularization parameter for lambda and mu
    snr : float, optional
        signal to noise ratio, :math:`inf` if None, by default None
    n_layers : int
        max number of iterations
    freq_bounds : tuple[float, float]
        freq. low cut-off and high cut-off
    device : torch.device, optional
        device to run the computation, by default AUTO_DEVICE

    Returns
    -------
    torch.Tensor
        reconstructed radon image
    """
    dataset = SRTDataset(snr=snr, device=device)
    shapes = dataset.shapes
    nFFT = shapes.nFFT
    dT = shapes.dT

    for sample in dataset:
        # init signal in frequency domain
        y_freq: torch.Tensor = time2freq(sample["y_noise"], nFFT)

        # init radon transform matrix
        radon3d = init_radon3d_mat(
            q=sample["q"], rayP=sample["rayP"], shapes=shapes, device=device
        )

        # determine a [ilow, ihigh) frequency range, bounded by
        # [1, nFFT // 2) <-> (nFFT // 2, nFFT - 1] such that it's symmetric
        # e.g. [1, 32) <-> (32, 63], so ilow = 1, ihigh = 32
        if freq_bounds is not None:
            ilow = max(int(freq_bounds[0] * dT * nFFT), 1)
            ihigh = min(int(freq_bounds[1] * dT * nFFT), nFFT // 2)
        else:
            ilow, ihigh = 1, nFFT // 2

        x0 = get_x0(
            y_freq=y_freq,
            radon3d=radon3d,
            ilow=ilow,
            ihigh=ihigh,
            mu=alphas[1],
            shapes=shapes,
        )

        metrics_out_kwargs = dict(
            log_path="log/fista.jsonl",
            log_settings={
                "snr": snr,
                "n_layers": n_layers,
                "lambd": alphas[0],
                "mu": alphas[1],
            },
        )

        eval_metrics(
            pred=x0,
            gt=sample["x"],
            **metrics_out_kwargs,
            **sample,
        )

        # run FISTA from x(0) -> x(K)
        for x_hat in fista(
            x0=x0,
            y_freq=y_freq,
            radon3d=radon3d,
            shapes=shapes,
            ilow=ilow,
            ihigh=ihigh,
            n_layers=n_layers,
            lambd=alphas[0],
        ):
            eval_metrics(
                pred=x_hat,
                gt=sample["x"],
                **metrics_out_kwargs,
                **sample,
            )

        # yield final x(K) of each sample
        yield sample | {"x_hat": x_hat}
