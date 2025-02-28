import torch
import torch.nn as nn

import os
from time import time_ns

from .plotting import plot_sample

FID_LOSS = nn.MSELoss(reduction="sum")
REG_LOSS = nn.L1Loss(reduction="sum")


def get_loss(pred: torch.Tensor, gt: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    l_fid = FID_LOSS(pred, gt)
    l_reg = REG_LOSS(pred, torch.zeros_like(pred))
    return l_fid + lambd * l_reg


def eval_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    fig_path: str = None,
    log_path: str = None,
    log_settings: dict = None,
    **kwargs,
) -> tuple[float, float, int]:
    mse = FID_LOSS(pred, gt)
    mse_0 = FID_LOSS(gt, torch.zeros_like(gt))
    nmse = mse / mse_0
    nonzeros = torch.count_nonzero(pred)

    if fig_path is not None:
        plot_sample(
            prefix_incl=("x_hat",), save_path=fig_path, **(kwargs | {"x_hat": pred})
        )
        print(f"fig saved to {fig_path}")

    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # does not exist then overwrite then add header
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("timestamp,settings,mse,nmse,nonzeros\n")

        with open(log_path, "a") as f:
            f.write(
                f'{time_ns()},"{log_settings}",{mse.item()},{nmse.item()},{nonzeros.item()}\n'
            )

    return mse.item(), nmse.item(), nonzeros.item()
