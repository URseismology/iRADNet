import torch
import torch.nn as nn

from .bridging import plot_sample

FID_LOSS = nn.MSELoss(reduction="sum")
REG_LOSS = nn.L1Loss(reduction="sum")


def get_loss(pred: torch.Tensor, gt: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    l_fid = FID_LOSS(pred, gt)
    l_reg = REG_LOSS(pred, torch.zeros_like(pred))
    return l_fid + lambd * l_reg


def eval_metrics(
    pred: torch.Tensor, gt: torch.Tensor, save_path: str = None, **kwargs
) -> tuple[float, float, int]:
    mse = FID_LOSS(pred, gt)
    mse_0 = FID_LOSS(pred, torch.zeros_like(pred))
    nmse = mse / mse_0
    nonzeros = torch.count_nonzero(pred)

    if save_path is not None:
        plot_sample(prefix_scope=("x",), save_path=save_path, x_pred=pred, **kwargs)
        print(f"fig saved to {save_path}")

    return mse.item(), nmse.item(), nonzeros.item()
