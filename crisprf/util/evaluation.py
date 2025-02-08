import torch
import torch.nn as nn


FID_LOSS = nn.MSELoss()
REG_LOSS = nn.L1Loss(reduction="sum")


def get_loss(pred: torch.Tensor, gt: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    l_fid = FID_LOSS(pred, gt)
    l_reg = REG_LOSS(pred, torch.zeros_like(pred))
    return l_fid + lambd * l_reg


def eval_metrics(
    pred: torch.Tensor, gt: torch.Tensor, save_path: str = None
) -> tuple[float, float, int]:
    mse = FID_LOSS(pred, gt)
    mse_0 = FID_LOSS(pred, torch.zeros_like(pred))
    nmse = mse / mse_0
    nonzeros = torch.count_nonzero(pred)

    if save_path is not None:
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig, (a0, a1) = plt.subplots(1, 2, sharey=True)
        sns.heatmap(pred.detach().cpu().numpy(), center=0, ax=a0)
        sns.heatmap(gt.detach().cpu().numpy(), center=0, ax=a1)
        a0.set_title("pred")
        a1.set_title("gt")
        plt.savefig(save_path)
        print(f"fig saved to {save_path}")

    return mse.item(), nmse.item(), nonzeros.item()
