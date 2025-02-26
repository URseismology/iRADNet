import torch

from tqdm import tqdm

from typing import Generator

from crisprf.model.FISTA import sparse_inverse_radon_fista
from crisprf.util import SRTDataset, eval_metrics

DEVICE = torch.device("cuda")


def grid_search():
    with open(f"log/fista_param_search.csv", "a") as f:
        for l in torch.linspace(0.8, 2.6, 10):
            for mu in torch.linspace(0.1, 0.9, 9):
                for d in inference(l, mu):
                    f.write(f"{l:.2f},{mu:.2f},")
                    f.write(",".join(map(str, d)) + "\n")


def inference(
    lambd: float, mu: float
) -> Generator[tuple[int, float, float, int], None, None]:
    dataset = SRTDataset(device=DEVICE)

    for i, sample in enumerate(dataset):
        for x_hat in sparse_inverse_radon_fista(
            sample,
            alphas=(lambd, mu),
            n_layers=10,
            device=DEVICE,
        ):
            pass
        mse, nmse, nonzeros = eval_metrics(x_hat, sample["x"])
        yield mse, nmse, nonzeros


if __name__ == "__main__":
    grid_search()
