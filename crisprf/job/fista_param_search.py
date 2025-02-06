import torch
from crisprf.model.FISTA import fista
from crisprf.model.radon_solver import sparse_inverse_radon_fista
from crisprf.util.evaluation import eval_metrics
from crisprf.util.dataloading import SRTDataset
from typing import Generator
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt = 0.02


def grid_search():
    for l in torch.linspace(0.8, 2.6, 10):
        for mu in tqdm(torch.linspace(0.5, 0.9, 9)):
            for d in inference(l, mu):
                with open(f"log/l={l:.2f}_mu={mu:.2f}.csv", "a") as f:
                    f.write(",".join(map(str, d)) + "\n")


def inference(
    lambd: float, mu: float
) -> Generator[tuple[int, float, float, int], None, None]:
    dataset = SRTDataset(device=DEVICE)

    for sample in dataset:
        for x_hat, elapsed in sparse_inverse_radon_fista(
            **sample,
            dt=dt,
            freq_bounds=(0, 1 / 2 / dt),
            alphas=(lambd, mu),
            n_layers=10,
            device=DEVICE,
            ista_fn=fista,
        ):
            mse, nmse, nonzeros = eval_metrics(
                x_hat,
                sample["x"],
            )
            break
        yield elapsed, mse, nmse, nonzeros


if __name__ == "__main__":
    grid_search()
