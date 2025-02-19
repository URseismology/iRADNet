import torch

from tqdm import tqdm

from typing import Generator

from crisprf.model.FISTA import fista
from crisprf.model.solver import sparse_inverse_radon_fista
from crisprf.util.dataloading import SRTDataset
from crisprf.util.evaluation import eval_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt = 0.02


def grid_search():
    for l in torch.linspace(0.8, 2.6, 10):
        for mu in tqdm(torch.linspace(0.1, 0.9, 9)):
            with open(f"log/l={l:.2f}_mu={mu:.2f}.csv", "a") as f:
                f.write(f"{l:.2f},{mu:.2f},")
                for d in inference(l, mu):
                    f.write(",".join(map(str, d)) + "\n")


def inference(
    lambd: float, mu: float
) -> Generator[tuple[int, float, float, int], None, None]:
    dataset = SRTDataset(device=DEVICE)

    for i, sample in enumerate(dataset):
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
                f"fig/samples/fista_l={lambd:.2f}_mu={mu:.2f}.png" if i == 0 else None,
            )
            break
        yield elapsed, mse, nmse, nonzeros


if __name__ == "__main__":
    grid_search()
