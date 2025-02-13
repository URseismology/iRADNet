import torch

import pandas as pd

from crisprf.model.FISTA import fista
from crisprf.model.radon_solver import sparse_inverse_radon_fista
from crisprf.util.bridging import retrieve_single_xy
from crisprf.util.evaluation import eval_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE = retrieve_single_xy("data/Ps_RF_syn1.mat", device=DEVICE)
dt = 0.02


def test_fista():
    stats = []

    for i, (x_hat, elapsed) in enumerate(
        sparse_inverse_radon_fista(
            **SAMPLE,
            dt=dt,
            freq_bounds=(0, 1 / 2 / dt),
            alphas=(1.0, 0.2),
            n_layers=10,
            device=DEVICE,
            ista_fn=fista,
        )
    ):
        mse, nmse, nonzeros = eval_metrics(
            x_hat,
            SAMPLE["x"],
            f"fig/fista_{i}.png",
        )

        print(f"{elapsed} {mse:e}/{nmse:4e} {nonzeros:2e}")
        if stats:
            assert nonzeros <= stats[-1][0]
        stats.append((elapsed, mse, nmse, nonzeros))
    pd.DataFrame(
        stats,
        columns=[
            "timestamp",
            "mse",
            "nmse",
            "nonzero",
        ],
    ).to_csv(
        "log/fista.csv",
        index=False,
    )


if __name__ == "__main__":
    test_fista()
