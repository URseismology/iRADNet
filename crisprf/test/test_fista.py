import torch

import pandas as pd

from crisprf.model.FISTA import fista
from crisprf.model.solver import sparse_inverse_radon_fista
from crisprf.util.bridging import retrieve_single_xy
from crisprf.util.constants import AUTO_DEVICE
from crisprf.util.evaluation import eval_metrics

SAMPLE = retrieve_single_xy(device=AUTO_DEVICE)
dT = 0.02


def test_fista():
    stats = []

    for i, (x_hat, elapsed) in enumerate(
        sparse_inverse_radon_fista(
            SAMPLE,
            alphas=(1.0, 0.2),
            n_layers=10,
            device=AUTO_DEVICE,
            ista_fn=fista,
        )
    ):
        mse, nmse, nonzeros = eval_metrics(
            pred=x_hat,
            gt=SAMPLE["x"],
            save_path=f"fig/fista_{i}.png",
            **SAMPLE,
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
