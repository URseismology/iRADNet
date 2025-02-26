from crisprf.model.FISTA import sparse_inverse_radon_fista
from crisprf.util.bridging import retrieve_single_xy
from crisprf.util.constants import AUTO_DEVICE
from crisprf.util.evaluation import eval_metrics

SAMPLE = retrieve_single_xy(device=AUTO_DEVICE)


def test_fista():
    for i, x_hat in enumerate(
        sparse_inverse_radon_fista(
            SAMPLE,
            alphas=(1.0, 0.2),
            n_layers=10,
            device=AUTO_DEVICE,
        )
    ):
        eval_metrics(
            pred=x_hat,
            gt=SAMPLE["x"],
            # fig_path=f"tmp/fista/fista_{i}.png",
            log_path=f"log/fista.csv",
            **SAMPLE,
        )


if __name__ == "__main__":
    test_fista()
