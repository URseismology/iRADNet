import argparse

from crisprf.model.FISTA import sparse_inverse_radon_fista
from crisprf.util import AUTO_DEVICE, eval_metrics, retrieve_single_xy

SAMPLE = retrieve_single_xy(device=AUTO_DEVICE)


def run_fista(args: argparse.Namespace):
    for i, x_hat in enumerate(
        sparse_inverse_radon_fista(
            SAMPLE,
            alphas=(args.lambd, args.mu),
            n_layers=args.n_layers,
            snr=args.snr,
            device=args.device,
        )
    ):
        eval_metrics(
            pred=x_hat,
            gt=SAMPLE["x"],
            # fig_path=f"tmp/fista/fista_{i}.png",
            log_path=f"log/fista.csv",
            **SAMPLE,
        )
    eval_metrics(
        pred=x_hat,
        gt=SAMPLE["x"],
        fig_path=f"tmp/fista_snr={args.snr}.png",
        **SAMPLE,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--n_layers", type=int, default=10)
    parser.add_argument("--lambd", type=float, default=1.4)
    parser.add_argument("--mu", type=float, default=0.4)
    parser.add_argument("--device", type=str, default=AUTO_DEVICE)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_fista(args)
