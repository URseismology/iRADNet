import argparse

from crisprf.model.FISTA import sparse_inverse_radon_fista
from crisprf.util import AUTO_DEVICE, eval_metrics, save_results


def run_fista(args: argparse.Namespace):
    for i, sample_with_result in enumerate(
        sparse_inverse_radon_fista(
            alphas=(args.lambd, args.mu),
            n_layers=args.n_layers,
            snr=args.snr,
            device=args.device,
        )
    ):
        eval_metrics(
            pred=sample_with_result["x_hat"],
            gt=sample_with_result["x"],
            # save the first sample as a reference
            fig_path=f"fig/fista_snr={args.snr}.png" if i == 0 else None,
            **sample_with_result,
        )
        save_results(sample_with_result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--n_layers", type=int, default=10)
    parser.add_argument("--lambd", type=float, default=1.4)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=AUTO_DEVICE)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_fista(args)
