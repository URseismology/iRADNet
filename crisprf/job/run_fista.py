from tqdm import tqdm

import argparse

from crisprf.model.FISTA import sparse_inverse_radon_fista
from crisprf.util import AUTO_DEVICE, SRTDataset, eval_metrics


def run_fista(args: argparse.Namespace):
    dataset = SRTDataset(
        # 'data/gt/meta.json',
        snr=args.snr,
        device=args.device,
    )
    shapes = dataset.shapes

    _saved_first = False
    for sample in tqdm(dataset, desc="FISTA/datapoints"):
        for x_hat in sparse_inverse_radon_fista(
            sample=sample,
            shapes=shapes,
            alphas=(args.lambd, args.mu),
            n_layers=args.n_layers,
        ):
            eval_metrics(
                pred=x_hat,
                gt=sample["x"],
                log_path="log/FISTA.jsonl",
                log_settings={
                    "snr": args.snr,
                    "n_layers": args.n_layers,
                    "lambd": args.lambd,
                    "mu": args.mu,
                },
                # save the first sample as a reference
                # fig_path=None if _saved_first else f"fig/fista_snr={args.snr}.png",
                **sample,
            )
            _saved_first = True


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
