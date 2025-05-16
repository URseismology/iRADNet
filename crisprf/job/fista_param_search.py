import torch

from tqdm import tqdm

import argparse

from crisprf.model.FISTA import sparse_inverse_radon_fista
from crisprf.util import SRTDataset, eval_metrics


def grid_search(args):
    pbar = tqdm(total=len(args.lambds) * len(args.mus))
    for l in args.lambds:
        for mu in args.mus:
            l = l.item() if isinstance(l, torch.Tensor) else l
            mu = mu.item() if isinstance(mu, torch.Tensor) else mu

            pbar.set_description(f"FISTA Param Search")
            pbar.set_postfix(lambd=l, mu=mu)
            inference(lambd=l, mu=mu, n_layers=args.n_layers, device=args.device)
            pbar.update(1)


def inference(lambd: float, mu: float, n_layers: int, device: torch.device) -> None:
    dataset = SRTDataset(device=device)
    for sample in dataset:
        for x_hat in sparse_inverse_radon_fista(
            sample=sample,
            shapes=dataset.shapes,
            alphas=(lambd, mu),
            n_layers=n_layers,
        ):
            # nothing for x^(k)
            pass
        # eval only x^(K)
        eval_metrics(
            pred=x_hat,
            gt=sample["x"],
            log_path="log/fista_param_search.jsonl",
            log_settings={"lambd": lambd, "mu": mu},
        )


def get_args():
    # default search space of lambd and mu
    LAMBDA_SPACE = torch.linspace(0.8, 2.6, 10)
    MU_SPACE = torch.linspace(0.1, 0.9, 9)

    parser = argparse.ArgumentParser()

    # train args
    parser.add_argument("--n_layers", type=int, default=10)
    parser.add_argument("--lambds", type=float, nargs="+", default=LAMBDA_SPACE)
    parser.add_argument("--mus", type=float, nargs="+", default=MU_SPACE)

    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    grid_search(get_args())
