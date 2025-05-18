import torch
from torch.utils.data import DataLoader

from tqdm import trange

import argparse

from crisprf.model import (
    SRT_LISTA,
    SRT_LISTA_CP,
    LISTA_base,
    SRT_AdaLFISTA,
    SRT_AdaLISTA,
    init_radon3d_mat,
    time2freq,
)
from crisprf.util import (
    SRTDataset,
    eval_metrics,
    get_loss,
    plot_outliers,
)


def train_lista(
    model_class: LISTA_base = SRT_LISTA_CP,
    device: torch.device = torch.device("cuda:0"),
    n_layers: int = 10,
    n_epochs: int = 20,
    lr: float = 5e-3,
) -> LISTA_base:
    dataset = SRTDataset(device=device)
    shapes = dataset.shapes
    sample = dataset[0]
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    radon3d = init_radon3d_mat(sample["q"], sample["rayP"], shapes, N=2, device=device)
    model: LISTA_base = model_class(
        radon3d=radon3d,
        n_layers=n_layers,
        shapes=shapes,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in trange(n_epochs):
        for batch in train_loader:
            optimizer.zero_grad()

            x: torch.Tensor = batch["x"].squeeze(0).to(device)
            y: torch.Tensor = batch["y"].squeeze(0).to(device)

            y_freq = time2freq(y, shapes.nFFT)

            x0 = torch.zeros_like(x)
            for x_hat in model(x0=x0, y_freq=y_freq):
                pass

            loss = get_loss(x_hat, x)
            loss.backward()
            optimizer.step()

        eval_metrics(
            x_hat,
            x,
            # fig_path=(
            #     f"tmp/{model_class.__name__}/e{epoch}.png"
            #     if epoch % 5 == 4
            #     else None
            # ),
            log_path=f"log/train/{model_class.__name__}.jsonl",
            log_settings={
                "epoch": epoch,
                "loss": loss.item(),
            },
            **sample,
        )

    model.save_checkpoint()
    return model


def eval_lista(
    model_class: LISTA_base = SRT_LISTA_CP,
    snr: float | None = None,
    device: torch.device = torch.device("cuda:0"),
    fig_path: str = None,
):
    dataset = SRTDataset(snr=snr, device=device)
    sample = dataset[0]
    shapes = dataset.shapes
    radon3d = init_radon3d_mat(sample["q"], sample["rayP"], shapes, N=2, device=device)

    y_freq = time2freq(sample["y_noise"], shapes.nFFT)
    x0 = torch.zeros_like(sample["x"])

    model = LISTA_base.load_checkpoint(model_class, radon3d)
    model.eval()
    model.to(device)

    for x_hat in model(x0, y_freq):
        eval_metrics(
            x_hat,
            sample["x"],
            log_path=f"log/eval/{model_class.__name__}.jsonl",
            log_settings={
                "snr": snr,
                "n_layers": model.n_layers,
            },
            **sample,
        )
    eval_metrics(x_hat, sample["x"], fig_path=fig_path, **sample)

    return model


def plot_difference(
    model_class: LISTA_base = SRT_LISTA_CP,
):
    model = LISTA_base.load_checkpoint(model_class)
    model.eval()

    plot_outliers(model.radon3d, model.W1)
    print(model.__class__.__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="SRT_LISTA_CP")

    # train args
    parser.add_argument("--train", action=argparse.BooleanOptionalAction)
    parser.add_argument("--n_layers", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-3)

    # eval args
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction)
    parser.add_argument("--snr", type=float, default=None)

    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.model == "SRT_LISTA_CP":
        model_class = SRT_LISTA_CP
    elif args.model == "SRT_AdaLISTA":
        model_class = SRT_AdaLISTA
    elif args.model == "SRT_AdaLFISTA":
        model_class = SRT_AdaLFISTA
    elif args.model == "SRT_LISTA":
        model_class = SRT_LISTA
    else:
        raise NotImplementedError

    if args.train is None and args.eval is None:
        print("Specify at least one task, `--train', `--eval' or both")
        exit(1)

    if args.train:
        train_lista(
            model_class=model_class,
            device=args.device,
            n_layers=args.n_layers,
            n_epochs=args.n_epochs,
            lr=args.lr,
        )

    if args.eval:
        eval_lista(
            model_class=model_class,
            snr=args.snr,
            device=args.device,
            # fig_path=f"fig/{args.model}_snr={args.snr}.png",
        )

    exit(0)
