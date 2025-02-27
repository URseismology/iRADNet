import torch
from torch.utils.data import DataLoader

import argparse

from crisprf.model import (
    SRT_LISTA_CP,
    LISTA_base,
    SRT_AdaLISTA,
    init_radon3d_mat,
    time2freq,
)
from crisprf.util import (
    SRTDataset,
    eval_metrics,
    get_loss,
    plot_outliers,
    plot_sample,
)


def train_lista(
    model_class: LISTA_base = SRT_LISTA_CP,
    device: torch.device = torch.device("cuda:0"),
) -> LISTA_base:
    dataset = SRTDataset(device=device)
    shapes = dataset.shapes
    sample = dataset[0]
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    radon3d = init_radon3d_mat(sample["q"], sample["rayP"], shapes, N=2, device=device)
    model: LISTA_base = model_class(
        radon3d=radon3d,
        n_layers=10,
        shapes=shapes,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    for epoch in range(20):
        for batch in train_loader:
            optimizer.zero_grad()

            x: torch.Tensor = batch["x"].squeeze(0).to(device)
            y: torch.Tensor = batch["y"].squeeze(0).to(device)

            # (nT, nP) -> (nFFT, nP)
            y_freq = time2freq(y, shapes.nFFT)

            x0 = torch.zeros_like(x)
            for x_hat in model(x0=x0, y_freq=y_freq):
                pass

            loss = get_loss(x_hat, x)
            loss.backward()
            optimizer.step()

        print(
            epoch,
            eval_metrics(
                x_hat,
                x,
                fig_path=f"tmp/{model_class.__name__}/e{epoch}.png",
                log_path=f"log/{model_class.__name__}.csv",
                **sample,
            ),
        )

    model.save_checkpoint()
    return model


def eval_lista(
    model_class: LISTA_base = SRT_LISTA_CP,
    snr: float | None = None,
    device: torch.device = torch.device("cuda:0"),
    fig_path: str = None,
):
    model = LISTA_base.load_checkpoint(model_class)
    model.eval()
    model.to(device)

    dataset = SRTDataset(snr=snr, device=device)
    sample = dataset[0]
    shapes = dataset.shapes

    y_freq = time2freq(sample["y_noise"], shapes.nFFT)
    x0 = torch.zeros_like(sample["x"])

    for x_hat in model(x0, y_freq):
        pass
    print(eval_metrics(x_hat, sample["x"], fig_path=fig_path, **sample))

    return model


def plot_difference(
    model_class: LISTA_base = SRT_LISTA_CP,
):
    model = LISTA_base.load_checkpoint(model_class)
    model.eval()

    # fig = plot_radon2d(model.radon3d[1], model.W1[1])
    # fig.savefig("tmp/w1_1.png")
    # fig = plot_radon2d(model.radon3d[4096], model.W1[4096])
    # fig.savefig("tmp/w1_4096.png")
    plot_outliers(model.radon3d, model.W1)
    print(model.__class__.__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="SRT_LISTA_CP")
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.model == "SRT_LISTA_CP":
        model_class = SRT_LISTA_CP
    elif args.model == "SRT_AdaLISTA":
        model_class = SRT_AdaLISTA
    else:
        raise NotImplementedError

    if args.eval is None or not args.eval:
        train_lista(
            model_class=model_class,
            device=args.device,
        )
    eval_lista(
        model_class=model_class,
        snr=args.snr,
        device=args.device,
        fig_path=f"tmp/{args.model}_snr={args.snr}.png",
    )
