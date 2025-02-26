import torch
from torch.utils.data import DataLoader

from crisprf.model import (
    SRT_LISTA_CP,
    LISTA_base,
    SRT_AdaLISTA,
    init_radon3d_mat,
    time2freq,
)
from crisprf.util import (
    RFDataShape,
    SRTDataset,
    eval_metrics,
    gen_noise,
    get_loss,
    plot_outliers,
    plot_sample,
)


def train_lista(
    model_class: LISTA_base = SRT_LISTA_CP,
    device: torch.device = torch.device("cuda:0"),
) -> LISTA_base:
    dataset = SRTDataset(device=device)
    sample = dataset[0]
    shapes = RFDataShape.from_sample(**sample)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    radon3d = init_radon3d_mat(sample["q"], sample["rayP"], shapes, N=2, device=device)
    model: LISTA_base = model_class(
        radon3d=radon3d,
        n_layers=10,
        shapes=shapes,
        device=device,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(20):
        for batch in train_loader:
            x: torch.Tensor = batch["x"].squeeze(0).to(device)
            y: torch.Tensor = batch["y"].squeeze(0).to(device)

            # (nT, nP) -> (nFFT, nP)
            y_freq = time2freq(y, shapes.nFFT)

            x0 = torch.zeros_like(x)
            for x_hat in model(x0=x0, y_freq=y_freq):
                pass

            loss = get_loss(x_hat, x)
            print(
                epoch,
                eval_metrics(
                    x_hat,
                    x,
                    f"tmp/{model_class.__name__}_e{epoch}.png",
                    log_path=f"log/{model_class.__name__}.csv",
                    **sample,
                ),
            )
            loss.backward()
            optimizer.step()

    print(model.etas.state_dict())
    print(model.gammas.state_dict())
    model.save_checkpoint()

    return model


def eval_lista(
    model_class: LISTA_base = SRT_AdaLISTA,
    snr: float = None,
    device: torch.device = torch.device("cuda:0"),
    fig_path: str = None,
):
    model = LISTA_base.load_checkpoint(model_class)
    model.eval()
    model.to(device)

    dataset = SRTDataset(device=device)
    sample = dataset[0]
    shapes = RFDataShape.from_sample(**sample)

    y = sample["y"]
    if snr is not None:
        noise = gen_noise(y, dT=shapes.dT, snr=snr)
        plot_sample(**sample, y_noise=y + noise, save_path="tmp/problem.png")
        y = y + noise

    y_freq = time2freq(y, shapes.nFFT)
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


if __name__ == "__main__":
    # train_lista()
    eval_lista(snr=1, fig_path="tmp/adalista_snr=1.0.png")
    # plot_difference()
