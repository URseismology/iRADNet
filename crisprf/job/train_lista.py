import torch
from torch.utils.data import DataLoader

from crisprf.model.LISTA_CP import SRT_LISTA_CP
from crisprf.model.AdaLISTA import SRT_AdaLISTA
from crisprf.model.LISTA_base import LISTA_base
from crisprf.model.radon3d import init_radon3d_mat, time2freq
from crisprf.util.bridging import RFDataShape
from crisprf.util.dataloading import SRTDataset
from crisprf.util.evaluation import eval_metrics, get_loss
from crisprf.util.plotting import plot_radon2d


def train_lista(
    srt_model: LISTA_base = SRT_LISTA_CP,
    device: torch.device = torch.device("cuda:0"),
) -> LISTA_base:
    dataset = SRTDataset(device=device)
    sample = dataset[0]
    shapes = RFDataShape.from_sample(**sample)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    radon3d = init_radon3d_mat(sample["q"], sample["rayP"], shapes, N=2, device=device)
    model: LISTA_base = srt_model(
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

            for x_hat in model(x, y_freq):
                pass

            loss = get_loss(x_hat, x)
            print(
                epoch,
                eval_metrics(
                    x_hat,
                    x,
                    # f"tmp/lista_e{epoch}.png",
                    **sample,
                ),
            )
            loss.backward()
            optimizer.step()

    print(model.etas.state_dict())
    print(model.gammas.state_dict())
    model.save_checkpoint()

    return model


def plot_difference(
    srt_model: LISTA_base = SRT_LISTA_CP,
):
    model = LISTA_base.load_checkpoint(srt_model)
    model.eval()

    fig = plot_radon2d(model.radon3d[1], model.W1[1])
    fig.savefig("tmp/w1_1.png")
    fig = plot_radon2d(model.radon3d[4096], model.W1[4096])
    fig.savefig("tmp/w1_4096.png")
    print(model.__class__.__name__)


if __name__ == "__main__":
    train_lista()
