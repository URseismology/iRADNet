import torch
import torch.nn.functional as F
from crisprf.model.FISTA import fista
from crisprf.model.LISTA_CP import lista
from crisprf.model.radon_solver import sparse_inverse_radon_fista
from crisprf.util.bridging import retrieve_single_xy, heatmap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = retrieve_single_xy("data/Ps_RF_syn1.mat")
nt = 5000
dt = 0.02
alpha0 = 1.6
alpha1 = 1
y = DATA["y"].T.to(DEVICE)
rayP = DATA["rayP"].to(DEVICE)
q = DATA["q"].to(DEVICE)


def test_fista():
    stats = []

    for x_hat, elapsed in sparse_inverse_radon_fista(
        y=y,
        rayP=rayP,
        q=q,
        seconds=nt * dt,
        freq_bounds=(0, 1 / 2 / dt),
        alphas=(1, 0.2),
        n_layers=10,
        device=DEVICE,
        ista_fn=fista,
    ):
        # get number of none-zero elements
        x_hat = x_hat.detach().cpu().T
        nonzeros = torch.count_nonzero(x_hat).item()
        mse = F.mse_loss(DATA["x"], x_hat)
        mse_0 = F.mse_loss(DATA["x"], torch.zeros_like(DATA["x"]))
        nmse = mse / mse_0

        if stats:
            assert nonzeros <= stats[-1][0]
        print(
            f"{nonzeros:2e} {mse.item():6e}/{mse_0.item():6e}={nmse.item():4e} {elapsed}"
        )
        stats.append((nonzeros, mse, mse_0, nmse, elapsed))


def test_lista():
    stats = []

    for x_hat, elapsed in sparse_inverse_radon_fista(
        y=y,
        rayP=rayP,
        q=q,
        seconds=nt * dt,
        freq_bounds=(0, 1 / 2 / dt),
        alphas=(1, 0.2),
        n_layers=10,
        device=DEVICE,
        ista_fn=lista,
    ):
        # get number of none-zero elements
        x_hat = x_hat.detach().cpu().T
        nonzeros = torch.count_nonzero(x_hat).item()
        mse = F.mse_loss(DATA["x"], x_hat)
        mse_0 = F.mse_loss(DATA["x"], torch.zeros_like(DATA["x"]))
        nmse = mse / mse_0

        if stats:
            assert nonzeros <= stats[-1][0]
        print(
            f"{nonzeros:2e} {mse.item():6e}/{mse_0.item():6e}={nmse.item():4e} {elapsed}"
        )
        stats.append((nonzeros, mse, mse_0, nmse, elapsed))


if __name__ == "__main__":
    test_lista()
