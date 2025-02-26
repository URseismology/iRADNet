import torch
import torch.nn as nn

from ..util.bridging import RFDataShape
from ..util.constants import AUTO_DEVICE


def _init_param_list(n: int, alpha: float, device: torch.device):
    return nn.ParameterList(
        [nn.Parameter(torch.ones(1, device=device) * alpha) for _ in range(n)]
    )


class LISTA_base(nn.Module):
    r"""
    Abstract class for LISTA-based models.
    In LISTA, we aim to solve optimization problem :math:`y = Dx + ϵ` with

    :math:`x(k+1) = ηθ(k) ( W1(k)y + W2(k)x(k) )`

    by unfolding the iteration, where
    - :math:`D ∈ R^{N×M}` is dictionary matrix,
    - :math:`y ∈ R^N` is observed signal,
    - :math:`x ∈ R^M` is sparse code,
    - :math:`ϵ` is noise,
    - :math:`θ` is thresholding parameter,
    - :math:`Θ = {W1, W2, θ}_k, k ∈ [0,K)` are parameters to learn.


    Parameters
    ----------
    radon3d : torch.Tensor
        radon 3d matrix. (nFFT, nP, nQ)
    n_layers : int
        number of layers or iterations (i.e., K)
    shapes : RFDataShape
        shape information wrapper
    shared_theta : bool, optional
        whether thetas(0, ..., K-1) are same, by default False
    shared_weight : bool, optional
        whether weights(0, ..., K-1) are same, by default True
    freq_index_bounds : tuple[int, int], optional
        frequency index bounds, by default None
    alpha : float, optional
        initializer for gamma, by default 0.9
    device : torch.device, optional
        device, by default AUTO_DEVICE
    """

    def __init__(
        self,
        radon3d: torch.Tensor,
        n_layers: int,
        shapes: RFDataShape,
        shared_theta: bool = False,
        shared_weight: bool = True,
        freq_index_bounds: tuple[int, int] = None,
        alpha: float = 0.9,
        device: torch.device = AUTO_DEVICE,
    ) -> None:
        super().__init__()
        assert radon3d is not None
        self.shapes = shapes
        if freq_index_bounds is not None:
            self.ilow, self.ihigh = freq_index_bounds
        else:
            self.ilow, self.ihigh = 1, shapes.nFFT // 2

        self.radon3d = radon3d
        self.n_layers = n_layers

        self.shared_theta = shared_theta
        self.shared_weight = shared_weight

        # set up thresholding parameters
        n_theta = 1 if shared_theta else n_layers
        self.gammas = _init_param_list(n_theta, alpha, device)
        self.etas = _init_param_list(n_theta, alpha, device)

    def get_gamma(self, k: int) -> torch.Tensor:
        return self.gammas[0 if self.shared_theta else k]

    def get_eta(self, k: int) -> torch.Tensor:
        return self.etas[0 if self.shared_theta else k]

    def save_checkpoint(self, path: str = None):
        path = path or f"cache/{self.__class__.__name__}.pt"
        torch.save(
            {
                "radon3d": self.radon3d,
                "n_layers": self.n_layers,
                "state_dict": self.state_dict(),
                "shapes": self.shapes,
                "shared_theta": self.shared_theta,
                "shared_weight": self.shared_weight,
                "freq_index_bounds": (self.ilow, self.ihigh),
            },
            path,
        )

    @staticmethod
    def load_checkpoint(model_class: "LISTA_base", path: str = None) -> "LISTA_base":
        path = path or f"cache/{model_class.__name__}.pt"
        checkpoint = torch.load(path)
        model: LISTA_base = model_class(
            radon3d=checkpoint["radon3d"],
            n_layers=checkpoint["n_layers"],
            shapes=checkpoint["shapes"],
            shared_theta=checkpoint["shared_theta"],
            shared_weight=checkpoint["shared_weight"],
            freq_index_bounds=checkpoint["freq_index_bounds"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def forward(self, x0: torch.Tensor, y_freq: torch.Tensor):
        raise NotImplementedError(
            "LISTA_base is an abstract class. Use one of its subclasses."
        )
