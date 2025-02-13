import torch
import torch.nn as nn


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
    n_iter : int
        number of iterations, i.e. K
    D : torch.Tensor
        dictionary matrix. If unknown, set `D=None` and provide `N, M`
    N : int, optional
        original signal dimension, by default None
    M : int, optional
        sparse code dimension, by default None
    shared_theta : bool, optional
        whether thetas(0, ..., K-1) are same, by default False
    shared_weight : bool, optional
        whether weights(0, ..., K-1) are same, by default False
    """

    def __init__(
        self,
        n_iter: int,
        D: torch.Tensor | None,
        N: int = None,
        M: int = None,
        shared_theta: bool = True,
        shared_weight: bool = True,
    ) -> None:
        super().__init__()
        self.n_iter = n_iter

        # either D or (M, N) should be provided,
        # mutually exclusive, hence XOR
        assert (D is not None) ^ (M is not None and N is not None)
        self.D = None if D is None else D.requires_grad_(False)
        # (N, M) matrix
        self.D_SHAPE = (N, M) if D is None else D.shape

        self.shared_theta = shared_theta
        self.shared_weight = shared_weight

    @property
    def N(self) -> int:
        """
        original signal dimension :math:`y ∈ R^N`

        Returns
        -------
        int
            dimension of y
        """
        return self.D_SHAPE[0]

    @property
    def M(self) -> int:
        """
        sparse code dimension :math:`x ∈ R^M`

        Returns
        -------
        int
            dimension of x
        """
        return self.D_SHAPE[1]

    def forward(self):
        raise NotImplementedError(
            "LISTA_base is an abstract class. Use one of its subclasses."
        )


if __name__ == "__main__":
    model = LISTA_base(1, torch.randn(20, 10))
