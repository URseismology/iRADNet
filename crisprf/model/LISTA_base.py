import torch
import torch.nn as nn


class LISTA_base(nn.Module):
    r"""
    Abstract class for LISTA-based models.  
    In LISTA, we aim to solve optimization problem :math:`y = Dx + ϵ` with

    :math:`x(k+1) = ηθ(k) ( W1(k)y + W2(k)x(k) )`

    by unfolding the iteration, where 
    - :math:`D ∈ R^{N×M}` is the dictionary matrix, 
    - :math:`y ∈ R^N` is the observed signal, 
    - :math:`x ∈ R^M` is the sparse signal,
    - :math:`ϵ` is the noise,
    - :math:`θ` is the thresholding parameter,
    - :math:`N << M`, but :math:`x` is sparse,
    - :math:`Θ = {W1, W2, θ}_k, k ∈ [0,K)` are parameters to learn.


    Parameters
    ----------
    n_iter : int
        number of iterations, i.e. K
    D : torch.Tensor
        dictionary matrix, if known
    shared_theta : bool, optional
        whether thetas(0, ..., K-1) are same, by default False
    shared_weight : bool, optional
        whether weights(0, ..., K-1) are same, by default False
    """

    def __init__(
        self,
        n_iter: int,
        D: torch.Tensor,
        shared_theta: bool = False,
        shared_weight: bool = False
    ) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.D = D.requires_grad_(False)

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
        return self.D.shape[0]

    @property
    def M(self) -> int:
        """
        sparse code dimension :math:`x ∈ R^M`

        Returns
        -------
        int
            dimension of x
        """
        return self.D.shape[1]

    def forward(self):
        raise NotImplementedError(
            "LISTA_base is an abstract class. Use one of its subclasses."
        )
