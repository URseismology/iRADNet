from crisprf.model.LISTA_base import LISTA_base
import torch
import torch.nn as nn
import torch.nn.functional as F


class LISTA(LISTA_base):
    r"""
    Learned Iterative Shrinkage-Thresholding Algorithm [1]_ AutoEncoder neural
    network for sparse coding.

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


    References
    ----------
    .. [1] Gregor, K., & LeCun, Y. (2010, June). Learning fast approximations
       of sparse coding. In Proceedings of the 27th international conference on
       international conference on machine learning (pp. 399-406).
    """

    def __init__(
        self,
        n_iter: int,
        D: torch.Tensor,
        shared_theta: bool = False,
        shared_weight: bool = False,
        alpha: float = 5,
    ) -> None:
        super(LISTA, self).__init__(
            n_iter=n_iter,
            D=D,
            shared_theta=shared_theta,
            shared_weight=shared_weight,
        )
        L = 1.001 * torch.norm(self.D, p=2) ** 2

        n_weights = 1 if shared_weight else n_iter
        self.w1: list[nn.Linear] = nn.ModuleList(
            nn.Linear(self.N, self.M, bias=False)
            for _ in range(n_weights))
        self.w2: list[nn.Linear] = nn.ModuleList(
            nn.Linear(self.M, self.M, bias=False)
            for _ in range(n_weights))

        n_thetas = 1 if shared_theta else n_iter
        self.theta = nn.ModuleList(
            nn.Parameter(torch.ones(1) * alpha / L)
            for _ in range(n_thetas)
        )

    def init_weights(self):
        for w in self.w1:
            nn.init.xavier_normal_(w.weight)
        for w in self.w2:
            nn.init.eye_(w.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_iter):
            theta = self.theta[i % len(self.theta)]
            w1 = self.w1[0 % len(self.w1)]
            w2 = self.w2[0 % len(self.w2)]

            x = F.softshrink(w1(x) + w2(x), theta)
        return x


if __name__ == '__main__':
    model = LISTA(n_iter=1, D=torch.randn(10, 5))
    model.cuda()

    assert model.w1[0].weight.is_cuda
