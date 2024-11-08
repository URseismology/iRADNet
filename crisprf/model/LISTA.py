from crisprf.model.LISTA_base import LISTA_base
from crisprf.util.shrink import shrink
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
        dictionary matrix, if known
    shared_theta : bool, optional
        whether thetas(0, ..., K-1) are same, by default True
    shared_weight : bool, optional
        whether weights(0, ..., K-1) are same, by default True


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
        N: int = None,
        M: int = None,
        shared_theta: bool = True,
        shared_weight: bool = True,
        alpha: float = 5,
    ) -> None:
        super(LISTA, self).__init__(
            n_iter=n_iter,
            D=D,
            N=N,
            M=M,
            shared_theta=shared_theta,
            shared_weight=shared_weight,
        )
        L = 1.001 if D is None else 1.001 * torch.linalg.matrix_norm(self.D) ** 2

        n_weights = 1 if shared_weight else n_iter
        self.w1: list[nn.Linear] = nn.ModuleList(
            nn.Linear(self.N, self.M, bias=False) for _ in range(n_weights)
        )
        self.w2: list[nn.Linear] = nn.ModuleList(
            nn.Linear(self.M, self.M, bias=False) for _ in range(n_weights)
        )

        n_thetas = 1 if shared_theta else n_iter
        self.theta = nn.Parameter(torch.ones(n_thetas) * alpha / L)

    def init_weights(self):
        for w in self.w1:
            if self.D is None:
                nn.init.kaiming_normal_(w.weight)
            else:
                w.weight.data = self.D.t() / (1.001 * torch.norm(self.D, p=2) ** 2)
        for w in self.w2:
            nn.init.eye_(w.weight)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_iter):
            # getting params for this iter
            theta = self.theta[0 if self.shared_theta else i]
            w1 = self.w1[0 if self.shared_weight else i]
            w2 = self.w2[0 if self.shared_weight else i]

            x = shrink(w1(y) + w2(x), theta)

        return x


if __name__ == "__main__":
    # model = LISTA(5, None, True, True)
    # model.cuda()

    # assert model.w1[0].weight.is_cuda

    z = torch.randn(1, 10).cuda()
    z_d = z.detach()
    print(z_d.is_cuda)
