# %%
import torch
import torch.nn.functional as F


def shrink(x: torch.Tensor,
           theta: torch.Tensor) -> torch.Tensor:
    """
    Soft shrinkage function, θ is auto-ReLU-ed.

    :math:`sign(x)max(0, |x|-ReLU(θ))`

    Parameters
    ----------
    x : torch.Tensor
        input tensor
    theta : torch.Tensor
        threshold, automatically ReLU to ensure non-negativity

    Returns
    -------
    torch.Tensor
        result
    """
    return x.sign() * F.relu(x.abs() - F.relu(theta))


def shrink_free(x: torch.Tensor,
                theta: torch.Tensor) -> torch.Tensor:
    """
    Soft shrinkage function.

    :math:`sign(x)max(0, |x|-θ)`

    Parameters
    ----------
    x : torch.Tensor
        input tensor
    theta : torch.Tensor
        threshold, automatically ReLU to ensure non-negativity

    Returns
    -------
    torch.Tensor
        result
    """
    return x.sign() * F.relu(x.abs() - theta)


def shrink_ss(data: torch.Tensor,
              theta: torch.Tensor,
              q: int,
              return_index: bool = False):
    """special shrink that does not apply soft shrinkage to entries of top q% magnitudes.

    Args:
        data (torch.Tensor): _description_
        theta (torch.Tensor): threshold (1,)
        q (float): top q% magnitudes `[0, 100]`
        return_index (bool, optional): cindex. Defaults to False.
    """
    _abs = torch.abs(data)
    _thres = torch.quantile(_abs, 1 - q / 100, dim=1, keepdim=True)

    # entries > threshold and in the top q% simultaneously will be selected into the support
    # and thus will not be sent to the shrinkage function

    _id = torch.logical_and(_abs > theta, _abs > _thres).float()
    _id = _id.detach()
    _cid = 1.0 - _id  # complementary index

    output = (_id * data + shrink(_cid * data, theta))
    if return_index:
        return output, _cid
    else:
        return output


# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = torch.arange(-10, 10, 0.1)
    theta = torch.tensor(-1.0)
    y = shrink(x, theta)

    plt.plot(x, y)
    plt.show()
