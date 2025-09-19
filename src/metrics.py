from typing import Tuple, Union

import ot
import torch


def wasserstein_distance_2d(R: torch.Tensor, G: torch.Tensor) -> float:
    mu = R.mean(0)
    std = R.std(0)
    Rn = (R - mu) / (std + 1e-8)
    Gn = (G - mu) / (std + 1e-8)

    M = torch.cdist(Rn, Gn, p=2)  # Euclidean distances
    n1, n2 = Rn.shape[0], Gn.shape[0]
    a = torch.full((n1,), 1.0 / n1, dtype=M.dtype, device=M.device)
    b = torch.full((n2,), 1.0 / n2, dtype=M.dtype, device=M.device)

    M_np = M.cpu().numpy()
    a_np = a.cpu().numpy()
    b_np = b.cpu().numpy()
    return float(ot.emd2(a_np, b_np, M_np))


def mmd_rbf(X: torch.Tensor, Y: torch.Tensor, gamma: float = 1.0) -> float:
    # Pairwise squared distances
    X_sq = (X * X).sum(dim=1, keepdim=True)  # (n,1)
    Y_sq = (Y * Y).sum(dim=1, keepdim=True)  # (m,1)

    dXX = (X_sq + X_sq.t() - 2.0 * (X @ X.t())).clamp_min(0.0)
    dYY = (Y_sq + Y_sq.t() - 2.0 * (Y @ Y.t())).clamp_min(0.0)
    dXY = (X_sq + Y_sq.t() - 2.0 * (X @ Y.t())).clamp_min(0.0)

    K_XX = torch.exp(-gamma * dXX)
    K_YY = torch.exp(-gamma * dYY)
    K_XY = torch.exp(-gamma * dXY)

    mmd = K_XX.mean() + K_YY.mean() - 2.0 * K_XY.mean()
    return mmd.item()


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count
