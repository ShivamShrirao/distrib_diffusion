import numpy as np
import ot
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde


def sinkhorn_distance_2d(samples1, samples2, reg=0.1, metric='euclidean', normalize=True, num_iter=200):
    """Entropic OT cost between two 2D point sets."""
    X = np.asarray(samples1, dtype=np.float64)
    Y = np.asarray(samples2, dtype=np.float64)

    if normalize:
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)

    M = cdist(X, Y, metric=metric)  # cost matrix
    n1, n2 = len(X), len(Y)
    a = np.ones(n1, dtype=np.float64) / n1
    b = np.ones(n2, dtype=np.float64) / n2

    val = ot.sinkhorn2(a, b, M, reg, numItermax=num_iter)
    return float(val)


def wasserstein_distance_2d(real_samples, gen_samples):
    mu = real_samples.mean(0)
    std = real_samples.std(0)
    real_samples = (real_samples - mu) / (std + 1e-8)
    gen_samples = (gen_samples - mu) / (std + 1e-8)

    M = cdist(real_samples, gen_samples, metric='euclidean')
    n1, n2 = len(real_samples), len(gen_samples)
    a, b = np.ones(n1) / n1, np.ones(n2) / n2
    return ot.emd2(a, b, M)


def mmd_rbf(X, Y, gamma=1.0):
    """Maximum Mean Discrepancy with RBF kernel."""
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)

    X_sqnorms = np.diag(XX)
    Y_sqnorms = np.diag(YY)

    K_XX = np.exp(-gamma * (X_sqnorms[:, None] + X_sqnorms[None, :] - 2 * XX))
    K_YY = np.exp(-gamma * (Y_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * YY))
    K_XY = np.exp(-gamma * (X_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * XY))

    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return mmd


def coverage_and_density(real_samples, gen_samples, k=5):
    """Density & Coverage (Naeem et al.): 
    - coverage: fraction of real samples with a generated sample inside their k-NN real radius
    - density: average (over generated samples) count of real samples for which the gen sample lies within that real sample's k-NN radius, normalized by k
    """
    R = np.asarray(real_samples, dtype=np.float64)
    G = np.asarray(gen_samples, dtype=np.float64)
    Nr = R.shape[0]

    # ensure valid k (k=1 means 'first non-self neighbor' because self-distance is 0)
    k = int(max(1, min(k, Nr - 1)))

    # k-NN radius per real sample (distance to its k-th nearest neighbor among real)
    RR = cdist(R, R)
    kth = np.sort(RR, axis=1)[:, k]  # shape (Nr,)

    # Coverage
    RG = cdist(R, G)                  # shape (Nr, Ng)
    min_r_to_g = RG.min(axis=1)       # shape (Nr,)
    coverage = np.mean(min_r_to_g <= kth)

    # Density
    GR = RG.T                         # shape (Ng, Nr)
    within = GR <= kth[None, :]       # broadcast kth radii across gens
    density = within.sum(axis=1).mean() / k

    return float(coverage), float(density)


def negative_log_likelihood(real_samples, gen_samples, bandwidth='scott'):
    """Estimate NLL using kernel density estimation."""
    kde = gaussian_kde(gen_samples.T, bw_method=bandwidth)
    log_probs = kde.logpdf(real_samples.T)
    return -np.mean(log_probs)


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
