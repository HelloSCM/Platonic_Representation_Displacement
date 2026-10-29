"""Normalized HSIC with linear and median-bandwidth RBF kernels."""

import torch


def centered_kernel(kernel):
    """Center a Gram matrix along both sample axes."""
    return (
        kernel - kernel.mean(dim=0, keepdim=True) - kernel.mean(dim=1, keepdim=True) + kernel.mean()
    )


def gram(x, kernel, eps=1e-12):
    """Compute a linear or median-bandwidth RBF Gram matrix."""
    if kernel == "linear":
        return x @ x.T
    if kernel != "rbf":
        raise ValueError("kernel must be linear or rbf.")
    x = x - x.mean(dim=0, keepdim=True)
    product = x @ x.T
    norm = (x * x).sum(dim=1)
    distances = (norm[:, None] + norm[None, :] - 2 * product).clamp_min(0)
    distances.fill_diagonal_(0)
    mask = ~torch.eye(len(x), dtype=torch.bool, device=x.device)
    positive = distances[mask & (distances > eps)]
    sigma2 = positive.quantile(0.5) if len(positive) else x.new_tensor(1.0)
    return torch.exp(-distances / (2 * sigma2))


@torch.no_grad()
def normalized_hsic(u, v, kernel="linear", eps=1e-12, device="cpu"):
    """Measure dependence in aligned-whitened coordinates with centered Gram matrices."""
    u = torch.as_tensor(u, dtype=torch.float64, device=device)
    v = torch.as_tensor(v, dtype=torch.float64, device=device)
    if len(u) != len(v) or len(u) < 2:
        raise ValueError("HSIC requires at least two matched rows.")
    ku, kv = centered_kernel(gram(u, kernel, eps)), centered_kernel(gram(v, kernel, eps))
    denominator = ((ku * ku).sum() * (kv * kv).sum()).clamp_min(eps).sqrt()
    return float(((ku * kv).sum() / denominator).clamp(0, 1).item())
