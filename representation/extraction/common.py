"""Capture complete transformer blocks and pool token features."""

import random

import numpy as np
import torch


def set_seed(seed):
    """Seed the Python, NumPy, and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pool_tokens(x, pool, prefix=0, mask=None):
    """Pool token features using the class token or a masked mean."""
    if pool == "cls":
        return x[:, 0]
    x = x.float()
    if mask is None:
        return x[:, prefix:].mean(dim=1)
    mask = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


class BlockHooks:
    """Capture pooled outputs after selected transformer blocks."""

    def __init__(self, blocks, layers, pool):
        """Register pooling hooks on the selected transformer blocks."""
        if layers == "all":
            self.indices = list(range(len(blocks)))
        elif layers == "last":
            self.indices = [len(blocks) - 1]
        else:
            self.indices = sorted({int(i) for i in layers.split(",")})
        self.features = {}
        self.pool = pool
        self.handles = [blocks[i].register_forward_hook(self._hook(i)) for i in self.indices]

    def _hook(self, index):
        """Create a hook that captures pooled features from one transformer block."""

        def capture(module, inputs, output):
            """Pool and store the output of the selected transformer block."""
            x = output[0] if isinstance(output, (tuple, list)) else output
            self.features[f"layer_{index:03d}"] = self.pool(x)

        return capture

    def clear(self):
        """Clear the features captured during the previous forward pass."""
        self.features.clear()

    def close(self):
        """Remove all registered hooks and clear captured features."""
        for handle in self.handles:
            handle.remove()
        self.features.clear()


def to_numpy(features):
    """Convert named feature tensors to float32 NumPy arrays on the CPU."""
    return {name: x.detach().float().cpu().numpy() for name, x in features.items()}
