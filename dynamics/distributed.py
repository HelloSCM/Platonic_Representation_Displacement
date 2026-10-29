"""Initialize single-machine distributed training and reduce epoch statistics."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class Runtime:
    """Manage device selection and single-machine distributed communication."""

    def __init__(self, device):
        """Initialize the local device, process rank, and distributed process group."""
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(
            f"cuda:{local_rank}" if self.world_size > 1 and device.startswith("cuda") else device
        )
        if self.device.type == "cuda":
            self.device = torch.device(
                "cuda", self.device.index if self.device.index is not None else 0
            )
            torch.cuda.set_device(self.device)
        if self.world_size > 1:
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo")

    @property
    def main(self):
        """Return whether this process is the primary rank."""
        return self.rank == 0

    def wrap(self, model):
        """Wrap the model in distributed data parallelism when using multiple processes."""
        if self.world_size == 1:
            return model
        devices = [self.device.index] if self.device.type == "cuda" else None
        return DistributedDataParallel(model, device_ids=devices)

    def barrier(self):
        """Synchronize all distributed processes."""
        if self.world_size > 1:
            dist.barrier()

    def sum(self, values):
        """Sum numeric values across processes and return them on the CPU."""
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(tensor)
        return tensor.cpu().tolist()

    def gather(self, value):
        """Collect one Python object from each process."""
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def close(self):
        """Destroy the distributed process group."""
        if self.world_size > 1:
            dist.destroy_process_group()
