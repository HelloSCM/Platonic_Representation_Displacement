"""Train linear heads on frozen reconstructed representation components."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from metric.io import make_output, write_csv, write_json

from .data import concepts, held_out_ids, open_inputs, parse_ranks
from .distributed import Runtime
from .model import Decomposition


class FeatureDataset(Dataset):
    """Provide indexed feature vectors and class labels for linear probing."""

    def __init__(self, array, indices, labels):
        """Store the feature matrix, selected row indices, and class labels."""
        self.array, self.indices, self.labels = array, indices, labels

    def __len__(self):
        """Return the number of selected feature rows."""
        return len(self.indices)

    def __getitem__(self, index):
        """Return one selected feature vector and its integer class label."""
        x = np.array(self.array[self.indices[index]], dtype=np.float32)
        return torch.from_numpy(x), int(self.labels[index])


class ComponentMap(nn.Module):
    """Reconstruct a fixed component in its model's original feature coordinates."""

    def __init__(self, bundle, model, component, rank, device):
        """Register fixed affine maps for reconstructing the selected component."""
        super().__init__()
        mean, forward, inverse = bundle.affine(model, component, rank)
        for name, value in (("mean", mean), ("forward_map", forward), ("inverse_map", inverse)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32, device=device))

    def forward(self, x):
        """Reconstruct the selected component in the original feature coordinates."""
        return ((x - self.mean) @ self.forward_map) @ self.inverse_map + self.mean


def loaders(array, indices, labels, args, runtime, training):
    """Build a feature data loader with optional distributed training sampling."""
    dataset = FeatureDataset(array, indices, labels)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            seed=args.seed,
            shuffle=True,
        )
        if training
        else None
    )
    generator = torch.Generator().manual_seed(args.seed + runtime.rank)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        generator=generator,
    )


def train_head(loader, transform, width, classes, args, runtime):
    """Train a linear classifier on fixed reconstructed component features."""
    torch.manual_seed(args.seed)
    head = nn.Linear(width, classes).to(runtime.device)
    nn.init.normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)
    model = runtime.wrap(head)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(runtime.device.type, enabled=args.dtype == "float16")
    dtype = getattr(torch, args.dtype)
    for epoch in range(args.epochs):
        loader.sampler.set_epoch(epoch)
        model.train()
        loss_sum, count = 0.0, 0
        for x, labels in loader:
            x, labels = x.to(runtime.device), labels.to(runtime.device)
            with torch.no_grad():
                x = transform(x)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(runtime.device.type, dtype=dtype, enabled=dtype != torch.float32):
                loss = F.cross_entropy(model(x).float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(labels)
            count += len(labels)
        loss_sum, count = runtime.sum([loss_sum, count])
        if runtime.main:
            print(f"Probe epoch {epoch + 1}: loss={loss_sum / count:.6f}", flush=True)
    return head


@torch.inference_mode()
def accuracy(head, loader, transform, device):
    """Compute top-one accuracy after applying a fixed component reconstruction."""
    head.eval()
    correct, count = 0, 0
    for x, labels in loader:
        prediction = head(transform(x.to(device))).argmax(dim=-1).cpu()
        correct += int((prediction == labels).sum())
        count += len(labels)
    return correct / count


def run(args):
    """Evaluate frozen or refitted linear probes across shared and private component ranks."""
    runtime = Runtime(args.device)
    bundle = Decomposition.load(args.params)
    ranks = parse_ranks(args.ranks, bundle.dim)
    train_paths = args.train_inputs or [model["path"] for model in bundle.models]
    train_stores, train_ids, train_rows, train_arrays = open_inputs(bundle, train_paths)
    val_stores, val_ids, val_rows, val_arrays = open_inputs(bundle, args.inputs)
    if any(a.path.resolve() == b.path.resolve() for a, b in zip(train_stores, val_stores)):
        raise ValueError("Training and validation feature stores must be different.")
    if set(train_ids).intersection(val_ids):
        raise ValueError("Linear-probe training and validation IDs must be disjoint.")
    held_out_ids(args.params, val_ids)
    train_keys = [str(key) for key in concepts(train_stores, train_rows)]
    val_keys = [str(key) for key in concepts(val_stores, val_rows)]
    classes = sorted(set(train_keys))
    lookup = {key: i for i, key in enumerate(classes)}
    if set(val_keys) - lookup.keys():
        raise ValueError("Validation classes must be present in the training set.")
    y_train = np.array([lookup[key] for key in train_keys])
    y_val = np.array([lookup[key] for key in val_keys])
    out = Path(args.output)
    if runtime.main:
        make_output(out)
        (out / "heads").mkdir()
        write_json(
            out / "config.json",
            {
                **vars(args),
                "params": str(Path(args.params).resolve()),
                "ranks": ranks,
                "classes": classes,
                "train_inputs": [str(store.path.resolve()) for store in train_stores],
                "inputs": [str(store.path.resolve()) for store in val_stores],
                "world_size": runtime.world_size,
                "global_batch_size": args.batch_size * runtime.world_size,
            },
        )
    runtime.barrier()
    results = []
    for i, model in enumerate(bundle.models):
        training = loaders(train_arrays[i], train_rows[i], y_train, args, runtime, True)
        validation = (
            loaders(val_arrays[i], val_rows[i], y_val, args, runtime, False)
            if runtime.main
            else None
        )
        original_map = ComponentMap(bundle, i, "original", None, runtime.device)
        original_head = train_head(
            training, original_map, train_arrays[i].shape[1], len(classes), args, runtime
        )
        original_path = out / "heads" / f"{i:03d}_original.pt"
        original_acc = None
        if runtime.main:
            original_acc = accuracy(original_head, validation, original_map, runtime.device)
            torch.save(
                {
                    "state_dict": original_head.state_dict(),
                    "model": model["name"],
                    "classes": classes,
                    "component": "original",
                },
                original_path,
            )
        runtime.barrier()
        for rank in ranks:
            if runtime.main:
                results.append(
                    {
                        "model": model["name"],
                        "rank": rank,
                        "ratio": rank / bundle.dim,
                        "component": "original",
                        "protocol": args.protocol,
                        "accuracy": original_acc,
                        "head": str(original_path.resolve()),
                    }
                )
            for component in ("shared", "private"):
                transform = ComponentMap(bundle, i, component, rank, runtime.device)
                head, head_path = original_head, original_path
                if args.protocol == "refit" and rank < bundle.dim:
                    head = train_head(
                        training, transform, train_arrays[i].shape[1], len(classes), args, runtime
                    )
                    head_path = out / "heads" / f"{i:03d}_{component}_{rank}.pt"
                    if runtime.main:
                        torch.save(
                            {
                                "state_dict": head.state_dict(),
                                "model": model["name"],
                                "classes": classes,
                                "component": component,
                                "rank": rank,
                            },
                            head_path,
                        )
                if runtime.main:
                    score = (
                        original_acc
                        if rank == bundle.dim
                        else accuracy(head, validation, transform, runtime.device)
                    )
                    results.append(
                        {
                            "model": model["name"],
                            "rank": rank,
                            "ratio": rank / bundle.dim,
                            "component": component,
                            "protocol": args.protocol,
                            "accuracy": score,
                            "head": str(head_path.resolve()),
                        }
                    )
                runtime.barrier()
    if runtime.main:
        write_csv(out / "probe.csv", results)
        if args.plot:
            from .plotting import plot_probe

            plot_probe(out)
    runtime.barrier()
    runtime.close()
    return results
