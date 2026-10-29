"""Select visible GPUs and launch one training process per device."""

import argparse
import os
import subprocess
import sys


def parse_gpus(value):
    """Parse and validate a comma-separated list of GPU indices."""
    ids = [part.strip() for part in value.split(",")]
    if not ids or any(not part.isdecimal() for part in ids):
        raise argparse.ArgumentTypeError("Use comma-separated GPU indices, such as 0,2,3.")
    ids = [str(int(part)) for part in ids]
    if len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("GPU indices must be unique.")
    return ",".join(ids)


def add_device_args(parser, default="cuda"):
    """Add device and visible GPU options to an argument parser."""
    parser.add_argument(
        "--gpus", type=parse_gpus, help="Physical GPU indices, such as 0,2,3; defaults to GPU 0."
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=default,
        help="Execution device; GPU selection is controlled by --gpus.",
    )


def launch(module, args, argv):
    """Configure GPU visibility and start local DDP workers when needed."""
    gpus = getattr(args, "gpus", None)
    if args.device == "cpu":
        args.gpus = None
        return
    worker = "LOCAL_RANK" in os.environ
    if gpus is None:
        if worker:
            gpus = os.environ.get(
                "CUDA_VISIBLE_DEVICES",
                ",".join(str(i) for i in range(int(os.environ["WORLD_SIZE"]))),
            )
        else:
            gpus = args.device.partition(":")[2] or "0"
    args.gpus, args.device = gpus, "cuda"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    count = len(gpus.split(","))
    if worker:
        if int(os.environ["WORLD_SIZE"]) != count:
            raise ValueError("The number of training processes must match --gpus.")
        return
    if count > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={count}",
            "--module",
            module,
            *argv,
        ]
        raise SystemExit(subprocess.run(command, check=False).returncode)
