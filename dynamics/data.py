"""Prepare shared image augmentations and ordered evaluation datasets."""

from pathlib import Path

import torch
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


class SharedViews:
    """Apply one spatial augmentation followed by model-specific normalization."""

    def __init__(self, models):
        """Build shared image augmentation and model-specific normalization transforms."""
        self.configs = [resolve_model_data_config(model) for model in models]
        size = self.configs[0]["input_size"][-2:]
        self.augment = T.Compose(
            [
                T.RandomResizedCrop(size, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.4, 0.4, 0.4),
            ]
        )
        self.transforms = [
            T.Compose(
                [
                    T.Resize(config["input_size"][-2:], interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(config["mean"], config["std"]),
                ]
            )
            for config in self.configs
        ]

    def __call__(self, image):
        """Apply a shared augmentation and return one normalized view per model."""
        image = self.augment(image)
        return tuple(transform(image) for transform in self.transforms)


def make_loader(dataset, batch_size, workers, training=False, world_size=1, rank=0, seed=42):
    """Build a data loader with optional distributed training sampling."""
    sampler = (
        DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, seed=seed, shuffle=True, drop_last=True
        )
        if training
        else None
    )
    generator = torch.Generator().manual_seed(seed + rank)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=training,
        generator=generator,
    ), sampler


def make_training_data(args, student, teacher, world_size, rank):
    """Load training images with shared student and teacher views."""
    models = [student] + ([teacher] if args.loss != "ce" else [])
    train = ImageFolder(args.train_dir, transform=SharedViews(models))
    loader, sampler = make_loader(
        train, args.batch_size, args.workers, True, world_size, rank, args.seed
    )
    if len(loader) == 0:
        raise ValueError("The training set must contain at least one full batch per rank.")
    return train, loader, sampler


def make_validation_data(args, model, classes):
    """Load validation images with the model's evaluation transform and class mapping."""
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)
    dataset = ImageFolder(Path(args.val_dir), transform=transform)
    if dataset.class_to_idx != classes:
        raise ValueError("Training and validation directories must have the same class mapping.")
    return make_loader(dataset, args.eval_batch_size, args.workers, seed=args.seed)[0]
