"""Extract timm vision transformer representations."""

import torch

from .common import BlockHooks, pool_tokens, set_seed


class VisionEncoder:
    """Encode complete blocks or the final pooled backbone representation."""

    def __init__(
        self,
        model,
        device="cuda",
        layers="all",
        pool="auto",
        dtype="float32",
        checkpoint=None,
        random=False,
        seed=42,
    ):
        """Initialize a frozen vision backbone, image transform, and optional block hooks."""
        import timm
        from timm.data import create_transform, resolve_model_data_config

        if random and checkpoint is not None:
            raise ValueError("Random initialization cannot be combined with a checkpoint.")
        set_seed(seed)
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.model = (
            timm.create_model(
                model,
                pretrained=not random and checkpoint is None,
                num_classes=0,
                checkpoint_path=str(checkpoint) if checkpoint else "",
            )
            .eval()
            .to(self.device)
            .requires_grad_(False)
        )
        config = resolve_model_data_config(self.model)
        self.transform = create_transform(**config, is_training=False)
        self.layers = layers
        self.prefix = self.model.num_prefix_tokens
        self.pool = pool
        if pool == "auto":
            self.pool = "cls" if self.model.global_pool == "token" else "mean"
        self.hooks = None
        if layers != "final":
            self.hooks = BlockHooks(
                self.model.blocks,
                layers,
                lambda x: pool_tokens(x, self.pool, prefix=self.prefix),
            )
        self.metadata = {
            "backend": "timm",
            "model": model,
            "layers": layers,
            "pool": "model" if layers == "final" and pool == "auto" else self.pool,
            "model_pool": self.model.global_pool,
            "num_prefix_tokens": self.prefix,
            "dtype": dtype,
            "random": random,
            "seed": seed,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "data_config": config,
            "normalized": False,
        }
        self.assets = {}

    @torch.inference_mode()
    def __call__(self, images):
        """Encode an image batch into the selected block or final features."""
        if self.hooks:
            self.hooks.clear()
        images = images.to(self.device, non_blocking=True)
        with torch.autocast(
            self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
        ):
            tokens = self.model.forward_features(images)
            if self.hooks:
                return dict(self.hooks.features)
            if self.metadata["pool"] == "model":
                x = self.model.forward_head(tokens, pre_logits=True)
            else:
                x = self.model.fc_norm(pool_tokens(tokens, self.pool, self.prefix))
        return {"final": x}

    def close(self):
        """Remove the vision model's feature capture hooks."""
        if self.hooks:
            self.hooks.close()
