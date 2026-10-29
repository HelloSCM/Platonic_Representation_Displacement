"""Construct ViT students and expose complete-block representations."""

import timm
import torch
from torch import nn

from representation.extraction.common import BlockHooks, pool_tokens


def build_backbone(name, classes=0, checkpoint=None, pretrained=True):
    """Build a timm backbone with optional pretrained or checkpoint weights."""
    model = timm.create_model(
        name, pretrained=pretrained and checkpoint is None, num_classes=classes
    )
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("model", payload.get("state_dict", payload))
        state = {key.removeprefix("module."): value for key, value in state.items()}
        if any(key.startswith("backbone.") for key in state):
            state = {
                key.removeprefix("backbone."): value
                for key, value in state.items()
                if key.startswith("backbone.")
            }
        own = model.state_dict()
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith("head.") or (key in own and value.shape == own[key].shape)
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith("head.") for key in missing):
            raise ValueError(
                f"Incompatible backbone checkpoint: missing={missing}, unexpected={unexpected}"
            )
    return model


def block_hooks(model, layers="all"):
    """Register pooled feature hooks on selected vision transformer blocks."""
    pool = "cls" if model.global_pool == "token" else "mean"
    return BlockHooks(model.blocks, layers, lambda x: pool_tokens(x, pool, model.num_prefix_tokens))


class Student(nn.Module):
    """Train a classifier or block representation with an optional FKD projection."""

    def __init__(self, backbone, classify, teacher_dim=None):
        """Attach block hooks and an optional teacher-dimension projection to the backbone."""
        super().__init__()
        self.backbone = backbone
        self.classify = classify
        self.hooks = block_hooks(backbone, "last")
        dim = backbone.num_features
        self.projector = nn.Identity()
        if teacher_dim is not None and teacher_dim != dim:
            self.projector = nn.Linear(dim, teacher_dim)
            nn.init.trunc_normal_(self.projector.weight, std=0.02)
            nn.init.zeros_(self.projector.bias)
        if not classify:
            for name, parameter in backbone.named_parameters():
                if name.split(".")[0] in {"norm", "fc_norm", "head", "pre_logits"}:
                    parameter.requires_grad_(False)

    def forward(self, images):
        """Return classification logits, final block features, and projected features."""
        self.hooks.clear()
        tokens = self.backbone.forward_features(images)
        features = next(iter(self.hooks.features.values()))
        logits = self.backbone.forward_head(tokens) if self.classify else None
        return logits, features, self.projector(features)

    def close(self):
        """Remove the student's feature capture hooks."""
        self.hooks.close()
