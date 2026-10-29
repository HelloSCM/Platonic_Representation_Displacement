"""Gated residual composition and training-only whitening."""

import torch
from torch import nn
from torch.nn import functional as F


class VisionConnector(nn.Module):
    """Fuse CLIP and DINO image features through a gated residual adapter."""

    def __init__(self, width, dino_dim, ratio=2):
        """Initialize the cross-model fusion projections and a zero-valued residual gate."""
        super().__init__()
        hidden = int(width * ratio)
        self.w_gate = nn.Linear(width, hidden, bias=False)
        self.w_value = nn.Linear(dino_dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, width, bias=False)
        self.gate = nn.Parameter(torch.zeros(width))

    def forward(self, clip, dino):
        """Add a gated DINO-conditioned residual to CLIP image features."""
        residual = self.w_down(F.silu(self.w_gate(clip)) * self.w_value(dino))
        return clip + self.gate.tanh() * residual


class LanguageGuider(nn.Module):
    """Adapt CLIP text features through a gated residual multilayer perceptron."""

    def __init__(self, width, ratio=2):
        """Initialize the text adapter projections and a zero-valued residual gate."""
        super().__init__()
        self.w_up = nn.Linear(width, int(width * ratio), bias=False)
        self.w_down = nn.Linear(int(width * ratio), width, bias=False)
        self.gate = nn.Parameter(torch.zeros(width))

    def forward(self, clip):
        """Add a gated nonlinear residual to CLIP text features."""
        return clip + self.gate.tanh() * self.w_down(F.gelu(self.w_up(clip)))


class ShadowCLIP(nn.Module):
    """Compose frozen backbone features in the original CLIP embedding space."""

    def __init__(self, config, assets=None):
        """Initialize the selected adapters, CLIP projections, and contrastive logit scale."""
        super().__init__()
        self.config = dict(config)
        variant = config.get("variant", "shadowclip")
        vision_dim, text_dim = config["vision_dim"], config["text_dim"]
        embed_dim = config["embed_dim"]
        self.vision = (
            VisionConnector(vision_dim, config["dino_dim"], config.get("hidden_ratio", 2))
            if variant in {"shadowclip", "no-rkd"}
            else None
        )
        self.language = (
            LanguageGuider(text_dim, config.get("hidden_ratio", 2))
            if variant == "shadowclip"
            else None
        )
        assets = assets or {}
        self.vision_proj = nn.Parameter(
            torch.as_tensor(assets.get("vision", torch.zeros(vision_dim, embed_dim)))
            .float()
            .clone(),
            requires_grad=variant == "clip-ft",
        )
        self.text_proj = nn.Parameter(
            torch.as_tensor(assets.get("language", torch.zeros(text_dim, embed_dim)))
            .float()
            .clone(),
            requires_grad=variant in {"clip-ft", "no-rkd"},
        )
        self.register_buffer(
            "logit_scale", torch.as_tensor(assets.get("logit_scale", 0.0)).float().clone()
        )

    def encode_image(self, clip, dino=None):
        """Fuse image features and project them into normalized CLIP embeddings."""
        hidden = self.vision(clip, dino) if self.vision is not None else clip
        return F.normalize(hidden @ self.vision_proj, dim=-1)

    def text_hidden(self, clip):
        """Apply the optional language adapter to unprojected CLIP text features."""
        return self.language(clip) if self.language is not None else clip

    def encode_text(self, clip):
        """Adapt and project text features into normalized CLIP embeddings."""
        return F.normalize(self.text_hidden(clip) @ self.text_proj, dim=-1)


class Whitening(nn.Module):
    """Center, whiten, and normalize features with a fixed or trainable matrix."""

    def __init__(self, mean, matrix, trainable=False):
        """Initialize the whitening mean and a fixed or trainable transform matrix."""
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean).float().clone())
        self.matrix = nn.Parameter(torch.as_tensor(matrix).float().clone(), requires_grad=trainable)

    def forward(self, x):
        """Center, whiten, and normalize features in float32."""
        with torch.autocast(x.device.type, enabled=False):
            return F.normalize((x.float() - self.mean) @ self.matrix, dim=-1)


class TrainingModel(nn.Module):
    """Attach fixed teacher and trainable student whitening to the adapters."""

    def __init__(self, core, whitening=None):
        """Attach optional student and teacher whitening transforms to the core adapters."""
        super().__init__()
        self.core = core
        self.student_white = self.teacher_white = None
        if whitening is not None:
            self.student_white = Whitening(
                whitening["student_mean"], whitening["student_matrix"], trainable=True
            )
            self.teacher_white = Whitening(whitening["teacher_mean"], whitening["teacher_matrix"])

    def forward(self, batch):
        """Compute image and text embeddings with optional whitened distillation features."""
        hidden = self.core.text_hidden(batch["clip_text"])
        out = {
            "image": self.core.encode_image(batch["clip_vision"], batch.get("dino")),
            "text": F.normalize(hidden @ self.core.text_proj, dim=-1),
        }
        if self.student_white is not None:
            out["student"] = self.student_white(hidden)
            with torch.no_grad():
                out["teacher"] = self.teacher_white(batch["teacher"])
        return out
