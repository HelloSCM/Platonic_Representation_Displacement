"""Exact global-batch contrastive and relational distillation objectives."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .distributed import gather_features


def contrastive_rows(image, text, images, texts, labels, scale):
    """Compute symmetric contrastive loss for local rows against global image and text features."""
    loss = F.cross_entropy(scale * (image @ texts.T), labels, reduction="sum")
    loss += F.cross_entropy(scale * (text @ images.T), labels, reduction="sum")
    return loss / 2


def relation_rows(student, teacher, students, teachers, labels, temperature):
    """Compare off-diagonal teacher and student cosine distributions."""
    keep = torch.arange(len(students), device=student.device)[None] != labels[:, None]
    shape = (len(student), len(students) - 1)
    s = (student @ students.T)[keep].view(shape) / temperature
    t = (teacher @ teachers.T)[keep].view(shape) / temperature
    return F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="sum")


class Objectives(nn.Module):
    """Combine global-batch contrastive and relational distillation losses in chunks."""

    def __init__(self, chunk_size=128, temperature=0.1, rkd_weight=1.0):
        """Configure the loss chunk size, distillation temperature, and relational loss weight."""
        super().__init__()
        if chunk_size < 1 or temperature <= 0 or rkd_weight < 0:
            raise ValueError("Use a positive chunk size/temperature and nonnegative RKD weight.")
        self.chunk_size = chunk_size
        self.temperature = temperature
        self.rkd_weight = rkd_weight

    def forward(self, out, logit_scale, rank=0):
        """Gather global features and compute contrastive and relational losses in chunks."""
        local = {key: value.float() for key, value in out.items()}
        global_ = {key: gather_features(value) for key, value in local.items()}
        n = len(local["image"])
        labels = torch.arange(n, device=local["image"].device) + rank * n
        contrastive = local["image"].new_zeros(())
        rkd = contrastive.clone()
        for start in range(0, n, self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            contrastive = (
                contrastive
                + checkpoint(
                    contrastive_rows,
                    local["image"][sl],
                    local["text"][sl],
                    global_["image"],
                    global_["text"],
                    labels[sl],
                    logit_scale.exp(),
                    use_reentrant=False,
                )
                / n
            )
            if "student" in local:
                if len(global_["student"]) < 2:
                    raise ValueError("RKD requires at least two global samples.")
                rkd = (
                    rkd
                    + checkpoint(
                        relation_rows,
                        local["student"][sl],
                        local["teacher"][sl],
                        global_["student"],
                        global_["teacher"],
                        labels[sl],
                        self.temperature,
                        use_reentrant=False,
                    )
                    / n
                )
        return contrastive + self.rkd_weight * rkd, contrastive.detach(), rkd.detach()
