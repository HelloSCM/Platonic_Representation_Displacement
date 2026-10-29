"""Feature and relational distillation objectives."""

import torch
from torch.nn import functional as F


def feature_loss(student, teacher):
    """Match teacher representations with mean squared error."""
    return F.mse_loss(student.float(), teacher.detach().float())


def relation_loss(student, teacher, temperature=0.1):
    """Match teacher distributions over off-diagonal batch cosine similarities."""
    with torch.autocast(student.device.type, enabled=False):
        student = F.normalize(student.float(), dim=-1)
        teacher = F.normalize(teacher.detach().float(), dim=-1)
        n = len(student)
        if n < 2:
            return student.sum() * 0
        mask = ~torch.eye(n, dtype=torch.bool, device=student.device)
        s = (student @ student.T)[mask].reshape(n, n - 1) / temperature
        t = (teacher @ teacher.T)[mask].reshape(n, n - 1) / temperature
        return F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="batchmean")
