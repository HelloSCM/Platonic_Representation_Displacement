"""Zero-shot classification, bidirectional retrieval, and MMVP pair accuracy."""

from collections import defaultdict

import torch
from torch.nn import functional as F


def class_id(row):
    """Read a class identifier from a classification record."""
    for key in ("class_id", "concept", "label"):
        if key in row:
            return str(row[key])
    raise ValueError("Classification records need class_id, concept, or label.")


def classification(images, texts, image_rows, text_rows, chunk_size=512):
    """Evaluate zero-shot classification using averaged text prototypes for each class."""
    groups = defaultdict(list)
    for i, row in enumerate(text_rows):
        groups[class_id(row)].append(i)
    classes = sorted(groups)
    lookup = {name: i for i, name in enumerate(classes)}
    prototypes = F.normalize(torch.stack([texts[groups[name]].mean(0) for name in classes]), dim=-1)
    labels = torch.tensor([lookup[class_id(row)] for row in image_rows], device=images.device)
    top1, top5 = [], []
    for start in range(0, len(images), chunk_size):
        logits = images[start : start + chunk_size] @ prototypes.T
        predictions = logits.topk(min(5, len(classes)), dim=-1).indices
        target = labels[start : start + len(logits), None]
        top1.append((predictions[:, :1] == target).any(-1))
        top5.append((predictions == target).any(-1))
    correct = torch.cat(top1).float()
    per_class = [correct[labels == i].mean().item() for i in labels.unique()]
    return {
        "top1": 100 * correct.mean().item(),
        "top5": 100 * torch.cat(top5).float().mean().item(),
        "mean_class_top1": 100 * sum(per_class) / len(per_class),
        "images": len(images),
        "classes": len(classes),
        "prompts": len(texts),
    }


def recall(queries, candidates, query_ids, candidate_ids, chunk_size):
    """Compute retrieval recall at selected ranks using matching query and candidate identities."""
    all_ids = sorted(set(query_ids) | set(candidate_ids))
    lookup = {name: i for i, name in enumerate(all_ids)}
    query_ids = torch.tensor([lookup[name] for name in query_ids], device=queries.device)
    candidate_ids = torch.tensor([lookup[name] for name in candidate_ids], device=queries.device)
    counts = {k: 0 for k in (1, 5, 10)}
    for start in range(0, len(queries), chunk_size):
        scores = queries[start : start + chunk_size] @ candidates.T
        top = scores.topk(min(10, len(candidates)), dim=-1).indices
        hits = candidate_ids[top] == query_ids[start : start + len(scores), None]
        for k in counts:
            counts[k] += hits[:, :k].any(-1).sum().item()
    return {f"R@{k}": 100 * count / len(queries) for k, count in counts.items()}


def retrieval(images, texts, image_rows, text_rows, chunk_size=512):
    """Evaluate image-to-text and text-to-image retrieval recall."""
    image_ids = [str(row["id"]) for row in image_rows]
    text_ids = [str(row["image_id"]) for row in text_rows]
    if len(set(image_ids)) != len(image_ids) or set(text_ids) != set(image_ids):
        raise ValueError("Every retrieval image must have captions linked by image_id.")
    return {
        "i2t": recall(images, texts, image_ids, text_ids, chunk_size),
        "t2i": recall(texts, images, text_ids, image_ids, chunk_size),
        "images": len(images),
        "captions": len(texts),
    }


def mmvp(images, texts, image_rows, text_rows):
    """Count a pair only when both captions prefer their corresponding images."""
    image_groups, text_groups = defaultdict(list), defaultdict(list)
    for i, row in enumerate(image_rows):
        image_groups[str(row["pair_id"])].append(i)
    for i, row in enumerate(text_rows):
        text_groups[str(row["pair_id"])].append(i)
    if image_groups.keys() != text_groups.keys():
        raise ValueError("MMVP image and caption pair IDs must match.")
    categories = defaultdict(list)
    results = []
    for pair, ii in image_groups.items():
        tt = text_groups[pair]
        if len(ii) != 2 or len(tt) != 2:
            raise ValueError("Each MMVP pair must contain two images and two captions.")
        ids = [str(image_rows[i]["id"]) for i in ii]
        expected = [str(text_rows[t]["image_id"]) for t in tt]
        if set(expected) != set(ids):
            raise ValueError("MMVP captions must link to the two distinct images by image_id.")
        scores = texts[tt] @ images[ii].T
        correct = all(
            scores[j, ids.index(target)] > scores[j, 1 - ids.index(target)]
            for j, target in enumerate(expected)
        )
        value = float(correct)
        results.append(value)
        category = image_rows[ii[0]].get("category")
        if category is not None:
            categories[str(category)].append(value)
    return {
        "pair_accuracy": 100 * sum(results) / len(results),
        "pairs": len(results),
        "categories": {key: 100 * sum(values) / len(values) for key, values in categories.items()},
    }
