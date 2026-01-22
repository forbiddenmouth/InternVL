from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def _entropy_from_logits(logits: torch.Tensor, eps: float = 1e-12) -> float:
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs + eps)).sum(dim=-1)
    return ent.item()


def _top2_margin_from_logits(logits: torch.Tensor) -> float:
    values, _ = torch.topk(logits, k=2, dim=-1)
    return (values[..., 0] - values[..., 1]).item()


def _topk_entropy_from_logits(logits: torch.Tensor, k: int, eps: float = 1e-12) -> float:
    values, _ = torch.topk(logits, k=k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    ent = -(probs * torch.log(probs + eps)).sum(dim=-1)
    return ent.item()


def compute_token_entropy(
    scores: List[torch.Tensor],
    topk: int | None = None,
) -> Dict[str, Dict[str, float] | List[float]]:
    per_token = []
    top2_margins = []
    topk_entropies = []
    for step_scores in scores:
        logits = step_scores.squeeze(0)
        per_token.append(_entropy_from_logits(logits))
        top2_margins.append(_top2_margin_from_logits(logits))
        if topk is not None:
            topk_entropies.append(_topk_entropy_from_logits(logits, topk))
    arr = np.array(per_token, dtype=np.float32)
    stats = {
        "per_token": per_token,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
    }
    optional = {
        "top2_margin": {
            "per_token_mean": float(np.mean(top2_margins)) if top2_margins else 0.0,
            "per_token_min": float(np.min(top2_margins)) if top2_margins else 0.0,
        }
    }
    if topk is not None:
        optional["topk_entropy"] = {
            "per_token_mean": float(np.mean(topk_entropies)) if topk_entropies else 0.0
        }
    return {"stats": stats, "optional": optional}
