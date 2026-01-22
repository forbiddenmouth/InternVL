from __future__ import annotations

from typing import List

import torch


def compute_sequence_nll(
    scores: List[torch.Tensor],
    token_ids: List[int],
) -> dict[str, float | int]:
    nll_sum = 0.0
    for step_scores, token_id in zip(scores, token_ids):
        logits = step_scores.squeeze(0)
        log_probs = torch.log_softmax(logits, dim=-1)
        nll_sum -= log_probs[token_id].item()
    length = len(token_ids)
    nll_mean = nll_sum / length if length else 0.0
    return {"nll_sum": float(nll_sum), "nll_mean": float(nll_mean), "length": length}
