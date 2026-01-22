import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from internvl_entropy.entropy_token import compute_token_entropy


def test_token_entropy_stats():
    logits_step1 = torch.tensor([[0.0, 0.0, 0.0]])
    logits_step2 = torch.tensor([[10.0, 0.0, 0.0]])
    result = compute_token_entropy([logits_step1, logits_step2], topk=2)
    per_token = result["stats"]["per_token"]
    expected_step1 = -np.sum((1 / 3) * np.log(1 / 3))
    expected_step2 = 0.0
    assert np.isclose(per_token[0], expected_step1, atol=1e-4)
    assert per_token[1] < 1e-3
    assert np.isclose(result["stats"]["mean"], np.mean(per_token), atol=1e-4)
    assert np.isclose(result["stats"]["max"], np.max(per_token), atol=1e-4)
    assert np.isclose(result["stats"]["p95"], np.percentile(per_token, 95), atol=1e-4)
