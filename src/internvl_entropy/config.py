from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class EntropyConfig(BaseModel):
    input: Optional[Path] = None
    output: Optional[Path] = None
    model: str = "OpenGVLab/InternVL2_5-8B"
    device: str = "cuda"
    dtype: str = "bf16"
    max_new_tokens: int = 128
    num_samples: int = 8
    sample_temperature: float = 0.7
    sample_top_p: float = 0.9
    seed: int = 42
    se_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    se_cluster_method: str = "agglomerative"
    se_threshold: float = 0.25
    se_cluster_metric: str = "cosine"
    log_every: int = 10
    resume: bool = False
    limit: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "EntropyConfig":
        data = yaml.safe_load(path.read_text())
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
