from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class InputSchema(BaseModel):
    image: Optional[str] = None
    prompt: str


class GenerationMain(BaseModel):
    text: str
    tokens: List[str]
    token_ids: List[int]


class GenerationSample(BaseModel):
    text: str


class GenerationParams(BaseModel):
    max_new_tokens: int
    main_decoding: Dict[str, Any]
    sample_decoding: Dict[str, Any]
    seed: int


class GenerationSchema(BaseModel):
    main: GenerationMain
    samples: List[GenerationSample]
    params: GenerationParams


class TokenEntropyStats(BaseModel):
    per_token: List[float]
    mean: float
    max: float
    p95: float


class SequenceEntropyStats(BaseModel):
    nll_sum: float
    nll_mean: float
    length: int


class SemanticEntropyCluster(BaseModel):
    method: str
    metric: str
    threshold: float


class SemanticEntropyStats(BaseModel):
    num_samples: int
    embedding_model: str
    clustering: SemanticEntropyCluster
    num_clusters: int
    cluster_sizes: List[int]
    p: List[float]
    value: float
    warnings: List[str] = Field(default_factory=list)


class OptionalEntropyStats(BaseModel):
    top2_margin: Optional[Dict[str, float]] = None
    topk_entropy: Optional[Dict[str, float]] = None
    sample_similarity: Optional[Dict[str, float]] = None


class EntropySchema(BaseModel):
    token_shannon: TokenEntropyStats
    sequence: SequenceEntropyStats
    semantic_entropy: SemanticEntropyStats
    optional: OptionalEntropyStats = Field(default_factory=OptionalEntropyStats)


class TimingSchema(BaseModel):
    main_gen_s: float
    sample_gen_s: float
    se_embed_s: float
    se_cluster_s: float


class EntropyLog(BaseModel):
    id: str
    meta: Dict[str, Any] = Field(default_factory=dict)
    input: InputSchema
    generation: GenerationSchema
    entropy: EntropySchema
    timing: TimingSchema
    errors: List[str] = Field(default_factory=list)
