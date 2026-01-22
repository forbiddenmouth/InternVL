from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer


@dataclass
class SemanticEntropyResult:
    num_clusters: int
    cluster_sizes: List[int]
    probabilities: List[float]
    value: float
    warnings: List[str]
    pairwise_mean: float | None = None
    pairwise_var: float | None = None
    embed_s: float = 0.0
    cluster_s: float = 0.0


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    return embeddings / norms


def _cluster_embeddings(
    embeddings: np.ndarray,
    method: str,
    metric: str,
    threshold: float,
) -> np.ndarray:
    if method != "agglomerative":
        raise ValueError(f"Unsupported clustering method: {method}")
    clustering = AgglomerativeClustering(
        metric=metric,
        linkage="average",
        distance_threshold=threshold,
        n_clusters=None,
    )
    labels = clustering.fit_predict(embeddings)
    return labels


def _entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> float:
    return float(-(probs * np.log(probs + eps)).sum())


def compute_semantic_entropy(
    texts: List[str],
    embedder: SentenceTransformer,
    method: str = "agglomerative",
    metric: str = "cosine",
    threshold: float = 0.25,
    cache: Dict[str, np.ndarray] | None = None,
) -> SemanticEntropyResult:
    warnings: List[str] = []
    cleaned = []
    for text in texts:
        if not text:
            warnings.append("empty_sample")
            continue
        cleaned.append(text)
        if len(text.split()) < 3:
            warnings.append("short_sample")
    if not cleaned:
        raise ValueError("No valid samples for semantic entropy.")
    embedding_list: List[np.ndarray] = []
    if cache is None:
        cache = {}
    new_texts = [text for text in cleaned if text not in cache]
    if new_texts:
        import time

        embed_start = time.perf_counter()
        new_embeddings = embedder.encode(new_texts, convert_to_numpy=True)
        embed_s = time.perf_counter() - embed_start
        for text, embedding in zip(new_texts, new_embeddings):
            cache[text] = embedding
    else:
        embed_s = 0.0
    embedding_list = [cache[text] for text in cleaned]
    embeddings = _normalize_embeddings(np.vstack(embedding_list))
    import time

    cluster_start = time.perf_counter()
    labels = _cluster_embeddings(embeddings, method, metric, threshold)
    cluster_s = time.perf_counter() - cluster_start
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = counts.tolist()
    probabilities = (counts / counts.sum()).tolist()
    value = _entropy_from_probs(np.array(probabilities))
    result = SemanticEntropyResult(
        num_clusters=len(unique),
        cluster_sizes=cluster_sizes,
        probabilities=probabilities,
        value=value,
        warnings=warnings,
    )
    if embeddings.shape[0] > 1:
        distances = pairwise_distances(embeddings, metric=metric)
        upper = distances[np.triu_indices_from(distances, k=1)]
        result.pairwise_mean = float(upper.mean()) if upper.size else 0.0
        result.pairwise_var = float(upper.var()) if upper.size else 0.0
    result.embed_s = embed_s
    result.cluster_s = cluster_s
    return result
