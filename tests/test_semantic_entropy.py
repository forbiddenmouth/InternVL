import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from internvl_entropy.semantic_entropy import compute_semantic_entropy


class DummyEmbedder:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, texts, convert_to_numpy=True):
        embeddings = [self.mapping[text] for text in texts]
        return np.array(embeddings, dtype=np.float32)


def test_semantic_entropy_clusters():
    texts = [
        "cluster1_a",
        "cluster1_b",
        "cluster1_c",
        "cluster1_d",
        "cluster2_a",
        "cluster2_b",
        "cluster2_c",
        "cluster3_a",
    ]
    mapping = {
        "cluster1_a": [1.0, 0.0],
        "cluster1_b": [1.0, 0.0],
        "cluster1_c": [1.0, 0.0],
        "cluster1_d": [1.0, 0.0],
        "cluster2_a": [0.0, 1.0],
        "cluster2_b": [0.0, 1.0],
        "cluster2_c": [0.0, 1.0],
        "cluster3_a": [-1.0, 0.0],
    }
    embedder = DummyEmbedder(mapping)
    result = compute_semantic_entropy(
        texts,
        embedder,
        method="agglomerative",
        metric="cosine",
        threshold=0.25,
    )
    assert result.num_clusters == 3
    assert 0.0 < result.value < np.log(len(texts))


def test_semantic_entropy_all_same():
    texts = ["same_1", "same_2", "same_3"]
    mapping = {text: [1.0, 0.0] for text in texts}
    embedder = DummyEmbedder(mapping)
    result = compute_semantic_entropy(
        texts,
        embedder,
        method="agglomerative",
        metric="cosine",
        threshold=0.25,
    )
    assert result.num_clusters == 1
    assert result.value < 1e-3
