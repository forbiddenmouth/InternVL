import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from internvl_entropy.schemas import (
    EntropyLog,
    EntropySchema,
    GenerationMain,
    GenerationParams,
    GenerationSample,
    GenerationSchema,
    InputSchema,
    OptionalEntropyStats,
    SemanticEntropyCluster,
    SemanticEntropyStats,
    SequenceEntropyStats,
    TimingSchema,
    TokenEntropyStats,
)


def test_logging_schema_fields():
    log = EntropyLog(
        id="sample_1",
        meta={"dataset": "dummy"},
        input=InputSchema(image=None, prompt="Question: test"),
        generation=GenerationSchema(
            main=GenerationMain(text="answer", tokens=["answer"], token_ids=[1]),
            samples=[GenerationSample(text="sample")],
            params=GenerationParams(
                max_new_tokens=16,
                main_decoding={"do_sample": False, "temperature": 0.0},
                sample_decoding={
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_samples": 1,
                },
                seed=42,
            ),
        ),
        entropy=EntropySchema(
            token_shannon=TokenEntropyStats(per_token=[0.1], mean=0.1, max=0.1, p95=0.1),
            sequence=SequenceEntropyStats(nll_sum=1.0, nll_mean=1.0, length=1),
            semantic_entropy=SemanticEntropyStats(
                num_samples=1,
                embedding_model="dummy",
                clustering=SemanticEntropyCluster(method="agglomerative", metric="cosine", threshold=0.25),
                num_clusters=1,
                cluster_sizes=[1],
                p=[1.0],
                value=0.0,
            ),
            optional=OptionalEntropyStats(
                top2_margin={"per_token_mean": 0.2, "per_token_min": 0.2},
                topk_entropy={"per_token_mean": 0.1},
                sample_similarity={"pairwise_mean": 0.0, "pairwise_var": 0.0},
            ),
        ),
        timing=TimingSchema(main_gen_s=0.1, sample_gen_s=0.2, se_embed_s=0.1, se_cluster_s=0.01),
        errors=[],
    )
    payload = log.model_dump()
    for key in ["id", "meta", "input", "generation", "entropy", "timing", "errors"]:
        assert key in payload
