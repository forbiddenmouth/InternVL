from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

from internvl_entropy.config import EntropyConfig
from internvl_entropy.entropy_sequence import compute_sequence_nll
from internvl_entropy.entropy_token import compute_token_entropy
from internvl_entropy.generation import generate_main, generate_samples
from internvl_entropy.io import load_existing_ids, read_jsonl, write_jsonl
from internvl_entropy.model import load_model, prepare_inputs
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
from internvl_entropy.semantic_entropy import SemanticEntropyResult, compute_semantic_entropy
from internvl_entropy.utils_seed import set_seed
from internvl_entropy.utils_text import clean_text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InternVL entropy logging")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=0.7)
    parser.add_argument("--sample_top_p", type=float, default=0.9)
    parser.add_argument("--se_embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--se_cluster_method", type=str, default="agglomerative")
    parser.add_argument("--se_threshold", type=float, default=0.25)
    parser.add_argument("--se_cluster_metric", type=str, default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> EntropyConfig:
    if args.config:
        config = EntropyConfig.from_yaml(args.config)
        overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
        return config.model_copy(update=overrides)
    return EntropyConfig(**vars(args))


def main() -> None:
    args = _parse_args()
    config = _load_config(args)
    if config.input is None or config.output is None:
        raise ValueError("Input and output paths are required.")
    set_seed(config.seed)

    existing_ids = load_existing_ids(config.output) if config.resume else set()

    model, tokenizer = load_model(config.model, config.device, config.dtype)
    try:
        embedder = SentenceTransformer(config.se_embed_model)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to load embedding model. Semantic entropy requires sentence-transformers."
        ) from exc

    rows = []
    embed_cache: Dict[str, Any] = {}
    for idx, sample in enumerate(read_jsonl(config.input)):
        if config.limit is not None and idx >= config.limit:
            break
        sample_id = str(sample.get("id", idx))
        if sample_id in existing_ids:
            continue
        prompt = sample.get("prompt", "")
        image = sample.get("image")
        meta = sample.get("meta", {})
        errors: List[str] = []

        inputs = prepare_inputs(image, prompt, tokenizer)
        inputs = {k: v.to(config.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        start_main = time.time()
        try:
            main_out = generate_main(model, tokenizer, inputs, config.max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"main_generation_failed:{exc}")
            main_out = None
        main_gen_s = time.time() - start_main

        start_samples = time.time()
        try:
            samples = generate_samples(
                model,
                tokenizer,
                inputs,
                config.max_new_tokens,
                config.num_samples,
                config.sample_temperature,
                config.sample_top_p,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"sample_generation_failed:{exc}")
            samples = []
        sample_gen_s = time.time() - start_samples

        cleaned_main = clean_text(main_out.text) if main_out else ""
        cleaned_samples = [clean_text(text) for text in samples]

        if main_out:
            token_entropy = compute_token_entropy(main_out.scores, topk=50)
            seq_entropy = compute_sequence_nll(main_out.scores, main_out.token_ids)
            main_tokens = main_out.tokens
            main_token_ids = main_out.token_ids
        else:
            token_entropy = {"stats": {"per_token": [], "mean": 0.0, "max": 0.0, "p95": 0.0}, "optional": {"top2_margin": {"per_token_mean": 0.0, "per_token_min": 0.0}, "topk_entropy": {"per_token_mean": 0.0}}}
            seq_entropy = {"nll_sum": 0.0, "nll_mean": 0.0, "length": 0}
            main_tokens = []
            main_token_ids = []

        try:
            se_result = compute_semantic_entropy(
                cleaned_samples,
                embedder,
                method=config.se_cluster_method,
                metric=config.se_cluster_metric,
                threshold=config.se_threshold,
                cache=embed_cache,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"semantic_entropy_failed:{exc}")
            se_result = None

        se_embed_s = se_result.embed_s if se_result else 0.0
        se_cluster_s = se_result.cluster_s if se_result else 0.0

        valid_samples = [text for text in cleaned_samples if text]

        if se_result is None:
            errors.append("semantic_entropy_placeholder")
            se_result = SemanticEntropyResult(
                num_clusters=0,
                cluster_sizes=[],
                probabilities=[],
                value=0.0,
                warnings=["semantic_entropy_failed"],
            )
        entropy_schema = EntropySchema(
            token_shannon=TokenEntropyStats(**token_entropy["stats"]),
            sequence=SequenceEntropyStats(**seq_entropy),
            semantic_entropy=SemanticEntropyStats(
                num_samples=len(valid_samples),
                embedding_model=config.se_embed_model,
                clustering=SemanticEntropyCluster(
                    method=config.se_cluster_method,
                    metric=config.se_cluster_metric,
                    threshold=config.se_threshold,
                ),
                num_clusters=se_result.num_clusters,
                cluster_sizes=se_result.cluster_sizes,
                p=se_result.probabilities,
                value=se_result.value,
                warnings=se_result.warnings,
            ),
            optional=OptionalEntropyStats(
                top2_margin=token_entropy["optional"].get("top2_margin"),
                topk_entropy=token_entropy["optional"].get("topk_entropy"),
                sample_similarity={
                    "pairwise_mean": se_result.pairwise_mean,
                    "pairwise_var": se_result.pairwise_var,
                },
            ),
        )

        generation_schema = GenerationSchema(
            main=GenerationMain(
                text=cleaned_main,
                tokens=main_tokens,
                token_ids=main_token_ids,
            ),
            samples=[GenerationSample(text=text) for text in cleaned_samples],
            params=GenerationParams(
                max_new_tokens=config.max_new_tokens,
                main_decoding={"do_sample": False, "temperature": 0.0},
                sample_decoding={
                    "do_sample": True,
                    "temperature": config.sample_temperature,
                    "top_p": config.sample_top_p,
                    "num_samples": config.num_samples,
                },
                seed=config.seed,
            ),
        )

        log_entry = EntropyLog(
            id=sample_id,
            meta=meta,
            input=InputSchema(image=image, prompt=prompt),
            generation=generation_schema,
            entropy=entropy_schema,
            timing=TimingSchema(
                main_gen_s=main_gen_s,
                sample_gen_s=sample_gen_s,
                se_embed_s=se_embed_s,
                se_cluster_s=se_cluster_s,
            ),
            errors=errors,
        )
        rows.append(log_entry.model_dump())

        if config.log_every and (idx + 1) % config.log_every == 0:
            print(f"Processed {idx + 1} samples", file=sys.stderr)

    if rows:
        write_jsonl(config.output, rows)


if __name__ == "__main__":
    main()
