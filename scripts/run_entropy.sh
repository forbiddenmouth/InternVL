#!/usr/bin/env bash
set -euo pipefail

python -m internvl_entropy.cli \
  --input data/pope_val.jsonl \
  --output runs/entropy/pope_val_entropy.jsonl \
  --model OpenGVLab/InternVL2_5-8B \
  --max_new_tokens 128 \
  --num_samples 8 \
  --sample_temperature 0.7 \
  --sample_top_p 0.9 \
  --se_embed_model sentence-transformers/all-MiniLM-L6-v2 \
  --se_cluster_method agglomerative \
  --se_threshold 0.25 \
  --seed 42 \
  --device cuda
