from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, device: str, dtype: str):
    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def prepare_inputs(image: str | None, prompt: str, tokenizer) -> Dict[str, Any]:
    if image is None:
        inputs = tokenizer(prompt, return_tensors="pt")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs["image"] = image
    return inputs
