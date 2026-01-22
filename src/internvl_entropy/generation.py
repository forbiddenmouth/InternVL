from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class GenerationOutput:
    text: str
    token_ids: List[int]
    tokens: List[str]
    scores: List[torch.Tensor]


def generate_main(
    model,
    tokenizer,
    inputs,
    max_new_tokens: int,
) -> GenerationOutput:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    sequences = outputs.sequences[0]
    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = sequences[prompt_len:]
    tokens = tokenizer.convert_ids_to_tokens(gen_ids)
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    scores = outputs.scores
    return GenerationOutput(text=text, token_ids=gen_ids.tolist(), tokens=tokens, scores=scores)


def generate_samples(
    model,
    tokenizer,
    inputs,
    max_new_tokens: int,
    num_samples: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    samples: List[str] = []
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                return_dict_in_generate=True,
                output_scores=False,
            )
            sequences = outputs.sequences[0]
            prompt_len = inputs["input_ids"].shape[-1]
            gen_ids = sequences[prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            samples.append(text)
    return samples
