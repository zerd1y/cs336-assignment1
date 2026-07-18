from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


def _get_model_device(model: nn.Module) -> torch.device:
    """Return the device of the first parameter or buffer in a module."""
    for tensor in list(model.parameters()) + list(model.buffers()):
        return tensor.device
    return torch.device("cpu")


def _normalize_prompt_ids(
    prompt_ids: Union[Tensor, list[int], list[list[int]]],
    *,
    device: torch.device,
) -> Tensor:
    """Convert prompt ids into a 2D ``LongTensor`` of shape ``(batch, seq_len)``."""
    if isinstance(prompt_ids, Tensor):
        normalized = prompt_ids.to(device=device, dtype=torch.long)
    elif isinstance(prompt_ids, Sequence):
        if len(prompt_ids) == 0:
            raise ValueError("prompt_ids must not be empty.")
        first_item = prompt_ids[0]
        if isinstance(first_item, Sequence) and not isinstance(first_item, (int, bool)):
            normalized = torch.tensor(prompt_ids, device=device, dtype=torch.long)
        else:
            normalized = torch.tensor(prompt_ids, device=device, dtype=torch.long).unsqueeze(0)
    else:
        raise TypeError("prompt_ids must be a Tensor, list[int], or list[list[int]].")

    if normalized.ndim == 1:
        normalized = normalized.unsqueeze(0)
    if normalized.ndim != 2:
        raise ValueError(f"prompt_ids must be 1D or 2D after normalization, got shape {tuple(normalized.shape)}.")
    if normalized.shape[1] == 0:
        raise ValueError("prompt_ids must have a non-zero sequence length.")
    return normalized


def sample_top_p(logits: Tensor, top_p: float) -> Tensor:
    """Sample token ids from logits using nucleus sampling."""
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (batch_size, vocab_size), got {tuple(logits.shape)}.")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}.")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        filtered_sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted_logits)
    else:
        filtered_logits = logits

    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.inference_mode()
def generate(
    model: nn.Module,
    prompt_ids: Union[Tensor, list[int], list[list[int]]],
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tensor:
    """Generate tokens autoregressively from a decoder-only language model.

    Args:
        model: Language model that maps token ids of shape ``(B, T)`` to logits of shape ``(B, T, V)``.
        prompt_ids: Prompt token ids as a 1D/2D tensor or Python list.
        max_new_tokens: Maximum number of tokens to append.
        eos_token_id: Optional end-of-sequence token id for early stopping.
        temperature: Sampling temperature. Values ``<= 1e-5`` use greedy decoding.
        top_p: Nucleus sampling threshold in ``(0, 1]``.

    Returns:
        A ``LongTensor`` of shape ``(batch_size, prompt_length + generated_length)`` containing the prompt and
        generated continuation.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}.")
    if temperature < 0.0:
        raise ValueError(f"temperature must be non-negative, got {temperature}.")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}.")

    device = prompt_ids.device if isinstance(prompt_ids, Tensor) else _get_model_device(model)
    generated = _normalize_prompt_ids(prompt_ids, device=device)

    batch_size = generated.shape[0]
    unfinished_sequences = torch.ones(batch_size, device=generated.device, dtype=torch.bool)
    context_length = getattr(model, "context_length", None)

    for _ in range(max_new_tokens):
        model_input = generated
        if isinstance(context_length, int) and context_length > 0 and generated.shape[1] > context_length:
            model_input = generated[:, -context_length:]

        logits = model(model_input)
        if not isinstance(logits, Tensor):
            raise TypeError("model(generated) must return a Tensor.")
        if logits.ndim != 3:
            raise ValueError(f"model(generated) must return logits of shape (B, T, V), got {tuple(logits.shape)}.")

        next_token_logits = logits[:, -1, :]

        if temperature <= 1e-5:
            next_tokens = torch.argmax(next_token_logits, dim=-1)
        else:
            scaled_logits = next_token_logits / temperature
            next_tokens = sample_top_p(scaled_logits, top_p=top_p)

        if eos_token_id is not None:
            eos_fill = torch.full_like(next_tokens, eos_token_id)
            next_tokens = torch.where(unfinished_sequences, next_tokens, eos_fill)

        generated = torch.cat((generated, next_tokens.unsqueeze(-1)), dim=-1)

        if eos_token_id is None:
            continue

        newly_finished = next_tokens.eq(eos_token_id)
        unfinished_sequences = unfinished_sequences & ~newly_finished
        if not torch.any(unfinished_sequences):
            break

    return generated


__all__ = ["generate", "sample_top_p"]
