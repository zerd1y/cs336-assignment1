from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

from llm_basics import generate, sample_top_p


class StepwiseLogitModel(nn.Module):
    def __init__(self, per_step_logits: list[Tensor]) -> None:
        super().__init__()
        self.register_buffer("_anchor", torch.zeros(1))
        self.per_step_logits = [logits.clone() for logits in per_step_logits]
        self.call_count = 0

    def forward(self, token_ids: Tensor) -> Tensor:
        batch_size, seq_len = token_ids.shape
        step_logits = self.per_step_logits[min(self.call_count, len(self.per_step_logits) - 1)].to(token_ids.device)
        logits = torch.zeros(batch_size, seq_len, step_logits.shape[-1], device=token_ids.device, dtype=step_logits.dtype)
        logits[:, -1, :] = step_logits
        self.call_count += 1
        return logits


def test_generate_normalizes_1d_list_and_uses_greedy_decoding() -> None:
    model = StepwiseLogitModel(
        [
            torch.tensor([[0.1, 2.0, 0.0, -1.0]]),
            torch.tensor([[0.0, -1.0, 3.0, 0.5]]),
        ]
    )

    output = generate(
        model=model,
        prompt_ids=[7, 8],
        max_new_tokens=2,
        temperature=0.0,
    )

    expected = torch.tensor([[7, 8, 1, 2]], dtype=torch.long)
    assert torch.equal(output, expected)


def test_sample_top_p_keeps_boundary_token_in_candidate_set() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, -0.1, -5.0]], dtype=torch.float32)

    samples = torch.stack([sample_top_p(logits, top_p=0.7) for _ in range(128)])

    assert torch.all(samples != 2)
    assert torch.any(samples == 0)
    assert torch.any(samples == 1)


def test_generate_overwrites_finished_sequences_with_eos_and_exits_early() -> None:
    eos_token_id = 4
    model = StepwiseLogitModel(
        [
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 5.0],
                    [0.0, 6.0, 0.0, 0.0, 0.0],
                ]
            ),
            torch.tensor(
                [
                    [0.0, 8.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 7.0],
                ]
            ),
            torch.tensor(
                [
                    [9.0, 0.0, 0.0, 0.0, 0.0],
                    [9.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ]
    )

    output = generate(
        model=model,
        prompt_ids=[[1, 2], [3, 4]],
        max_new_tokens=5,
        eos_token_id=eos_token_id,
        temperature=0.0,
    )

    expected = torch.tensor(
        [
            [1, 2, 4, 4],
            [3, 4, 1, 4],
        ],
        dtype=torch.long,
    )
    assert torch.equal(output, expected)
    assert model.call_count == 2


def test_generate_preserves_tensor_device_and_long_dtype() -> None:
    model = StepwiseLogitModel([torch.tensor([[0.0, 1.0]])])
    prompt = torch.tensor([0, 1], dtype=torch.int32)

    output = generate(model=model, prompt_ids=prompt, max_new_tokens=1, temperature=0.0)

    assert output.device == prompt.device
    assert output.dtype == torch.long
