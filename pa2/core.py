"""Core SFT / DPO math shared by the training code and ``submission_adapter.py``.

This module deliberately depends on nothing but PyTorch so the private grader can
import it with only ``PYTHONPATH=.``.

Conventions (documented in the report):

* A sequence is ``prompt_ids + response_ids + [eos]`` and is then right-truncated to
  ``max_length``. EOS is appended *before* truncation, so it can be cut off.
* Tokens are encoded with ``add_special_tokens=False`` (no BOS is inserted).
* Labels are ``-100`` on prompt tokens and padding. Masks are built from lengths,
  never by comparing ids with ``pad_token_id``: Qwen2.5 uses the same id for pad
  and EOS, so an id comparison would silently drop the EOS label.
* Rows are right-padded to the longest row in the batch (never beyond
  ``max_length``). Padding never affects real positions of a causal LM.
* Response log-probabilities are summed over response tokens only (label shift by
  one), in float32.
* SFT loss reduction: mean negative log-probability over all unmasked response
  tokens in the batch (token-mean).
* DPO loss: ``-log sigmoid(beta * ((pi_c - pi_r) - (ref_c - ref_r)))`` averaged over
  the batch; the reference forward runs under ``torch.no_grad()``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100

Batch = Dict[str, torch.Tensor]


# --------------------------------------------------------------------------- #
# Tokenizer helpers
# --------------------------------------------------------------------------- #
def encode_text(tokenizer, text: str) -> List[int]:
    """Encode ``text`` without special tokens; works for HF and toy tokenizers."""
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(i) for i in ids]


def _first_int(value):
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) else None
    return None if value is None else int(value)


def get_eos_id(tokenizer):
    return _first_int(getattr(tokenizer, "eos_token_id", None))


def get_pad_id(tokenizer) -> int:
    pad = _first_int(getattr(tokenizer, "pad_token_id", None))
    if pad is None:
        pad = get_eos_id(tokenizer)
    return 0 if pad is None else pad


# --------------------------------------------------------------------------- #
# Batch construction
# --------------------------------------------------------------------------- #
def tokenize_example(tokenizer, prompt: str, response: str, max_length: int) -> Tuple[List[int], List[int]]:
    """Return ``(input_ids, labels)`` for one prompt/response pair (unpadded)."""
    if max_length is None or int(max_length) < 1:
        raise ValueError(f"max_length must be a positive integer, got {max_length!r}")
    max_length = int(max_length)
    prompt_ids = encode_text(tokenizer, prompt)
    response_ids = encode_text(tokenizer, response)
    eos = get_eos_id(tokenizer)
    if eos is not None:
        response_ids = response_ids + [eos]  # EOS goes on before truncation
    input_ids = (prompt_ids + response_ids)[:max_length]
    n_prompt = min(len(prompt_ids), len(input_ids))
    labels = [IGNORE_INDEX] * n_prompt + input_ids[n_prompt:]
    return input_ids, labels


def build_lm_batch(tokenizer, prompts: Sequence[str], responses: Sequence[str], max_length: int) -> Batch:
    """Right-padded LM batch with response-only labels; preserves row order."""
    prompts = list(prompts)
    responses = list(responses)
    if len(prompts) != len(responses):
        raise ValueError(f"got {len(prompts)} prompts but {len(responses)} responses")
    rows = [tokenize_example(tokenizer, p, r, max_length) for p, r in zip(prompts, responses)]
    width = max([len(ids) for ids, _ in rows], default=0)
    pad_id = get_pad_id(tokenizer)
    n = len(rows)
    input_ids = torch.full((n, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, width), dtype=torch.long)
    labels = torch.full((n, width), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, lab) in enumerate(rows):
        length = len(ids)
        if length:
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1
            labels[i, :length] = torch.tensor(lab, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length: int) -> Dict[str, Batch]:
    prompts = list(prompts)
    chosen_responses = list(chosen_responses)
    rejected_responses = list(rejected_responses)
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen_responses and rejected_responses must have equal length")
    return {
        "chosen": build_lm_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_lm_batch(tokenizer, prompts, rejected_responses, max_length),
    }


def pad_batch_to(batch: Batch, width: int, pad_id: int) -> Batch:
    """Right-pad an LM batch to ``width`` columns."""
    extra = width - batch["input_ids"].shape[1]
    if extra <= 0:
        return batch
    return {
        "input_ids": F.pad(batch["input_ids"], (0, extra), value=pad_id),
        "attention_mask": F.pad(batch["attention_mask"], (0, extra), value=0),
        "labels": F.pad(batch["labels"], (0, extra), value=IGNORE_INDEX),
    }


def concat_batches(first: Batch, second: Batch, pad_id: int) -> Batch:
    width = max(first["input_ids"].shape[1], second["input_ids"].shape[1])
    a = pad_batch_to(first, width, pad_id)
    b = pad_batch_to(second, width, pad_id)
    return {key: torch.cat([a[key], b[key]], dim=0) for key in ("input_ids", "attention_mask", "labels")}


# --------------------------------------------------------------------------- #
# Log-probabilities and losses
# --------------------------------------------------------------------------- #
def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def batch_to_device(batch: Batch, device) -> Batch:
    return {key: value.to(device) for key, value in batch.items()}


def forward_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    return output.logits if hasattr(output, "logits") else output[0]


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Summed response-token log-probs per row and the number of scored tokens.

    ``logits[:, t]`` predicts token ``t + 1``; positions whose (shifted) label is
    ``-100`` contribute nothing. Computed in float32 for fp16/bf16 stability.
    """
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and labels {tuple(labels.shape)} disagree")
    labels = labels.to(logits.device)
    shift_logits = logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)
    safe_targets = targets.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask.to(token_logps.dtype)
    return token_logps.sum(dim=-1), mask.sum(dim=-1)


def batch_logprobs(model, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run ``model`` on an LM batch (moved to the model's device)."""
    batch = batch_to_device(batch, model_device(model))
    logits = forward_logits(model, batch["input_ids"], batch["attention_mask"])
    return sequence_logprobs(logits, batch["labels"])


def response_logprobs(model, tokenizer, prompts, responses, max_length: int) -> torch.Tensor:
    batch = build_lm_batch(tokenizer, prompts, responses, max_length)
    logps, _ = batch_logprobs(model, batch)
    return logps


def sft_loss_from_batch(model, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Token-mean response NLL and the number of response tokens scored."""
    logps, counts = batch_logprobs(model, batch)
    n_tokens = counts.sum()
    return -(logps.sum() / n_tokens.clamp_min(1)), n_tokens


def sft_loss(model, tokenizer, prompts, responses, max_length: int) -> torch.Tensor:
    loss, _ = sft_loss_from_batch(model, build_lm_batch(tokenizer, prompts, responses, max_length))
    return loss


def dpo_loss_from_logps(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Scalar batch-mean DPO objective."""
    device, dtype = policy_chosen_logps.device, policy_chosen_logps.dtype
    policy_rejected_logps = policy_rejected_logps.to(device=device, dtype=dtype)
    reference_chosen_logps = reference_chosen_logps.to(device=device, dtype=dtype)
    reference_rejected_logps = reference_rejected_logps.to(device=device, dtype=dtype)
    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_gap - reference_gap)
    return -F.logsigmoid(logits).mean()


def dpo_rewards(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    """Implicit rewards r_w, r_l (detached) used for reward accuracy / margin."""
    device = policy_chosen_logps.device
    chosen = beta * (policy_chosen_logps.detach() - reference_chosen_logps.detach().to(device))
    rejected = beta * (policy_rejected_logps.detach() - reference_rejected_logps.detach().to(device))
    return chosen, rejected


def concatenated_logprobs(model, dpo_batch: Dict[str, Batch], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """One forward over [chosen; rejected]; returns (chosen_logps, rejected_logps)."""
    n = dpo_batch["chosen"]["input_ids"].shape[0]
    merged = concat_batches(dpo_batch["chosen"], dpo_batch["rejected"], pad_id)
    logps, _ = batch_logprobs(model, merged)
    return logps[:n], logps[n:]


def dpo_forward(policy_model, reference_model, dpo_batch: Dict[str, Batch], pad_id: int, beta: float):
    """Policy forward with grad, reference forward under no_grad.

    Returns ``(loss, policy_chosen, policy_rejected, reference_chosen, reference_rejected)``.
    """
    policy_chosen, policy_rejected = concatenated_logprobs(policy_model, dpo_batch, pad_id)
    with torch.no_grad():
        reference_chosen, reference_rejected = concatenated_logprobs(reference_model, dpo_batch, pad_id)
    reference_chosen = reference_chosen.to(policy_chosen.device)
    reference_rejected = reference_rejected.to(policy_chosen.device)
    loss = dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
    return loss, policy_chosen, policy_rejected, reference_chosen, reference_rejected


def dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    loss, *_ = dpo_forward(policy_model, reference_model, batch, get_pad_id(tokenizer), beta)
    return loss
