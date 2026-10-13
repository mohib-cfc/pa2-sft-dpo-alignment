"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function is a thin wrapper
around ``pa2/core.py`` (the same code the training scripts use).

Documented choices:
* ``compute_sft_loss`` reduction: mean negative log-probability over all
  unmasked response tokens in the batch (token-mean).
* Batches are right-padded to the longest row (never longer than ``max_length``);
  EOS is appended to every response before truncation.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:  # make ``pa2`` importable with or without ``pip install -e .``
    sys.path.insert(0, _ROOT)

from pa2 import core  # noqa: E402


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return core.build_lm_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    return core.response_logprobs(model, tokenizer, prompts, responses, max_length)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    return core.sft_loss(model, tokenizer, prompts, responses, max_length)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return core.build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    return core.dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    return core.dpo_loss(
        policy_model,
        reference_model,
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        beta,
        max_length,
    )
