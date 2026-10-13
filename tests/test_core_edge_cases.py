"""Extra edge-case checks beyond the public smoke tests (truncation, odd tokenizers,
row order, DPO numerics, gradient direction). Offline, CPU-only."""

from __future__ import annotations

import math

import torch

import submission_adapter as adapter
from pa2 import core
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class PadIsEosTokenizer(ToyTokenizer):
    """Like Qwen2.5: pad_token_id == eos_token_id."""

    pad_token_id = 1


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


class PositionalToyLM(torch.nn.Module):
    """Toy causal LM whose logits depend on the current token (so order matters)."""

    def __init__(self, vocab_size, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.table = torch.nn.Parameter(torch.randn(vocab_size, vocab_size, generator=g))

    def forward(self, input_ids, attention_mask=None, labels=None):
        from types import SimpleNamespace

        return SimpleNamespace(logits=self.table[input_ids])


def _manual_logp(model, tok, prompt, response, max_length):
    ids, labels = core.tokenize_example(tok, prompt, response, max_length)
    x = torch.tensor([ids])
    logits = model(input_ids=x).logits[0]
    total = 0.0
    for t in range(1, len(ids)):
        if labels[t] != -100:
            total += torch.log_softmax(logits[t - 1].float(), -1)[ids[t]].item()
    return total


def test_pad_equal_to_eos_keeps_eos_label():
    tok = PadIsEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: ", "Longer question: "], ["a", "b"], max_length=64)
    for row, (p, r) in enumerate([("Q: ", "a"), ("Longer question: ", "b")]):
        plen = len(tok.encode(p))
        rlen = len(tok.encode(r)) + 1
        assert batch["labels"][row, plen + rlen - 1].item() == tok.eos_token_id
        assert batch["attention_mask"][row, : plen + rlen].eq(1).all()
        assert batch["attention_mask"][row, plen + rlen :].eq(0).all()


def test_eos_appended_before_truncation_and_can_be_cut():
    tok = ToyTokenizer()
    prompt, response = "abc", "defgh"  # 3 + 5 + eos = 9 tokens
    full = adapter.build_sft_batch(tok, [prompt], [response], max_length=9)
    assert full["input_ids"].shape[1] == 9
    assert full["labels"][0, -1].item() == tok.eos_token_id
    cut = adapter.build_sft_batch(tok, [prompt], [response], max_length=7)
    assert cut["input_ids"].shape[1] == 7
    assert tok.eos_token_id not in cut["labels"][0].tolist()
    assert cut["labels"][0, :3].eq(-100).all() and cut["labels"][0, 3:].ne(-100).all()


def test_prompt_longer_than_max_length_masks_everything():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["abcdefghij"], ["xyz"], max_length=5)
    assert batch["input_ids"].shape == (1, 5)
    assert batch["labels"].eq(-100).all()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.3)
    logps = adapter.compute_response_logprobs(model, tok, ["abcdefghij"], ["xyz"], max_length=5)
    assert logps.shape == (1,) and logps.item() == 0.0


def test_missing_pad_and_missing_eos_tokenizers():
    nopad = NoPadTokenizer()
    b = adapter.build_sft_batch(nopad, ["a", "abc"], ["b", "c"], max_length=16)
    assert b["input_ids"][0, -1].item() == nopad.eos_token_id  # padded with eos id
    assert b["labels"][0].tolist().count(nopad.eos_token_id) == 1
    noeos = NoEosTokenizer()
    b2 = adapter.build_sft_batch(noeos, ["ab"], ["cd"], max_length=16)
    assert b2["input_ids"].shape[1] == 4  # no EOS appended
    assert b2["labels"][0, 2:].ne(-100).all()


def test_mixed_lengths_attention_and_row_order():
    tok = ToyTokenizer()
    prompts = ["p1: ", "a much longer prompt: ", "x"]
    responses = ["r", "resp", "longest response here"]
    batch = adapter.build_sft_batch(tok, prompts, responses, max_length=128)
    expected = [len(tok.encode(p)) + len(tok.encode(r)) + 1 for p, r in zip(prompts, responses)]
    assert batch["attention_mask"].sum(1).tolist() == expected
    assert batch["input_ids"].shape[1] == max(expected)
    model = PositionalToyLM(tok.vocab_size)
    forward = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=128)
    order = [2, 0, 1]
    permuted = adapter.compute_response_logprobs(
        model, tok, [prompts[i] for i in order], [responses[i] for i in order], max_length=128
    )
    torch.testing.assert_close(permuted, forward[order])
    for i, (p, r) in enumerate(zip(prompts, responses)):
        assert math.isclose(forward[i].item(), _manual_logp(model, tok, p, r, 128), rel_tol=1e-5, abs_tol=1e-5)


def test_sft_loss_is_token_mean():
    tok = ToyTokenizer()
    model = PositionalToyLM(tok.vocab_size, seed=3)
    prompts, responses = ["Q: ", "Question two: "], ["short", "a bit longer one"]
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, max_length=64)
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=64)
    n_tokens = sum(len(tok.encode(r)) + 1 for r in responses)
    torch.testing.assert_close(loss, -logps.sum() / n_tokens)


def test_dpo_loss_matches_formula_and_beta():
    torch.manual_seed(0)
    pc, pr, rc, rr = (torch.randn(5) for _ in range(4))
    for beta in (0.1, 0.5):
        expected = -torch.nn.functional.logsigmoid(beta * ((pc - pr) - (rc - rr))).mean()
        torch.testing.assert_close(adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta), expected)
    # preferring chosen more than the reference does lowers the loss below log(2)
    loss = adapter.compute_dpo_loss_from_logps(torch.tensor([0.0]), torch.tensor([-2.0]), torch.tensor([0.0]), torch.tensor([0.0]), 0.1)
    assert loss.item() < math.log(2)


def test_compute_dpo_loss_matches_separate_forwards():
    tok = ToyTokenizer()
    policy, reference = PositionalToyLM(tok.vocab_size, seed=1), PositionalToyLM(tok.vocab_size, seed=2)
    prompts = ["Pick one: ", "Choose: "]
    chosen, rejected = ["good answer", "yes"], ["bad", "no thanks at all"]
    loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, beta=0.1, max_length=64)
    pc = adapter.compute_response_logprobs(policy, tok, prompts, chosen, 64)
    pr = adapter.compute_response_logprobs(policy, tok, prompts, rejected, 64)
    with torch.no_grad():
        rc = adapter.compute_response_logprobs(reference, tok, prompts, chosen, 64)
        rr = adapter.compute_response_logprobs(reference, tok, prompts, rejected, 64)
    torch.testing.assert_close(loss, adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, 0.1))


def test_dpo_step_increases_margin_and_never_touches_reference():
    tok = ToyTokenizer()
    policy, reference = PositionalToyLM(tok.vocab_size, seed=4), PositionalToyLM(tok.vocab_size, seed=4)
    ref_before = reference.table.detach().clone()
    prompts, chosen, rejected = ["Q: "] * 3, ["yes", "sure", "ok"], ["no", "nope", "never"]
    opt = torch.optim.SGD(policy.parameters(), lr=0.5)

    def margin():
        with torch.no_grad():
            pc = adapter.compute_response_logprobs(policy, tok, prompts, chosen, 32)
            pr = adapter.compute_response_logprobs(policy, tok, prompts, rejected, 32)
            rc = adapter.compute_response_logprobs(reference, tok, prompts, chosen, 32)
            rr = adapter.compute_response_logprobs(reference, tok, prompts, rejected, 32)
            return ((pc - rc) - (pr - rr)).mean().item()

    before = margin()
    for _ in range(3):
        opt.zero_grad()
        loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, beta=0.1, max_length=32)
        loss.backward()
        opt.step()
    assert margin() > before
    assert torch.equal(reference.table, ref_before) and reference.table.grad is None


def test_dpo_batch_rows_and_masks():
    tok = PadIsEosTokenizer()
    batch = adapter.build_dpo_batch(tok, ["A: ", "Bb: "], ["x", "yy"], ["zzz", "w"], max_length=32)
    for side, responses in (("chosen", ["x", "yy"]), ("rejected", ["zzz", "w"])):
        for row, (p, r) in enumerate(zip(["A: ", "Bb: "], responses)):
            plen, rlen = len(tok.encode(p)), len(tok.encode(r)) + 1
            labels = batch[side]["labels"][row]
            assert labels[:plen].eq(-100).all()
            assert labels[plen : plen + rlen].ne(-100).all()
            assert labels[plen + rlen :].eq(-100).all()
            assert labels[plen + rlen - 1].item() == tok.eos_token_id
