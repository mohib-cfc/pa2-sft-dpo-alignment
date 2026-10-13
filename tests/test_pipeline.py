"""Fast tests for the pipeline pieces around pa2.core (no network, no GPU)."""

import json
import random
from collections import Counter

import torch

from pa2 import core
from pa2.build_pairs import build_pairs
from pa2.data import parse_hh_pair, split_turns
from pa2.judge import parse_judgments, run_judging, summarize_judgments
from pa2.prompting import qwen_prompt


class CharTok:
    eos_token_id = 1
    pad_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 90 + 5 for c in text]


def test_qwen_prompt_format():
    p = qwen_prompt("Hi there")
    assert p == (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n"
    )


def test_collate_matches_core_and_eos_before_truncation():
    tok = CharTok()
    b = core.build_lm_batch(tok, ["ab", "c"], ["xyz", "q"], 16)
    assert b["input_ids"].shape == (2, 6)
    # row 0: prompt 2 tokens then response 3 tokens then eos
    assert b["input_ids"][0, -1].item() == tok.eos_token_id
    assert (b["labels"][0, :2] == core.IGNORE_INDEX).all()
    # truncation can cut the eos
    t = core.build_lm_batch(tok, ["ab"], ["xyz"], 4)
    assert t["input_ids"].shape == (1, 4)
    assert t["input_ids"][0, -1].item() != tok.eos_token_id


HH_OK = "\n\nHuman: What is 2+2?\n\nAssistant: It is 4."
HH_BAD = "\n\nHuman: What is 2+2?\n\nAssistant: Four."


def test_hh_parser_single_turn():
    r = parse_hh_pair(HH_OK, HH_BAD)
    assert r == {"prompt": "What is 2+2?", "chosen": "It is 4.", "rejected": "Four."}


def test_hh_parser_rejects_multiturn_and_mismatch_and_identical():
    multi = "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: more\n\nAssistant: ok"
    assert parse_hh_pair(multi, multi.replace("ok", "no")) is None
    other = "\n\nHuman: What is 3+3?\n\nAssistant: 6."
    assert parse_hh_pair(HH_OK, other) is None
    assert parse_hh_pair(HH_OK, HH_OK) is None
    assert len(split_turns(multi)) == 4


def _gens(name, n=200):
    return [{"id": f"p{i:03d}", "instruction": f"q{i}", "output": f"{name}-{i}", "generator": name,
             "dataset": "d"} for i in range(n)]


def test_pairs_keep_hidden_mapping_and_balance():
    first, second = _gens("dpo"), _gens("sft")
    rows = build_pairs(first, second, seed=42)
    assert len(rows) == 200
    for r in rows:
        a, b = r["hidden"]["A"], r["hidden"]["B"]
        assert {a, b} == {"dpo", "sft"}
        assert r["response_a"].startswith(a) and r["response_b"].startswith(b)
    n_a = Counter(r["hidden"]["A"] for r in rows)["dpo"]
    assert 70 <= n_a <= 130
    assert build_pairs(first, second, seed=42) == rows  # deterministic


def test_judge_parser_variants():
    ids = ["a", "b"]
    ok = '[{"pair_id":"a","winner":"A","reason":"x"},{"pair_id":"b","winner":"tie","reason":"y"}]'
    assert parse_judgments(ok, ids)["b"]["winner"] == "tie"
    fenced = "```json\n" + ok + "\n```"
    assert parse_judgments(fenced, ids)["a"]["winner"] == "A"
    prose = "Here you go: " + ok + " hope it helps"
    assert len(parse_judgments(prose, ids)) == 2
    partial = '[{"pair_id":"a","winner":"B","reason":""},{"pair_id":"zzz","winner":"A"}]'
    assert list(parse_judgments(partial, ids)) == ["a"]
    invalid = '[{"pair_id":"a","winner":"maybe"}]'
    assert parse_judgments(invalid, ids)["a"]["winner"] == "unknown"
    assert parse_judgments("totally not json", ids) is None
    assert parse_judgments("", ids) is None


def test_run_judging_resume_and_summary(tmp_path):
    first, second = _gens("dpo", 12), _gens("sft", 12)
    pairs = build_pairs(first, second)
    pp = tmp_path / "pairs.jsonl"
    pp.write_text("\n".join(json.dumps(r) for r in pairs), encoding="utf-8")
    calls = {"n": 0}

    def fake(batch):
        calls["n"] += 1
        if calls["n"] == 2:  # second call is garbage -> retried
            return "garbage"
        return json.dumps([{"pair_id": p["pair_id"], "winner": "A", "reason": "r"} for p in batch])

    out = tmp_path / "j.jsonl"
    s = run_judging(pp, out, "fake-judge", batch_size=5, sleep_seconds=0, max_retries=3, call_fn=fake,
                    summary_path=tmp_path / "s.json")
    assert s["n_judged"] == 12 and s["failed_batches"] == 0 and s["retries"] == 1
    assert sum(s["wins"].values()) == 12
    # resume: nothing left to do, no new calls
    n_before = calls["n"]
    s2 = run_judging(pp, out, "fake-judge", batch_size=5, sleep_seconds=0, reuse_judgments=True, call_fn=fake)
    assert calls["n"] == n_before and s2["n_judged"] == 12


def test_reward_accuracy_is_zero_when_policy_equals_reference():
    """Identical policy/reference log-probs give r_w == r_l == 0, so the strict accuracy (r_w > r_l) is exactly 0."""
    logps_chosen = torch.tensor([-3.0, -7.5, -1.25])
    logps_rejected = torch.tensor([-4.0, -2.0, -9.0])
    rc, rr = core.dpo_rewards(logps_chosen, logps_rejected, logps_chosen.clone(), logps_rejected.clone(), 0.1)
    assert torch.equal(rc, torch.zeros(3)) and torch.equal(rr, torch.zeros(3))
    assert (rc > rr).float().mean().item() == 0.0
    # once the policy prefers the chosen answer more than the reference does, the pair counts as correct
    rc2, rr2 = core.dpo_rewards(logps_chosen + 0.5, logps_rejected, logps_chosen, logps_rejected, 0.1)
    assert (rc2 > rr2).float().mean().item() == 1.0
