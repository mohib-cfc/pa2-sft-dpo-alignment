"""Batched deterministic generation -> JSON.

    python -m pa2.generate --model outputs/sft/best --generator qwen2.5-0.5b-sft \
        --eval_set eval_sets/alpaca_eval_300_seed42.json eval_sets/hh_rlhf_eval_300_seed42.json \
        --out results/generations/sft_alpaca.json results/generations/sft_hh.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pa2.prompting import QWEN_STOP_IDS, qwen_prompt, smol_prompt
from pa2.utils import read_json, write_json


def load_model(model_path: str, adapter: str | None, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    return model.to(device).eval()


@torch.no_grad()
def generate_set(model, tokenizer, rows, fmt, max_new_tokens, batch_size, stop_ids, rep_penalty, generator, device):
    prompts = [qwen_prompt(r["instruction"]) if fmt == "qwen" else smol_prompt(tokenizer, r["instruction"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]), reverse=True)
    outputs = [None] * len(rows)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=rep_penalty,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
        new = out[:, enc["input_ids"].shape[1] :]
        for j, i in enumerate(idx):
            outputs[i] = tokenizer.decode(new[j], skip_special_tokens=True).strip()
        print(f"[gen] {generator}: {min(s + batch_size, len(order))}/{len(order)}", flush=True)
    result = []
    for r, o in zip(rows, outputs):
        item = {"id": r["id"], "instruction": r["instruction"], "output": o, "generator": generator,
                "dataset": r.get("dataset")}
        if "category" in r:
            item["category"] = r["category"]
        result.append(item)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--eval_set", nargs="+", required=True)
    ap.add_argument("--out", nargs="+", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--prompt_format", choices=["qwen", "smol"], default=None)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    if len(args.eval_set) != len(args.out):
        raise SystemExit("--eval_set and --out must have the same length")

    fmt = args.prompt_format or ("smol" if "smollm" in (args.model + (args.generator or "")).lower() else "qwen")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if device.type == "cpu":
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if fmt == "qwen":
        stop_ids = QWEN_STOP_IDS
    else:
        stop_ids = [tokenizer.eos_token_id]
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and im_end != tokenizer.unk_token_id and im_end not in stop_ids:
            stop_ids.append(im_end)
    model = load_model(args.model, args.adapter, dtype, device)

    for eval_path, out_path in zip(args.eval_set, args.out):
        rows = read_json(eval_path)
        if args.limit:
            rows = rows[: args.limit]
        t0 = time.time()
        res = generate_set(model, tokenizer, rows, fmt, args.max_new_tokens, args.batch_size, stop_ids,
                           args.repetition_penalty, args.generator, device)
        write_json(out_path, res)
        print(f"[gen] wrote {out_path} ({len(res)} rows, {time.time()-t0:.0f}s)", flush=True)
        if args.limit:
            for r in res[:3]:
                print("---", r["instruction"][:100], "\n>>>", r["output"][:300])


if __name__ == "__main__":
    main()
