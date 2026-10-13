"""Memory / speed probe with real Qwen: SFT micro-steps on cuda:0, DPO micro-steps on cuda:1."""

from __future__ import annotations

import argparse
import contextlib
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pa2 import core
from pa2.prompting import qwen_prompt
from pa2.utils import write_json

TEXT = ("The quick brown fox jumps over the lazy dog while discussing machine learning and alignment. " * 60)


def probe_sft(model_id, tok, dev, steps=3):
    torch.cuda.reset_peak_memory_stats(dev)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(dev)
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")
    times = []
    for _ in range(steps):
        batch = core.build_lm_batch(tok, [qwen_prompt("Explain things.")] * 2, [TEXT] * 2, 512)
        torch.cuda.synchronize(dev)
        t = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _ = core.sft_loss_from_batch(model, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(dev)
        times.append(time.time() - t)
    res = {"peak_gb": torch.cuda.max_memory_allocated(dev) / 1e9, "micro_step_s": times, "seq_len": int(batch["input_ids"].shape[1])}
    del model, opt
    torch.cuda.empty_cache()
    return res


def probe_dpo(model_id, tok, dev, steps=3, optimizer="rmsprop"):
    torch.cuda.reset_peak_memory_stats(dev)
    policy = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(dev)
    policy.config.use_cache = False
    policy.train()
    ref = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(dev).eval()
    ref.requires_grad_(False)
    opt = (torch.optim.RMSprop if optimizer == "rmsprop" else torch.optim.AdamW)(policy.parameters(), lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    pad = core.get_pad_id(tok)
    times = []
    for _ in range(steps):
        batch = core.build_dpo_batch(tok, [qwen_prompt("Explain things.")], [TEXT], [TEXT + " more"], 512)
        torch.cuda.synchronize(dev)
        t = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            loss, *_ = core.dpo_forward(policy, ref, batch, pad, 0.1)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(dev)
        times.append(time.time() - t)
    res = {"peak_gb": torch.cuda.max_memory_allocated(dev) / 1e9, "micro_step_s": times, "optimizer": optimizer}
    del policy, ref, opt
    torch.cuda.empty_cache()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", default="outputs/probe.json")
    a = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(a.model)
    n = torch.cuda.device_count()
    out = {"n_gpus": n, "gpus": [torch.cuda.get_device_name(i) for i in range(n)]}
    out["sft_adamw_fp16_autocast_b2x512"] = probe_sft(a.model, tok, torch.device("cuda:0"))
    d1 = torch.device("cuda:1" if n > 1 else "cuda:0")
    out["dpo_rmsprop_fp16ref_same_gpu"] = probe_dpo(a.model, tok, d1)
    write_json(a.out, out)
    print(out)


if __name__ == "__main__":
    main()
