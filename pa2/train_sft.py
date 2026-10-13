"""Own SFT loop (no HF Trainer).

    python -m pa2.train_sft --config configs/sft.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from pa2 import core
from pa2.data import ROOT, build_sft, load_cached
from pa2.prompting import qwen_prompt
from pa2.utils import (
    append_jsonl,
    fmt_eta,
    gpu_peak_gb,
    load_yaml_config,
    save_model,
    seed_everything,
    write_json,
)

DEFAULTS = dict(
    model="Qwen/Qwen2.5-0.5B",
    out_dir="outputs/sft",
    cache_dir="outputs/data",
    n_train=30000,
    n_val=1000,
    max_length=512,
    per_device_batch=2,
    grad_accum=16,
    lr=2e-5,
    weight_decay=0.1,
    warmup_frac=0.03,
    clip=1.0,
    epochs=1,
    seed=42,
    precision="fp16",  # fp16 (autocast+GradScaler, fp32 weights) | bf16 | fp32
    optimizer="adamw",  # adamw | rmsprop | sgd (the last two only for laptop smoke runs)
    grad_checkpointing=False,
    val_every=150,
    val_batch=4,  # 8 OOMs on a T4 next to AdamW state; the token-weighted NLL is identical
    log_every=10,
    max_steps=None,
    max_train_examples=None,
    max_val_examples=None,
)


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower()) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, lr=cfg["lr"])
    if cfg["optimizer"] == "rmsprop":
        return torch.optim.RMSprop(groups, lr=cfg["lr"])
    return torch.optim.SGD(groups, lr=cfg["lr"])


@torch.no_grad()
def evaluate(model, tokenizer, val, cfg, autocast_ctx):
    model.eval()
    nll, ntok = 0.0, 0
    bs = cfg["val_batch"]
    for i in range(0, len(val), bs):
        chunk = val[i : i + bs]
        batch = core.build_lm_batch(
            tokenizer, [qwen_prompt(x["prompt"]) for x in chunk], [x["response"] for x in chunk], cfg["max_length"]
        )
        with autocast_ctx():
            loss, n = core.sft_loss_from_batch(model, batch)
        nll += float(loss) * int(n)
        ntok += int(n)
    model.train()
    return nll / max(ntok, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = dict(DEFAULTS)
    cfg.update(load_yaml_config(args.config, args.overrides))
    seed_everything(cfg["seed"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    write_json(out_dir / "config_used.json", cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train, val, counts = load_cached(
        build_sft, tokenizer, Path(cfg["cache_dir"]) / "sft.json", n_train=cfg["n_train"], n_val=cfg["n_val"]
    )
    if cfg["max_train_examples"]:
        train = train[: cfg["max_train_examples"]]
    if cfg["max_val_examples"]:
        val = val[: cfg["max_val_examples"]]
    print(f"[sft] train={len(train)} val={len(val)} counts={json.dumps(counts)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    if cfg["grad_checkpointing"]:
        model.gradient_checkpointing_enable()
    model.train()

    use_scaler = cfg["precision"] == "fp16" and device.type == "cuda"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(cfg["precision"])

    def autocast_ctx():
        if amp_dtype is None or device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=amp_dtype)

    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    optimizer = make_optimizer(model, cfg)

    bs, accum = cfg["per_device_batch"], cfg["grad_accum"]
    g = torch.Generator()
    g.manual_seed(cfg["seed"])

    def collate(items):
        return core.build_lm_batch(
            tokenizer, [qwen_prompt(x["prompt"]) for x in items], [x["response"] for x in items], cfg["max_length"]
        ), sum(1 for _ in items)

    loader = DataLoader(train, batch_size=bs, shuffle=True, generator=g, collate_fn=collate, num_workers=0)
    steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    if cfg["max_steps"]:
        total_steps = min(total_steps, int(cfg["max_steps"]))
    warmup = max(1, int(round(cfg["warmup_frac"] * total_steps)))

    def lr_at(step):  # step is 0-based optimizer step about to be taken
        if step < warmup:
            return cfg["lr"] * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    print(f"[sft] total_steps={total_steps} warmup={warmup} precision={cfg['precision']} opt={cfg['optimizer']}", flush=True)
    base_cfg = {**cfg, "total_steps": total_steps, "warmup_steps": warmup, "counts": counts}

    best_val = float("inf")
    best_step = -1
    t0 = time.time()
    step = 0
    win_loss, win_tokens, win_n = 0.0, 0, 0
    done = False

    def run_val(step):
        nonlocal best_val, best_step
        optimizer.zero_grad(set_to_none=True)  # free gradients before the eval forward
        if device.type == "cuda":
            torch.cuda.empty_cache()
        v = evaluate(model, tokenizer, val, cfg, autocast_ctx)
        append_jsonl(metrics_path, {"type": "val", "step": step, "val_loss": v})
        print(f"[sft] step {step}/{total_steps} val_loss={v:.4f}", flush=True)
        if v < best_val:
            best_val, best_step = v, step
            save_model(model, tokenizer, out_dir / "best", torch.float32, {**base_cfg, "step": step, "val_loss": v, "dtype": "fp32"})
        return v

    for epoch in range(cfg["epochs"]):
        micro = []
        for mi, (batch, _) in enumerate(loader, start=1):
            micro.append(batch)
            # window is full, or this is the last micro-batch of the epoch
            if len(micro) < accum and mi < len(loader):
                continue
            window = micro
            micro = []
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(step)
            optimizer.zero_grad(set_to_none=True)
            win_loss, win_tokens = 0.0, 0
            for b in window:
                with autocast_ctx():
                    loss, n = core.sft_loss_from_batch(model, b)
                scaler.scale(loss / len(window)).backward()
                win_loss += float(loss)
                win_tokens += int(n)
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % cfg["log_every"] == 0 or step == 1 or step == total_steps:
                el = time.time() - t0
                row = {
                    "type": "train",
                    "step": step,
                    "loss": win_loss / len(window),
                    "lr": lr_at(step - 1),
                    "grad_norm": float(gn),
                    "scale": float(scaler.get_scale()) if use_scaler else 1.0,
                    "resp_tokens": win_tokens,
                    "elapsed_s": el,
                }
                append_jsonl(metrics_path, row)
                eta = el / step * (total_steps - step)
                print(
                    f"[sft] step {step}/{total_steps} loss={row['loss']:.4f} lr={row['lr']:.2e} gn={row['grad_norm']:.2f} "
                    f"scale={row['scale']:.0f} eta={fmt_eta(eta)} peak={gpu_peak_gb():.1f}GB",
                    flush=True,
                )
            if step % cfg["val_every"] == 0 and step != total_steps:
                run_val(step)
            if step >= total_steps:
                done = True
                break
        if done:
            break

    run_val(step)
    save_model(model, tokenizer, out_dir / "final", torch.float16, {**base_cfg, "step": step, "dtype": "fp16", "best_step": best_step})
    write_json(out_dir / "summary.json", {"steps": step, "best_val_loss": best_val, "best_step": best_step,
                                          "elapsed_s": time.time() - t0, "peak_gpu_gb": gpu_peak_gb()})
    print(f"[sft] done. best_val={best_val:.4f} @ step {best_step}", flush=True)


if __name__ == "__main__":
    main()
