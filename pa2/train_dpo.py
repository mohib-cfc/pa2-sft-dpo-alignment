"""Own DPO loop (no TRL).

    python -m pa2.train_dpo --config configs/dpo_hh.yaml [key=value ...]
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
from pa2.data import build_hh, build_uf, load_cached
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
    sft_model="outputs/sft/best",
    dataset="hh",  # hh | uf
    out_dir="outputs/dpo_hh",
    cache_dir="outputs/data",
    n_train=15000,
    n_val=1000,
    max_length=512,
    per_device_batch=1,
    grad_accum=32,
    lr=1e-6,
    beta=0.1,
    clip=1.0,
    epochs=1,
    seed=42,
    precision="fp16",  # autocast dtype for the policy forward: fp16 | bf16
    ref_dtype="fp16",  # fp16 | bf16
    optimizer="rmsprop",  # rmsprop | adamw | sgd
    grad_checkpointing=False,
    ref_device=None,  # default: same device as the policy
    val_every=50,
    val_batch=4,
    log_every=1,
    max_steps=None,
    max_train_examples=None,
    max_val_examples=None,
)


def make_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"] == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg["lr"])
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"])
    return torch.optim.SGD(params, lr=cfg["lr"])


def split_batch(items, tokenizer, max_length):
    return core.build_dpo_batch(
        tokenizer,
        [qwen_prompt(x["prompt"]) for x in items],
        [x["chosen"] for x in items],
        [x["rejected"] for x in items],
        max_length,
    )


@torch.no_grad()
def reference_logps(reference, tokenizer, data, cfg, pad_id):
    chosen, rejected = [], []
    reference.eval()
    bs = cfg["val_batch"]
    for i in range(0, len(data), bs):
        batch = split_batch(data[i : i + bs], tokenizer, cfg["max_length"])
        c, r = core.concatenated_logprobs(reference, batch, pad_id)
        chosen.append(c.float().cpu())
        rejected.append(r.float().cpu())
    return torch.cat(chosen), torch.cat(rejected)


@torch.no_grad()
def evaluate(policy, tokenizer, val, ref_c, ref_r, cfg, pad_id, autocast_ctx):
    policy.eval()
    bs = cfg["val_batch"]
    pc, pr = [], []
    for i in range(0, len(val), bs):
        batch = split_batch(val[i : i + bs], tokenizer, cfg["max_length"])
        with autocast_ctx():
            c, r = core.concatenated_logprobs(policy, batch, pad_id)
        pc.append(c.float().cpu())
        pr.append(r.float().cpu())
    pc, pr = torch.cat(pc), torch.cat(pr)
    loss = core.dpo_loss_from_logps(pc, pr, ref_c, ref_r, cfg["beta"])
    rc, rr = core.dpo_rewards(pc, pr, ref_c, ref_r, cfg["beta"])
    policy.train()
    return {
        "val_loss": float(loss),
        "val_reward_acc": float((rc > rr).float().mean()),
        "val_margin": float((rc - rr).mean()),
        "val_chosen_reward": float(rc.mean()),
        "val_rejected_reward": float(rr.mean()),
    }


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
    ref_device = torch.device(cfg["ref_device"]) if cfg["ref_device"] else device
    tokenizer = AutoTokenizer.from_pretrained(cfg["sft_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = core.get_pad_id(tokenizer)

    builder = build_hh if cfg["dataset"] == "hh" else build_uf
    # the dataset split uses the base Qwen tokenizer's length filter; SFT tokenizer is identical
    train, val, counts = load_cached(
        builder, tokenizer, Path(cfg["cache_dir"]) / f"{cfg['dataset']}.json", n_train=cfg["n_train"], n_val=cfg["n_val"]
    )
    if cfg["max_train_examples"]:
        train = train[: cfg["max_train_examples"]]
    if cfg["max_val_examples"]:
        val = val[: cfg["max_val_examples"]]
    print(f"[dpo] dataset={cfg['dataset']} train={len(train)} val={len(val)} counts={json.dumps(counts)}", flush=True)

    ref_dt = torch.float16 if cfg["ref_dtype"] == "fp16" else torch.bfloat16
    policy = AutoModelForCausalLM.from_pretrained(cfg["sft_model"], torch_dtype=torch.float32).to(device)
    policy.config.use_cache = False
    if cfg["grad_checkpointing"]:
        policy.gradient_checkpointing_enable()
    policy.train()
    reference = AutoModelForCausalLM.from_pretrained(cfg["sft_model"], torch_dtype=ref_dt).to(ref_device)
    reference.config.use_cache = False
    reference.eval()
    reference.requires_grad_(False)

    amp_dtype = torch.float16 if cfg["precision"] == "fp16" else torch.bfloat16
    use_scaler = cfg["precision"] == "fp16" and device.type == "cuda"

    def autocast_ctx():
        if device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=amp_dtype)

    scaler = torch.amp.GradScaler("cuda", init_scale=2.0**10, enabled=use_scaler)
    optimizer = make_optimizer(policy, cfg)

    t0 = time.time()
    print("[dpo] precomputing reference val log-probs ...", flush=True)
    with autocast_ctx():
        ref_vc, ref_vr = reference_logps(reference, tokenizer, val, cfg, pad_id)
    print(f"[dpo] reference val done in {time.time()-t0:.0f}s", flush=True)

    bs, accum = cfg["per_device_batch"], cfg["grad_accum"]
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    loader = DataLoader(train, batch_size=bs, shuffle=True, generator=g, collate_fn=lambda items: items, num_workers=0)
    total_steps = math.ceil(len(loader) / accum) * cfg["epochs"]
    if cfg["max_steps"]:
        total_steps = min(total_steps, int(cfg["max_steps"]))
    base_cfg = {**cfg, "total_steps": total_steps, "counts": counts}
    print(f"[dpo] total_steps={total_steps} opt={cfg['optimizer']} beta={cfg['beta']} lr={cfg['lr']}", flush=True)

    best = (-1.0, -1e9)  # (reward acc, margin)
    best_step = -1

    def run_val(step):
        nonlocal best, best_step
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        m = evaluate(policy, tokenizer, val, ref_vc, ref_vr, cfg, pad_id, autocast_ctx)
        append_jsonl(metrics_path, {"type": "val", "step": step, **m})
        print(
            f"[dpo] step {step}/{total_steps} val_loss={m['val_loss']:.4f} acc={m['val_reward_acc']:.3f} "
            f"margin={m['val_margin']:.4f}",
            flush=True,
        )
        key = (m["val_reward_acc"], m["val_margin"])
        if step > 0 and key > best:
            best, best_step = key, step
            save_model(policy, tokenizer, out_dir / "best", torch.float16, {**base_cfg, "step": step, **m, "dtype": "fp16"})
        return m

    run_val(0)
    step = 0
    done = False
    t_train = time.time()
    for epoch in range(cfg["epochs"]):
        micro = []
        for mi, items in enumerate(loader, start=1):
            micro.append(items)
            if len(micro) < accum and mi < len(loader):
                continue
            window = micro
            micro = []
            optimizer.zero_grad(set_to_none=True)
            agg = dict(loss=0.0, acc=0.0, margin=0.0, rc=0.0, rr=0.0)
            n_pairs = 0
            for it in window:
                batch = split_batch(it, tokenizer, cfg["max_length"])
                with autocast_ctx():
                    loss, pc, pr, rc_, rr_ = core.dpo_forward(policy, reference, batch, pad_id, cfg["beta"])
                scaler.scale(loss / len(window)).backward()
                with torch.no_grad():
                    wc, wr = core.dpo_rewards(pc, pr, rc_, rr_, cfg["beta"])
                    k = len(it)
                    agg["loss"] += float(loss) * k
                    agg["acc"] += float((wc > wr).float().sum())
                    agg["margin"] += float((wc - wr).sum())
                    agg["rc"] += float(wc.sum())
                    agg["rr"] += float(wr.sum())
                    n_pairs += k
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["clip"])
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % cfg["log_every"] == 0 or step == total_steps:
                el = time.time() - t_train
                row = {
                    "type": "train",
                    "step": step,
                    "loss": agg["loss"] / n_pairs,
                    "reward_acc": agg["acc"] / n_pairs,
                    "margin": agg["margin"] / n_pairs,
                    "chosen_reward": agg["rc"] / n_pairs,
                    "rejected_reward": agg["rr"] / n_pairs,
                    "grad_norm": float(gn),
                    "scale": float(scaler.get_scale()) if use_scaler else 1.0,
                    "elapsed_s": el,
                }
                append_jsonl(metrics_path, row)
                if step % 5 == 0 or step == total_steps or step == 1:
                    eta = el / step * (total_steps - step)
                    print(
                        f"[dpo] step {step}/{total_steps} loss={row['loss']:.4f} acc={row['reward_acc']:.2f} "
                        f"margin={row['margin']:.3f} gn={row['grad_norm']:.2f} eta={fmt_eta(eta)} peak={gpu_peak_gb():.1f}GB",
                        flush=True,
                    )
            if step % cfg["val_every"] == 0 and step != total_steps:
                run_val(step)
            if step >= total_steps:
                done = True
                break
        if done:
            break

    final = run_val(step)
    if best_step < 0:  # nothing beat step 0 (e.g. very short smoke run)
        save_model(policy, tokenizer, out_dir / "best", torch.float16, {**base_cfg, "step": step, **final, "dtype": "fp16"})
        best_step = step
    save_model(policy, tokenizer, out_dir / "final", torch.float16, {**base_cfg, "step": step, **final, "dtype": "fp16"})
    write_json(
        out_dir / "summary.json",
        {"steps": step, "best_step": best_step, "best_val_reward_acc": best[0], "best_val_margin": best[1],
         "final": final, "elapsed_s": time.time() - t0, "peak_gpu_gb": gpu_peak_gb()},
    )
    print(f"[dpo] done. best step {best_step} acc={best[0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
