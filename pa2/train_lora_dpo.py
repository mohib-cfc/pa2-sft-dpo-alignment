"""LoRA persona DPO with TRL DPOTrainer + PEFT (the only place TRL is used).

    python -m pa2.train_lora_dpo --config configs/lora_persona.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from pa2.data import ROOT
from pa2.utils import load_yaml_config, seed_everything, write_json

DEFAULTS = dict(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    data_path="data/persona_lora/ramsay_style_preferences.json",
    out_dir="outputs/lora_persona",
    seed=42,
    n_eval=20,
    beta=0.1,
    lr=5e-5,
    epochs=3,
    per_device_batch=4,
    grad_accum=2,
    max_length=512,
    max_prompt_length=256,
    warmup_ratio=0.1,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    precision="fp16",  # fp16 | bf16
    max_steps=-1,
    logging_steps=5,
)


def to_records(rows):
    return [
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
        }
        for r in rows
    ]


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

    rows = json.loads(Path(cfg["data_path"]).read_text(encoding="utf-8"))
    random.Random(cfg["seed"]).shuffle(rows)
    n_eval = cfg["n_eval"]
    train_rows, eval_rows = rows[n_eval:], rows[:n_eval]
    print(f"[lora] train={len(train_rows)} eval={len(eval_rows)}", flush=True)
    train_ds, eval_ds = Dataset.from_list(to_records(train_rows)), Dataset.from_list(to_records(eval_rows))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    on_cuda = torch.cuda.is_available()
    # fp32 base weights; mixed precision via the trainer flag (fp16 on T4, bf16 locally)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32)

    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    kw = dict(
        output_dir=str(out_dir / "trainer"),
        beta=cfg["beta"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=cfg["warmup_ratio"],
        num_train_epochs=cfg["epochs"],
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_batch"],
        per_device_eval_batch_size=cfg["per_device_batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        max_length=cfg["max_length"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=cfg["seed"],
        fp16=on_cuda and cfg["precision"] == "fp16",
        bf16=on_cuda and cfg["precision"] == "bf16",
        use_cpu=not on_cuda,
    )
    params = inspect.signature(DPOConfig.__init__).parameters
    if "max_prompt_length" in params:
        kw["max_prompt_length"] = cfg["max_prompt_length"]
    dpo_args = DPOConfig(**{k: v for k, v in kw.items() if k in params})

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"[lora] trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)
    trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))
    write_json(out_dir / "log_history.json", trainer.state.log_history)
    write_json(out_dir / "config_used.json", {**cfg, "trainable_params": trainable, "total_params": total,
                                              "trainable_pct": 100 * trainable / total})
    print("[lora] done", flush=True)


if __name__ == "__main__":
    main()
