"""Small shared helpers: seeding, jsonl io, timers, checkpoint saving."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Iterable, List

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path) -> List[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0


def fmt_eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def save_model(model, tokenizer, out_dir, dtype: torch.dtype, run_config: dict) -> None:
    """Save weights in ``dtype`` (safetensors) without disturbing the live model."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to("cpu", dtype=dtype).contiguous() for k, v in model.state_dict().items()}
    model.save_pretrained(out, state_dict=state, safe_serialization=True)
    tokenizer.save_pretrained(out)
    write_json(out / "run_config.json", run_config)


def load_yaml_config(path, overrides=None) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    for item in overrides or []:
        key, _, value = item.partition("=")
        cfg[key.lstrip("-")] = yaml.safe_load(value)
    return cfg


def gpu_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9
