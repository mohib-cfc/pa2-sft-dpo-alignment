"""Judge the three evaluation tracks sequentially (resumable)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa2.judge import main  # noqa: E402

TRACKS = [
    ("hh_dpo_vs_sft", "hh_dpo_vs_sft", "safety"),
    ("uf_dpo_vs_sft", "uf_dpo_vs_sft", "general"),
]

if __name__ == "__main__":
    only = sys.argv[1:] or [t[0] for t in TRACKS]
    for pairs, name, mode in TRACKS:
        if name not in only:
            continue
        main([
            "--pairs_path", f"results/pairs/{pairs}.jsonl",
            "--output_path", f"results/judgments/{name}.jsonl",
            "--summary_path", f"results/judgments/{name}_summary.json",
            "--failures_path", f"results/judgments/{name}_failures.jsonl",
            "--mode", mode,
            "--reuse_judgments",
        ])
