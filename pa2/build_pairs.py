"""Randomized, blinded A/B pair files with a hidden mapping.

    python -m pa2.build_pairs --first results/generations/dpo_hh_hh.json \
        --second results/generations/sft_hh.json --out results/pairs/hh_dpo_vs_sft.jsonl
"""

from __future__ import annotations

import argparse
import random

from pa2.utils import read_json, write_jsonl


def build_pairs(first, second, seed: int = 42):
    """``first``/``second`` are generation lists; returns pair rows (one per shared prompt id)."""
    by_id_b = {r["id"]: r for r in second}
    rng = random.Random(seed)
    rows = []
    for ra in sorted(first, key=lambda r: r["id"]):
        rb = by_id_b.get(ra["id"])
        if rb is None:
            continue
        flip = rng.random() < 0.5
        a, b = (ra, rb) if not flip else (rb, ra)
        rows.append(
            {
                "pair_id": ra["id"],
                "dataset": ra.get("dataset"),
                "category": ra.get("category"),
                "instruction": ra["instruction"],
                "response_a": a["output"],
                "response_b": b["output"],
                "hidden": {"A": a["generator"], "B": b["generator"]},
            }
        )
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True, help="generations of the model under test")
    ap.add_argument("--second", required=True, help="generations of the baseline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    rows = build_pairs(read_json(a.first), read_json(a.second), a.seed)
    write_jsonl(a.out, rows)
    n_a = sum(1 for r in rows if r["hidden"]["A"] == read_json(a.first)[0]["generator"])
    print(f"wrote {len(rows)} pairs to {a.out}; first-model is A in {n_a}")


if __name__ == "__main__":
    main()
