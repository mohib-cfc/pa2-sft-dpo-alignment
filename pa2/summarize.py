"""Win/tie/loss tables -> results/summary.{json,md}."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from pa2.judge import summarize_judgments
from pa2.utils import read_json, read_jsonl, write_json

TRACKS = {
    "hh_dpo_vs_sft": {"pairs": "hh_dpo_vs_sft", "judged": "hh_dpo_vs_sft", "title": "HH-RLHF (safety mode): DPO-HH vs SFT",
                      "focus": "qwen2.5-0.5b-dpo-hh"},
    "uf_dpo_vs_sft": {"pairs": "uf_dpo_vs_sft", "judged": "uf_dpo_vs_sft", "title": "AlpacaEval-300 (general mode): DPO-UF vs SFT",
                      "focus": "qwen2.5-0.5b-dpo-uf"},
    "persona_style": {"pairs": "persona_lora_vs_base", "judged": "persona_style", "title": "Persona style: LoRA-DPO vs SmolLM2-Instruct",
                      "focus": "smollm2-360m-lora-dpo"},
    "persona_quality": {"pairs": "persona_lora_vs_base", "judged": "persona_quality", "title": "Persona quality: LoRA-DPO vs SmolLM2-Instruct",
                        "focus": "smollm2-360m-lora-dpo"},
}


def words(s: str) -> int:
    return len(s.split())


def track_summary(root: Path, name: str, spec: dict) -> dict | None:
    pairs_path = root / "pairs" / f"{spec['pairs']}.jsonl"
    judged_path = root / "judgments" / f"{spec['judged']}.jsonl"
    if not pairs_path.exists() or not judged_path.exists():
        return None
    pairs = read_jsonl(pairs_path)
    judged = {r["pair_id"]: r for r in read_jsonl(judged_path)}
    meta_path = root / "judgments" / f"{spec['judged']}_summary.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    base = summarize_judgments(pairs, judged, meta.get("judge_model", "unknown"),
                               {"failed_batches": meta.get("failed_batches"), "retries": meta.get("retries"),
                                "mode": meta.get("mode")})
    focus = spec["focus"] if spec["focus"] in base["wins"] else sorted(base["wins"])[0]
    base["focus_model"] = focus
    # per-category breakdown
    cats = defaultdict(lambda: {"n": 0, "focus_wins": 0, "other_wins": 0, "ties": 0, "unknown": 0})
    lens = defaultdict(list)
    for p in pairs:
        c = p.get("category") or "all"
        j = judged.get(p["pair_id"])
        d = cats[c]
        d["n"] += 1
        if j is None or j["winner"] == "unknown":
            d["unknown"] += 1
        elif j["winner"] == "tie":
            d["ties"] += 1
        else:
            if p["hidden"][j["winner"]] == focus:
                d["focus_wins"] += 1
            else:
                d["other_wins"] += 1
        for side in ("A", "B"):
            lens[p["hidden"][side]].append(words(p["response_" + side.lower()]))
    for d in cats.values():
        dec = d["n"] - d["unknown"]
        d["win_rate_ties_half"] = (d["focus_wins"] + 0.5 * d["ties"]) / dec if dec else None
    base["by_category"] = dict(cats)
    base["mean_words"] = {m: sum(v) / len(v) for m, v in lens.items() if v}
    return base


def to_markdown(summary: dict) -> str:
    out = ["# Evaluation summary", ""]
    for name, t in summary["tracks"].items():
        spec = TRACKS[name]
        f = t["focus_model"]
        other = [m for m in t["wins"] if m != f][0]
        out += [f"## {spec['title']}", "",
                f"- judge: `{t['judge_model']}` (mode `{t.get('mode')}`), pairs n={t['n_pairs']}, judged={t['n_judged']}, "
                f"unknown={t['n_unknown']}, missing={t['n_missing']}, failed batches={t.get('failed_batches')}, retries={t.get('retries')}",
                f"- wins `{f}` = {t['wins'][f]}, wins `{other}` = {t['wins'][other]}, ties = {t['ties']}",
                f"- win rate of `{f}` (ties = 0.5): {t['win_rate_ties_half'][f]:.3f}; excluding ties: "
                f"{(t['win_rate_excl_ties'][f] if t['win_rate_excl_ties'][f] is not None else float('nan')):.3f}",
                f"- mean words: " + ", ".join(f"`{m}` {v:.0f}" for m, v in t["mean_words"].items()), ""]
        if len(t["by_category"]) > 1:
            out += ["| category | n | focus wins | other wins | ties | unknown | win rate (ties .5) |", "|---|---|---|---|---|---|---|"]
            for c, d in sorted(t["by_category"].items()):
                wr = "n/a" if d["win_rate_ties_half"] is None else f"{d['win_rate_ties_half']:.2f}"
                out.append(f"| {c} | {d['n']} | {d['focus_wins']} | {d['other_wins']} | {d['ties']} | {d['unknown']} | {wr} |")
            out.append("")
    return "\n".join(out)


def length_table(root: Path) -> dict:
    stats = {}
    for p in sorted(glob.glob(str(root / "generations" / "*.json"))):
        rows = read_json(p)
        if rows:
            stats[Path(p).stem] = {"n": len(rows), "generator": rows[0]["generator"],
                                   "mean_words": sum(words(r["output"]) for r in rows) / len(rows),
                                   "empty": sum(1 for r in rows if not r["output"].strip())}
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    root = Path(a.results)
    tracks = {}
    for name, spec in TRACKS.items():
        t = track_summary(root, name, spec)
        if t:
            tracks[name] = t
    summary = {"tracks": tracks, "generation_lengths": length_table(root)}
    write_json(root / "summary.json", summary)
    md = to_markdown(summary)
    if summary["generation_lengths"]:
        md += "\n## Generation length statistics\n\n| file | generator | n | mean words | empty |\n|---|---|---|---|---|\n"
        for k, v in summary["generation_lengths"].items():
            md += f"| {k} | {v['generator']} | {v['n']} | {v['mean_words']:.0f} | {v['empty']} |\n"
    (root / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
