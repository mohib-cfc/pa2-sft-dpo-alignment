# PA2: Instruction Tuning and DPO on a Small Language Model

- **Training:**
  - SFT of `Qwen/Qwen2.5-0.5B` on 30k UltraChat examples.
  - Two from-scratch DPO runs starting from the SFT model: HH-RLHF and UltraFeedback, 15k pairs each.
  - A LoRA-DPO persona run with TRL on `HuggingFaceTB/SmolLM2-360M-Instruct`.
- **Compute:** all training and generation ran on Kaggle (2× Tesla T4).
- **Evaluation:** pairwise LLM-as-judge through the Kaggle Model Proxy (`google/gemini-3-flash-preview`).

**Report:** submitted separately as a PDF. Its figures are in `report/figures/` (made by `scripts/report_figures.py`).

## Results

| Comparison | Eval set | n | wins | ties | losses | win rate (tie = ½) |
|---|---|---|---|---|---|---|
| DPO-HH vs SFT | HH-style set (provided) | 300 | 120 | 76 | 104 | 0.527 |
| DPO-UF vs SFT | AlpacaEval, 300 prompts, seed 42 | 300 | 124 | 72 | 104 | 0.533 |
| LoRA-DPO vs SmolLM2-Instruct (style) | persona set | 24 | 8 | 14 | 2 | 0.625 |
| LoRA-DPO vs SmolLM2-Instruct (quality) | persona set | 24 | 3 | 1 | 20 | 0.146 |

More detail:
- `results/summary.md`: per-category tables and lengths.
- `results/eval_diagnostics.json`: judge position bias, self-consistency, loop rates.

## Layout

| Path | What it is |
|---|---|
| `pa2/core.py` | batching, response log-probs, SFT and DPO losses |
| `submission_adapter.py` | the six grading functions (thin wrappers around `pa2/core.py`) |
| `pa2/train_sft.py`, `pa2/train_dpo.py` | my SFT and DPO training loops (plain PyTorch) |
| `pa2/train_lora_dpo.py` | LoRA persona run with TRL `DPOTrainer` + PEFT |
| `pa2/data.py`, `pa2/prompting.py` | dataset loading/filtering, eval sets, Qwen prompt template |
| `pa2/generate.py`, `pa2/build_pairs.py`, `pa2/judge.py`, `pa2/summarize.py` | generation, blinded A/B pairs, judge, summaries |
| `pa2/stage.py`, `kaggle/notebooks/` | what each Kaggle notebook runs (`setup`, `sft`, `dpo`) |
| `configs/` | hyperparameters (each run also saves `config_used.json`) |
| `data/persona_lora/` | the provided persona preference data |
| `eval_sets/` | AlpacaEval 300 subset (seed 42), provided HH-style set, persona prompts |
| `results/` | metrics, generations, pairs, judgments, summaries, Kaggle logs |
| `scripts/` | judge CLI, diagnostics, report figures |
| `report/figures/` | the plots used in the report |
| `tools/`, `config/local.json` | helpers: build the Kaggle notebooks, wait for a Kaggle run, tiny model for local smoke tests |
| `tests/` | public adapter tests + my own tests |

## Reproducing

```bash
pip install -r requirements.txt && pip install -e . --no-deps   # use Kaggle's PyTorch (don't reinstall it)

python -m pa2.data                                             # datasets + eval sets (seed 42)
python -m pa2.train_sft --config configs/sft.yaml              # -> outputs/sft/best
python -m pa2.train_dpo --config configs/dpo_hh.yaml sft_model=outputs/sft/best
python -m pa2.train_dpo --config configs/dpo_uf.yaml sft_model=outputs/sft/best
python -m pa2.train_lora_dpo --config configs/lora_persona.yaml

python -m pa2.generate --model outputs/sft/best --generator qwen2.5-0.5b-sft \
    --eval_set eval_sets/alpaca_eval_300_seed42.json eval_sets/hh_rlhf_eval_300_seed42.json \
    --out results/generations/sft_alpaca.json results/generations/sft_hh.json
python -m pa2.generate --model outputs/dpo_hh/best --generator qwen2.5-0.5b-dpo-hh \
    --eval_set eval_sets/hh_rlhf_eval_300_seed42.json --out results/generations/dpo_hh_hh.json
python -m pa2.build_pairs --first results/generations/dpo_hh_hh.json \
    --second results/generations/sft_hh.json --out results/pairs/hh_dpo_vs_sft.jsonl

pip install kaggle && kaggle benchmarks auth -y --env-file .env.kaggle   # Model Proxy credentials (not committed)
python scripts/judge_pairwise_kaggle.py --pairs_path results/pairs/hh_dpo_vs_sft.jsonl \
    --output_path results/judgments/hh_dpo_vs_sft.jsonl --failures_path results/judgments/hh_dpo_vs_sft_failures.jsonl \
    --judge_model google/gemini-3-flash-preview --mode safety --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments

python -m pa2.summarize                 # results/summary.{json,md}
python scripts/eval_diagnostics.py      # results/eval_diagnostics.json
python scripts/report_figures.py        # report/figures/*.png
```

### On Kaggle

The notebooks in `kaggle/notebooks/` run `python -m pa2.stage --stage setup|sft|dpo` with the GPU T4 ×2 accelerator. Their logs are in `results/kaggle_logs/`. The `dpo` notebook uses the output of `pa2-sft` (`outputs/sft/best`) as its input.

`pa2-sft` ran twice:
- Version 1 ran every task. Its SFT step ran out of memory at the first validation; the base and LoRA outputs from this version were kept.
- Version 2 reran only `--only sft,gen_sft`, after the validation batch was reduced and gradients were freed before evaluation.

## Kaggle setup check (Task 1)

Every notebook clones this repository and runs `pip uninstall -y torchao`. It then installs `requirements.txt` and runs `pip install -e . --no-deps`, so Kaggle's PyTorch is not reinstalled. Finally it prints the GPUs and package versions before starting its stage.

From `results/kaggle_logs/pa2-setup_v1.log` (the other logs show the same):

```
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   51C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
|   1  Tesla T4                       Off |   00000000:00:05.0 Off |                    0 |
| N/A   49C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
python 3.13.15
torch 2.11.0+cu128 | cuda 12.8
transformers 4.57.6 | datasets 4.8.5 | trl 0.29.1 | peft 0.18.1
GPUs: 2 ['Tesla T4', 'Tesla T4']
```

## Tests

```bash
PYTHONPATH=. python -m pytest -q tests
```

This runs the 7 public adapter tests plus 18 of my own. Mine cover truncation boundaries, pad = EOS tokenizers, missing pad/EOS tokens, mixed lengths, row order, DPO numerics and the data/pair/judge helpers.

Model weights are not committed.
