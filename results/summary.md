# Evaluation summary

## HH-RLHF (safety mode): DPO-HH vs SFT

- judge: `google/gemini-3-flash-preview` (mode `safety`), pairs n=300, judged=300, unknown=0, missing=0, failed batches=0, retries=0
- wins `qwen2.5-0.5b-dpo-hh` = 120, wins `qwen2.5-0.5b-sft` = 104, ties = 76
- win rate of `qwen2.5-0.5b-dpo-hh` (ties = 0.5): 0.527; excluding ties: 0.536
- mean words: `qwen2.5-0.5b-dpo-hh` 111, `qwen2.5-0.5b-sft` 133

| category | n | focus wins | other wins | ties | unknown | win rate (ties .5) |
|---|---|---|---|---|---|---|
| benign_helpfulness | 30 | 12 | 18 | 0 | 0 | 0.40 |
| calibration_and_uncertainty | 30 | 15 | 10 | 5 | 0 | 0.58 |
| cyber_abuse | 30 | 11 | 9 | 10 | 0 | 0.53 |
| harmful_instructions | 30 | 13 | 3 | 14 | 0 | 0.67 |
| hate_or_harassment | 30 | 15 | 11 | 4 | 0 | 0.57 |
| legal_financial_safety | 30 | 12 | 10 | 8 | 0 | 0.53 |
| medical_safety | 30 | 13 | 9 | 8 | 0 | 0.57 |
| privacy_and_doxxing | 30 | 12 | 8 | 10 | 0 | 0.57 |
| safe_transformation | 30 | 10 | 12 | 8 | 0 | 0.47 |
| self_harm_support | 30 | 7 | 14 | 9 | 0 | 0.38 |

## AlpacaEval-300 (general mode): DPO-UF vs SFT

- judge: `google/gemini-3-flash-preview` (mode `general`), pairs n=300, judged=300, unknown=0, missing=0, failed batches=0, retries=0
- wins `qwen2.5-0.5b-dpo-uf` = 124, wins `qwen2.5-0.5b-sft` = 104, ties = 72
- win rate of `qwen2.5-0.5b-dpo-uf` (ties = 0.5): 0.533; excluding ties: 0.544
- mean words: `qwen2.5-0.5b-dpo-uf` 197, `qwen2.5-0.5b-sft` 136

## Persona style: LoRA-DPO vs SmolLM2-Instruct

- judge: `google/gemini-3-flash-preview` (mode `persona_style`), pairs n=24, judged=24, unknown=0, missing=0, failed batches=0, retries=0
- wins `smollm2-360m-lora-dpo` = 8, wins `smollm2-360m-instruct` = 2, ties = 14
- win rate of `smollm2-360m-lora-dpo` (ties = 0.5): 0.625; excluding ties: 0.800
- mean words: `smollm2-360m-lora-dpo` 133, `smollm2-360m-instruct` 146

| category | n | focus wins | other wins | ties | unknown | win rate (ties .5) |
|---|---|---|---|---|---|---|
| coding | 6 | 1 | 0 | 5 | 0 | 0.58 |
| cooking | 8 | 5 | 0 | 3 | 0 | 0.81 |
| general | 6 | 0 | 0 | 6 | 0 | 0.50 |
| safety | 4 | 2 | 2 | 0 | 0 | 0.50 |

## Persona quality: LoRA-DPO vs SmolLM2-Instruct

- judge: `google/gemini-3-flash-preview` (mode `persona_quality`), pairs n=24, judged=24, unknown=0, missing=0, failed batches=0, retries=0
- wins `smollm2-360m-lora-dpo` = 3, wins `smollm2-360m-instruct` = 20, ties = 1
- win rate of `smollm2-360m-lora-dpo` (ties = 0.5): 0.146; excluding ties: 0.130
- mean words: `smollm2-360m-lora-dpo` 133, `smollm2-360m-instruct` 146

| category | n | focus wins | other wins | ties | unknown | win rate (ties .5) |
|---|---|---|---|---|---|---|
| coding | 6 | 1 | 4 | 1 | 0 | 0.25 |
| cooking | 8 | 0 | 8 | 0 | 0 | 0.00 |
| general | 6 | 0 | 6 | 0 | 0 | 0.00 |
| safety | 4 | 2 | 2 | 0 | 0 | 0.50 |

## Generation length statistics

| file | generator | n | mean words | empty |
|---|---|---|---|---|
| base_alpaca | qwen2.5-0.5b-base | 300 | 264 | 0 |
| base_hh | qwen2.5-0.5b-base | 300 | 255 | 0 |
| dpo_hh_hh | qwen2.5-0.5b-dpo-hh | 300 | 111 | 0 |
| dpo_uf_alpaca | qwen2.5-0.5b-dpo-uf | 300 | 197 | 0 |
| persona_base | smollm2-360m-instruct | 24 | 146 | 0 |
| persona_lora | smollm2-360m-lora-dpo | 24 | 133 | 0 |
| sft_alpaca | qwen2.5-0.5b-sft | 300 | 136 | 0 |
| sft_hh | qwen2.5-0.5b-sft | 300 | 133 | 0 |
