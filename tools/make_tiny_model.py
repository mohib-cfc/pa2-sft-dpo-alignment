"""Build a tiny random Qwen2 model + synthetic data caches for offline laptop smoke tests."""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/smoke/tiny_qwen")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                  eos_token_id=tok.eos_token_id, tie_word_embeddings=True)
torch.manual_seed(0)
model = Qwen2ForCausalLM(cfg)
model.save_pretrained(out)
tok.save_pretrained(out)
data = Path("outputs/smoke/data")
data.mkdir(parents=True, exist_ok=True)
sft = [{"idx": i, "prompt": f"Tell me fact number {i} about cooking.", "response": f"Fact {i}: always salt your pasta water. " * 3} for i in range(80)]
(data / "sft.json").write_text(json.dumps({"train": sft[:64], "val": sft[64:], "counts": {"synthetic": True}}))
hh = [{"idx": i, "prompt": f"How do I fix problem {i}?", "chosen": "Here is a careful step by step answer.", "rejected": "I do not know."} for i in range(40)]
(data / "hh.json").write_text(json.dumps({"train": hh[:32], "val": hh[32:], "counts": {"synthetic": True}}))
print("tiny model ->", out)
