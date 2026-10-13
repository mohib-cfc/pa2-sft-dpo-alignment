"""Prompt templates and stop ids."""

from __future__ import annotations

QWEN_SYSTEM = "You are a helpful assistant."
QWEN_STOP_IDS = [151643, 151645]  # <|endoftext|>, <|im_end|>


def qwen_prompt(instruction: str) -> str:
    """Qwen chat format built by hand (apply_chat_template injects another system prompt)."""
    return (
        f"<|im_start|>system\n{QWEN_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def smol_prompt(tokenizer, instruction: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
    )
