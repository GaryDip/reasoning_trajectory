"""Minimal local Llama chat wrapper used by decompose_to_nl.py.

Vendored (trimmed) from the old multihop_trajectory/llama_infer_reasoning
pipeline's PipelineConfig/LlamaChat so decompose_to_nl.py doesn't depend on
anything outside this repo.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineConfig:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    device_map: str | None = "auto"
    dtype: str = "bfloat16"
    seed: int = 42
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    # transformers: flash_attention_2 (flash-attn), sdpa (PyTorch), or eager
    attn_implementation: str = "flash_attention_2"


class LlamaChat:
    def __init__(self, cfg: PipelineConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(cfg.seed)
        dtype = getattr(torch, cfg.dtype) if hasattr(torch, cfg.dtype) else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if cfg.device_map:
            kwargs["device_map"] = cfg.device_map
        ai = (getattr(cfg, "attn_implementation", None) or "").strip()
        if ai:
            kwargs["attn_implementation"] = ai
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
        self.model.eval()
        self.cfg = cfg
        self._torch = torch
        self._seed = int(cfg.seed)

    def set_seed(self, seed: int) -> None:
        self._seed = int(seed)

    def chat(self, user_text: str, max_new_tokens: int) -> str:
        import torch

        # Seed per call so retries (sampling) differ.
        torch.manual_seed(int(self._seed))
        try:
            torch.cuda.manual_seed_all(int(self._seed))
        except Exception:
            pass

        messages = [{"role": "user", "content": user_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=bool(self.cfg.do_sample),
                temperature=float(self.cfg.temperature) if self.cfg.do_sample else None,
                top_p=float(self.cfg.top_p) if self.cfg.do_sample else None,
                repetition_penalty=float(self.cfg.repetition_penalty) if self.cfg.do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def chat_batch(self, user_texts: list[str], max_new_tokens: int) -> list[str]:
        """Same decoding settings as ``chat``, but batched ``generate`` for independent user strings."""
        if not user_texts:
            return []
        import torch

        torch.manual_seed(int(self._seed))
        try:
            torch.cuda.manual_seed_all(int(self._seed))
        except Exception:
            pass

        if len(user_texts) == 1:
            return [self.chat(user_texts[0], max_new_tokens)]

        prompts: list[str] = []
        for ut in user_texts:
            messages = [{"role": "user", "content": ut}]
            prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        gen_kw: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(self.cfg.do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.cfg.do_sample:
            gen_kw["temperature"] = float(self.cfg.temperature)
            gen_kw["top_p"] = float(self.cfg.top_p)
            gen_kw["repetition_penalty"] = float(self.cfg.repetition_penalty)
        else:
            gen_kw["repetition_penalty"] = 1.0

        with torch.inference_mode():
            out = self.model.generate(**enc, **gen_kw)

        in_len = int(enc["input_ids"].shape[1])
        gen_ids = out[:, in_len:]
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def unload(self) -> None:
        del self.model
        gc.collect()
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass
