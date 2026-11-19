"""LLM backends.

hf          Hugging Face transformers pipeline (default). Chat models (Qwen, Llama, Phi...) and
            encoder-decoder models (flan-t5) are both supported.
extractive  No model: answers by quoting the best-matching context sentences with citations.
            Offline fallback used by the tests; also a safe degraded mode.
vertex      Gemini on Vertex AI through google-genai (optional).
"""
from __future__ import annotations

import math
from typing import Protocol

from .config import Settings
from .prompt import REFUSAL, PromptBundle
from .text_utils import split_sentences, tokenize


class LLM(Protocol):
    name: str

    def generate(self, bundle: PromptBundle) -> str: ...


class ExtractiveLLM:
    name = "extractive"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, bundle: PromptBundle) -> str:
        q = set(tokenize(bundle.question))
        scored = []
        for n, hit in enumerate(bundle.hits, 1):
            for pos, sent in enumerate(split_sentences(hit.text)):
                overlap = len(q & set(tokenize(sent)))
                if overlap:
                    scored.append((overlap + 0.5 / n, n, pos, sent))
        if not scored:
            return REFUSAL
        best = sorted(scored, key=lambda t: -t[0])[: self.max_sentences]
        best.sort(key=lambda t: (t[1], t[2]))  # read in document order
        return " ".join(f"{sent.rstrip('.')}. [{n}]" for _, n, _, sent in best)


class HFLLM:
    def __init__(self, model_name: str, device: str = "auto", max_new_tokens: int = 256):
        try:
            from transformers import AutoConfig, pipeline
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("LLM_BACKEND=hf needs: pip install transformers torch") from e
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = 0 if device == "cuda" else -1
        self.name = model_name
        self.max_new_tokens = max_new_tokens
        cfg = AutoConfig.from_pretrained(model_name)
        self.seq2seq = bool(getattr(cfg, "is_encoder_decoder", False))
        task = "text2text-generation" if self.seq2seq else "text-generation"
        self._pipe = pipeline(task, model=model_name, device=dev)

    def generate(self, bundle: PromptBundle) -> str:
        if self.seq2seq:
            out = self._pipe(bundle.full_text, max_new_tokens=self.max_new_tokens, do_sample=False)
            return out[0]["generated_text"].strip()
        messages = [{"role": "system", "content": bundle.system}, {"role": "user", "content": bundle.user}]
        out = self._pipe(messages, max_new_tokens=self.max_new_tokens, do_sample=False, return_full_text=False)
        gen = out[0]["generated_text"]
        if isinstance(gen, list):  # some transformers versions return the chat list
            gen = gen[-1]["content"]
        return gen.strip()


class VertexLLM:  # pragma: no cover - needs GCP credentials
    def __init__(self, project: str, location: str, model: str, max_new_tokens: int = 512):
        from google import genai

        self._client = genai.Client(vertexai=True, project=project, location=location)
        self.name, self._max = f"vertex:{model}", max_new_tokens
        self._model = model

    def generate(self, bundle: PromptBundle) -> str:
        from google.genai import types

        res = self._client.models.generate_content(
            model=self._model,
            contents=bundle.user,
            config=types.GenerateContentConfig(
                system_instruction=bundle.system, temperature=0.1, max_output_tokens=self._max
            ),
        )
        return (res.text or "").strip()


def get_llm(s: Settings) -> LLM:
    if s.llm_backend == "extractive":
        return ExtractiveLLM()
    if s.llm_backend == "hf":
        return HFLLM(s.llm_model, s.device, s.max_new_tokens)
    if s.llm_backend == "vertex":
        return VertexLLM(s.gcp_project, s.gcp_location, s.vertex_llm_model)
    raise ValueError(f"Unknown LLM_BACKEND '{s.llm_backend}' (hf | extractive | vertex)")
