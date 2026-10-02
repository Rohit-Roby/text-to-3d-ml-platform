from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EncoderSpec:
    backend: str
    model_name: str
    embedding_dim: int


class HashTextEncoder:
    """Deterministic, dependency-light encoder used for smoke tests and demos.

    It is not intended as the portfolio ML encoder. The production path uses T5,
    but this backend makes the repository fully runnable without downloading model
    weights and guarantees that CI can exercise training and inference end to end.
    """

    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.spec = EncoderSpec("hash", "deterministic-hash-v1", embedding_dim)

    def encode(self, texts: list[str], max_length: int = 64) -> torch.Tensor:
        del max_length
        output = torch.zeros((len(texts), self.embedding_dim), dtype=torch.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9-]+", (text or "").lower()) or ["empty"]
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
                index = int.from_bytes(digest[:8], "little") % self.embedding_dim
                sign = 1.0 if digest[8] % 2 == 0 else -1.0
                output[row, index] += sign
            norm = float(torch.linalg.vector_norm(output[row]))
            if norm > 0:
                output[row] /= norm
        return output


class T5TextEncoder:
    def __init__(self, model_name: str = "google-t5/t5-base", device: str | None = None):
        from transformers import AutoTokenizer, T5EncoderModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device).eval()
        dim = int(self.model.config.d_model)
        self.spec = EncoderSpec("t5", model_name, dim)

    @torch.inference_mode()
    def encode(self, texts: list[str], max_length: int = 64) -> torch.Tensor:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.cpu()


def create_text_encoder(
    backend: str = "t5",
    model_name: str = "google-t5/t5-base",
    embedding_dim: int = 256,
    device: str | None = None,
):
    backend = backend.lower()
    if backend == "t5":
        return T5TextEncoder(model_name=model_name, device=device)
    if backend == "hash":
        return HashTextEncoder(embedding_dim=embedding_dim)
    raise ValueError(f"Unknown text encoder backend: {backend}")


# Backwards-compatible alias for the original scaffold.
TextEncoder = T5TextEncoder
