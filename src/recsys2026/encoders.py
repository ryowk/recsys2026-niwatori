"""Qwen3 text encoder used by preprocessing and reranker features."""

from __future__ import annotations

import numpy as np
import torch


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)


class Qwen3TextEncoder:
    """Encode text as normalized 1024-dimensional Qwen3 embeddings."""

    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = (
            AutoModel.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        )

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**enc)
            hidden = outputs.last_hidden_state[:, -1]
            hidden = _l2_normalize(hidden.float())
            out.append(hidden.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)
