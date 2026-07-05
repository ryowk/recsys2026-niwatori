"""事前学習エンコーダのラッパ (text → embedding)。

各 encoder は ``encode(texts: list[str]) -> np.ndarray`` を提供し、
出力は float32, L2 正規化済み (cosine = dot product として使える)。

track 側の事前計算済み embedding (talkpl-ai/TalkPlayData-Challenge-Track-Embeddings) と
同一 model 想定。同一空間でない場合は 004 を smoke で走らせて nDCG@20 が公式 BERT
baseline (~0.14) を顕著に下回るので、その時点で track 側を再エンコードする方針に切替。
"""

from __future__ import annotations

import numpy as np
import torch


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)


class Qwen3TextEncoder:
    """Qwen/Qwen3-Embedding-0.6B (1024-dim)。

    Qwen3 系 embedding は last hidden state の最後のトークンを取る (left-padding 済 EOS)。
    track 側 attributes/lyrics/metadata-qwen3 と同空間想定。
    """

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
            hidden = outputs.last_hidden_state[:, -1]  # left-padded → last token = EOS
            hidden = _l2_normalize(hidden.float())
            out.append(hidden.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


def _extract_text_embedding(feats: object) -> torch.Tensor:
    """``model.get_text_features`` の戻り値を tensor に揃える。

    transformers の version によって Tensor を直接返したり
    BaseModelOutputWithPooling を返したりするので両対応。
    """
    if isinstance(feats, torch.Tensor):
        return feats
    pooler = getattr(feats, "pooler_output", None)
    if pooler is not None:
        return pooler
    last = getattr(feats, "last_hidden_state", None)
    if last is not None:
        return last[:, 0]  # CLS fallback
    raise TypeError(f"unexpected text-features type: {type(feats).__name__}")


class ClapTextEncoder:
    """laion/larger_clap_general の text branch (pooler_output 512-dim).

    track 側 audio-laion_clap と同空間想定。
    """

    DEFAULT_MODEL = "laion/larger_clap_general"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 77,
    ) -> None:
        from transformers import AutoProcessor, ClapModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.processor(
                text=batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            feats = self.model.get_text_features(**enc)
            text_emb = _extract_text_embedding(feats)
            text_emb = _l2_normalize(text_emb.float())
            out.append(text_emb.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


class SigLIPTextEncoder:
    """google/siglip2-base-patch16-224 の text branch (pooler_output 768-dim).

    track 側 image-siglip2 と同空間想定 (アルバム画像 ↔ テキスト)。
    AutoProcessor が SigLIP2 の image processor を解決できない transformers 版があるので、
    text encoding は AutoTokenizer 単体で行う。
    """

    DEFAULT_MODEL = "google/siglip2-base-patch16-224"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 64,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            feats = self.model.get_text_features(**enc)
            text_emb = _extract_text_embedding(feats)
            text_emb = _l2_normalize(text_emb.float())
            out.append(text_emb.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)
