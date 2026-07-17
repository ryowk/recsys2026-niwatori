"""Qwen3-Embedding-0.6B LoRA two-tower retriever implementation.

The query tower and projection heads are trained on the caller's fit split.
The track tower consumes the organizer-provided modalities and catalog metadata.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from recsys2026.artifacts import npz_dump
from recsys2026.data import load
from recsys2026.retrieval import chat_to_query_text
from recsys2026.submission import InferenceInput

QUERY_DIM = 1024
TRACK_MODALITY_DIMS = {
    "audio-laion_clap": 512,
    "image-siglip2": 768,
    "cf-bpr": 128,
    "attributes-qwen3_embedding_0.6b": 1024,
    "lyrics-qwen3_embedding_0.6b": 1024,
    "metadata-qwen3_embedding_0.6b": 1024,
}
TRACK_DIM = sum(TRACK_MODALITY_DIMS.values()) + 1
EMBED_DIM = 512
HIDDEN_DIM = 1024
MAX_TURNS = 8

QWEN3_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _to_dense(values, dim=None):
    if dim is None:
        lengths = [len(v) for v in values if v is not None and len(v) > 0]
        if not lengths:
            raise ValueError("no non-empty embeddings")
        dim = Counter(lengths).most_common(1)[0][0]
    out = np.zeros((len(values), dim), dtype=np.float32)
    for i, v in enumerate(values):
        if v is None or len(v) != dim:
            continue
        out[i] = np.asarray(v, dtype=np.float32)
    return out


def _l2_normalize(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    return (x / n).astype(np.float32)


def build_track_features(artifact_path):
    if artifact_path.exists():
        try:
            with np.load(artifact_path, allow_pickle=False) as data:
                track_ids = [str(value) for value in data["track_ids"]]
                features = data["features"].astype(np.float32)
            if (
                features.shape == (len(track_ids), TRACK_DIM)
                and len(track_ids) == len(set(track_ids))
                and np.isfinite(features).all()
            ):
                print(f"[existing] {artifact_path}")
                return track_ids, features
        except (KeyError, OSError, ValueError):
            pass
        print(f"rebuilding incomplete track feature cache: {artifact_path}")
    emb = load("track_emb", split="all_tracks")
    track_ids = list(emb["track_id"])
    n = len(track_ids)
    parts = []
    for col, dim in TRACK_MODALITY_DIMS.items():
        if col not in emb.column_names:
            parts.append(np.zeros((n, dim), dtype=np.float32))
            continue
        m = _to_dense(emb[col], dim=dim)
        parts.append(_l2_normalize(m))
    meta = load("track", split="all_tracks")
    pop_by_id = {row["track_id"]: float(row.get("popularity") or 0.0) for row in meta}
    pop = np.asarray([pop_by_id.get(tid, 0.0) for tid in track_ids], dtype=np.float32)
    log_pop = np.log1p(pop)[:, None]
    parts.append(log_pop)
    features = np.concatenate(parts, axis=1).astype(np.float32)
    npz_dump(
        artifact_path,
        {"track_ids": np.asarray(track_ids, dtype=np.str_), "features": features},
    )
    return track_ids, features


def query_text(inp: InferenceInput) -> str:
    """Render only fields available consistently in Blind B."""
    return chat_to_query_text(inp, mode="full").strip() or inp.user_query


# -------------------- model --------------------


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, r=16, alpha=32):
        super().__init__()
        self.lora_a = nn.Linear(in_dim, r, bias=False)
        self.lora_b = nn.Linear(r, out_dim, bias=False)
        nn.init.normal_(self.lora_a.weight, std=0.02)
        nn.init.zeros_(self.lora_b.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.lora_b(self.lora_a(x)) * self.scale


def add_lora_to_qwen3(model, r=16, alpha=32):
    """Add LoRA to all q_proj/k_proj/v_proj/o_proj in attention layers."""
    # The base model is loaded on one device and dtype.
    sample_param = next(model.parameters())
    target_device = sample_param.device
    target_dtype = sample_param.dtype
    n_added = 0
    for name, module in model.named_modules():
        if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
            in_dim = module.q_proj.in_features
            out_dim = module.q_proj.out_features
            if not hasattr(module.q_proj, "lora"):
                # Keep LoRA weights in float32 for stable gradients.
                module.q_proj.lora = (
                    LoRALayer(in_dim, out_dim, r=r, alpha=alpha)
                    .to(target_device)
                    .float()
                )
                module.k_proj.lora = (
                    LoRALayer(in_dim, module.k_proj.out_features, r=r, alpha=alpha)
                    .to(target_device)
                    .float()
                )
                module.v_proj.lora = (
                    LoRALayer(in_dim, module.v_proj.out_features, r=r, alpha=alpha)
                    .to(target_device)
                    .float()
                )
                if hasattr(module, "o_proj"):
                    module.o_proj.lora = (
                        LoRALayer(
                            module.o_proj.in_features,
                            module.o_proj.out_features,
                            r=r,
                            alpha=alpha,
                        )
                        .to(target_device)
                        .float()
                    )
                # patch forward
                _patch_lora_forward(module.q_proj)
                _patch_lora_forward(module.k_proj)
                _patch_lora_forward(module.v_proj)
                if hasattr(module, "o_proj"):
                    _patch_lora_forward(module.o_proj)
                n_added += 1
    print(
        f"  LoRA added to {n_added} attention modules (device={target_device}, base_dtype={target_dtype})"
    )
    return model


def _patch_lora_forward(linear: nn.Linear):
    """Patch a linear layer to return its base output plus LoRA."""
    if getattr(linear, "_lora_patched", False):
        return
    base_forward = nn.Linear.forward

    def forward(self, x):
        out = base_forward(self, x)
        if hasattr(self, "lora"):
            out = out + self.lora(x.to(self.lora.lora_a.weight.dtype)).to(out.dtype)
        return out

    linear.forward = forward.__get__(linear, type(linear))
    linear._lora_patched = True


def freeze_non_lora(model):
    """Freeze every Qwen3 parameter except the LoRA adapters."""
    for name, p in model.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"  trainable params (encoder): {n_train:,} / {n_total:,} ({n_train / n_total * 100:.2f}%)"
    )


# -------------------- track tower --------------------


class TrackHead(nn.Module):
    def __init__(
        self, track_dim=TRACK_DIM, hidden=HIDDEN_DIM, out_dim=EMBED_DIM, dropout=0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(track_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, t):
        return F.normalize(self.net(t), dim=-1)


class QueryHead(nn.Module):
    def __init__(
        self, in_dim=QUERY_DIM, hidden=HIDDEN_DIM, out_dim=EMBED_DIM, dropout=0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# -------------------- training --------------------


class TrainPairsDataset(Dataset):
    def __init__(self, q_texts, gold_idxs):
        self.q_texts = q_texts
        self.gold = gold_idxs

    def __len__(self):
        return len(self.gold)

    def __getitem__(self, idx):
        return self.q_texts[idx], int(self.gold[idx])


def collate(batch, track_features):
    texts = [b[0] for b in batch]
    golds = [b[1] for b in batch]
    t_pos = torch.from_numpy(track_features[golds])
    return texts, t_pos


def train_lora_two_tower(
    q_texts,
    gold_idxs,
    track_features,
    epochs=2,
    batch_size=64,
    lr_lora=5e-5,
    lr_head=1e-4,
    temperature=0.05,
    lora_r=16,
    lora_alpha=32,
    device="cuda",
):
    from transformers import AutoModel, AutoTokenizer

    print(f"loading Qwen3 + LoRA r={lora_r} ...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL, padding_side="left")
    qwen = AutoModel.from_pretrained(QWEN3_MODEL, dtype=torch.bfloat16).to(device)
    add_lora_to_qwen3(qwen, r=lora_r, alpha=lora_alpha)
    freeze_non_lora(qwen)
    qwen.train()

    q_head = QueryHead().to(device)
    t_head = TrackHead().to(device)

    lora_params = [p for n, p in qwen.named_parameters() if p.requires_grad]
    head_params = list(q_head.parameters()) + list(t_head.parameters())
    optim = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lr_lora},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=1e-4,
    )

    ds = TrainPairsDataset(q_texts, gold_idxs)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        collate_fn=lambda b: collate(b, track_features),
    )
    print(
        f"training: {len(ds)} pairs, batch={batch_size}, epochs={epochs}, lr_lora={lr_lora}, lr_head={lr_head}"
    )

    for ep in range(epochs):
        ep_losses = []
        for texts, t_pos in tqdm(loader, desc=f"epoch {ep + 1}"):
            t_pos = t_pos.to(device)
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            outputs = qwen(**enc)
            hidden = outputs.last_hidden_state[:, -1].float()  # [B, 1024]
            qz = q_head(hidden)
            tz = t_head(t_pos)
            sim = qz @ tz.T / temperature
            labels = torch.arange(len(qz), device=device)
            loss = F.cross_entropy(sim, labels)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            ep_losses.append(float(loss.item()))
        print(f"  ep {ep + 1} mean loss: {np.mean(ep_losses):.4f}")

    return tokenizer, qwen, q_head, t_head


def encode_all_queries(tokenizer, qwen, q_head, q_texts, device, batch_size=16):
    qwen.eval()
    q_head.eval()
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(q_texts), batch_size), desc="encode queries"):
            chunk = q_texts[i : i + batch_size]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            outputs = qwen(**enc)
            hidden = outputs.last_hidden_state[:, -1].float()
            z = q_head(hidden)
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def encode_all_tracks(t_head, track_features, device, batch_size=512):
    t_head.eval()
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(track_features), batch_size), desc="encode tracks"):
            batch = torch.from_numpy(track_features[i : i + batch_size]).to(device)
            z = t_head(batch)
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)
