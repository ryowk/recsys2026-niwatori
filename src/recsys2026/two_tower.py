"""113_two_tower_lora_thought: Two-Tower LoRA with current thought in query text.

095_two_tower_lora は learned retriever として zoo 内で最大の marginal contribution を持つが、
query text は chat_to_query_text(InferenceInput) 由来で current user thought を含まない。
current thought は blind 入力にも提供される legal feature なので、conversation_goal と一緒に
query text へ足し、LoRA retriever の recall が上がるかを見る。

設計:
- Query encoder: Qwen3-Embedding-0.6B (LoRA r=16, target=q_proj, k_proj, v_proj, o_proj)
- Track tower: 081 と同じ 6-modality concat → MLP (frozen weights from 081 で warm start可)
- Loss: InfoNCE in-batch (batch=64 since LoRA needs grad)
- 1 epoch = 1900 step at bs=64, ~20-30 min on RTX 6000.

リーク:
- train pairs only (121592 turns from train split).
- LoRA weights, head MLP, track tower 全部 train でしか学習しない.
- track features (6 modality + popularity): 081 と同じ.

出力: output/086_retriever_zoo_v2/cand/two_tower_lora_thought__n8000.npz
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from recsys2026.data import load
from recsys2026.paths import OUTPUT_DIR as _OUTPUT_ROOT, RESULTS_DIR as _RESULTS_ROOT
from recsys2026.retrieval import chat_to_query_text
from recsys2026.submission import InferenceInput

OUT_DIR = _OUTPUT_ROOT / "two_tower"
RESULTS_DIR = _RESULTS_ROOT / "two_tower"

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


def build_track_features(cache_path):
    if cache_path.exists():
        print(f"[cache] {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return list(data["track_ids"]), data["features"].astype(np.float32)
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
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, track_ids=np.asarray(track_ids, dtype=object), features=features)
    return track_ids, features


def build_train_pairs():
    ds = load("dataset", split="train")
    pairs = []
    for item in ds:
        conversations = list(item["conversations"])
        for target_turn in range(1, MAX_TURNS + 1):
            current = [c for c in conversations if c["turn_number"] == target_turn]
            user_turn = next((c for c in current if c["role"] == "user"), None)
            music_turn = next((c for c in current if c["role"] == "music"), None)
            if user_turn is None or music_turn is None:
                continue
            history = [c for c in conversations if c["turn_number"] < target_turn]
            inp = InferenceInput(
                session_id=item["session_id"],
                user_id=item["user_id"],
                turn_number=target_turn,
                chat_history=history,
                user_query=user_turn["content"],
            )
            q_text = query_text_with_goal_thought(
                inp,
                item.get("conversation_goal") or {},
                user_turn.get("thought") or "",
            )
            gold_tid = music_turn["content"]
            if gold_tid:
                pairs.append((q_text, gold_tid))
    return pairs


def query_text_with_goal_thought(
    inp: InferenceInput,
    conversation_goal: dict,
    current_thought: str,
) -> str:
    safe = True  # blind-B-safe fixed: goal/thought are never used
    parts: list[str] = []
    goal = conversation_goal or {}
    goal_text = "" if safe else " ".join(
        str(goal.get(k) or "")
        for k in ("category", "specificity", "listener_goal")
    ).strip()
    if goal_text:
        parts.append(f"conversation goal: {goal_text}")
    base = chat_to_query_text(inp, mode="full")
    if base:
        parts.append(base)
    thought = "" if safe else str(current_thought or "").strip()
    if thought:
        parts.append(f"current user thought: {thought}")
    return "\n".join(parts).strip() or inp.user_query


def build_devset_examples():
    ds = load("dataset", split="test")
    out = []
    for item in ds:
        conversations = list(item["conversations"])
        for target_turn in range(1, MAX_TURNS + 1):
            current = [c for c in conversations if c["turn_number"] == target_turn]
            user_turn = next(c for c in current if c["role"] == "user")
            gold = next(c["content"] for c in current if c["role"] == "music")
            inp = InferenceInput(
                session_id=item["session_id"],
                user_id=item["user_id"],
                turn_number=target_turn,
                chat_history=[c for c in conversations if c["turn_number"] < target_turn],
                user_query=user_turn["content"],
            )
            q_text = query_text_with_goal_thought(
                inp,
                item.get("conversation_goal") or {},
                user_turn.get("thought") or "",
            )
            out.append({
                "session_id": item["session_id"],
                "user_id": item["user_id"],
                "turn_number": target_turn,
                "chat_history": [c for c in conversations if c["turn_number"] < target_turn],
                "user_query": user_turn["content"],
                "gold_track_id": gold,
                "q_text": q_text,
            })
    return out


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
    # 推測: model 全体が同じ device/dtype に乗っているとして, parameters() の最初を見る
    sample_param = next(model.parameters())
    target_device = sample_param.device
    target_dtype = sample_param.dtype
    n_added = 0
    for name, module in model.named_modules():
        if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
            in_dim = module.q_proj.in_features
            out_dim = module.q_proj.out_features
            if not hasattr(module.q_proj, "lora"):
                # LoRA は float32 で持つ (bf16 だと grad 不安定なので)
                module.q_proj.lora = LoRALayer(in_dim, out_dim, r=r, alpha=alpha).to(target_device).float()
                module.k_proj.lora = LoRALayer(in_dim, module.k_proj.out_features, r=r, alpha=alpha).to(target_device).float()
                module.v_proj.lora = LoRALayer(in_dim, module.v_proj.out_features, r=r, alpha=alpha).to(target_device).float()
                if hasattr(module, "o_proj"):
                    module.o_proj.lora = LoRALayer(module.o_proj.in_features, module.o_proj.out_features, r=r, alpha=alpha).to(target_device).float()
                # patch forward
                _patch_lora_forward(module.q_proj)
                _patch_lora_forward(module.k_proj)
                _patch_lora_forward(module.v_proj)
                if hasattr(module, "o_proj"):
                    _patch_lora_forward(module.o_proj)
                n_added += 1
    print(f"  LoRA added to {n_added} attention modules (device={target_device}, base_dtype={target_dtype})")
    return model


def _patch_lora_forward(linear: nn.Linear):
    """Linear.forward を base + lora に書き換える."""
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
    """LoRA 以外の Qwen3 weights を freeze."""
    for name, p in model.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params (encoder): {n_train:,} / {n_total:,} ({n_train/n_total*100:.2f}%)")


def encode_with_qwen3(model, tokenizer, texts, device, max_length=512, batch_size=16):
    """Qwen3 で texts を encode (LoRA 適用)."""
    out = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        outputs = model(**enc)
        hidden = outputs.last_hidden_state[:, -1]  # last token (EOS, left-padded)
        out.append(hidden)
    return torch.cat(out, dim=0)


# -------------------- track tower --------------------


class TrackHead(nn.Module):
    def __init__(self, track_dim=TRACK_DIM, hidden=HIDDEN_DIM, out_dim=EMBED_DIM, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(track_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, t):
        return F.normalize(self.net(t), dim=-1)


class QueryHead(nn.Module):
    def __init__(self, in_dim=QUERY_DIM, hidden=HIDDEN_DIM, out_dim=EMBED_DIM, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# -------------------- training --------------------


class TrainPairsDataset(Dataset):
    def __init__(self, q_texts, gold_idxs, track_features):
        self.q_texts = q_texts
        self.gold = gold_idxs
        self.t = track_features

    def __len__(self):
        return len(self.gold)

    def __getitem__(self, idx):
        return self.q_texts[idx], int(self.gold[idx])


def collate(batch, track_features):
    texts = [b[0] for b in batch]
    golds = [b[1] for b in batch]
    t_pos = torch.from_numpy(track_features[golds])
    return texts, t_pos


def load_lora_two_tower(ckpt_path, *, device="cuda", lora_r=16, lora_alpha=32):
    """Reconstruct (tokenizer, qwen, q_head, t_head) from a saved checkpoint
    WITHOUT training. Mirrors the load path of `train_lora_two_tower` but returns
    the model in eval mode so callers can encode/retrieve directly.

    Used to reproduce two-tower candidates from shipped weights (skipping the
    hours-long LoRA training). The checkpoint must carry `qwen_lora` / `q_head`
    / `t_head` (as written by `train_lora_two_tower`)."""
    from transformers import AutoModel, AutoTokenizer

    ckpt_path = Path(ckpt_path)
    print(f"loading two-tower checkpoint (no training): {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL, padding_side="left")
    qwen = AutoModel.from_pretrained(QWEN3_MODEL, dtype=torch.bfloat16).to(device)
    add_lora_to_qwen3(qwen, r=lora_r, alpha=lora_alpha)
    freeze_non_lora(qwen)

    ckpt = torch.load(ckpt_path, map_location=device)
    qwen_params = dict(qwen.named_parameters())
    missing: list[str] = []
    for name, value in ckpt.get("qwen_lora", {}).items():
        param = qwen_params.get(name)
        if param is None:
            missing.append(name)
            continue
        param.data.copy_(value.to(device=param.device, dtype=param.dtype))
    if missing:
        raise RuntimeError(f"checkpoint has unknown LoRA params: {missing[:5]}")

    q_head = QueryHead().to(device)
    t_head = TrackHead().to(device)
    q_head.load_state_dict(ckpt["q_head"])
    t_head.load_state_dict(ckpt["t_head"])

    qwen.eval()
    q_head.eval()
    t_head.eval()
    return tokenizer, qwen, q_head, t_head


def train_lora_two_tower(
    q_texts, gold_idxs, track_features,
    epochs=2, batch_size=64, lr_lora=5e-5, lr_head=1e-4,
    temperature=0.05, lora_r=16, lora_alpha=32,
    device="cuda", init_checkpoint=None,
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

    if init_checkpoint is not None:
        ckpt_path = Path(init_checkpoint)
        print(f"loading two-tower init checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        qwen_params = dict(qwen.named_parameters())
        missing: list[str] = []
        for name, value in ckpt.get("qwen_lora", {}).items():
            param = qwen_params.get(name)
            if param is None:
                missing.append(name)
                continue
            param.data.copy_(value.to(device=param.device, dtype=param.dtype))
        if missing:
            raise RuntimeError(f"init checkpoint has unknown LoRA params: {missing[:5]}")
        q_head.load_state_dict(ckpt["q_head"])
        t_head.load_state_dict(ckpt["t_head"])

    lora_params = [p for n, p in qwen.named_parameters() if p.requires_grad]
    head_params = list(q_head.parameters()) + list(t_head.parameters())
    optim = torch.optim.AdamW([
        {"params": lora_params, "lr": lr_lora},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=1e-4)

    ds = TrainPairsDataset(q_texts, gold_idxs, track_features)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2,
        collate_fn=lambda b: collate(b, track_features),
    )
    print(f"training: {len(ds)} pairs, batch={batch_size}, epochs={epochs}, lr_lora={lr_lora}, lr_head={lr_head}")

    for ep in range(epochs):
        ep_losses = []
        for texts, t_pos in tqdm(loader, desc=f"epoch {ep+1}"):
            t_pos = t_pos.to(device)
            enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
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
        print(f"  ep {ep+1} mean loss: {np.mean(ep_losses):.4f}")

    return tokenizer, qwen, q_head, t_head


def encode_all_queries(tokenizer, qwen, q_head, q_texts, device, batch_size=16):
    qwen.eval()
    q_head.eval()
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(q_texts), batch_size), desc="encode queries"):
            chunk = q_texts[i:i + batch_size]
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
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
            batch = torch.from_numpy(track_features[i:i + batch_size]).to(device)
            z = t_head(batch)
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def run_inference(query_emb_devset, track_emb, devset, track_id_to_idx, top_k):
    n = len(devset)
    cand = np.full((n, top_k), -1, dtype=np.int32)
    sizes = np.zeros(n, dtype=np.int32)
    sims = query_emb_devset @ track_emb.T
    for i, ex in enumerate(tqdm(devset, desc="infer")):
        played: set[int] = set()
        for c in ex["chat_history"]:
            if c.get("role") == "music":
                idx = track_id_to_idx.get(c.get("content"))
                if idx is not None:
                    played.add(idx)
        score = sims[i].copy()
        if played:
            score[np.fromiter(played, dtype=np.int32)] = -np.inf
        k = min(top_k, len(score))
        part = np.argpartition(-score, k - 1)[:k]
        order = np.argsort(-score[part])
        kept = part[order].astype(np.int32)
        cand[i, :len(kept)] = kept
        sizes[i] = len(kept)
    return cand, sizes


def compute_recall(cand, gold_idxs, ks=(20, 50, 100, 200)):
    valid = gold_idxs >= 0
    n_v = max(int(valid.sum()), 1)
    out = {}
    for k in ks:
        cand_k = cand[:, :k]
        hit = (cand_k == gold_idxs[:, None]).any(axis=1) & valid
        out[f"recall@{k}"] = float(hit.sum() / n_v)
    return out


