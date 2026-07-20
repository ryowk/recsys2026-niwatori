"""Dataset, prompt-rendering, and output helpers for the Qwen3.6 responder."""

from __future__ import annotations

from pathlib import Path

import yaml

from recsys2026.artifacts import records_from_ranked_artifact
from recsys2026.data import load
from recsys2026.paths import REPO_ROOT


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_track_meta() -> dict[str, dict]:
    return {r["track_id"]: r for r in load("track", split="all_tracks")}


def _ctx(item: dict, cur: dict, hist: list[dict], gpa: list[dict], tt: int) -> dict:
    return {
        "chat_history": hist,
        "user_query": str(cur["content"]),
        "thought": str(cur.get("thought") or "").strip(),
        "user_profile": dict(item.get("user_profile") or {}),
        "conversation_goal": dict(item.get("conversation_goal") or {}),
        "prior_gpa": [
            g.get("goal_progress_assessment")
            for g in gpa
            if g.get("turn_number", 0) < tt
        ],
    }


def load_dataset_context(target: str) -> dict[tuple[str, int], dict]:
    """Return {(session_id, turn_number): context_dict}.

    context_dict has keys: chat_history, user_query, thought, user_profile,
    conversation_goal, prior_gpa.

    target: "devset" | "blind_b"
    """
    if target == "devset":
        ds = load("dataset", split="test")
    elif target == "blind_b":
        ds = load("blind_b", split="test")
    else:
        raise ValueError(f"unknown target: {target}")

    out: dict[tuple[str, int], dict] = {}
    for item in ds:
        convs = list(item["conversations"])
        gpa = list(item.get("goal_progress_assessments") or [])
        if target == "blind_b":
            cur = convs[-1]
            tt = int(cur["turn_number"])
            out[(item["session_id"], tt)] = _ctx(item, cur, convs[:-1], gpa, tt)
        else:
            for tt in range(1, 9):
                user_turn = next(
                    (
                        c
                        for c in convs
                        if c["turn_number"] == tt and c["role"] == "user"
                    ),
                    None,
                )
                if user_turn is None:
                    continue
                hist = [c for c in convs if c["turn_number"] < tt]
                out[(item["session_id"], tt)] = _ctx(item, user_turn, hist, gpa, tt)
    return out


def load_predictions_from_config(config: dict, target: str) -> tuple[list[dict], str]:
    """Load the reranked track-id records referenced by ``ranked_artifact``.

    Configs point at a ranked candidate artifact directory:

      ranked_artifact: artifacts/runs/reranker/<name>/<cfg>/<fit_mode>/{target}
    """
    artifact = REPO_ROOT / str(config["ranked_artifact"]).format(target=target)
    return records_from_ranked_artifact(
        artifact, target, top_k=20, response=""
    ), f"ranked:{artifact}"


# -------------------- standard responder helpers --------------------


def build_chat_history(context: dict, track_meta: dict) -> str:
    parts: list[str] = []
    for c in context.get("chat_history", []):
        role = c.get("role", "user")
        content = c.get("content", "")
        if role == "music":
            md = track_meta.get(content, {})
            tn = ", ".join(md.get("track_name") or []) or "?"
            an = ", ".join(md.get("artist_name") or []) or "?"
            content = f'(played: "{tn}" by {an})'
            role = "assistant"
        parts.append(f"  {role}: {content}")
    return "\n".join(parts) if parts else "  (no history yet)"


def _bytes_to_unicode() -> dict[int, str]:
    """Return the reversible GPT-2 byte-to-Unicode mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_TO_UNICODE = _bytes_to_unicode()
_UNICODE_TO_BYTE = {v: k for k, v in _BYTE_TO_UNICODE.items()}


def fix_byte_bpe_artifacts(text: str) -> str:
    """Convert byte-level BPE artifacts back to UTF-8 text."""
    bytes_arr = bytearray()
    for ch in text:
        if ch in _UNICODE_TO_BYTE:
            bytes_arr.append(_UNICODE_TO_BYTE[ch])
        else:
            bytes_arr.extend(ch.encode("utf-8"))
    try:
        return bytes_arr.decode("utf-8")
    except UnicodeDecodeError:
        return bytes_arr.decode("utf-8", errors="replace")


def build_top_tracks(track_ids: list[str], track_meta: dict, top_k: int) -> str:
    lines: list[str] = []
    for i, tid in enumerate(track_ids[:top_k], 1):
        md = track_meta.get(tid, {})
        tn = ", ".join(md.get("track_name") or []) or "?"
        an = ", ".join(md.get("artist_name") or []) or "?"
        al = ", ".join(md.get("album_name") or []) or "?"
        tags = ", ".join((md.get("tag_list") or [])[:3])
        rd = str(md.get("release_date") or "")
        year = rd[:4] if len(rd) >= 4 and rd[:4].isdigit() else ""
        year_part = f", released: {year}" if year else ""
        lines.append(f'  {i}. "{tn}" by {an} (album: {al}{year_part}, tags: {tags})')
    return "\n".join(lines)


def build_user_profile_block(ctx: dict) -> str:
    prof = ctx.get("user_profile", {}) or {}
    keys = (
        "age",
        "country_name",
        "gender",
        "preferred_language",
        "preferred_musical_culture",
    )
    lines = [f"  {k}: {prof[k]}" for k in keys if prof.get(k) not in (None, "")]
    return "\n".join(lines) if lines else "  (unknown)"


def build_goal_block(ctx: dict) -> str:
    goal = ctx.get("conversation_goal", {}) or {}
    keys = ("category", "specificity", "listener_goal")
    lines = [f"  {k}: {goal[k]}" for k in keys if goal.get(k) not in (None, "")]
    return "\n".join(lines) if lines else "  (none)"


def build_gpa_block(ctx: dict) -> str:
    gpa = [g for g in (ctx.get("prior_gpa", []) or []) if g is not None]
    if not gpa:
        return "  (no prior assistant turns)"
    return f"  Last {len(gpa)} assistant turn(s): {' -> '.join(gpa)}"


def load_model_and_tokenizer(model_name: str, device: str):
    """Load the responder LLM in bf16 on a single GPU.

    The submitted system runs Qwen3.6-27B unquantized (~54GB weights → 80GB-class
    GPU).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16
    ).eval()
    model = model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model
