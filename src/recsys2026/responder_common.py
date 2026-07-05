"""responder/common.py — utility shared across standard responders.

dataset / IO / run log の helper に加え、`run_standard_responder` で
"YAML 読む → LLM ロード → 1 record ずつ生成 → 保存" の標準フローを束ねる.

独自フロー (multi-step generation, special prompt logic 等) を要する responder は
自前 main.py で `run_standard_responder` を使わず実装すれば良い.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
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
            g.get("goal_progress_assessment") for g in gpa
            if g.get("turn_number", 0) < tt
        ],
    }


def load_dataset_context(target: str) -> dict[tuple[str, int], dict]:
    """Return {(session_id, turn_number): context_dict}.

    context_dict has keys: chat_history, user_query, thought, user_profile,
    conversation_goal, prior_gpa.

    target: "devset" | "blind_a" | "blind_b"
    """
    if target == "devset":
        ds = load("dataset", split="test")
    elif target in ("blind_a", "blind_b"):
        ds = load(target, split="test")
    else:
        raise ValueError(f"unknown target: {target}")

    out: dict[tuple[str, int], dict] = {}
    for item in ds:
        convs = list(item["conversations"])
        gpa = list(item.get("goal_progress_assessments") or [])
        if target in ("blind_a", "blind_b"):
            cur = convs[-1]
            tt = int(cur["turn_number"])
            out[(item["session_id"], tt)] = _ctx(item, cur, convs[:-1], gpa, tt)
        else:
            for tt in range(1, 9):
                user_turn = next(
                    (c for c in convs if c["turn_number"] == tt and c["role"] == "user"),
                    None,
                )
                if user_turn is None:
                    continue
                hist = [c for c in convs if c["turn_number"] < tt]
                out[(item["session_id"], tt)] = _ctx(item, user_turn, hist, gpa, tt)
    return out


def load_predictions(retriever: str, target: str) -> list[dict]:
    """Read output/<retriever>/<target>.json from the source retriever exp."""
    path = Path(__file__).parents[1] / "output" / retriever / f"{target}.json"
    with open(path) as f:
        return json.load(f)


def load_predictions_from_config(config: dict, target: str) -> tuple[list[dict], str]:
    """Resolve legacy retriever output or new ranked artifact input.

    Existing responder configs use ``retriever: <exp_name>`` and keep working.
    New configs may use either:

      ranked_artifact: output/reranker/<name>/<cfg>/<fit_mode>/{target}

    or:

      upstream:
        reranker: {name: legacy_098, config: current_thought_profile, fit_mode: legacy}
    """
    has_legacy = "retriever" in config
    has_ranked = "ranked_artifact" in config
    has_upstream = isinstance(config.get("upstream"), dict) and "reranker" in config["upstream"]
    if sum(bool(x) for x in (has_legacy, has_ranked, has_upstream)) != 1:
        raise ValueError("responder config must define exactly one of retriever, ranked_artifact, upstream.reranker")

    if has_legacy:
        retriever = config["retriever"]
        return load_predictions(retriever, target), f"legacy:{retriever}"

    top_k = int(config.get("preserve_top_k", 20))
    if has_ranked:
        artifact = REPO_ROOT / str(config["ranked_artifact"]).format(target=target)
        return records_from_ranked_artifact(artifact, target, top_k=top_k, response=""), f"ranked:{artifact}"

    ref = config["upstream"]["reranker"]
    fit_mode = ref.get("fit_mode", config.get("fit_mode", "legacy"))
    artifact = REPO_ROOT / "output" / "reranker" / ref["name"] / ref["config"] / fit_mode / target
    return records_from_ranked_artifact(artifact, target, top_k=top_k, response=""), f"reranker:{ref['name']}/{ref['config']}/{fit_mode}"


def save_predictions(records: list[dict], output_dir: Path, target: str) -> Path:
    """Write <target>.json + <target>.submission.zip (Codabench format)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(records, ensure_ascii=False)
    (output_dir / f"{target}.json").write_text(body)
    zip_path = output_dir / f"{target}.submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", body)
    return zip_path


def save_manifest(
    output_dir: Path,
    *,
    target: str,
    responder_name: str,
    config_name: str,
    config: dict,
    source_label: str,
    argv: list[str],
    started_at: float,
    elapsed: float,
    n_records: int,
) -> None:
    manifest = {
        "schema_version": 1,
        "artifact_type": "predictions",
        "stage": "responder",
        "name": responder_name,
        "config": config_name,
        "target": target,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "elapsed_sec": round(elapsed, 2),
        "n_records": n_records,
        "source": source_label,
        "argv": argv,
        "config_body": config,
        "outputs": {
            "json": str((output_dir / f"{target}.json").relative_to(REPO_ROOT)),
            "zip": str((output_dir / f"{target}.submission.zip").relative_to(REPO_ROOT)),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def append_run_log(
    results_dir: Path,
    started_at: float,
    elapsed: float,
    status: str,
    config: dict,
    argv: list[str],
) -> None:
    """Append one JSON line to results_dir/runs.jsonl."""
    results_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "elapsed_sec": round(elapsed, 2),
        "status": status,
        "config": config,
        "argv": argv,
    }
    with open(results_dir / "runs.jsonl", "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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
    """GPT-2 byte → printable-unicode mapping (transformers.GPT2Tokenizer 由来)."""
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
    """`Ġ`, `Ċ`, `âĢĶ` 等の GPT-2 BPE byte 文字列を真の UTF-8 に直す.

    DeepSeek-R1 distill 等の一部 fast tokenizer が decode で byte 表現を残す挙動への対策.
    通常のテキスト ASCII / 通常 UTF-8 では no-op.
    """
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
    return build_top_tracks_with_options(track_ids, track_meta, top_k, tag_limit=3)


def build_top_tracks_with_options(
    track_ids: list[str],
    track_meta: dict,
    top_k: int,
    *,
    tag_limit: int = 3,
) -> str:
    lines: list[str] = []
    for i, tid in enumerate(track_ids[:top_k], 1):
        md = track_meta.get(tid, {})
        tn = ", ".join(md.get("track_name") or []) or "?"
        an = ", ".join(md.get("artist_name") or []) or "?"
        al = ", ".join(md.get("album_name") or []) or "?"
        tags = ", ".join((md.get("tag_list") or [])[:tag_limit])
        rd = str(md.get("release_date") or "")
        year = rd[:4] if len(rd) >= 4 and rd[:4].isdigit() else ""
        year_part = f", released: {year}" if year else ""
        lines.append(f'  {i}. "{tn}" by {an} (album: {al}{year_part}, tags: {tags})')
    return "\n".join(lines)


def build_user_profile_block(ctx: dict) -> str:
    prof = ctx.get("user_profile", {}) or {}
    keys = ("age", "country_name", "gender", "preferred_language", "preferred_musical_culture")
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


def build_recent_music_block(
    context: dict,
    track_meta: dict,
    *,
    max_items: int = 5,
    tag_limit: int = 5,
) -> str:
    music_turns = [
        c for c in context.get("chat_history", [])
        if c.get("role") == "music" and c.get("content")
    ]
    if not music_turns:
        return "  (no prior music turns)"

    selected = music_turns[-max_items:]
    lines: list[str] = []
    for c in selected:
        tid = str(c.get("content"))
        md = track_meta.get(tid, {})
        tn = ", ".join(md.get("track_name") or []) or "?"
        an = ", ".join(md.get("artist_name") or []) or "?"
        al = ", ".join(md.get("album_name") or []) or "?"
        tags = ", ".join((md.get("tag_list") or [])[:tag_limit])
        rd = str(md.get("release_date") or "")
        year = rd[:4] if len(rd) >= 4 and rd[:4].isdigit() else ""
        bits = [f'"{tn}" by {an}', f"album: {al}"]
        if year:
            bits.append(f"released: {year}")
        if tags:
            bits.append(f"tags: {tags}")
        lines.append("  - " + "; ".join(bits))
    return "\n".join(lines)


def _load_model_and_tokenizer(model_name: str, quantization: str | None, device: str):
    """Load the responder LLM in bf16 on a single GPU.

    The submitted system runs Qwen3.6-27B unquantized (~54GB weights → 80GB-class
    GPU); `quantization` must be None.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if quantization is not None:
        raise ValueError(
            f"quantization={quantization!r} is not supported in this repository "
            "(the submitted system runs bf16 unquantized)"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).eval()
    model = model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def run_standard_responder(responder_name: str, here: Path) -> None:
    """Standard flow: parse args, load YAML, generate responses, save.

    YAML schema (required keys):
      retriever, model, top_k, max_new_tokens, temperature, top_p, seed, prompt_template
    optional: quantization ("bnb_4bit" | "bnb_8bit" | null)
    """
    import torch
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--target", choices=("blind_a", "blind_b", "devset"), default="blind_a"
    )
    parser.add_argument("--max_records", type=int, default=None)
    args = parser.parse_args()

    started_at = time.time()
    config = load_yaml(here / "configs" / f"{args.config}.yaml")
    project_root = here.parents[1]
    output_dir = project_root / "output" / "responder" / responder_name / args.config
    results_dir = project_root / "results" / "responder" / responder_name / args.config
    records, source_label = load_predictions_from_config(config, args.target)

    print(
        f"=== responder/{responder_name}/{args.config} target={args.target} "
        f"source={source_label} ==="
    )
    if args.max_records is not None:
        records = records[: args.max_records]
    print(f"  {len(records)} records to enrich")

    track_meta = load_track_meta()
    contexts = load_dataset_context(args.target)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantization = config.get("quantization")
    print(
        f"loading {config['model']} on {device} "
        f"(quantization={quantization or 'bf16'}) ..."
    )
    tokenizer, model = _load_model_and_tokenizer(config["model"], quantization, device)
    torch.manual_seed(config.get("seed", 0))

    t0 = time.time()
    for rec in tqdm(records, desc="response"):
        ctx = contexts.get((rec["session_id"], rec["turn_number"]), {})
        thought = ctx.get("thought", "")
        thought_block = f"User's inner thought: {thought}" if thought else ""
        # 全 placeholder を渡す. 各 config の template は使うものだけ書く.
        track_tag_limit = int(config.get("track_tag_limit", 3))
        prompt = config["prompt_template"].format(
            chat_history=build_chat_history(ctx, track_meta),
            user_query=ctx.get("user_query", ""),
            thought_block=thought_block,
            user_profile_block=build_user_profile_block(ctx),
            goal_block=build_goal_block(ctx),
            gpa_block=build_gpa_block(ctx),
            recent_music_block=build_recent_music_block(
                ctx,
                track_meta,
                max_items=int(config.get("recent_music_k", 5)),
                tag_limit=int(config.get("recent_music_tag_limit", 5)),
            ),
            top_tracks=build_top_tracks(
                rec["predicted_track_ids"], track_meta, config["top_k"]
            ) if track_tag_limit == 3 else build_top_tracks_with_options(
                rec["predicted_track_ids"],
                track_meta,
                config["top_k"],
                tag_limit=track_tag_limit,
            ),
            top_k=config["top_k"],
        )
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **config.get("chat_template_kwargs", {}),
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                do_sample=config["temperature"] > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
        ).strip()
        # 一部 tokenizer (DeepSeek-R1 distill 等) では GPT-2 byte-level BPE 記号が残る.
        text = fix_byte_bpe_artifacts(text)
        # Strip <think>...</think> blocks (R1 distill / Qwen3 thinking mode 等).
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1].strip()
        rec["predicted_response"] = text.strip()

    elapsed = time.time() - t0
    print(f"generated in {elapsed:.1f}s ({elapsed / max(len(records), 1):.2f}s/record)")
    save_predictions(records, output_dir, args.target)
    save_manifest(
        output_dir,
        target=args.target,
        responder_name=responder_name,
        config_name=args.config,
        config=config,
        source_label=source_label,
        argv=sys.argv,
        started_at=started_at,
        elapsed=elapsed,
        n_records=len(records),
    )
    print(f"wrote {output_dir / f'{args.target}.submission.zip'}")
    append_run_log(results_dir, started_at, elapsed, "success", config, sys.argv)
