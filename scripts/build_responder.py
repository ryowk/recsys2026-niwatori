#!/usr/bin/env python3
"""Generate 10 Qwen baseline responses and select a high-diversity ensemble.

This is a local responder packaging script. It never submits to Codabench.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from recsys2026.responder_common import (
    build_chat_history,
    build_goal_block,
    build_gpa_block,
    build_recent_music_block,
    build_top_tracks,
    build_top_tracks_with_options,
    build_user_profile_block,
    fix_byte_bpe_artifacts,
    load_dataset_context,
    load_predictions_from_config,
    load_track_meta,
    load_yaml,
    save_predictions,
)

from recsys2026.artifacts import file_ref, json_dump
from recsys2026.paths import REPO_ROOT


def load_ensemble_module() -> Any:
    from recsys2026 import responder_ensemble

    return responder_ensemble


def load_model_and_tokenizer(model_name: str, quantization: str | None, device: str):
    from recsys2026.responder_common import _load_model_and_tokenizer  # noqa: PLC0415

    return _load_model_and_tokenizer(model_name, quantization, device)


def render_prompt(config: dict[str, Any], rec: dict[str, Any], ctx: dict[str, Any], track_meta: dict[str, dict]) -> str:
    thought = ctx.get("thought", "")
    thought_block = f"User's inner thought: {thought}" if thought else ""
    track_tag_limit = int(config.get("track_tag_limit", 3))
    top_tracks = (
        build_top_tracks(rec["predicted_track_ids"], track_meta, config["top_k"])
        if track_tag_limit == 3
        else build_top_tracks_with_options(
            rec["predicted_track_ids"],
            track_meta,
            config["top_k"],
            tag_limit=track_tag_limit,
        )
    )
    return config["prompt_template"].format(
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
        top_tracks=top_tracks,
        top_k=config["top_k"],
    )


def generate_run(
    *,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    contexts: dict[tuple[str, int], dict[str, Any]],
    track_meta: dict[str, dict],
    tokenizer: Any,
    model: Any,
    seed: int,
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    out_records = [dict(rec) for rec in records]
    for rec in tqdm(out_records, desc=f"response seed={seed}"):
        ctx = contexts.get((rec["session_id"], rec["turn_number"]), {})
        prompt = render_prompt(config, rec, ctx, track_meta)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **config.get("chat_template_kwargs", {}),
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                do_sample=config["temperature"] > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            generated[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip()
        response = fix_byte_bpe_artifacts(response)
        if "</think>" in response:
            response = response.rsplit("</think>", 1)[-1].strip()
        rec["predicted_response"] = response.strip()
    return out_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="rich_context_hierpop_tagchain")
    parser.add_argument("--ranked-artifact", type=Path, required=True)
    parser.add_argument("--target", choices=("blind_a", "blind_b", "devset"), default="blind_b")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection-objective", choices=("lexdiv", "efficiency", "raw_unique"), default="lexdiv")
    parser.add_argument("--n-random-orders", type=int, default=30)
    args = parser.parse_args()

    started_at = time.time()
    ranked_artifact = args.ranked_artifact if args.ranked_artifact.is_absolute() else REPO_ROOT / args.ranked_artifact
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    config_path = REPO_ROOT / "responder" / "qwen36_27b" / "configs" / f"{args.base_config}.yaml"
    config = load_yaml(config_path)
    config = {**config, "ranked_artifact": str(ranked_artifact.relative_to(REPO_ROOT))}
    config.pop("retriever", None)
    records, source_label = load_predictions_from_config(config, args.target)
    track_meta = load_track_meta()
    contexts = load_dataset_context(args.target)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model_and_tokenizer(config["model"], config.get("quantization"), device)
    all_runs: list[list[dict[str, Any]]] = []
    for run_i in range(args.n_runs):
        run_seed = args.seed + run_i
        run_records = generate_run(
            records=records,
            config=config,
            contexts=contexts,
            track_meta=track_meta,
            tokenizer=tokenizer,
            model=model,
            seed=run_seed,
        )
        run_path = runs_dir / f"run{run_i:02d}.json"
        run_path.write_text(json.dumps(run_records, ensure_ascii=False))
        all_runs.append(run_records)

    keys = [(r["session_id"], r["turn_number"]) for r in records]
    candidates_per_record: list[list[dict[str, Any]]] = []
    for rec_i, key in enumerate(keys):
        cands = []
        for run_i, run_records in enumerate(all_runs):
            r = run_records[rec_i]
            if (r["session_id"], r["turn_number"]) != key:
                raise ValueError(f"run{run_i:02d} key mismatch at row {rec_i}")
            cands.append({**r, "_source": f"qwen36_seed{args.seed + run_i}"})
        candidates_per_record.append(cands)

    ens = load_ensemble_module()
    best_records = None
    best_meta = None
    best_score = float("-inf")
    best_trial = -1
    for trial in range(args.n_random_orders):
        selected, meta = ens.select_diverse(
            candidates_per_record,
            seed=args.seed + 1000 + trial,
            objective=args.selection_objective,
        )
        score = float(meta["lexdiv_unigram"])
        if score > best_score:
            best_score = score
            best_meta = meta
            best_records = selected
            best_trial = trial
    if best_records is None or best_meta is None:
        raise RuntimeError("ensemble selection failed")

    source_counts: dict[str, int] = {}
    for rec in best_records:
        source_counts[str(rec["_source"])] = source_counts.get(str(rec["_source"]), 0) + 1
    output_records = [{k: v for k, v in rec.items() if k != "_source"} for rec in best_records]
    zip_path = save_predictions(output_records, out_dir, args.target)

    elapsed = time.time() - started_at
    manifest = {
        "schema_version": 1,
        "artifact_type": "predictions",
        "stage": "responder",
        "name": "qwen36_10run_diverse",
        "target": args.target,
        "base_config": args.base_config,
        "base_config_file": file_ref(config_path),
        "ranked_artifact": str(ranked_artifact.relative_to(REPO_ROOT)),
        "source": source_label,
        "n_runs": int(args.n_runs),
        "run_seeds": [args.seed + i for i in range(args.n_runs)],
        "selection": {
            "objective": args.selection_objective,
            "n_random_orders": int(args.n_random_orders),
            "best_trial": int(best_trial),
            "meta": best_meta,
            "source_counts": source_counts,
        },
        "elapsed_sec": elapsed,
        "outputs": {
            "json": str((out_dir / f"{args.target}.json").relative_to(REPO_ROOT)),
            "zip": str(zip_path.relative_to(REPO_ROOT)),
            "runs_dir": str(runs_dir.relative_to(REPO_ROOT)),
        },
        "submission_note": "Local submission file only; not submitted to Codabench.",
    }
    json_dump(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest["outputs"], indent=2))
    print(json.dumps(manifest["selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
