#!/usr/bin/env python3
"""Generate the final Qwen responses and select a high-diversity ensemble.

This is a local responder packaging script. It never submits to Codabench.
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from typing import Any

import torch
from tqdm import tqdm

from responder.qwen36_27b import ensemble
from responder.qwen36_27b.component import (
    build_chat_history,
    build_goal_block,
    build_gpa_block,
    build_top_tracks,
    build_user_profile_block,
    fix_byte_bpe_artifacts,
    load_dataset_context,
    load_model_and_tokenizer,
    load_predictions_from_config,
    load_track_meta,
    load_yaml,
)

from recsys2026.artifacts import (
    artifact_complete,
    component_output_dir,
    file_ref,
    json_dump,
)
from recsys2026.paths import REPO_ROOT
from recsys2026.submission import (
    validate_predictions,
    write_predictions,
    zip_submission,
)


def render_prompt(
    config: dict[str, Any],
    rec: dict[str, Any],
    ctx: dict[str, Any],
    track_meta: dict[str, dict],
) -> str:
    thought = ctx.get("thought", "")
    thought_block = f"User's inner thought: {thought}" if thought else ""
    top_tracks = build_top_tracks(
        rec["predicted_track_ids"], track_meta, config["top_k"]
    )
    return config["prompt_template"].format(
        chat_history=build_chat_history(ctx, track_meta),
        user_query=ctx.get("user_query", ""),
        thought_block=thought_block,
        user_profile_block=build_user_profile_block(ctx),
        goal_block=build_goal_block(ctx),
        gpa_block=build_gpa_block(ctx),
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
    argparse.ArgumentParser(description=__doc__).parse_args()
    target = "blind_b"
    base_config = "default"
    started_at = time.time()
    config_path = (
        REPO_ROOT / "responder" / "qwen36_27b" / "configs" / f"{base_config}.yaml"
    )
    config = load_yaml(config_path)
    ranked_artifact = REPO_ROOT / str(config["ranked_artifact"]).format(target=target)
    out_dir = component_output_dir(
        "responder", "qwen36_27b", base_config, target=target
    )
    json_path = out_dir / "prediction.json"
    zip_path = out_dir / "submission.zip"
    if artifact_complete(out_dir, json_path.name, zip_path.name):
        try:
            existing = json.loads(json_path.read_text())
            validate_predictions(existing, target)
            with zipfile.ZipFile(zip_path) as archive:
                if archive.namelist() != ["prediction.json"]:
                    raise ValueError(
                        "submission archive must contain prediction.json only"
                    )
                if archive.read("prediction.json") != json_path.read_bytes():
                    raise ValueError(
                        "submission archive does not match prediction JSON"
                    )
            print(f"[skip] complete responder artifact {out_dir}")
            return
        except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
            print(f"[rebuild] invalid responder artifact {out_dir}: {exc}")
    (out_dir / "manifest.json").unlink(missing_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    n_runs = int(config["n_runs"])
    seed = int(config["seed"])
    n_random_orders = int(config["n_random_orders"])
    records, source_label = load_predictions_from_config(config, target)
    track_meta = load_track_meta()
    contexts = load_dataset_context(target)

    all_runs: list[list[dict[str, Any]]] = []
    tokenizer = None
    model = None
    for run_i in range(n_runs):
        run_seed = seed + run_i
        run_path = runs_dir / f"run{run_i:02d}.json"
        run_records = None
        if run_path.exists():
            try:
                loaded = json.loads(run_path.read_text())
                validate_predictions(loaded, target)
                run_records = loaded
                print(f"[skip] complete responder run {run_path}")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                print(f"[rebuild] invalid responder run {run_path}: {exc}")
        if run_records is None:
            if model is None or tokenizer is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tokenizer, model = load_model_and_tokenizer(config["model"], device)
            run_records = generate_run(
                records=records,
                config=config,
                contexts=contexts,
                track_meta=track_meta,
                tokenizer=tokenizer,
                model=model,
                seed=run_seed,
            )
            write_predictions(run_records, run_path, target)
        all_runs.append(run_records)

    keys = [(r["session_id"], r["turn_number"]) for r in records]
    candidates_per_record: list[list[dict[str, Any]]] = []
    for rec_i, key in enumerate(keys):
        cands = []
        for run_i, run_records in enumerate(all_runs):
            r = run_records[rec_i]
            if (r["session_id"], r["turn_number"]) != key:
                raise ValueError(f"run{run_i:02d} key mismatch at row {rec_i}")
            cands.append({**r, "_source": f"qwen36_seed{seed + run_i}"})
        candidates_per_record.append(cands)

    best_records = None
    best_meta = None
    best_score = float("-inf")
    best_trial = -1
    for trial in range(n_random_orders):
        selected, meta = ensemble.select_diverse(
            candidates_per_record,
            seed=seed + 1000 + trial,
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
        source_counts[str(rec["_source"])] = (
            source_counts.get(str(rec["_source"]), 0) + 1
        )
    output_records = [
        {k: v for k, v in rec.items() if k != "_source"} for rec in best_records
    ]
    write_predictions(output_records, json_path, target)
    zip_path = zip_submission(json_path, zip_path)

    elapsed = time.time() - started_at
    manifest = {
        "schema_version": 1,
        "artifact_type": "predictions",
        "stage": "responder",
        "name": "qwen36_27b",
        "config": base_config,
        "target": target,
        "base_config": base_config,
        "base_config_file": file_ref(config_path),
        "ranked_artifact": str(ranked_artifact.relative_to(REPO_ROOT)),
        "source": source_label,
        "n_runs": n_runs,
        "run_seeds": [seed + i for i in range(n_runs)],
        "selection": {
            "objective": "lexdiv",
            "n_random_orders": n_random_orders,
            "best_trial": int(best_trial),
            "meta": best_meta,
            "source_counts": source_counts,
        },
        "elapsed_sec": elapsed,
        "outputs": {
            "json": str(json_path.relative_to(REPO_ROOT)),
            "zip": str(zip_path.relative_to(REPO_ROOT)),
            "runs_dir": str(runs_dir.relative_to(REPO_ROOT)),
        },
    }
    json_dump(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest["outputs"], indent=2))
    print(json.dumps(manifest["selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
