# recsys2026-niwatori

Validation repository for our **RecSys Challenge 2026 (Music CRS)** Top-5 entry. It contains exactly one system — the one behind our final Blind-B submission:

- **Blind-B Codabench score ≈ 0.59** (overall; breakdown not published)
- **Local 5-fold nDCG@20 = 0.2743** on public labeled rows (129,592 rows)
- config name: `blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5`

```text
Blind-B input
  -> retriever/union            14 candidate sources, ordered_unique merge
  -> reranker (LightGBM)        LambdaRank, 200 trees, 176 features -> top 20
  -> responder (Qwen3.6-27B)    10 seeded generations -> lexical-diversity pick
  -> prediction.json (submission zip)
```

Method details: [`docs/method.md`](docs/method.md).

## What you can run

| flow | entrypoint | needs | wall time (measured/estimated) |
| --- | --- | --- | --- |
| 1. Blind-B inference (load-only) | `run_inference.sh` | shipped weights + union artifact + dense caches; **no GPU for the ranking**, 80GB GPU only for the responder | rank ≈ 10 min CPU; responder ≈ 1.5–2 h GPU |
| 2. 5-fold CV validation | `run_cv5.sh` | **public sources built first** (run_preprocess + build_stage1/2 — GPU for two-tower); ≥128GB RAM | source builds + several hours CPU (refits 5 fold models) |
| 3. Train from scratch | `run_full.sh` | full downloads; GPU (two-tower) + ≥128GB RAM | ≈ half a day |

The ranking part of flow 1 is **deterministic and CPU-only**: the reranker loads the shipped LightGBM model (`weights/reranker_lgbm.txt`, `--load-model`) and predicts over the shipped union artifact + dense caches. On the pinned environment (`uv.lock`) this reproduces the submitted Blind-B top-20 **bit-for-bit** (`verify_blind_b_ranking.py --strict` → 80/80); the default check is a soft top-20 overlap report, which is what the evaluation relies on (regenerating anything on a GPU, or a different library build, is not bit-reproducible). Responder text is seeded but GPU-nondeterministic; the submitted responses are in `reference/blind_b.json` for diffing (track lists must match; wording may not be bit-identical).

## Hardware

- **Responder**: single ~80GB GPU (A100/H100 80GB class). Qwen3.6-27B in bf16, unquantized (~54GB weights). 10 runs × 80 rows × 200 new tokens.
- **Reranker training / CV**: CPU, **≥128GB RAM** (peak ~96GB: full candidate pool ≈ 850 candidates/row × 129,592 rows × 176 features), all cores. *Load-only inference does not need this* (a few GB).
- **Two-tower LoRA training**: single 16–24GB GPU, bf16, r=16, 2 epochs.
- Everything else (bm25/tfidf/cooc/union/splits/map): CPU, moderate RAM.

## Environment

- Python **3.12**, env manager **uv**: `uv sync` creates `.venv/` from the committed `uv.lock` (exact pins: torch 2.11.0, transformers 5.7.0, lightgbm 4.6.0, numpy 2.3.5, datasets 4.8.5, scikit-learn 1.8.0, …).
- The responder uses `transformers` (no vLLM, no quantization libraries).
- CUDA-capable PyTorch build is assumed for the GPU steps.

## Download

```bash
export HF_TOKEN=...            # access to the gated talkpl-ai challenge repos
bash download_datasets.sh      # challenge datasets + TalkPlayData-1/2 + Qwen models -> HF cache
bash download_weights.sh       # our weights + union artifact + dense caches (~0.3GB) -> artifacts/
export HF_HUB_OFFLINE=1        # all further runs are offline
```

`download_datasets.sh` materializes everything through the `datasets` / `huggingface_hub` caches (`HF_HOME`), so all pipeline steps run with `HF_HUB_OFFLINE=1` afterwards. External inputs are referenced by ID only and never re-uploaded:

- gated challenge repos: `talkpl-ai/TalkPlayData-Challenge-{Dataset,Track-Metadata,User-Metadata,Track-Embeddings,User-Embeddings,Blind-B}`
- external data: `talkpl-ai/TalkPlayData-1` (train split), `talkpl-ai/TalkPlayData-2`
- pretrained: `Qwen/Qwen3.6-27B`, `Qwen/Qwen3-Embedding-0.6B`

## Quickstart: Blind-B inference (flow 1)

```bash
bash run_inference.sh
```

Steps (see the script):

1. **reranker** `--load-model artifacts/weights/reranker_lgbm.txt` builds the Blind-B feature matrix from the shipped union artifact and predicts with the shipped model — no fitting, no GPU. The feature layout is checked against the 176 feature names stored inside the model before predicting.
2. **responder** generates 10 seeded responses per row with Qwen3.6-27B and picks the lexical-diversity-optimal set (GPU).

Check the ranking against the submission:

```bash
uv run python scripts/verify_blind_b_ranking.py            # top-20 overlap report vs the submission
# on the pinned environment (uv.lock) the load-only path also passes the exact check:
uv run python scripts/verify_blind_b_ranking.py --strict   # -> PASS (80/80 identical)
```

Expected outputs under `artifacts/runs/`:

```text
reranker/protocol_098_union_rich_lgbm/.../full_public/blind_b/{ranked.npz,model.txt,manifest.json}
responder/qwen36_10run_diverse/.../blind_b.json + blind_b.submission.zip   (member: prediction.json)
```

No GPU? Replace step C with the canned-text packager (identical track IDs):

```bash
uv run python scripts/build_ranked_submission.py \
  --ranked-artifact artifacts/runs/reranker/protocol_098_union_rich_lgbm/blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5/full_public/blind_b \
  --target blind_b --out-dir artifacts/runs/pipeline/blind_b
```

## 5-fold CV validation (flow 2)

```bash
bash run_cv5.sh
```

**Not a load-only flow.** `run_cv5.sh` chains `run_preprocess.sh` → `run_retriever_cv5.sh` (builds the public-labeled sources, including two-tower training on a GPU) → `run_reranker_cv5.sh` (fits the reranker per fold over the public union). Self-contained but heavy: hours of CPU, a GPU for the two-tower, and ≥128GB RAM for the reranker fit. Expected `scores.json`:

- `ndcg@20 ≈ 0.2743` — folds ≈ 0.27674 / 0.27546 / 0.27405 / 0.27212 / 0.27321
- `candidate_recall@20 = 0.410234`

Last-digit wobble is expected (LightGBM histogram nondeterminism with `n_jobs=-1`); the shipped reference numbers are in `docs/method.md` and were produced with seed 20260520.

## Train from scratch (flow 3)

```bash
bash run_full.sh          # preprocess -> retrievers -> reranker (fit) -> responder -> submission
```

`run_full.sh` runs the submission path from scratch — **no HF weights**: the two-tower is trained, the reranker is fit. Stages (each script documents its own dependencies in its header):

- `run_preprocess.sh` — cv5 split, TPD1→catalog map, dense track embeddings.
- `run_retriever_cv5.sh` — public_labeled sources (two-tower 5-fold OOF trained; cooc/transition incl. TalkPlayData-1) + public union.
- `run_retriever_blind_b.sh` — blind_b sources (two-tower full-public trained; cooc/transition) + blind_b union.
- `run_reranker_blind_b.sh` — fit the final reranker on the public union, rank blind_b (needs **both** retriever stages: cv5 for the fit, blind_b for the prediction).
- `run_responder_blind_b.sh` — Qwen3.6-27B → submission zip.

The 5-fold CV is a separate side branch, not on the submission path: `run_reranker_cv5.sh` (reranker CV over an already-built public union), or `run_cv5.sh` (retriever + reranker CV).

`artifacts/cache/dense_track_emb.npz` (Qwen3-Embedding-0.6B encoding of track metadata) is shipped and reused rather than rebuilt — `preprocessing/dense_track_encoder.py` regenerates it if absent (GPU). Dense query features (`artifacts/cache/dense_qfeat/`) are re-encoded by the reranker automatically for any rows not in the cache (only `blind_b.npz` is shipped; train/devset are regenerated during training).

## Repository layout

```text
src/recsys2026/          shared library
  data/paths/artifacts/submission/eval/retrieval/splits/encoders  framework
  zoo.py                 source scoring lib (bm25/history/last/exact/cooc/tag)
  reranker_features.py   TrackIndex / FeatureEncoder / example builders
  reranker_protocol.py   candidate/dense materialization + eval helpers
  fast_features.py       vectorized 098 base features (numpy/scipy)
  two_tower.py           two-tower LoRA model + track features
  responder_common.py / responder_ensemble.py
preprocessing/           build_splits (via scripts/), dense_track_encoder
retriever/<component>/   13 candidate sources + union (main.py + configs)
reranker/protocol_098_union_rich_lgbm/   config wrapper + configs
responder/qwen36_27b/    prompt-template config (rich_context_hierpop_tagchain)
scripts/                 cross-cutting python builders + the reranker runner
  run_reranker.py        the reranker runner (supports --load-model)
run_preprocess / run_retriever_{cv5,blind_b} / run_reranker_{cv5,blind_b} /
run_responder_blind_b / run_full   per-stage from-scratch drivers (repo root)
artifacts/               all generated/downloaded state (gitignored)
  weights/               reranker_lgbm.txt, two_tower/{full_public,fold0..4}.pt
  cache/                 derived caches (mirrors the HF dataset repo)
  runs/                  pipeline artifacts + predictions
reference/blind_b.json   the submitted responses (for diffing)
```

## HF dataset repo manifest

**Official load-only inference set** (~0.3GB, what `download_weights.sh` pulls — enough for `run_inference.sh` to reproduce the Blind-B top-20 bit-for-bit):

| path | size | role |
| --- | ---: | --- |
| `weights/reranker_lgbm.txt` | 1.5MB | final LightGBM model (loaded by `--load-model`) |
| `cache/dense_track_emb.npz` | 118MB | dense track embeddings (reranker `TrackIndex`) |
| `cache/dense_qfeat/blind_b.npz` | 0.2MB | Blind-B dense query features (reranker) |
| `cache/runs_seed/retriever/union/.../blind_b/` | 18MB | Blind-B union artifact — the reranker's candidate input |
| `cache/runs_seed/retriever/union/.../public_labeled/candidates.npz` | 164MB | public candidate ids/keys/folds for the reranker's feature-stack fit |

**Also on the repo** (for the auxiliary `run_blind_b.sh` and full training — not pulled by the load-only download):

| path | size | role |
| --- | ---: | --- |
| `weights/two_tower/{full_public,fold0..4}.pt` | 6×44MB | trained two-tower LoRA models — loaded by `build_two_tower_lora_oof.py --load-models-dir` (used by `run_blind_b.sh`) to regenerate the two-tower candidates without retraining |

Everything else is re-derivable and **not needed for the official reproduction**: the public-labeled per-source retriever artifacts, the union `source_features.npz` (36GB), `two_tower/track_features.npz`, the train/devset dense features, the CV splits, and the Spotify map. `run_blind_b.sh` / `run_train.sh` rebuild them locally.

## Reproducibility caveats

- **Ranking is deterministic** under `--load-model`: same inputs + shipped `model.txt` → identical top-20 (verified against the submission artifact).
- **Responder text is not bit-stable** across GPUs (sampling at temperature 0.7; all seeds fixed, but CUDA kernels are nondeterministic). Selection is exactly seeded. Compare against `reference/blind_b.json`: track lists match, wording may differ.
- **From-scratch training** reproduces scores to ~4th decimal (LightGBM `n_jobs=-1` histogram nondeterminism; bf16 GPU training for the two-tower).
- The feature vector is fixed-width; blind-B-safe fields are neutralized as *values*, never dropped as columns, so the shipped model's 176-feature layout always matches (asserted at load time).

## Compliance

- `track_emb.test_tracks` (target-side track set) is **not used** anywhere; the candidate universe is the full catalog / train-derivable sets only.
- No evaluation-split information is used to fit any component (TF-IDF fits on the track catalog only; supervised retrievers use 5-fold OOF for train-row features; the reranker trains on public labeled rows only).
- External data is limited to TalkPlayData-1/2 (permitted by the challenge).

## License / attribution

- Challenge data & TalkPlayData-1/2: © talkpl-ai, per their respective dataset licenses (accessed via Hugging Face; never redistributed here).
- Qwen3.6-27B and Qwen3-Embedding-0.6B: Qwen model licenses (referenced by ID).
- Code in this repository: MIT (see `LICENSE`).
