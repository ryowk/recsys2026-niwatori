# recsys2026-niwatori

Validation repository for the pipeline that ranked **third overall** in the RecSys Challenge 2026 Music CRS task. The repository supports the final Blind-B submission and the Train-to-Devset evaluation reported in the paper:

- **Blind-B composite score = 0.5859** (third overall)
- **Blind-B nDCG@20 = 0.4934** (third on the ranking metric)
- config name: `combined_tpd1_parts_cooc500_t200_cv5`

Method details are in [`docs/method.md`](docs/method.md); fit scope is in [`docs/folds.md`](docs/folds.md).

## What you can run

| flow | entrypoint |
| --- | --- |
| Blind-B submission from scratch | `run_full.sh` |
| Paper Devset evaluation | `run_paper_devset.sh` |

See [`docs/paper_evaluation.md`](docs/paper_evaluation.md) for the paper evaluation and [`docs/runtime.md`](docs/runtime.md) for compute requirements.

## Setup

```bash
uv sync
```

## Download

```bash
bash download_datasets.sh      # challenge datasets + TalkPlayData-1/2 + Qwen models -> HF cache
```

`download_datasets.sh` materializes everything through the `datasets` / `huggingface_hub` caches (`HF_HOME`), so all pipeline steps run with `HF_HUB_OFFLINE=1` afterwards. External inputs are referenced by ID only and never re-uploaded:

- challenge dataset: `talkpl-ai/TalkPlayData-Challenge-{Dataset,Track-Metadata,Track-Embeddings,User-Embeddings,Blind-B}`
- external dataset: `talkpl-ai/TalkPlayData-1`, `talkpl-ai/TalkPlayData-2`
- pretrained: `Qwen/Qwen3.6-27B`, `Qwen/Qwen3-Embedding-0.6B`

## Blind-B submission from scratch (flow 1)

```bash
bash run_full.sh          # preprocess -> retrievers -> reranker (fit) -> responder -> submission
```

`run_full.sh` trains the task-specific two-tower and reranker locally and writes the submission ZIP under `artifacts/runs/responder/`.

## Repository layout

```text
src/recsys2026/          shared library
  data/paths/artifacts/submission/retrieval/splits/encoders  core utilities
  retriever_common.py    shared catalog/example/history primitives
  *_runner.py            shared fit/OOF/artifact runners
preprocessing/           build_splits (via scripts/), dense_track_encoder
retriever/<component>/   executable source logic + README/config
retriever/fit_free_sources.yaml   final fit-free source registry
reranker/union_lambdarank/   executable ranker/features + configs
responder/qwen36_27b/    executable prompt/generation/ensemble + config
scripts/                 preprocessing and paper-analysis utilities
run_preprocess.sh / run_retriever_{fit,blind_b}.sh /
run_reranker_blind_b.sh / run_responder_blind_b.sh / run_full.sh
                         per-stage Blind-B drivers (repo root)
artifacts/               locally generated state (gitignored)
  preprocessed/          splits, ID map, and embedding features
  results/               paper and component metrics
  runs/                  pipeline artifacts + predictions
```

## Compliance

- `track_emb.test_tracks` (target-side track set) is **not used** anywhere; the candidate universe is the full catalog / train-derivable sets only.

## License / attribution

- Challenge data & TalkPlayData-1/2: © talkpl-ai, per their respective dataset licenses (accessed via Hugging Face; never redistributed here).
- Qwen3.6-27B and Qwen3-Embedding-0.6B: Qwen model licenses (referenced by ID).
- Code in this repository: MIT (see `LICENSE`).
