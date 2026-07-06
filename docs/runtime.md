# Runtime & compute cost (from-scratch `run_full.sh`)

Wall-clock for one full from-scratch run — `run_full.sh` (submission path) plus the CV branch (`run_reranker_cv5.sh`) — on a single ~97GB GPU (RTX PRO 6000-class) and a 64-core / 251GB-RAM host, fully offline (challenge datasets, TalkPlayData-1/2, and Qwen in the HF cache). The two-tower is **trained** and the reranker is **fit** — no shipped weights are loaded.

| stage | script | wall-clock | notes |
| --- | --- | ---: | --- |
| preprocessing | `run_preprocess.sh` | ~5 min | cv5 split + Spotify→catalog map + dense track embeddings (GPU encode of 47k tracks) |
| retriever (public) | `run_retriever_cv5.sh` | ~5.75 h | two-tower LoRA 5-fold OOF **training** (the long pole) + cooc/transition (+TPD1) + public union |
| retriever (blind) | `run_retriever_blind_b.sh` | ~1.4 h | two-tower full-public **training** + cooc/transition + blind_b union |
| reranker CV | `run_reranker_cv5.sh` | ~2.8 h | dense query encode (train+devset, GPU) + 5-fold LightGBM fit over the full candidate pool |
| reranker submission | `run_reranker_blind_b.sh` | ~33 min | final LightGBM fit on all public rows + blind_b ranking |
| responder | `run_responder_blind_b.sh` | ~2 h | Qwen3.6-27B, 10 seeded generations × 80 rows + lexical-diversity selection (80GB GPU) |
| **total** | `run_full.sh` (+ CV branch) | **~12 h** | within the ~9–14 h estimate |

Notes:

- Two-tower training dominates (~7 h across the two retriever stages). The auxiliary `run_blind_b.sh` loads the shipped two-tower weights (`--load-models-dir`) and replaces training with an encode pass (minutes); the official `run_inference.sh` skips all retriever/reranker fitting by loading the shipped model + union artifact (~10 min CPU for the ranking, no GPU).
- The reranker fit needs **≥128GB RAM** (peak ~96GB with the full ~850-candidate pool over 129,592 rows × 176 features).
- Every stage is resumable (per-artifact skip guards), so a re-run continues from the first missing artifact — the two-tower training is not repeated once cached.

## Reproduced 5-fold CV (this run)

`run_reranker_cv5.sh` on the from-scratch public union:

| metric | this from-scratch run | reference (submission) |
| --- | ---: | ---: |
| nDCG@20 | 0.274136 | 0.2743 |
| nDCG@10 | 0.248909 | 0.2491 |
| candidate recall@20 | 0.410234 | 0.410234 |

Per-fold nDCG@20: 0.2756 / 0.2753 / 0.2739 / 0.2722 / 0.2737. The ~2e-4 gap to the reference is the expected wobble from bf16 two-tower retraining and LightGBM's `n_jobs=-1` histogram nondeterminism; the candidate recall is identical. Per-source retriever recall for the same run is in `docs/retriever_metrics.md`.
