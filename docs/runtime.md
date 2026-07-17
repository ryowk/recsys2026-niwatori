# Runtime & compute cost (from-scratch `run_full.sh`)

Wall-clock for one full from-scratch `run_full.sh` submission run on a single 96GB GPU and a 64-core / 256GB-RAM host, fully offline after input download. The two-tower and reranker are trained locally. A clean `run_paper_devset.sh` run took 9 hours 41 minutes on this host.

| stage | script | wall-clock | notes |
| --- | --- | ---: | --- |
| preprocessing | `run_preprocess.sh` | ~5 min | cv5 split + catalog ID map + dense track embeddings (GPU encode of 47k tracks) |
| retriever (fit) | `run_retriever_fit.sh` | ~5.75 h | two-tower LoRA 5-fold OOF **training** (the long pole) + cooc/transition (+TPD1) + public union |
| retriever (blind) | `run_retriever_blind_b.sh` | ~1.4 h | two-tower full-public **training** + cooc/transition + blind_b union |
| reranker submission | `run_reranker_blind_b.sh` | ~33 min | final LightGBM fit on all public rows + blind_b ranking |
| responder | `run_responder_blind_b.sh` | ~2 h | Qwen3.6-27B, 10 seeded generations × 80 rows + lexical-diversity selection (80GB GPU) |
| **total** | `run_full.sh` | **~9--10 h** | submission workflow only |

Notes:

- Two-tower training dominates (~7 h across the two retriever stages).
- The reranker fit needs **≥192GB RAM** (observed process peak ~148GB with the full candidate pool and 176 features).
- Every stage is resumable. A stage is complete only when its manifest and required files exist; responder generations additionally resume per seed.
