# Fold handling — retriever / reranker / responder × CV vs submission

This document states exactly how cross-validation folds are used at each stage, and how that differs between **local CV** (measuring nDCG on public labeled rows) and the **Blind-B submission**. The single guarantee behind all of it: *no component that scores or trains on a public row uses a fitted signal that was derived from that same row.*

Fold assignments come from `artifacts/cache/splits/cv5` (`public_labeled_v2_5fold`, 5 folds, seed 20260515). The tfidf source is fit-free — its candidates are label-free — so it enumerates the same public rows but its output does not depend on the fold split.

## Summary table

| stage | component kind | local CV (public_labeled) | Blind-B submission |
| --- | --- | --- | --- |
| retriever | fit-free (bm25, tfidf, tag_intent, history_*, last_*, exact_*) | one `fit_free_all_rows` artifact, no folds | same artifact, applied to blind_b |
| retriever | supervised / statistical (two_tower, cooc_track/album/artist, transition) | `cv5_oof`: per fold, fit on the other 4 folds → candidates for the held-out fold | `full_public`: fit on all public rows → candidates for blind_b |
| reranker | LightGBM LambdaRank | 5-fold: train on `folds != f`, predict `folds == f`; concat → OOF nDCG | one model fit on all public rows → predict blind_b (shipped `model.txt`) |
| responder | Qwen3.6-27B generation | not run (CV scores rankings only) | run once on the blind_b ranking |

## Retriever

Two classes of source, distinguished by whether they fit anything on labels.

**Fit-free sources** — `bm25_5field_thought`, `protocol_tfidf_lgbm_k300`, `tag_intent_bm25`, `history_artist`, `history_album`, `last_music_artist`, `last_music_album`, `exact_album_artist_source`, `exact_title_artist_source`. These are pure functions of the query row + track catalog; they use **no labels**, so there is nothing to hold out. They write a single `fit_free_all_rows` artifact used identically for CV train rows and for blind_b. (The tfidf source additionally fits a per-fold LightGBM on the cv3 split, but that model is *not* consumed by the union — the union only reads the label-free TF-IDF `candidates.npz`. cv3 is a load-bearing default of that runner, not a second CV protocol.)

**Supervised / statistical sources** — `two_tower_lora_thought` (LoRA dense retriever), `cooc_track_combined_tpd1`, `transition_track_combined_tpd1`, `cooc_album`, `cooc_artist_name`. These fit on public labels (a trained model, or co-occurrence / transition counts), so they must be held out:

- **Local CV → `cv5_oof`.** For each fold `f`, the source is fit on the rows where `fold != f` and produces candidates only for the rows where `fold == f`. Concatenating the five held-out slices gives OOF candidates that cover every public row, where **no row's candidates were produced by a model or a count table that saw that row**. Concretely:
  - two-tower: a separate LoRA model is trained per fold on the non-fold rows (`build_two_tower_lora_oof.py`, fold loop: `train_rows = folds != f`, `valid_rows = folds == f`), then used to encode and retrieve the held-out queries → `models/fold{0..4}.pt` + `folds/fold{f}.npz`.
  - cooc / transition: counts are built from the non-fold public rows only.
  - TalkPlayData-1 (external) counts are added to **all** folds identically — they are not public labels, so they carry no fold leakage.
- **Blind-B → `full_public`.** The source is fit on **all** public rows (one two-tower model `models/full_public.pt`; cooc/transition counts from all public rows + TPD1) and applied to the 80 blind_b rows. Using all public data here is safe because blind_b rows are disjoint from public rows.

## Reranker (LightGBM LambdaRank)

The reranker consumes the union of the retriever candidates and their per-source scores as features. Its fold ids come from the `folds` array baked into the union `candidates.npz` (which is derived from `splits/cv5`).

- **Local CV → 5-fold OOF.** For each fold `f` (`run_reranker.py`, `valid_rows = folds == f`, `train_rows = folds != f`):
  1. fit the model on `train_rows`, whose retriever features come from the **`cv5_oof`** artifacts (so each training row's supervised-source signal already excluded that row);
  2. predict `valid_rows`. Concatenating the five held-out prediction slices gives the OOF ranking used for the reported metrics (`ndcg@20 = 0.2743`, folds 0.27674 / 0.27546 / 0.27405 / 0.27212 / 0.27321). The double hold-out — OOF retriever features *and* OOF reranker prediction — is what keeps the CV number honest.
- **Blind-B → one full-public model.** A single model is fit on **all** public rows (positive-target filter, below) using their **`cv5_oof`** retriever features — the same OOF features the CV fold models fit on, so no training row sees its own supervised-source signal — and then ranks the blind_b rows from the **`full_public`** blind_b union. So the fit still reads the public (`cv5_oof`) union; only the prediction targets the `full_public` blind_b artifacts. This is the shipped `weights/reranker_lgbm.txt`; `--load-model` loads it and skips the fit, so Blind-B inference reproduces the submitted top-20 bit-exactly.

**Positive-target training filter (orthogonal to folds).** Blind-A/B evaluation is skewed to the positive-current distribution, so the reranker trains only on rows in that distribution (`train_positive_only`: rows whose gold track is present in the candidate pool). This selects *which rows* train the model; it does not change the fold mechanics above. It applies to both the CV per-fold fits and the full-public submission fit.

## Responder (Qwen3.6-27B)

The responder has **no folds and no fit**. It reads the reranker's top-k tracks for a row and generates natural-language text; it never touches labels or fold assignments.

- **Local CV**: the responder is **not run**. CV measures ranking quality (nDCG@20 on the reranker OOF output); the generated text is not part of the scored metric.
- **Blind-B submission**: run once on the `full_public` blind_b ranking. For each of the 80 rows it produces 10 seeded generations (`torch.manual_seed(seed + run)`, runs 0..9) and selects the set that maximizes corpus lexical diversity (`select_diverse`, seeded `random.Random(seed + 1000 + trial)`, 30 trials). Fold-independent by construction.

## What lives where (fit scope recap)

| artifact mode | fit scope | used by |
| --- | --- | --- |
| `fit_free_all_rows` | none (label-free) | CV + submission (fit-free sources) |
| `cv5_oof` | per fold, other 4 folds | reranker CV **train rows** |
| `full_public` | all public rows | reranker **submission** fit + blind_b |

The reranker's CV train rows must read `cv5_oof` (never `full_public`) or the supervised sources would leak in-sample. The Blind-B path reads `full_public` throughout. These are the same retriever/reranker/responder components with different fit-scope artifacts — not different pipelines.

## Which build produces which artifact (CV vs submission dependency)

The `cv5_oof` (CV) and `full_public` (submission) artifacts come from the same source logic run for different targets; the per-stage drivers group them by target:

| stage script | produces | consumed by |
| --- | --- | --- |
| `run_preprocess.sh` | `splits/cv5`, `spotify_uuid_map`, `dense_track_emb` | every source build |
| `run_retriever_cv5.sh` | public-labeled sources — `fit_free_all_rows` (bm25 / tfidf / tag / history / last / exact) + `cv5_oof` (two-tower fold0-4, cooc/transition) — and the **public union** | reranker CV **and** the reranker submission fit |
| `run_retriever_blind_b.sh` | blind_b sources — `fit_free_all_rows` + `full_public` (two-tower full, cooc/transition) — and the **blind_b union** | reranker submission |
| `run_reranker_cv5.sh` | 5-fold OOF ranking + nDCG scores | the CV metric (not the submission) |
| `run_reranker_blind_b.sh` | the submitted ranking (fits the final model on the public union, ranks blind_b) | responder |

`run_full.sh` chains preprocess → retriever_cv5 → retriever_blind_b → reranker_blind_b → responder (the submission path). `run_reranker_blind_b.sh` needs **both** retriever stages: the public union (`cv5_oof`) for the final-model fit, and the blind_b union (`full_public`) for the prediction. `run_inference.sh` skips all of it by loading the shipped model + the shipped blind_b union artifact.
