# Final Blind-B System — Method Description

## Overview

A three-stage pipeline:

```text
Blind-B input
  -> retriever/union            14 candidate sources merged with ordered_unique
  -> reranker (LightGBM)        LambdaRank over the full union pool -> top 20
  -> responder (Qwen3.6-27B)    10 seeded generations -> lexical-diversity selection
```

Missing optional Blind-B context remains unavailable. The exact-match retrievers and responder consume those fields only when supplied.

## Retriever (union of 14 sources)

Defined by `retriever/union/configs/combined_tpd1_parts_cooc500_cv5.yaml`. Candidates are concatenated in source order with `ordered_unique`; no global cap. Per-source scores/ranks are kept as reranker features.

| source | component / config | cap | fit scope | signal |
| --- | --- | ---: | --- | --- |
| `bm25` | `bm25_5field/top500` | 200 | fit-free | BM25 between safe query/context text and 5-field track metadata |
| `tfidf` | `tfidf_catalog/top300` | 200 | track metadata only | TF-IDF lexical retrieval (vectorizer fit on the catalog only) |
| `twotower` | `two_tower_lora/oof5_top500` | 200 | public labeled query–gold pairs | supervised dense retriever (LoRA on Qwen3-Embedding-0.6B query tower; track tower from official embeddings) |
| `history_artist` | `history_artist/top500` | src top500 | fit-free | tracks by artists in the listening history |
| `history_album` | `history_album/top500` | src top500 | fit-free | tracks from albums in the history |
| `last_artist` | `last_music_artist/top500` | src top500 | fit-free | artist continuation from the last music turn |
| `last_album` | `last_music_album/top500` | src top500 | fit-free | album continuation from the last music turn |
| `exact_album_artist` | `exact_album_artist_source/top500` | src top500 | fit-free | album/artist surface match in the current message and available thought |
| `tag_intent` | `tag_intent_bm25/top500` | 100 | fit-free | genre/mood/descriptor intent-tag match |
| `cooc_track` | `cooc_track_combined_tpd1/oof5_top500_parts` | 500 | public OOF + TPD1 | tracks co-occurring with history tracks |
| `transition_track` | `transition_track_combined_tpd1/oof5_top500_prob_parts` | src top500 | public OOF + TPD1 | Markov next-track counts from the last music track |
| `cooc_album` | `cooc_album/oof5_top500` | 200 | public OOF | album co-occurrence (`score__primary >= 5`) |
| `cooc_artist_name` | `cooc_artist_name/oof5_top500` | 100 | public OOF | artist-name co-occurrence |
| `exact_title` | `exact_title_artist_source/top500` | src top500 | fit-free | title/artist surface match in the current message and available thought |

### How TalkPlayData-1 is used

TPD1 is not added as an independent source; its counts are **added into** the `cooc_track` / `transition_track` statistics. External track IDs are mapped to the challenge catalog via `artifacts/preprocessed/catalog_id_map.parquet` (built from TalkPlayData-2); unmapped tracks are dropped, so the candidate universe stays inside the challenge catalog. Per-candidate parts are stored as `score__challenge` / `score__tpd1` (+ `score__transition_probability`).

TPD1 counts are added identically to all folds. Artifact fit scope is defined in [`folds.md`](folds.md).

### Union size / candidate recall (5-fold OOF)

| target | rows | mean cands | median | p90 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| public_labeled | 129,592 | 853.4 | 807 | 1220 | 266 | 1835 |
| blind_b | 80 | 669.0 | 644.5 | 797.7 | 419 | 1405 |

recall@20 = 0.4102, @50 = 0.5129, @100 = 0.5693, @200 = 0.6129.

## Reranker (LightGBM LambdaRank, 176 features)

Defined by `reranker/union_lambdarank/configs/combined_tpd1_parts_cooc500_t200_cv5.yaml`. The full union pool is reranked and the top 20 emitted.

| parameter | value |
| --- | --- |
| objective | lambdarank |
| n_estimators / num_leaves | 200 / 63 |
| learning_rate | 0.04 |
| subsample / colsample_bytree | 0.85 / 0.85 |
| min_child_samples | 20 |
| seed | 20260520 |
| training rows | gold-in-pool rows only |
| primary score mode | zero |
| source features | enabled |

Feature groups (176 total): track/user/turn basics; history consistency (same artist/album/track, last-music match, tag overlap); query–metadata similarity (TF-IDF and dense cosine on Qwen3-Embedding vectors); metadata extensions; hierarchical popularity; tag-chain features; and per-source presence, rank, score, and support features, including the challenge/TPD1 score parts.

The `candidate_rank` family, prior-GPA counts, and `goal_track_tfidf_sim` are fixed to neutral values while retaining the submitted feature schema.

Five-fold OOF retriever artifacts supply reranker training features; Blind-B inference uses full-public retriever artifacts. See [`folds.md`](folds.md).

## Responder (`qwen36_27b`)

The base config is `responder/qwen36_27b/configs/default.yaml`. The top-3 ranked tracks and available conversation context are passed to **Qwen/Qwen3.6-27B** in bf16.

| parameter | value |
| --- | --- |
| top-k tracks in prompt | 3 |
| max_new_tokens / temperature / top_p | 200 / 0.7 / 0.9 |
| n_runs (seeds) | 10 (0..9) |
| per-row selection objective | unigram diversity + 0.5 x bigram diversity (greedy) |
| random-order trial selection | highest corpus-level unigram diversity |
| random-order trials | 30 (seeded) |
