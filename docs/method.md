# Final Blind-B System — Method Description

This document describes the system that produced our final Blind-B submission (Codabench overall score ≈ **0.59**; local 5-fold nDCG@20 = **0.2743**), configuration name `blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5`.

## Overview

A three-stage pipeline:

```text
Blind-B input
  -> retriever/union            14 candidate sources merged with ordered_unique
  -> reranker (LightGBM)        LambdaRank over the full union pool -> top 20
  -> responder (Qwen3.6-27B)    10 seeded generations -> lexical-diversity selection
```

Blind B withholds `conversation_goal`, `goal_progress_assessments` and `thoughts` for some users. The system is therefore **blind-B-safe throughout**: those fields are never used in query construction or ranking features. The responder prompt template has slots for goal/thought/GPA; missing fields render as empty blocks and are never imputed.

External data: **TalkPlayData-1** (co-occurrence/transition statistics and two-tower training mix) and **TalkPlayData-2** (only to build the Spotify-ID → challenge-catalog map). LFM-2b listening histories are **not** used.

## Retriever (union of 14 sources)

Defined by `retriever/union/configs/blind_b_safe_combined_tpd1_parts_cooc500_cv5.yaml`. Candidates are concatenated in source order with `ordered_unique`; no global cap. Per-source scores/ranks are kept as reranker features.

| source | component / config | cap | fit scope | signal |
| --- | --- | ---: | --- | --- |
| `bm25` | `bm25_5field_thought/top500_bsafe` | 200 | fit-free | BM25 between safe query/context text and 5-field track metadata |
| `tfidf` | `protocol_tfidf_lgbm_k300/protocol_v1_bsafe` | 200 | track metadata only | TF-IDF lexical retrieval (vectorizer fit on the catalog only) |
| `twotower` | `two_tower_lora_thought/oof5_top500_bsafe` | 200 | public labeled query–gold pairs | supervised dense retriever (LoRA on Qwen3-Embedding-0.6B query tower; track tower from official embeddings) |
| `history_artist` | `history_artist/top500` | src top500 | fit-free | tracks by artists in the listening history |
| `history_album` | `history_album/top500` | src top500 | fit-free | tracks from albums in the history |
| `last_artist` | `last_music_artist/top500` | src top500 | fit-free | artist continuation from the last music turn |
| `last_album` | `last_music_album/top500` | src top500 | fit-free | album continuation from the last music turn |
| `exact_album_artist` | `exact_album_artist_source/top500` | src top500 | fit-free | album/artist surface match in the current text |
| `tag_intent` | `tag_intent_bm25/top500_bsafe` | 100 | fit-free | genre/mood/descriptor intent-tag match |
| `cooc_track` | `cooc_track_combined_tpd1/oof5_top500_parts` | 500 | public OOF + TPD1 | tracks co-occurring with history tracks |
| `transition_track` | `transition_track_combined_tpd1/oof5_top500_prob_parts` | src top500 | public OOF + TPD1 | Markov next-track counts from the last music track |
| `cooc_album` | `cooc_album/oof5_top500` | 200 | public OOF | album co-occurrence (`score__primary >= 5`) |
| `cooc_artist_name` | `cooc_artist_name/oof5_top500` | 100 | public OOF | artist-name co-occurrence |
| `exact_title` | `exact_title_artist_source/top500` | src top500 | fit-free | title/artist surface match in the current text |

### How TalkPlayData-1 is used

TPD1 is not added as an independent source; its counts are **added into** the `cooc_track` / `transition_track` statistics. TPD1 Spotify IDs are mapped to the challenge catalog via `artifacts/cache/spotify_uuid_map.parquet` (built from TalkPlayData-2); unmapped tracks are dropped, so the candidate universe stays inside the challenge catalog. Per-candidate parts are stored as `score__challenge` / `score__tpd1` (+ `score__transition_probability`).

For the local-CV train-row artifacts, challenge-side statistics are 5-fold OOF (the target row's fold is excluded from the fit); TPD1 counts are added identically to all folds as external data. For Blind-B artifacts, statistics are built from all public labeled rows + TPD1 and applied to Blind-B rows. TPD1 vs public-labeled full-sequence and full-conversation-text exact overlaps are 0.

### Union size / candidate recall (5-fold OOF)

| target | rows | mean cands | median | p90 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| public_labeled | 129,592 | 853.4 | 807 | 1220 | 266 | 1835 |
| blind_b | 80 | 669.0 | 644.5 | 797.7 | 419 | 1405 |

recall@20 = 0.4102, @50 = 0.5129, @100 = 0.5693, @200 = 0.6129.

## Reranker (LightGBM LambdaRank, 176 features)

Defined by `reranker/protocol_098_union_rich_lgbm/configs/blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5.yaml`. The full union pool is reranked (`max_candidates: all`) and the top 20 emitted.

| parameter | value |
| --- | --- |
| objective | lambdarank |
| n_estimators / num_leaves | 200 / 63 |
| learning_rate | 0.04 |
| subsample / colsample_bytree | 0.85 / 0.85 |
| min_child_samples | 20 |
| seed | 20260520 |
| training rows | positive-target rows only (gold present in candidates) |
| primary score mode | zero |
| source features | enabled |

Feature groups (176 total): track/user/turn basics; history-consistency (same artist/album/track, last-music match, tag overlap); query–metadata similarity (TF-IDF, intent tags, dense cosine on Qwen3-Embedding vectors); metadata extensions (ISRC year/country, duration buckets, age–release-year consistency, seasonal tags); hierarchical popularity (within-artist/album popularity and counts); tag-chain features (history-tag Jaccard/cosine, PPMI graph neighbor overlap); per-source presence/rank/score transforms incl. `score__challenge` / `score__tpd1` / `score__transition_probability`.

Blind-B-safe neutralization: `candidate_rank` family, prior-GPA counts and `goal_track_tfidf_sim` are neutralized **as values in a fixed-width feature vector** (columns are kept so the layout always matches the shipped model). `current_thought`, `conversation_goal`, target GPA, `track_emb.test_tracks` and popularity tie-breakers are never used.

Local CV uses `artifacts/cache/splits/cv5` (5-fold); train-fitted retriever sources feed the reranker via their OOF artifacts, while Blind-B inference uses full-public artifacts. The shipped `model.txt` was fit on all public labeled rows; `run_inference.sh` loads it (`--load-model`) and reproduces the submitted top-20 ranking bit-exactly.

## Responder (`qwen36_10run_diverse`)

Base config `responder/qwen36_27b/configs/rich_context_hierpop_tagchain.yaml`; the top-3 ranked tracks are passed to **Qwen/Qwen3.6-27B** (bf16, single GPU) with user profile, conversation history and current message.

| parameter | value |
| --- | --- |
| top-k tracks in prompt | 3 |
| max_new_tokens / temperature / top_p | 200 / 0.7 / 0.9 |
| n_runs (seeds) | 10 (0..9) |
| selection objective | lexdiv (distinct-1 + distinct-2, greedy) |
| random-order trials | 30 (seeded) |

For each Blind-B row, 10 responses are generated under the same ranked top-3; the per-row selection maximizing corpus-level unigram diversity is chosen. All sampling is seeded (`torch.manual_seed(seed + run)`, `random.Random(seed + 1000 + trial)`); exact text reproduction is still subject to CUDA kernel nondeterminism, so the submitted responses are shipped as `reference/blind_b.json` for diffing. Track IDs (the nDCG part) are fully deterministic given the shipped model.

## Scores

| metric | value |
| --- | ---: |
| 5-fold nDCG@1 / @10 / @20 | 0.0995 / 0.2491 / 0.2743 |
| fold0..4 nDCG@20 | 0.27674 / 0.27546 / 0.27405 / 0.27212 / 0.27321 |
| train-rows / devset-rows nDCG@20 | 0.2791 / 0.2020 |
| Blind-B Codabench (overall) | ≈ 0.59 |

## Compliance notes

- `track_emb.test_tracks` (the target-side track set) is **not** used anywhere (candidate universe = full catalog / train-derivable sets only).
- Evaluation-split information is never used to fit anything; the TF-IDF vectorizer fits on the track catalog only; supervised components are fit on public labeled rows with OOF handling for train-row features.
- External data (TalkPlayData-1/2) is permitted by the challenge; attribution in the README.
