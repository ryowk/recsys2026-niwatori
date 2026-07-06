# reranker/protocol_098_union_rich_lgbm

## ロジック

この reranker は、既存の `098_current_thought_profile_ablation` の rich feature 群を candidate artifact 上で再計算し、さらに `retriever/union` が出した source feature pack を結合して LightGBM LambdaRank で順位学習します。

主な feature は次の通りです。

- 098 系 feature: goal / query / history / track metadata / user embedding / dense query 類似度など。
- source feature: `src__<source>__present`, `src__<source>__rank`, `src__<source>__score__primary`。
- 追加 source signal: source artifact が `score__*`, `sim__*`, `count__*`, `feat__*` などの複数 per-candidate 配列を持つ場合、それらも union candidate 位置に揃えて使う。
- meta feature: `meta__source_count`, `meta__best_source_rank`, `meta__mean_source_rank`。

`score__primary` は retriever 内の primary ordering score であり、全 retriever に共通の意味を持つ正規化済み score ではありません。reranker では raw 値に加えて row 内 z-score も作り、source 間のスケール差を model が扱えるようにします。
現行 config は `feature_build.drop_cross_source_score_meta: true` を設定しており、source をまたいだ `max(score__primary)` のような scale 混在集約は feature から外します。source 別の raw score / sim / count / feature は source 名付きで渡します。

## 意図

旧 098 は candidate generation 側に artist / album boost などの人為的な混合ロジックを持っていました。この component は「候補生成は独立 retriever の広い union に任せ、boost や source weighting は reranker feature として学習させる」方針を検証します。

## 設定

現行 repo の唯一の config は `blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5.yaml` です。

- 候補は `retriever/union` の `blind_b_safe_combined_tpd1_parts_cooc500_cv5` union artifact を全幅 (`max_candidates: all`) で読み、`primary_score_mode: zero` で primary candidate score を 0 にして、source feature と 098 rich feature で LightGBM LambdaRank を 200 trees (`num_leaves: 63`, `learning_rate: 0.04`) 学習します。`train_positive_only: true` で positive を含む row だけを学習に使います。
- `feature_build` では `drop_cross_source_score_meta: true` で source をまたいだ `max(score__primary)` meta を外し、`extra_metadata_features` / `extra_hier_pop_features` / `extra_tag_chain_features` を有効化し、`candidate_rank` 系や `prior_gpa_*` / `goal_track_tfidf_sim` を neutralize します。
- `leak_policy` は BlindB-safe (`blind_b_safe: true`) で、`conversation_goal` / GPA / current `thought` を使いません。external data は TalkPlayData-1 を `no_purge_all_tpd1_all_folds` で許可します。
- `--target public_labeled` では 5-fold (`cv_artifact_mode: cv5_oof`) の CV スコアを、`--target blind_b` では public 全体で fit した最終 model による blind_b 予測 (`full_public`) を出します。

## 入出力 artifact

入力:

- `artifacts/runs/retriever/union/blind_b_safe_combined_tpd1_parts_cooc500_cv5/public_labeled/candidates.npz`
- `artifacts/runs/retriever/union/blind_b_safe_combined_tpd1_parts_cooc500_cv5/public_labeled/source_features.npz`
- blind_b 予測時は同 union の `blind_b/` artifact
- 098 系 dense query cache

出力:

- CV (public_labeled): `artifacts/runs/reranker/protocol_098_union_rich_lgbm/blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5/cv5_oof/public_labeled/ranked.npz` と `ranked_top100.jsonl`
- CV スコア: `artifacts/results/reranker/protocol_098_union_rich_lgbm/blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5/cv5_oof/public_labeled/scores.json`
- blind_b 予測: `artifacts/runs/reranker/protocol_098_union_rich_lgbm/blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5/full_public/blind_b/ranked.npz`

## fit / leak 確認

reranker fit は public_labeled の固定 5-fold (`cv5_oof`) で CV します。train rows には、source 側が supervised / train-statistical な場合は cv5 OOF artifact、same-user memory の場合は strict date-censored artifact を使う前提です。devset score を見た weight / threshold / source gate tuning はしません。`track_emb.test_tracks` は使いません。現行 config は BlindB-safe なので `conversation_goal` / GPA / current thought を feature に使いません。external data の TalkPlayData-1 は cooc / transition retriever source 経由でのみ入り、public / blind label を scorer fit には使いません。最終 model の fit は public union (`cv5_oof` features) で行い、blind_b 予測だけ `full_public` union を対象にします。

## 結果と学び

現行 config `blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5` は public labeled 5-fold (`cv5_oof`) reranker nDCG@20 = 0.274316 です。TPD1 なし BlindB-safe baseline (nDCG@20 = 0.271911)、combined TPD1 basecap (0.273478)、cooc cap200 (0.273960) に対し、combined TPD1 cooc / transition を混ぜて cooc_track source cap を 500 に広げた本 config が最良でした。候補削減は union 後 truncate ではなく source 側 cap で行い、reranker は zero primary score + source feature + 098 rich feature で順位を学習します。TPD1 は reranker training group へ直接混ぜず、combined cooc / transition retriever source として union に足す形で使います。
