# retriever/transition_track_combined_tpd1

## ロジック

challenge public labeled の train music outcome から作る `transition_track_last` 統計と、TalkPlayData-1 train の mapped track だけから作る external transition 統計を単純加算する。TalkPlayData-1 の Spotify track ID は `data/derived/spotify_uuid_map_v1.parquet` で challenge catalog の `track_idx` に写し、mapping できない track は統計から除外する。

target row では history の最後の music track を起点に、challenge遷移count + TPD1遷移count を足した候補を返す。`score__transition_probability` は加算後countで正規化する。TPD1 の採用集合は全fold共通で、foldごとに変えない。

## 意図

TPD1を別sourceとしてunionに足すのではなく、既存 `transition_track_last` と同じ意味の Markov signal に外部データを混ぜる。BlindB-safeで使える履歴track列だけに基づくため、goal/thought/GPAが無いBlindBでも同じロジックを使える。

## 設定

- `configs/oof3_top500_prob.yaml`: top500、単純加算後の `score__primary` と `score__transition_probability` を保存。
- `configs/oof3_top500_prob_parts.yaml`: 上記に加えて `score__challenge` / `score__tpd1` を保存する。
- `configs/oof3_top500_prob_purge3_parts.yaml`: `parts` と同じ候補生成だが、public labeled 全体と3-track以上の連続部分列が重なるTPD1 sessionを除外する。除外済みTPD1集合は全fold共通。これは3-gram頻出モチーフも落とすため、本命policyではなく感度分析用。

## 入出力 artifact

入力:

- `splits/public_labeled_v1`
- challenge public labeled music outcome
- `data/derived/spotify_uuid_map_v1.parquet`
- `talkpl-ai/TalkPlayData-1`, split `train`
- target row の観測済み最後の music track

出力:

- `artifacts/runs/retriever/transition_track_combined_tpd1/<config>/cv3_oof/public_labeled/candidates.npz`
- `artifacts/runs/retriever/transition_track_combined_tpd1/<config>/full_public/blind_b/candidates.npz`

## fit / leak 確認

challenge由来統計は public labeled の fixed 3-fold OOF で作る。TPD1由来統計は external train の mapped track のみで作り、public labeled row / blind row は fit に使わない。`track_emb.test_tracks`、target future turn、current thought、conversation goal、GPA は使わない。候補宇宙は challenge `all_tracks` のみ。

TPD1 mapped session と public labeled session の exact full sequence overlap は0件。full conversation text 完全一致も0件。3-gram overlap は3,873 TPD1 rowsで見つかるが、同一artist / album近傍の頻出局所パターンが多い。first user utterance や短い text shingle の一致も一般的な依頼文の一致が中心で、session重複の強い証拠ではない。したがって本命policyはTPD1除外なしで、`purge3` configは安全側の感度分析として扱う。

## 結果と学び

`oof3_top500_prob` の public_labeled local recall:

- mean_size: 39.188
- recall@20: 0.186539
- recall@100: 0.214782
- recall@200: 0.220569
- recall@all: 0.223532

`oof3_top500_prob_parts` は candidate set は同じで、`score__challenge` / `score__tpd1` を reranker feature として残すための config。

combined cooc / transition を baseline union に差し替えた `blind_b_safe_combined_tpd1_parts_basecap_t200` は、3-fold reranker local nDCG@20=0.271759。既存 separate TPD1 all-source 3-fold 0.271189 は少し上回ったが、元baseline 3-fold 0.273327 には届かない。fold0 は 0.274086 と良かったが、fold1/2 で伸びが落ちた。

`blind_b_safe_combined_tpd1_purge3_parts_basecap_t200` は旧実装の fold依存 purge による参考値として 3-fold reranker local nDCG@20=0.271027。非purge版より低く、3-gram一致だけで外部sessionを落とすのは保守的すぎる可能性が高い。今後 purge3 を再評価する場合は、public labeled 全体に対する共通除外集合で作り直す。Markov signal はBlindB-safeな履歴track列だけで作れるが、この形だけでは主要baselineを置き換えるほどの改善ではない。
