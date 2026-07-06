# retriever/transition_track_combined_tpd1

## ロジック

challenge public labeled の train music outcome から作る `transition_track_last` 統計と、TalkPlayData-1 train の mapped track だけから作る external transition 統計を単純加算する。TalkPlayData-1 の Spotify track ID は `data/derived/spotify_uuid_map_v1.parquet` で challenge catalog の `track_idx` に写し、mapping できない track は統計から除外する。

target row では history の最後の music track を起点に、challenge遷移count + TPD1遷移count を足した候補を返す。`score__transition_probability` は加算後countで正規化する。TPD1 の採用集合は全fold共通で、foldごとに変えない。

## 意図

TPD1を別sourceとしてunionに足すのではなく、既存 `transition_track_last` と同じ意味の Markov signal に外部データを混ぜる。BlindB-safeで使える履歴track列だけに基づくため、goal/thought/GPAが無いBlindBでも同じロジックを使える。

## 設定

- `configs/oof5_top500_prob_parts.yaml`: 現行 pipeline が `--config oof5_top500_prob_parts --artifact-mode cv5_oof` で使う 5-fold config。top500、単純加算後の `score__primary` と `score__transition_probability` に加えて `score__challenge` / `score__tpd1` を保存する。

## 入出力 artifact

入力:

- `artifacts/cache/splits/cv5`
- challenge public labeled music outcome
- `data/derived/spotify_uuid_map_v1.parquet`
- `talkpl-ai/TalkPlayData-1`, split `train`
- target row の観測済み最後の music track

出力:

- `artifacts/runs/retriever/transition_track_combined_tpd1/<config>/cv5_oof/public_labeled/candidates.npz`
- `artifacts/runs/retriever/transition_track_combined_tpd1/<config>/full_public/blind_b/candidates.npz`

## fit / leak 確認

challenge由来統計は public labeled の fixed 5-fold OOF (`cv5_oof`) で作る。TPD1由来統計は external train の mapped track のみで作り、public labeled row / blind row は fit に使わない。`track_emb.test_tracks`、target future turn、current thought、conversation goal、GPA は使わない。候補宇宙は challenge `all_tracks` のみ。

TPD1 mapped session と public labeled session の exact full sequence overlap は0件。full conversation text 完全一致も0件。3-gram overlap は3,873 TPD1 rowsで見つかるが、同一artist / album近傍の頻出局所パターンが多い。first user utterance や短い text shingle の一致も一般的な依頼文の一致が中心で、session重複の強い証拠ではない。したがって本命policyはTPD1除外なしとし、TPD1 を除外する purge 変種は安全側の感度分析としてのみ扱う。

## 結果と学び

この source は現行 union `blind_b_safe_combined_tpd1_parts_cooc500_cv5` に `oof5_top500_prob_parts` として入り、reranker `protocol_098_union_rich_lgbm/blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5` の 5-fold (`cv5_oof`) nDCG@20 = 0.274316 に寄与する。Markov transition signal は BlindB-safe な履歴 track 列だけで作れるが、この source 単体で主要 baseline を置き換えるほどの改善ではなく、challenge + TPD1 の combined cooc / transition を union に足す構成の一部として使う。
