# retriever/cooc_track_combined_tpd1

## ロジック

challenge public labeled の train music outcome から作る `cooc_track` 統計と、TalkPlayData-1 train の mapped track だけから作る external cooc 統計を単純加算する。TalkPlayData-1 の Spotify track ID は `data/derived/spotify_uuid_map_v1.parquet` で challenge catalog の `track_idx` に写し、mapping できない track は統計から除外する。

public labeled artifact は `cv5_oof` で作る。各 row には、その row の fold を除いた challenge cooc 統計に、TPD1全量の外部統計を足したものを使う。blind artifact は `full_public` として challenge public labeled 全体 + TPD1全量で作る。TPD1 の採用集合は全fold共通で、foldごとに変えない。

## 意図

TPD1を別sourceとしてunionに足すのではなく、既存 `cooc_track` と同じ意味の統計に外部データを混ぜる。source数を増やすより、cooc signalそのものを強くする仮説。

## 設定

- `configs/oof5_top500_parts.yaml`: 現行 pipeline が `--config oof5_top500_parts --artifact-mode cv5_oof` で使う 5-fold config。top500、単純加算後の `score__primary` に加えて `score__challenge` / `score__tpd1` を保存する。TPD1 cooc は mapped track session 内の全組み合わせを数える。
- `configs/oof5_top1000_parts.yaml` / `configs/oof5_top2000_parts.yaml`: `oof5_top500_parts` と同じ統計・score field のまま、TPD1 combined cooc の深い候補を保存する cap 深掘り用。
- `configs/oof5_top500_parts_pmi.yaml`: 候補集合は全組み合わせ cooc のまま、TPD1 cooc PPMI を `score__tpd1_pmi` として追加する。
- `configs/oof5_top500_parts_window3_inverse.yaml`: TPD1 cooc を mapped track 列の距離3以内に限定し、距離 `d` の寄与を `1/d` にする。A→unmapped→B のような mapping 抜けは、mapped 部分列上の距離として扱う。

union 側で source cap を変える。現行 union `blind_b_safe_combined_tpd1_parts_cooc500_cv5` はこの source を `max_candidates=500` で使う。

## 入出力 artifact

入力:

- `artifacts/cache/splits/cv5`
- challenge public labeled music outcome
- `data/derived/spotify_uuid_map_v1.parquet`
- `talkpl-ai/TalkPlayData-1`, split `train`
- target row の観測済み music history

出力:

- `artifacts/runs/retriever/cooc_track_combined_tpd1/<config>/cv5_oof/public_labeled/candidates.npz`
- `artifacts/runs/retriever/cooc_track_combined_tpd1/<config>/full_public/blind_b/candidates.npz`

## fit / leak 確認

challenge由来統計は public labeled の fixed 5-fold OOF (`cv5_oof`) で作る。TPD1由来統計は external train の mapped track のみで作り、public labeled row / blind row は fit に使わない。`track_emb.test_tracks`、target future turn、current thought、conversation goal、GPA は使わない。候補宇宙は challenge `all_tracks` のみ。

TPD1 mapped session と public labeled session の exact full sequence overlap は0件。full conversation text 完全一致も0件。3-gram overlap は3,873 TPD1 rowsで見つかるが、同一artist / album近傍の頻出局所パターンが多い。first user utterance や短い text shingle の一致も一般的な依頼文の一致が中心で、session重複の強い証拠ではない。したがって本命policyはTPD1除外なしとし、TPD1 を除外する purge 変種は安全側の感度分析としてのみ扱う。

## 結果と学び

この source は現行 union `blind_b_safe_combined_tpd1_parts_cooc500_cv5` に `oof5_top500_parts` (source cap 500) として入り、reranker `protocol_098_union_rich_lgbm/blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5` の 5-fold (`cv5_oof`) nDCG@20 = 0.274316 に寄与する。TPD1 を別 source として union に足すより、challenge cooc 統計に外部 cooc 統計を単純加算して cooc signal そのものを強める形を本命にしている。`oof5_top1000_parts` / `oof5_top2000_parts` は TPD1 combined cooc の深さ上限を測る cap 深掘り用。
