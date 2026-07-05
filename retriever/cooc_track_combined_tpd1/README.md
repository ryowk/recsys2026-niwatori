# retriever/cooc_track_combined_tpd1

## ロジック

challenge public labeled の train music outcome から作る `cooc_track` 統計と、TalkPlayData-1 train の mapped track だけから作る external cooc 統計を単純加算する。TalkPlayData-1 の Spotify track ID は `data/derived/spotify_uuid_map_v1.parquet` で challenge catalog の `track_idx` に写し、mapping できない track は統計から除外する。

public labeled artifact は `cv3_oof` で作る。各 row には、その row の fold を除いた challenge cooc 統計に、TPD1全量の外部統計を足したものを使う。blind artifact は `full_public` として challenge public labeled 全体 + TPD1全量で作る。TPD1 の採用集合は全fold共通で、foldごとに変えない。

## 意図

TPD1を別sourceとしてunionに足すのではなく、既存 `cooc_track` と同じ意味の統計に外部データを混ぜる。source数を増やすより、cooc signalそのものを強くする仮説。

## 設定

- `configs/oof3_top500.yaml`: top500、単純加算後の `score__primary` だけを保存。
- `configs/oof3_top500_parts.yaml`: top500、`score__primary` に加えて `score__challenge` / `score__tpd1` を保存する。
- `configs/oof3_top500_purge3_parts.yaml`: `parts` と同じ候補生成だが、public labeled 全体と3-track以上の連続部分列が重なるTPD1 sessionを除外する。除外済みTPD1集合は全fold共通。これは3-gram頻出モチーフも落とすため、本命policyではなく感度分析用。
- `configs/oof5_top500_parts.yaml`: 5-fold split 用。TPD1 cooc は mapped track session 内の全組み合わせを数える。
- `configs/oof5_top1000_parts.yaml` / `configs/oof5_top2000_parts.yaml`: 5-fold split 用。`top500_parts` と同じ統計・score field のまま、TPD1 combined cooc の深い候補を保存する cap 深掘り用。
- `configs/oof5_top500_parts_pmi.yaml`: 5-fold split 用。候補集合は全組み合わせ cooc のまま、TPD1 cooc PPMI を `score__tpd1_pmi` として追加する。
- `configs/oof5_top500_parts_window3_inverse.yaml`: 5-fold split 用。TPD1 cooc を mapped track 列の距離3以内に限定し、距離 `d` の寄与を `1/d` にする。A→unmapped→B のような mapping 抜けは、mapped 部分列上の距離として扱う。

union 側で source cap を変える。既存 baseline の `cooc_track` は `max_candidates=100` なので、まず100/200/500をfold0で比較する。

## 入出力 artifact

入力:

- `splits/public_labeled_v1`
- challenge public labeled music outcome
- `data/derived/spotify_uuid_map_v1.parquet`
- `talkpl-ai/TalkPlayData-1`, split `train`
- target row の観測済み music history

出力:

- `artifacts/runs/retriever/cooc_track_combined_tpd1/<config>/cv3_oof/public_labeled/candidates.npz`
- `artifacts/runs/retriever/cooc_track_combined_tpd1/<config>/full_public/blind_b/candidates.npz`

## fit / leak 確認

challenge由来統計は public labeled の fixed 3-fold OOF で作る。TPD1由来統計は external train の mapped track のみで作り、public labeled row / blind row は fit に使わない。`track_emb.test_tracks`、target future turn、current thought、conversation goal、GPA は使わない。候補宇宙は challenge `all_tracks` のみ。

TPD1 mapped session と public labeled session の exact full sequence overlap は0件。full conversation text 完全一致も0件。3-gram overlap は3,873 TPD1 rowsで見つかるが、同一artist / album近傍の頻出局所パターンが多い。first user utterance や短い text shingle の一致も一般的な依頼文の一致が中心で、session重複の強い証拠ではない。したがって本命policyはTPD1除外なしで、`purge3` configは安全側の感度分析として扱う。

## 結果と学び

`oof3_top500` の public_labeled local recall:

- mean_size: 245.084
- recall@20: 0.353718
- recall@100: 0.472491
- recall@200: 0.505494
- recall@all: 0.537387

`oof3_top500_parts` は candidate set は同じで、`score__challenge` / `score__tpd1` を reranker feature として残すための config。

combined cooc / transition を baseline union に差し替えた `blind_b_safe_combined_tpd1_parts_basecap_t200` は、3-fold reranker local nDCG@20=0.271759。既存 separate TPD1 all-source 3-fold 0.271189 は少し上回ったが、元baseline 3-fold 0.273327 には届かない。fold0 は 0.274086 と良かったが、fold1/2 で伸びが落ちた。

`blind_b_safe_combined_tpd1_purge3_parts_basecap_t200` は旧実装の fold依存 purge による参考値として 3-fold reranker local nDCG@20=0.271027。非purge版より低く、3-gram一致だけで外部sessionを落とすのは保守的すぎる可能性が高い。今後 purge3 を再評価する場合は、public labeled 全体に対する共通除外集合で作り直す。cooc cap 200/500 は union recall@all だけ増え、recall@20-200 は変わらないため、重い reranker 評価は現時点では追加しない。

2026-06-26 final sprint では、BlindB 実提出で cooc500 が効いたことを受けて `oof5_top1000_parts` / `oof5_top2000_parts` を追加した。まず retriever recall と候補数を見て、fold0 reranker に進めるかを判定する。
