# retriever/exact_title_artist_source

## ロジック

現在の user turn のテキストから track title / artist 名の surface を抽出し、`track` metadata の `track_name` / `artist_name` と exact に一致する catalog tracks を返す。label を使わない fit-free source で、`score__primary` は一致の種別 / 強度に由来する。

## 意図

「(曲名)をかけて」のように具体曲を名指ししたケースを高精度で拾う。`exact_album_artist_source` が album / artist 粒度なのに対し、こちらは title 粒度で target track を直接当てにいく。

## 設定

- `top500`: 最大 500 candidates を保存する config。

## 入出力 artifact

入力: 現在 turn のテキスト、`track` metadata の `track_name` / `artist_name`。

出力:

- `artifacts/runs/retriever/exact_title_artist_source/top500/fit_free_all_rows/public_labeled/candidates.npz`
- `artifacts/runs/retriever/exact_title_artist_source/top500/fit_free_all_rows/blind_b/candidates.npz`

`score__primary` は一致種別 / 強度用 score。cross-source では揃っていないため reranker に source 固有 feature として渡す。

## fit / leak 確認

fit-free retriever なので OOF 不要。current user turn のテキストのみを使い、target turn 自身の GPA / 未来 turn / `track_emb.test_tracks` は使わない。popularity tie-breaker なし。

## 結果と学び

title 名指しに対する最高精度の source。ヒット行は少ないが gold を直接含みやすく、union / reranker で precision 補強として残す。
