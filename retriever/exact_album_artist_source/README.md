# retriever/exact_album_artist_source

## ロジック

現在の user turn のテキストから album 名 / artist 名の surface(表層文字列)を抽出し、`track` metadata の album / artist と exact に一致する catalog tracks を返す。label を使わない fit-free source で、`score__primary` は一致の種別 / 強度に由来する。

## 意図

「◯◯(album 名)を流して」「△△(artist 名)の曲」のように、ユーザが具体名を明示したケースを高精度で拾う。BM25 の soft match と違い exact 一致に限定するため、precision の高い candidates を供給する。

## 設定

- `top500`: 最大 500 candidates を保存する config。

## 入出力 artifact

入力: 現在 turn のテキスト、`track` metadata の album / artist 情報。

出力:

- `artifacts/runs/retriever/exact_album_artist_source/top500/fit_free_all_rows/public_labeled/candidates.npz`
- `artifacts/runs/retriever/exact_album_artist_source/top500/fit_free_all_rows/blind_b/candidates.npz`

`score__primary` は一致種別 / 強度用 score。cross-source では揃っていないため reranker に source 固有 feature として渡す。

## fit / leak 確認

fit-free retriever なので OOF 不要。current user turn のテキストのみを使い、target turn 自身の GPA / 未来 turn / `track_emb.test_tracks` は使わない。popularity tie-breaker なし。

## 結果と学び

高 precision な surface-match source。ヒットする行は少ないが当たると強いため、`exact_title_artist_source` と共に union の precision 補強に使う。
