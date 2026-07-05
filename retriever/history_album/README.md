# retriever/history_album

## ロジック

target turn までの chat history に登場した music tracks の album を集め、その album に属する catalog tracks を返す。label を使わない fit-free source で、候補順と `score__primary` は history 内での album 出現頻度 / 一致強度に由来する。

## 意図

ユーザが聴いた album の他収録曲を求めるケースを拾う。`history_artist` より粒度が細かく(artist ではなく album 単位)、同一作品内の連続再生 / album 志向を捉える。

## 設定

- `top500`: 最大 500 candidates を保存する広めの config。

## 入出力 artifact

入力: target turn より前の history music tracks、`track` metadata の album 情報。

出力:

- `artifacts/runs/retriever/history_album/top500/fit_free_all_rows/public_labeled/candidates.npz`
- `artifacts/runs/retriever/history_album/top500/fit_free_all_rows/blind_b/candidates.npz`

`score__primary` は source 内の一致強度用 score。cross-source では揃っていないため reranker に source 固有 feature として渡す。

## fit / leak 確認

fit-free retriever なので OOF 不要。history は `turn_number < target_turn` に限定し、target turn 自身 / 未来 turn の情報は使わない。`track_emb.test_tracks` / popularity tie-breaker は使わない。

## 結果と学び

album 粒度の history signal。`history_artist` と親子関係にあるため、union では album 単位の追加寄与を feature として見る。
