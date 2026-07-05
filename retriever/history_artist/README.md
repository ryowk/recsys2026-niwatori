# retriever/history_artist

## ロジック

target turn までの chat history に登場した music tracks の artist(artist ID / artist name)を集め、その artist に属する catalog tracks を返す。label を使わない fit-free source で、候補順と `score__primary` は history 内での出現頻度 / 一致強度に由来する。

## 意図

ユーザがこれまで聴いた artist の別曲を求めるケースを拾う。`last_music_artist` が直近 music turn だけに絞るのに対し、`history_artist` は history 全体の artist を使う長期嗜好 signal。

## 設定

- `top500`: 最大 500 candidates を保存する広めの config(union / reranker で後段に絞る)。

## 入出力 artifact

入力: target turn より前の history music tracks、`track` metadata の artist 情報。

出力:

- `artifacts/runs/retriever/history_artist/top500/fit_free_all_rows/public_labeled/candidates.npz`
- `artifacts/runs/retriever/history_artist/top500/fit_free_all_rows/blind_b/candidates.npz`

`score__primary` は source 内の一致強度用 score。cross-source では揃っていないため reranker に source 固有 feature として渡す。

## fit / leak 確認

fit-free retriever なので OOF 不要。history は `turn_number < target_turn` に限定し、target turn 自身 / 未来 turn の情報は使わない。`track_emb.test_tracks` / popularity tie-breaker は使わない。

## 結果と学び

長期 artist 嗜好の signal。`last_music_artist` と重なりやすいため、union では recency との差分(長期 vs 直近)を feature として見る。
