# retriever/last_music_artist

## ロジック

target turn の chat history から最後の music track を取り、その track と同じ artist ID / artist name に属する catalog tracks を返す。学習 label は使わない fit-free source で、候補順と `score__primary` は source 内の一致強度 / rank に由来する。

## 意図

ユーザが直前の曲と同じ artist の別曲を求めているケースを拾う。`history_artist` は history 全体の artist を使うが、`last_music_artist` は直近 music turn だけに絞るため、recency signal として扱える。

## 設定

- `basic.yaml`: 最大 200 candidates を保存する標準 config。
- `top500.yaml`: union / reranker で後段に絞れるよう、最大 500 candidates を保存する広めの config。

## 入出力 artifact

入力:

- target turn の history music tracks
- `track` metadata の artist 情報

出力:

- `artifacts/runs/retriever/last_music_artist/top500/fit_free_all_rows/public_labeled/candidates.npz`
- `artifacts/runs/retriever/last_music_artist/top500/fit_free_all_rows/blind_b/candidates.npz`

`score__primary` は source 内の一致強度 / rank 用 score。異なる retriever の score とスケールを揃えたものではないため、union 側で cross-source max などには使わず、reranker には source 固有 feature として渡す。

## fit / leak 確認

label outcome を使わない fit-free retriever なので OOF は不要。target turn より未来の情報や `track_emb.test_tracks` は使わない。popularity tie-breaker は使わない。

## 結果と学び

devset 単体評価では mean size 約25、recall@all 約0.299。単体でも強いが、`history_artist` と重なりやすいため、union では recency feature としての追加価値を見る。
