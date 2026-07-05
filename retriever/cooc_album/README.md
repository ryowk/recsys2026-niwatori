# retriever/cooc_album

## ロジック

public labeled sessions から album ID 同士の共起 count を作り、target turn の過去 music history に含まれる album から、共起 album の catalog tracks へ展開する retriever。

train-row artifact は固定 3-fold の OOF で作る。つまり各 public labeled row の候補生成では、その row が属する fold の sessions を共起 table の fit から外す。blind 用 artifact は public labeled 全体で fit した `full_public` を使う。

## 意図

`history_album` は候補数が少なく precision が高い。`cooc_album` はその近傍版として、同じ album そのものではなく「同じ session に出やすい別 album」を拾う。`cooc_track` より粗く、`cooc_artist` より album 単位で絞れるため、低候補数 high precision source になることを期待する。

## 設定

- `oof3_top500.yaml`: source ごとに最大 500 candidates を保存する。実際の union では source cap や `min_score` threshold でさらに絞る。

## 入出力 artifact

入力:

- `track` metadata の `album_id`
- public labeled sessions の music outcome
- `splits/public_labeled_v1`

出力:

- `artifacts/runs/retriever/cooc_album/oof3_top500/cv3_oof/public_labeled/candidates.npz`
- `artifacts/runs/retriever/cooc_album/oof3_top500/cv3_oof/public_labeled/turns.jsonl`
- `artifacts/runs/retriever/cooc_album/oof3_top500/full_public/blind_b/candidates.npz`

`score__primary` は、history album から共起 album へ入った album-album 共起 count の合計。

## fit / leak 確認

label を使う train-statistical retriever なので、reranker fit 用 public labeled rows には `cv3_oof` artifact を使う。blind inference には `full_public` を使う。`track_emb.test_tracks` は使わない。same-user future memory は使わない。popularity tie-breaker は使わない。

## 結果と学び

未評価。まず public labeled の recall / precision / 候補数を `cooc_track`, `history_album`, `cooc_artist_name` と比較する。
