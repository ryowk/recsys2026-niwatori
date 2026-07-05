# retriever/bm25_5field_thought

## ロジック

現在の user turn のテキスト(会話文 + 直近 context)を query にして、track metadata 5 field(`track_name` / `artist_name` / `album_name` / `tag_list` / `release_date`)を連結した corpus に対し BM25 で検索する。`bm25s` で index を張り、上位を candidates として返す。label は使わない fit-free source。

## 意図

lexical に強い汎用 retriever。曲名・アーティスト名・タグなどが query 文面に現れるケースを広く拾う union の土台。dense / cooc 系が拾えない表層一致を補完する。

## 設定

- `top500_bsafe`: **blind-B-safe** な query text(`conversation_goal` / `thought` を使わない、message-only)で最大 500 candidates を保存する config。union では cap 200 で採用。

## 入出力 artifact

入力: 現在 turn のテキスト、`track` metadata(5 field)。

出力:

- `artifacts/runs/retriever/bm25_5field_thought/top500_bsafe/fit_free_all_rows/public_labeled/candidates.npz`
- `artifacts/runs/retriever/bm25_5field_thought/top500_bsafe/fit_free_all_rows/blind_b/candidates.npz`

`score__primary` は BM25 スコア。source 内の相対強度であり cross-source では揃っていないため、reranker には source 固有 feature として渡す。

## fit / leak 確認

label outcome を使わない fit-free retriever。BM25 の統計は track catalog corpus のみから作り、valid / devset / blind の文面で index を作らない(vocabulary leak なし)。`conversation_goal` / current `thought` / target GPA / `track_emb.test_tracks` は使わない。popularity tie-breaker なし。

## 結果と学び

union の中核 lexical source。単体 recall は中程度だが、他 source と直交する表層一致を供給するため union / reranker で残す価値が高い。
