# retriever/tag_intent_bm25

## ロジック

現在の user turn のテキストから genre / mood / descriptor などの intent tag(例: acoustic, chill, workout, jazz)を検出し、その tag を query にして `track` metadata の `tag_list` を BM25 で検索する。label を使わない fit-free source で、`score__primary` は tag BM25 スコアに由来する。

## 意図

「作業用の落ち着いた曲」「パーティー向け」のように、具体名ではなく雰囲気 / 用途で要求されたケースを拾う。曲名 / artist が出てこない intent-driven な turn を、tag 空間の一致で捉える。

## 設定

- `top500_bsafe`: **blind-B-safe** な query text(`conversation_goal` / `thought` を使わない、message-only)で最大 500 candidates を保存する config。union では cap 100 で採用。

## 入出力 artifact

入力: 現在 turn のテキスト(から抽出した intent tag)、`track` metadata の `tag_list`。

出力:

- `artifacts/runs/retriever/tag_intent_bm25/top500_bsafe/fit_free_all_rows/public_labeled/candidates.npz`
- `artifacts/runs/retriever/tag_intent_bm25/top500_bsafe/fit_free_all_rows/blind_b/candidates.npz`

`score__primary` は tag BM25 スコア。cross-source では揃っていないため reranker に source 固有 feature として渡す。

## fit / leak 確認

fit-free retriever。BM25 統計は track catalog の `tag_list` corpus のみから作り、valid / devset / blind の文面で index を作らない。`conversation_goal` / current `thought` / target GPA / `track_emb.test_tracks` は使わない。popularity tie-breaker なし。

## 結果と学び

intent-driven turn 用の source。lexical(bm25_5field)や exact 系が拾えない雰囲気要求を補完し、union で mood / genre 軸の recall を supply する。
