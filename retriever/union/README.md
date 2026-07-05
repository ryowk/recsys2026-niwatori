# retriever/union

## ロジック

`retriever/union` は、複数の retriever artifact を同じ row key で揃え、指定順に候補を重複除去しながら結合する retriever component です。`ordered_unique` では source list の順番を優先し、同じ source 内では元 artifact の rank 順を保ちます。`round_robin` では各 source の同 rank を順番に拾います。

## 意図

単体 retriever の信号を失わず、reranker 側で「どの source に出たか」「source 内 rank は何位か」「source 固有の score / 類似度 / count は何か」を使える広い候補集合を作ります。union は最終 top20 を決める stage ではないため、config で明示しない限り union 後に強い size cutoff を入れません。

## 設定

- `union_v1.yaml`: 代表的な多数 retriever の ordered union。
- `union_v1_top500.yaml`: 各 source の top500 artifact を使う広めの union。
- `fit_free_top500.yaml`: label-free source 中心の軽量 union。
- `independent_full_union_cv3.yaml`: BM25 / Two-Tower / 履歴 / 共起 / personal などの代表 source を、union 後の人工 topK なしで全て残す public_labeled CV 用 config。
- `independent_balanced_union_cv3.yaml`: 同じ source を使うが、source 側の出力上限を調整して union size を LightGBM が扱いやすい範囲に抑える config。
- `independent_compact_union_cv3.yaml`: `balanced` よりさらに source cap を小さくし、平均候補数を約650まで抑える config。
- `independent_compact_plus_transition_track_prob_cv3.yaml`: compact-plus surface union に `transition_track_last/oof3_top500_prob` を足し、Markov の count と transition probability を source feature として渡す config。
- `independent_compact_plus_markov_prob_cv3.yaml`: `transition_track_last`, `transition_album_last`, `transition_artist_id_last`, `transition_track_bigram_last2` の count + transition probability artifact を追加する Markov 深掘り config。
- `blind_b_safe_tpd1_text_bm25_cv3.yaml`: BlindB-safe baseline union に `tpd1_track_text_bm25/oof3_no_ngram_purge` を cap200 で足す外部データ評価config。TPD1除外なしで全fold共通集合を使う。
- `blind_b_safe_combined_tpd1_parts_basecap_cv5.yaml`: BlindB-safe cv5 baseline に、public labeled + TPD1 を混ぜて作った cooc / transition track 統計 source を追加する 5-fold 本命系 config。
- `blind_b_safe_combined_tpd1_parts_cooc200_cv5.yaml` / `blind_b_safe_combined_tpd1_parts_cooc500_cv5.yaml`: basecap から `cooc_track_combined_tpd1` の source cap を 200 / 500 に広げる。union 後 truncate ではなく source cap で候補幅を制御する。
- `blind_b_safe_combined_tpd1_parts_cooc1000_textbm25_cv5.yaml` / `blind_b_safe_combined_tpd1_parts_cooc2000_textbm25_cv5.yaml`: 現行 best `cooc500 + TPD1 text BM25` の cooc source だけを 1000 / 2000 に広げる final sprint 用 config。TPD1 cooc の深さ上限を測るための事前固定実験。

`independent_full_union_cv3.yaml` と `independent_balanced_union_cv3.yaml` は supervised / train-statistical source に OOF artifact を指定し、same-user personal source には strict date-censored artifact を指定します。これは reranker fit 用 train rows で in-sample signal を使わないためです。

候補数を減らす場合は、union 後の global topK truncate ではなく、config の各 source entry に `max_candidates` を持たせて source prefix を制御します。これにより ordered_unique union そのものはそのまま reranker に渡しつつ、source 側の出力設計で自然な union size を作れます。

source artifact は原則として raw path を直接書かず、`component`, `config`, `source_policy.preferred_train_row_artifact_mode`, `source_policy.preferred_inference_artifact_mode` から resolver が次の形式で決めます。

```text
artifacts/runs/retriever/<component>/<config>/<artifact_mode>/<target>
```

`target=public_labeled` では train-row-safe な `preferred_train_row_artifact_mode`、`target=blind_a/blind_b` では提出用の `preferred_inference_artifact_mode` を使います。raw `artifact:` path は古い artifact や特殊な一時実験を読むための escape hatch としてだけ使います。

## 入出力 artifact

入力は `artifacts/runs/retriever/<source>/<config>/<mode>/<target>/candidates.npz` です。出力は `artifacts/runs/retriever/union/<config>/<target>/` に以下を作ります。

- `candidates.npz`: `track_idx`, `sizes`, `keys` と、public_labeled の場合は `source_split`, `folds`。
- `turns.jsonl`: public_labeled の row metadata。
- `source_features.npz`: union 後 candidate 位置に揃えた source feature pack。
- `manifest.json`: source artifact、union rule、fit / leak policy。

`source_features.npz` は `score__primary` だけを特別扱いしません。各 source artifact に `score__bm25`, `sim__dense`, `count__cooc`, `feat__*`, `eligible_mask__*` のような 2D per-candidate 配列があれば、`src__<source>__<field>` として可能な限り伝播します。`score__primary` はあくまで source 内の primary ordering score です。

## fit / leak 確認

union 自体は学習しません。ただし source artifact の fit scope を引き継ぎます。train row に使う source feature は、supervised retriever / train 統計 / same-user memory がその row 自身を見ていない artifact から取る必要があります。`track_emb.test_tracks` は使いません。popularity は tie-breaker として混ぜず、使う場合は独立 source として扱います。

## 結果と学び

`explore_core10_rr_top500_with_features` は人工的に top500 へ切った round-robin union で、098-rich LGBM と組み合わせると 3-fold nDCG@20 が 0.264 まで伸びました。`independent_full_union_cv3` は同じ代表 source を top500 に再切断せずに全 union する検証用 config です。事前集計では平均候補数は約 1706、最大 3021、`recall@all` は約 0.841 です。

docs の methodology を読み直すと、LightGBM へ渡す候補 pool は「union 後 truncate なし」かつ「source 側 cap で数百〜千数百へ自然に収める」が推奨です。`independent_balanced_union_cv3` はその修正版で、prefix analysis では平均候補数約 985、p90 約 1311、`recall@all` 約 0.802 です。

`independent_compact_union_cv3` はさらに候補数を抑えた現行提出 baseline 用 config です。public labeled では平均候補数 655.0、`recall@all=0.7724`、`recall@500=0.7423`。この retriever と `protocol_098_union_rich_lgbm/independent_compact_union_zeroscore_posonly_t200` を組み合わせた pipeline は、Blind A submission `740634` で composite 0.5495 / nDCG@20 0.4554 を記録しました。

Markov 深掘りでは、`independent_compact_plus_markov_prob_cv3` が public labeled 平均候補数 930.9、`recall@all=0.7900` でした。compact-plus surface からの recall 増分は小さい一方、`score__primary` の遷移 count と `score__transition_probability` を source feature として reranker へ渡せる点が主目的です。

BlindB-safe + TPD1 text BM25 では、`blind_b_safe_tpd1_text_bm25_cv3` が public labeled 平均候補数 824.9、`recall@all=0.7739`。BlindB-safe baseline より候補は広がるが、reranker fold0 の上積みは小さいため現時点では full 3-fold reranker の優先度は低い。

BlindB-safe cv5 + combined TPD1 parts では、`blind_b_safe_combined_tpd1_parts_cooc500_cv5` が public labeled 平均候補数 853.4、`recall@500=0.7082`、`recall@all=0.7819`。cooc source cap 500 は basecap より候補を広げるが、削減は reranker 側に任せる方針に沿っている。
