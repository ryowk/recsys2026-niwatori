# retriever/union

## ロジック

`retriever/union` は、複数の retriever artifact を同じ row key で揃え、指定順に候補を重複除去しながら結合する retriever component です。`ordered_unique` では source list の順番を優先し、同じ source 内では元 artifact の rank 順を保ちます。`round_robin` では各 source の同 rank を順番に拾います。

## 意図

単体 retriever の信号を失わず、reranker 側で「どの source に出たか」「source 内 rank は何位か」「source 固有の score / 類似度 / count は何か」を使える広い候補集合を作ります。union は最終 top20 を決める stage ではないため、config で明示しない限り union 後に強い size cutoff を入れません。

## 設定

- `union_v1.yaml`: 多数の代表 retriever source を `ordered_unique` で結合する広い union。labeled-fit source は train-row 用に OOF、pretrained / metadata source は `fit_free_all_rows` を使う (`source_policy` で解決)。
- `blind_b_safe_cv5.yaml`: BlindB にない `conversation_goal` / GPA / thought / cold-start profile 依存を落とした BlindB-safe baseline。public_labeled fold assignment と train-fitted retriever artifact を 5-fold (`artifacts/cache/splits/cv5`, `cv5_oof`) で解決し、bm25 / tfidf / two_tower / history / last / exact / tag_intent / cooc_track / transition_track / cooc_album / cooc_artist_name などに per-source `max_candidates` cap を掛ける。
- `blind_b_safe_combined_tpd1_parts_cooc500_cv5.yaml`: 上記 BlindB-safe cv5 baseline の cooc_track / transition_track source を、challenge + TalkPlayData-1 を混ぜた `cooc_track_combined_tpd1` (`oof5_top500_parts`) / `transition_track_combined_tpd1` (`oof5_top500_prob_parts`) に差し替え、cooc_track source cap を 500 に広げた現行 config。TPD1 は no-purge で全 fold 共通に使う。

`blind_b_safe_cv5.yaml` と `blind_b_safe_combined_tpd1_parts_cooc500_cv5.yaml` は supervised / train-statistical source に cv5 OOF artifact (`cv5_oof`) を、same-user personal source には strict date-censored artifact を指定します。これは reranker fit 用 train rows で in-sample signal を使わないためです。

候補数を減らす場合は、union 後の global topK truncate ではなく、config の各 source entry に `max_candidates` を持たせて source prefix を制御します。これにより ordered_unique union そのものはそのまま reranker に渡しつつ、source 側の出力設計で自然な union size を作れます。

source artifact は原則として raw path を直接書かず、`component`, `config`, `source_policy.preferred_train_row_artifact_mode`, `source_policy.preferred_inference_artifact_mode` から resolver が次の形式で決めます。

```text
artifacts/runs/retriever/<component>/<config>/<artifact_mode>/<target>
```

`target=public_labeled` では train-row-safe な `preferred_train_row_artifact_mode`、`target=blind_b` では提出用の `preferred_inference_artifact_mode` を使います。raw `artifact:` path は特殊な artifact を読むための escape hatch としてだけ使います。

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

現行 `blind_b_safe_combined_tpd1_parts_cooc500_cv5` は public labeled で平均候補数 853.4、`recall@500=0.7082`、`recall@all=0.7819` です。cooc source cap 500 は BlindB-safe baseline より候補を広げますが、最終 top20 への削減は union ではなく reranker 側に任せる方針に沿っています。候補数は union 後の global topK truncate ではなく source 側 cap で制御します。
