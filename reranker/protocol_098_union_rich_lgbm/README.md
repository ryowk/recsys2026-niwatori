# reranker/protocol_098_union_rich_lgbm

## ロジック

この reranker は、既存の `098_current_thought_profile_ablation` の rich feature 群を candidate artifact 上で再計算し、さらに `retriever/union` が出した source feature pack を結合して LightGBM LambdaRank で順位学習します。

主な feature は次の通りです。

- 098 系 feature: goal / query / history / track metadata / user embedding / dense query 類似度など。
- source feature: `src__<source>__present`, `src__<source>__rank`, `src__<source>__score__primary`。
- 追加 source signal: source artifact が `score__*`, `sim__*`, `count__*`, `feat__*` などの複数 per-candidate 配列を持つ場合、それらも union candidate 位置に揃えて使う。
- meta feature: `meta__source_count`, `meta__best_source_rank`, `meta__mean_source_rank`。

`score__primary` は retriever 内の primary ordering score であり、全 retriever に共通の意味を持つ正規化済み score ではありません。reranker では raw 値に加えて row 内 z-score も作り、source 間のスケール差を model が扱えるようにします。
今回提出済み baseline の config では、legacy feature plan 互換のため `meta__max_source_score__primary` も残っています。ただし、source をまたいだ `max(score__primary)` のような集約は scale が混ざるため、次の baseline では `feature_build.drop_cross_source_score_meta: true` を明示して外します。source 別の raw score / sim / count / feature は source 名付きで渡します。

## 意図

旧 098 は candidate generation 側に artist / album boost などの人為的な混合ロジックを持っていました。この component は「候補生成は独立 retriever の広い union に任せ、boost や source weighting は reranker feature として学習させる」方針を検証します。

## 設定

- `independent_full_union_zeroscore_posonly_t200.yaml`: `retriever/union/independent_full_union_cv3` の自然 union を全幅で読み、primary candidate score は 0 にして、source feature と 098 rich feature で LightGBM を 200 trees 学習する config。
- `independent_balanced_union_zeroscore_posonly_t200.yaml`: source 側 cap で候補数を約1000件規模に抑えた ordered union を使う config。
- `independent_compact_union_zeroscore_posonly_t200.yaml`: source 側 cap をさらに小さくし、候補数を約650件規模に抑えた ordered union を使う旧提出 baseline の reranker config。
- `independent_compact_union_cleanrank_zeroscore_posonly_t200.yaml`: 旧 compact union baseline と同じ候補 pool を使うが、union storage order 由来の `candidate_rank` 系 feature を 0 固定し、`meta__max_source_score__primary` を外す clean baseline 候補。
- `independent_compact_plus_transition_track_prob_cleanrank_t200.yaml`: compact-plus surface + `transition_track_last` の count / transition probability feature を使う Markov track-only 検証 config。
- `independent_compact_plus_markov_prob_cleanrank_t200.yaml`: track / album / artist / track bigram(last2) の Markov count / transition probability feature を使う検証 config。
- `independent_compact_plus_transition_track_prob_last_metadata_a_wave1_cleanrank_t200.yaml`: 直前 baseline の reranker config。`transition_track_last` の count / probability と `last_music_artist` / `last_music_album` source を使い、union storage order 由来の `candidate_rank` 系 feature を neutralize し、A.Wave1 metadata feature を追加する。
- `independent_compact_plus_transition_track_prob_last_metadata_hierpop_tagchain_b1_a11_cleanrank_t200.yaml`: 現行 baseline の reranker config。A.Wave1 metadata に加えて、track / artist / album 階層 popularity と、history tag overlap / tag TF-IDF cosine / catalog tag PPMI graph 由来の tag-chain feature を追加する。
- `blind_b_safe_t200.yaml`: BlindB-safe reference baseline。conversation_goal / GPA / thought / user profile cold-start 依存を落とした `blind_b_safe_cv3` union を使う。
- `blind_b_safe_tpd1_text_bm25_t200.yaml`: BlindB-safe baseline に `tpd1_track_text_bm25/oof3_no_ngram_purge` を追加した union を読むfold0評価config。
- `blind_b_safe_t200_cv5.yaml`: BlindB-safe reference baseline の 5-fold 版。BlindB にない `conversation_goal` / GPA / `thoughts` / user profile 依存を落とす。
- `blind_b_safe_combined_tpd1_parts_basecap_t200_cv5.yaml`: BlindB-safe cv5 baseline に combined TPD1 cooc / transition source を加えた本命外部データ baseline。
- `blind_b_safe_combined_tpd1_parts_cooc200_t200_cv5.yaml` / `blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5.yaml`: basecap から cooc source cap を 200 / 500 に広げた候補幅の評価。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_t200_cv5.yaml`: cooc500 本命に `tpd1_track_text_bm25/oof5_no_ngram_purge` を source cap 200 で追加した現行 local best。BlindB-safe 制約は維持し、TPD1 duplicate policy は no-purge all-fold 共通。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_fold0_t200_cv5.yaml` / `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_t200_cv5.yaml`: 上記 current best の candidate artifact を使い、reranker 入力を先頭 1000 候補に制限する提出可能幅の診断 / 5-fold / BlindB final config。`max_candidates: all` の full-public single model はメモリ危険域に入ったため、同じ retriever/reranker logic のまま候補幅だけを固定する。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_lgbmbinary_fold0_t200_cv5.yaml`: 上記 `k1000` の candidate / feature stack を固定し、LightGBM objective だけ binary logloss に変える fold0 診断 config。candidate row の predicted probability で query 内 rerank する。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_lgbmbinary_noweight_fold0_t200_cv5.yaml`: 上記 binary logloss を sample weight なしで学習する診断 config。query-balanced negative weight が hard negative の勾配を薄めている可能性を確認する。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_rankxendcg_fold0_t200_cv5.yaml`: 同じ `k1000` candidate / feature stack を固定し、LightGBM objective を `rank_xendcg` に変える fold0 診断 config。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_xgbrank_fold0_t200_cv5.yaml` / `..._xgbbinary_fold0_t200_cv5.yaml` / `..._catrank_fold0_t200_cv5.yaml` / `..._catbinary_fold0_t200_cv5.yaml`: 同じ `k1000` candidate / feature stack を使い、XGBoost / CatBoost の rank loss と binary logloss を fold0 で比較する診断 config。初期値は memory を見て depth 6 に抑え、見込みがあれば小範囲で tuning してから 5-fold 化する。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_xgbrank_eta08_t100_fold0_cv5.yaml`: XGBoost rank 初期値が基準より弱いものの崩壊はしていないため、learning rate を上げて tree 数を減らした短時間の大胆調整 config。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_xgbrank_gpu_fold0_t200_cv5.yaml`: XGBoost rank 初期値と同じ parameter で `device=cuda` を使う GPU smoke config。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_xgbrank_gpu_d4_eta06_t300_fold0_cv5.yaml`: XGBoost rank を浅め・強め正則化・多めの tree で試す tuning config。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_xgbrank_gpu_d8_eta03_t300_fold0_cv5.yaml`: XGBoost rank を深め・低 learning rate で試す high-capacity tuning config。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_xgbrank_gpu_d8_eta03_t300_cv5.yaml`: 上記 XGBoost high-capacity tuning を固定して 5-fold 評価する config。LightGBM lambdarank が既に tuning 済みであることを踏まえ、XGBoost も適切な depth / learning rate / regularization を選んだ状態で比較する。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_xgbrank_gpu_d8_pairwise_t300_fold0_cv5.yaml`: 上記 XGBoost d8 の capacity / regularization は維持し、objective だけ `rank:pairwise` に変える診断 config。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_xgbrank_gpu_d8_eta03_t300_lowreg_fold0_cv5.yaml`: 上記 XGBoost d8 の `rank:ndcg` を維持し、正則化を少し緩める診断 config。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_catrank_gpu_t50_fold0_cv5.yaml`: CPU CatBoost が重すぎたため、GPU + 50 iterations で速度と期待値だけを見る smoke config。
- `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_catrank_gpu_pairlogit_t100_fold0_cv5.yaml`: CatBoost GPU の `PairLogit` 100 iterations で rank-family の期待値を見る tuning / smoke config。
- `blind_b_safe_combined_tpd1_parts_cooc500_tpd1mix20k_w1_t200_cv5.yaml`: cooc500 の候補を使い、TalkPlayData-1 mapped music turns を 20k sample で reranker training group に追加する fold0 first-pass config。TPD1 group 側は source feature を 0 埋めし、dense query も 0 vector にして自己参照を避ける。
- `blind_b_safe_combined_tpd1_parts_cooc500_tpd1twostack_t200_cv5.yaml`: cooc500 の候補を使い、TalkPlayData-1 だけで pretrain した TwoTower checkpoint の query-track cosine score を `extra_candidate_feature_npz` から per-candidate feature として追加する fold0 first-pass config。
- `blind_b_safe_combined_tpd1_parts_cooc500_tpd1pairlgbm_n50k_t200_cv5.yaml`: cooc500 の候補を使い、TalkPlayData-1 だけで学習した pair-feature LightGBM scorer の score を `extra_candidate_feature_npz` から per-candidate feature として追加する fold0 first-pass config。
- `blind_b_safe_combined_tpd1_parts_cooc500_tpd1stack_combo_s1s3_t200_cv5.yaml`: cooc500 の候補を使い、TPD1-only TwoTower score と TPD1-only pair-LGBM score の両方を per-candidate feature として追加する fold0 first-pass config。

## 入出力 artifact

入力:

- `output/retriever/union/<config>/public_labeled/candidates.npz`
- `output/retriever/union/<config>/public_labeled/source_features.npz`
- 098 系 dense query cache

出力:

- `output/reranker/protocol_098_union_rich_lgbm/<config>/cv3_oof/public_labeled/ranked.npz`
- `output/reranker/protocol_098_union_rich_lgbm/<config>/cv3_oof/public_labeled/ranked_top100.jsonl`
- `results/reranker/protocol_098_union_rich_lgbm/<config>/cv3_oof/public_labeled/scores.json`

## fit / leak 確認

reranker fit は public_labeled の固定 fold で CV します。train rows には、source 側が supervised / train-statistical な場合は OOF artifact、same-user memory の場合は strict date-censored artifact を使う前提です。devset score を見た weight / threshold / source gate tuning はしません。`track_emb.test_tracks` は使いません。current user turn の `thought` は blind 入力にも提供されるため使用可です。

TPD1 mixed-train config は例外的に external row を reranker の教師 group に足します。TPD1 row 自身に対して TPD1 cooc / transition / BM25 source feature を in-sample に読ませると分布が壊れるため、first-pass では TPD1 group 側の source feature 列を 0 埋めしています。これは TPD1 内部 OOF を実装したという意味ではありません。public row / BlindB row では既存どおり selected candidate artifact の source features を使います。

TPD1 TwoTower stacking config は scorer の fit に TalkPlayData-1 train だけを使います。public / BlindB label は scorer fit に使わず、public / BlindB candidate artifact 上で score を計算して reranker feature として渡すだけです。query text は BlindB-safe にし、conversation_goal / GPA / thought / cold-start profile 依存を使いません。

TPD1 pair-LGBM stacking config は scorer の fit に TalkPlayData-1 train の mapped music turns だけを使います。first-pass では 50k sample に fit-free BM25 k64 候補を作り、gold が無い場合だけ候補内へ挿入しました。scorer 側では dense query feature と source feature を使わず、candidate rank / candidate score / prior GPA / goal sim 系を neutralize しています。public / BlindB candidate artifact は score 計算対象として使うだけで、scorer fit には使いません。

## 結果と学び

人工的に round-robin top500 へ切った `explore_core10_rr_top500_with_features` 上では、`zero score + source feature + positive rows only + 200 trees` が 3-fold nDCG@20 = 0.264 でした。`independent_full_union_zeroscore_posonly_t200` は同じ方針を、union 後 topK で再切断しない候補集合に適用する検証です。

`independent_full_union_zeroscore_posonly_t200` の fold0 は nDCG@20 = 0.2641、devset row nDCG@20 = 0.1950 でした。top500 版 fold0 の 0.2658 / 0.1970 より少し低く、追加候補の recall 増加を現状の LGBM が top20 改善に変換できていません。full union は強い reranker 用の候補 pool として残し、現時点の実用 baseline は top500 版を優先します。

docs 再確認後、union 後 truncate ではなく source 側 cap で候補数を制御する `independent_balanced_union_zeroscore_posonly_t200` と `independent_compact_union_zeroscore_posonly_t200` も追加しました。fold0 はそれぞれ nDCG@20 = 0.2641 / 0.2640 で、098 baseline より大きく強い一方、round-robin top500 fold0 の 0.2658 には少し届きませんでした。round-robin の強さは devset weight tuning ではなく均一 interleave + source feature によるものと見ています。

`independent_compact_union_zeroscore_posonly_t200` は 3-fold まで完走し、public labeled CV nDCG@20 = 0.2629 でした。旧 `protocol_098/current_thought_profile` の 0.2468 から大きく改善し、round-robin top500 版 0.2641 とほぼ同水準です。Blind A submission `740634` では、この compact reranker と `qwen36_27b/rich_context_compact_union` responder の組み合わせで composite 0.5495 / nDCG@20 0.4554 を記録し、当時の提出 baseline として使いました。

2026-05-19 の Markov 深掘りでは、`independent_compact_plus_transition_track_cleanrank_t200` が 3-fold nDCG@20 = 0.2657、count に transition probability を足した `independent_compact_plus_transition_track_prob_cleanrank_t200` が 0.2658、track / album / artist / last2 bigram を足した `independent_compact_plus_markov_prob_cleanrank_t200` が 0.2659 でした。改善幅は小さいものの、clean-rank 系では現時点で最良ラインです。

2026-05-21 時点の現行 baseline は `independent_compact_plus_transition_track_prob_last_metadata_hierpop_tagchain_b1_a11_cleanrank_t200` です。3-fold nDCG@20 = 0.273327、Blind A submission `745476` で nDCG@20 = 0.532119 / composite = 0.596125 でした。直前の A.Wave1 baseline `743971` は 3-fold nDCG@20 = 0.267744、Blind A nDCG@20 = 0.523241 / composite = 0.569266 で、hier-pop + tag-chain 追加により local CV と Blind A の両方で改善しました。

BlindB-safe reference baseline `blind_b_safe_t200` は 3-fold nDCG@20 = 0.268796。fold0 は 0.271152。`blind_b_safe_tpd1_text_bm25_t200` は fold0 nDCG@20 = 0.272541 で BlindB-safe fold0 は上回るが、combined TPD1 parts fold0 0.274086 には届かない。candidate 行数が fold0 で約54Mと重いため、単独では full 3-fold へ進める優先度は低い。

BlindB-safe cv5 では、TPD1 なし `blind_b_safe_t200_cv5` が nDCG@20 = 0.271911。combined TPD1 basecap `blind_b_safe_combined_tpd1_parts_basecap_t200_cv5` は 0.273478、cooc cap200 は 0.273960、cooc cap500 は 0.274316。cooc500 は 2026-06-25 時点の local 最良 BlindB-safe TPD1 variant で、BlindB full_public ranked artifact と template submission zip も同じ config で作成済み。

2026-06-26 の final local sprint で `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_t200_cv5` を 5-fold 実行した。nDCG@20 = 0.276162 で、cooc500 から +0.001846、全foldで cooc500 を上回った。BlindB 実提出では TPD1 なし baseline 約0.53、cooc500 約0.59 だったため、この `cooc500 + textBM25 cap200` を current best とした。ただし `max_candidates: all` の full-public single model は RSS 約166GB / available 約23GBまで上がり artifact を残せなかった。提出可能幅として `blind_b_safe_combined_tpd1_parts_cooc500_textbm25_k1000_t200_cv5` を実行し、5-fold nDCG@20 = 0.276672 で all幅 current best から +0.000510。fold0 は 0.277916 で all幅 fold0 0.278043 から -0.000127 だが、fold1-4 はすべて改善した。`k1000` の `full_public/blind_b` ranked artifact も local で作成済みで、final train matrix peak RSS は約162GB、LGBM fit 中は約105GB。

同じ `k1000` candidate / feature stack で GBDT family を比較した。LightGBM binary logloss は fold0 nDCG@20 = 0.253933、negative weight なしでも 0.258061、LightGBM `rank_xendcg` は 0.270273 で lambdarank に届かない。XGBoost `rank:ndcg` は GPU で深め低LR (`depth=8`, `eta=0.03`, `n_estimators=300`, `min_child_weight=3`, `reg_lambda=5`) にすると fold0 0.277689、5-fold 0.276406 まで来た。LightGBM k1000 5-fold 0.276672 から -0.000266 で単体ではわずかに下だが、ほぼ同等圏。LightGBM / XGBoost を対等に扱う rank-only fusion では、RRF k=20/60/100 が 0.276870-0.276881、mean rank が 0.276874。改善幅は小さいが、RRF の k と candidate cutoff 200/500 に対して安定している。CatBoost は GPU YetiRank t50 が fold0 0.235583、GPU PairLogit t100 が 0.252515 で遠く、現状は撤退。

追加で、XGBoost d8 の objective を `rank:pairwise` に変えた fold0 は 0.272661、同じ `rank:ndcg` で正則化を緩めた lowreg fold0 は 0.277355 だった。どちらも標準 d8 `rank:ndcg` fold0 0.277689 を下回るため、XGBoost の現行妥当設定は `rank:ndcg`, `depth=8`, `eta=0.03`, `n_estimators=300`, `min_child_weight=3`, `reg_lambda=5`, `subsample/colsample=0.85` とする。

2026-06-26 に TPD1 rows を reranker training group へ混ぜる first-pass として `blind_b_safe_combined_tpd1_parts_cooc500_tpd1mix20k_w1_t200_cv5` を fold0 実行した。結果は nDCG@20 = 0.273838 で、既存 cooc500 fold0 0.276740 と TPD1 なし baseline fold0 0.274001 の両方を下回った。見込み薄として 5-fold / BlindB full_public へは進めない。次は TPD1-only supervised scorer を作り、その score を reranker feature として積む stacking 方向を試す。

同日に S1 stacking として `blind_b_safe_combined_tpd1_parts_cooc500_tpd1twostack_t200_cv5` を fold0 実行した。TPD1-only TwoTower pretrain checkpoint の score feature を raw / row-z の 2 列で追加し、feature 数は 178。結果は nDCG@20 = 0.276473、train nDCG@20 = 0.281873、devset nDCG@20 = 0.194398。既存 cooc500 fold0 0.276740 を少し下回るため、この設定では 5-fold / BlindB full_public へ進めない。

続けて S3 stacking として `blind_b_safe_combined_tpd1_parts_cooc500_tpd1pairlgbm_n50k_t200_cv5` を fold0 実行した。TPD1-only pair-LGBM scorer の score feature を raw / row-z の 2 列で追加し、feature 数は 178。結果は nDCG@20 = 0.276404、train nDCG@20 = 0.281873、devset nDCG@20 = 0.193273。既存 cooc500 fold0 0.276740 を下回るため、この設定では 5-fold / BlindB full_public へ進めない。

S1 と S3 を同時に入れた `blind_b_safe_combined_tpd1_parts_cooc500_tpd1stack_combo_s1s3_t200_cv5` も fold0 実行した。feature 数は 180。結果は nDCG@20 = 0.276456、train nDCG@20 = 0.281844、devset nDCG@20 = 0.194561。S1 単体 / S3 単体とほぼ同水準で、既存 cooc500 fold0 を超えなかった。現時点では TPD1 を reranker に直接混ぜる / scorer feature として積むより、combined cooc / transition retriever source として使う設定を本命にする。
