# qwen36_27b responder

Qwen3.6-27B (Alibaba, latest dense, Apache 2.0) で `predicted_response` を生成.

Qwen3.6 はデフォルトで thinking mode が有効なので、`chat_template_kwargs.enable_thinking: false` で direct response モードに切替.

## configs
- `default.yaml`: top_k=3, max_new_tokens=80, temperature=0.7, thinking off
- `rich_context.yaml`: ranked tracks に加えて user profile / conversation goal / history / current thought / prior GPA を渡す旧 098 baseline 用 config。
- `rich_context_compact_union.yaml`: `protocol_098_union_rich_lgbm/independent_compact_union_zeroscore_posonly_t200` の ranked artifact を読む旧 compact union baseline 用 config。Blind A submission `740634` で composite 0.5495 / LLM Judge 4.30。
- `rich_context_hierpop_tagchain.yaml`: 現行提出 baseline。`protocol_098_union_rich_lgbm/independent_compact_plus_transition_track_prob_last_metadata_hierpop_tagchain_b1_a11_cleanrank_t200` の ranked artifact を読み、rich context prompt で response を生成する。Blind A submission `745476` で composite 0.5961 / LLM Judge 4.40。
- `rich_context_transition_last_metadata_awave1.yaml`: 直前提出 baseline。A.Wave1 metadata reranker の ranked artifact を読み、rich context prompt で response を生成する。Blind A submission `743971` で composite 0.5693 / LLM Judge 4.10。
- `tone_discovery_critic_compact_union.yaml`: 旧 responder baseline。compact union ranking を固定し、discovery-minded music critic tone で response を生成する。Blind A submission `741898` で composite 0.5785 / LLM Judge 4.65。
- `tone_*_compact_union.yaml`: 現行 compact union ranking を固定し、tone だけを変える Blind A responder sweep 用 config。`calm_curator` / `warm_friend` / `music_critic` / `empathic_coach` / `minimal_confident` / `discovery_dj` / `discovery_critic` / `discovery_critic_tight`。
- `top1_compact_union.yaml`: 現行 compact union ranking の top1 だけを prompt に渡す config。
- `richer_context_compact_union.yaml`: 現行 compact union ranking を固定し、recent music detail と多めの track tags を追加する config。
- `rich_free_length_compact_union.yaml`: 現行 compact union ranking を固定し、文数指定を外して自然な長さに寄せる config。
- `thinking_rich_context_compact_union.yaml`: 現行 compact union ranking を固定し、Qwen thinking mode を有効化する config。提出前に `<think>` 残り、空 response、異常長を確認する。

## 実行
```bash
python responder/qwen36_27b/main.py --config default --target blind_a
```
