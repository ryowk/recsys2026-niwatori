# Per-source retriever recall (public_labeled, 5-fold OOF)

Recall of the gold music track among each source's candidates, over the 129,592 public-labeled rows (train + devset). Supervised sources (two-tower, cooc/transition) use their `cv5_oof` artifacts; fit-free sources use `fit_free_all_rows`. Computed by `scripts/build_retriever_metrics.py`.

| source | mean cands | recall@20 | recall@50 | recall@100 | recall@200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 500 | 0.4102 | 0.5129 | 0.5693 | 0.6129 |
| tfidf | 300 | 0.3433 | 0.4640 | 0.5329 | 0.5762 |
| two_tower | 500 | 0.2798 | 0.4258 | 0.5087 | 0.5690 |
| history_artist | 67 | 0.2676 | 0.4268 | 0.5241 | 0.5537 |
| history_album | 16 | 0.3769 | 0.4159 | 0.4173 | 0.4173 |
| last_music_artist | 37 | 0.2874 | 0.4310 | 0.5041 | 0.5143 |
| last_music_album | 7 | 0.3213 | 0.3277 | 0.3277 | 0.3277 |
| exact_album_artist | 2 | 0.0636 | 0.0652 | 0.0652 | 0.0652 |
| tag_intent | 340 | 0.0076 | 0.0163 | 0.0291 | 0.0500 |
| cooc_track_tpd1 | 248 | 0.3585 | 0.4378 | 0.4793 | 0.5124 |
| transition_track_tpd1 | 40 | 0.1937 | 0.2134 | 0.2224 | 0.2282 |
| cooc_album | 334 | 0.2261 | 0.3396 | 0.3981 | 0.4381 |
| cooc_artist_name | 413 | 0.0220 | 0.0382 | 0.0550 | 0.0786 |
| exact_title | 0 | 0.0114 | 0.0114 | 0.0114 | 0.0114 |
| union (all 14 sources) | 853 | 0.4102 | 0.5129 | 0.5693 | 0.6129 |

The union's recall@20 = candidate recall@20 in the reranker CV report. Individual sources are intentionally narrow (each returns candidates only where its signal fires); the value is the orthogonal coverage they add to the union, not standalone recall.
