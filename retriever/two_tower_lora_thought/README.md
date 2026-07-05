# retriever/two_tower_lora_thought

## ロジック

`exp/113_two_tower_lora_thought` の Qwen3-Embedding-0.6B LoRA query tower と、track metadata / audio / image / CF / attribute / lyrics / popularity feature 由来の track tower を使う supervised dense retriever。query text は会話履歴と current user query から作り、BlindB-safe 実行では `BLIND_B_SAFE=1` により `conversation_goal` と `thought` を使わない。

`scripts/build_two_tower_lora_oof.py` が `cv5_oof` / `full_public` artifact を作る。`cv5_oof` では fold held-out row をその fold 外の public labeled rows で学習した model で検索し、`full_public` では public labeled 全体で fit した model を blind/test target に適用する。

## 意図

BM25 / cooc / Markov では拾いにくい semantic match を、query tower と track tower の内積で候補化する。BlindB-safe baseline では learned semantic retriever として使うが、BlindB にない goal/thought/profile 情報には依存させない。

## 設定

- `configs/basic.yaml`: 旧 wrapper 用の基本 config。
- `scripts/build_two_tower_lora_oof.py --config oof5_top500_bsafe`: BlindB-safe cv5 baseline 用。出力 top500。
- `--tpd1-mix --tpd1-max-pairs 200000 --config oof5_top500_bsafe_tpd1_sample200k`: public labeled train pairs に TalkPlayData-1 mapped pairs を reservoir sample で混ぜる試行。
- `--mode tpd1_pretrain --config oof5_top500_bsafe_tpd1_pretrain_all`: TalkPlayData-1 mapped pairs だけで 1 回 pretrain checkpoint を作る試行。後続 fold fine-tune では `--init-checkpoint` で読み込む。

## 入出力 artifact

入力:

- `output/113_two_tower_lora_thought/track_features.npz`
- public labeled dataset / `splits/public_labeled_v2_5fold`
- optional: `talkpl-ai/TalkPlayData-1`, `data/derived/spotify_uuid_map_v1.parquet`

出力:

- `artifacts/runs/retriever/two_tower_lora_thought/<config>/cv5_oof/public_labeled/candidates.npz`
- `artifacts/runs/retriever/two_tower_lora_thought/<config>/full_public/blind_b/candidates.npz`
- `artifacts/runs/retriever/two_tower_lora_thought/<config>/pretrain_tpd1/tpd1/models/pretrain_tpd1.pt`

## fit / leak 確認

public labeled train rows は OOF。fold k の retrieval model は fold k を見ずに学習する。BlindB inference は OOF artifact ではなく full_public model を使う。TPD1 mapped pairs は challenge catalog に map できる track のみ採用し、catalog 外 track は候補から除外する。TPD1 と public labeled の duplicate audit では exact full sequence / full text duplicate が見つからなかったため、TPD1 全量または固定 sample を全 fold に同じ外部集合として入れる。`track_emb.test_tracks`、BlindB labels、target future turn は使わない。

## 結果と学び

`oof5_top500_bsafe` fold0 の standalone recall は `recall@20=0.277045`、`recall@100=0.506713`、`recall@500=0.651466`、`mrr@500=0.072122`。

TPD1 mixed sample200k は fold0 で `recall@20=0.232909`、`recall@100=0.462654`、`recall@500=0.619252`、`mrr@500=0.060686` と悪化したため、simple mixed の 5-fold 展開は止めた。

TPD1-only pretrain all は `TalkPlayData-1` mapped pairs `1,221,738` 件で 1 epoch checkpoint を作り、public labeled fold0 fine-tune まで確認した。fold0 は `recall@20=0.271721`、`recall@100=0.506481`、`recall@500=0.648688`、`mrr@500=0.071144` で、public-only `oof5_top500_bsafe` よりやや弱い。`recall@200` だけは `0.569599` で public-only `0.569406` と同等だが、primary に見たい shallow recall / MRR が改善していないため、pretrain all の 5-fold 展開も止めた。
