"""出力ファイルのパス解決ヘルパー。

このリポジトリでは生成物を全て <repo>/artifacts/ 配下に置く:

  artifacts/weights/  学習済み model(HF dataset repo の weights/ と 1:1)
  artifacts/cache/    前処理産の派生 cache(HF dataset repo の cache/ と 1:1)
  artifacts/runs/     pipeline 実行の中間 artifact + 予測(local のみ)
  artifacts/results/  scores.json などの小さい結果
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
OUTPUT_DIR = ARTIFACTS_DIR / "runs"
RESULTS_DIR = ARTIFACTS_DIR / "results"
CACHE_DIR = ARTIFACTS_DIR / "cache"
WEIGHTS_DIR = ARTIFACTS_DIR / "weights"


def _experiment_dir(caller_file: str | Path, base: Path) -> Path:
    name = Path(caller_file).resolve().parent.name
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def experiment_output_dir(caller_file: str | Path) -> Path:
    """<repo>/artifacts/runs/<name>/ を返す。gitignore 配下、任意の成果物用。"""
    return _experiment_dir(caller_file, OUTPUT_DIR)


def experiment_results_dir(caller_file: str | Path) -> Path:
    """<repo>/artifacts/results/<name>/ を返す。小さい結果ファイル用。"""
    return _experiment_dir(caller_file, RESULTS_DIR)
