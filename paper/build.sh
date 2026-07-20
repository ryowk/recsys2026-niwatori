#!/bin/bash
# Build the paper or its generated assets; see paper/README.md.
set -euo pipefail

PAPER_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$PAPER_DIR/.." && pwd)
FIGURE_SOURCE_DIR="$PAPER_DIR/figure_sources"
FIGURE_DIR="$PAPER_DIR/figures"
MODE=${1:-tex}

if [ -n "${TECTONIC_BIN:-}" ]; then
  engine=("$TECTONIC_BIN")
elif command -v tectonic >/dev/null 2>&1; then
  engine=(tectonic)
else
  engine=()
fi

compile_paper() {
  cd "$PAPER_DIR"
  if [ "${#engine[@]}" -gt 0 ]; then
    "${engine[@]}" --keep-intermediates --keep-logs main.tex
  elif command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
  else
    echo "Install tectonic or latexmk, or set TECTONIC_BIN=/path/to/tectonic." >&2
    exit 127
  fi
}

compile_figure() {
  local name=$1
  mkdir -p "$FIGURE_DIR"
  if [ "${#engine[@]}" -gt 0 ]; then
    cd "$FIGURE_SOURCE_DIR"
    "${engine[@]}" --outdir "$FIGURE_DIR" "$name.tex"
  elif command -v latexmk >/dev/null 2>&1; then
    cd "$FIGURE_SOURCE_DIR"
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -outdir="$FIGURE_DIR" "$name.tex"
  else
    echo "Install tectonic or latexmk, or set TECTONIC_BIN=/path/to/tectonic." >&2
    exit 127
  fi
}

case "$MODE" in
  tex)
    compile_paper
    ;;
  figures)
    compile_figure pipeline
    compile_figure responder_example
    ;;
  refresh)
    cd "$REPO_ROOT"
    uv run python scripts/analyze_paper_results.py
    compile_paper
    ;;
  *)
    echo "usage: $0 [tex|figures|refresh]" >&2
    exit 2
    ;;
esac

if [ "$MODE" = figures ]; then
  echo "paper figures: $FIGURE_DIR"
else
  echo "paper: $PAPER_DIR/main.pdf"
fi
