# Paper Build

The paper uses the ACM SIGCONF two-column `acmart` format.

## Build modes

From the repository root, compile the checked-in LaTeX assets with:

```bash
bash paper/build.sh tex
```

Other modes are:

```bash
bash paper/build.sh figures
bash paper/build.sh refresh
```

`figures` rebuilds Figures 1 and 2 from `figure_sources/`. `refresh` regenerates tables and Figure 3 from existing evaluation artifacts, then compiles the paper. The full evaluation workflow is documented in [`docs/paper_evaluation.md`](../docs/paper_evaluation.md).

Either `tectonic` or `latexmk` must be available. To use a Tectonic binary outside `PATH`, run:

```bash
TECTONIC_BIN=/path/to/tectonic bash paper/build.sh tex
```

The selected TeX toolchain must provide the official ACM Primary Article Template (`acmart.cls`).

## Generated assets

`main.tex` consumes the checked-in `generated_*.tex` files and PDFs under `figures/`, so `tex` does not need evaluation artifacts. Do not edit generated assets manually.

## Validation

Run the paper-analysis tests and shell checks with:

```bash
uv run python -m pytest -q
bash -n run_paper_devset.sh paper/build.sh
```
