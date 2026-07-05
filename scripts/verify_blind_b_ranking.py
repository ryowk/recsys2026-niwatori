#!/usr/bin/env python3
"""Compare a produced Blind-B ranked artifact against the submitted reference.

Two modes:

- default (report): print how close the produced top-20 is to the submitted
  ranking and always exit 0. Use this for the from-weights path
  (`run_blind_b.sh`), where the two-tower / dense encodes run on the GPU in
  bf16 and are not bit-reproducible — the ranking should be *close* (same
  logic), not bit-identical.
- --strict: require every row's top-20 to match the submission exactly, exit 1
  otherwise. Use this for the load-only path (shipped candidates + caches, no
  GPU regeneration), where the LightGBM predict is deterministic and the
  ranking IS bit-identical.

Usage:
    uv run python scripts/verify_blind_b_ranking.py [--strict] [produced.npz] [reference.npz]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCED = (
    REPO_ROOT
    / "artifacts/runs/reranker/protocol_098_union_rich_lgbm"
    / "blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5/full_public/blind_b/ranked.npz"
)
DEFAULT_REFERENCE = REPO_ROOT / "reference/blind_b_ranked.npz"


def _keys(npz):
    return [k.decode() if isinstance(k, bytes) else str(k) for k in npz["keys"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="require bit-exact top-20 (exit 1 on any mismatch)")
    parser.add_argument("produced", nargs="?", default=str(DEFAULT_PRODUCED))
    parser.add_argument("reference", nargs="?", default=str(DEFAULT_REFERENCE))
    args = parser.parse_args()

    a = np.load(args.produced, allow_pickle=False)
    b = np.load(args.reference, allow_pickle=False)
    ref_by_key = {k: i for i, k in enumerate(_keys(b))}
    ref_idx = np.asarray(b["track_idx"])

    keys_a = _keys(a)
    idx_a = np.asarray(a["track_idx"])
    missing = [k for k in keys_a if k not in ref_by_key]
    if missing:
        raise SystemExit(f"produced rows absent from reference: {len(missing)} (e.g. {missing[:3]})")

    n = len(keys_a)
    overlaps, exact_set, exact_ordered = [], 0, 0
    for i, k in enumerate(keys_a):
        pa = idx_a[i, :20]
        pb = ref_idx[ref_by_key[k], :20]
        sa, sb = set(pa.tolist()), set(pb.tolist())
        overlaps.append(len(sa & sb) / 20.0)
        exact_set += sa == sb
        exact_ordered += bool(np.array_equal(pa, pb))

    mean_overlap = float(np.mean(overlaps))
    print(f"rows: {n}")
    print(f"top-20 mean overlap (order-independent): {mean_overlap:.4f}")
    print(f"rows with identical top-20 set:          {exact_set}/{n}")
    print(f"rows with identical top-20 order:        {exact_ordered}/{n}")

    if args.strict:
        if exact_ordered == n:
            print("PASS (strict): blind_b top-20 identical to the submitted ranking")
        else:
            raise SystemExit(f"FAIL (strict): {n - exact_ordered}/{n} rows differ from the submission")
        return

    # report mode: the from-weights path is not bit-reproducible (GPU bf16
    # encodes). A high overlap means the logic matches; small positional
    # differences are expected noise.
    if mean_overlap >= 0.98 and exact_set >= n - 2:
        verdict = "MATCH (logic reproduced; positional noise only)"
    elif mean_overlap >= 0.9:
        verdict = "CLOSE (minor drift from GPU nondeterminism)"
    else:
        verdict = "DIVERGENT — investigate (possible logic difference, not just noise)"
    print(f"VERIFY SUMMARY: {verdict} (mean overlap {mean_overlap:.4f}, "
          f"exact-order {exact_ordered}/{n})")


if __name__ == "__main__":
    main()
