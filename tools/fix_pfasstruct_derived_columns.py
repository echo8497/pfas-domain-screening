# -*- coding: utf-8 -*-
"""
Repair the derived structural columns of data/pfasstruct_v6.csv.

Why this exists
---------------
pfasstruct_v6.csv carries seven derived columns (n_CF2, n_ether_O, n_rings, n_C,
n_F, F_per_C, headgroup) that were written by an early build step. All seven are
recomputable from the `smiles` column, and the analysis pipeline already
recomputes them on load (screen_domain.py: "结构标志一律从 SMILES 现算，保证与
训练集口径完全一致（不依赖上游 CSV 列）"; screen_alternatives.characterize() does the
same). So no result in the manuscript depends on the stored values.

They are nevertheless repaired here because the CSV itself is a released
artifact: a third party reading it directly would otherwise pick up a stale
n_CF2 and reproduce the very bug described in pfas_qsar.structural_flags().

Diagnosis (2026-10-13): only n_CF2 is stale. The early implementation counted
`GetSubstructMatches("[CX4](F)(F)")`, which enumerates pairs of F on the same
carbon, so a CF3 terminus is counted three times. Measured on all 19,767 rows:
  stored  median 6, max 96
  correct median 3, max 60
  agreement 4,214 / 19,767 = 21.3%  (matches the figure documented in the
  structural_flags docstring). The other six columns agree 19,767/19,767.

Usage
-----
    python tools/fix_pfasstruct_derived_columns.py            # dry run: report only
    python tools/fix_pfasstruct_derived_columns.py --apply    # write, with backup
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)

import pfas_qsar as P  # noqa: E402

DATA = os.path.join(CODE, "data")
CSV = os.path.join(DATA, "pfasstruct_v6.csv")
BACKUP_DIR = os.path.join(CODE, "_archive_derived_fix_20261013")

DERIVED = ["n_CF2", "n_ether_O", "n_rings", "n_C", "n_F", "F_per_C", "headgroup"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the repaired CSV (default: report only)")
    args = ap.parse_args()

    df = pd.read_csv(CSV, encoding="utf-8-sig")
    print(f"[read] {os.path.relpath(CSV, CODE)}  {df.shape[0]} 行 × {df.shape[1]} 列")

    missing = [c for c in DERIVED if c not in df.columns]
    if missing:
        print(f"[error] 缺少派生列：{missing}；本脚本只修列，不新增列。")
        return 2

    flags = pd.DataFrame([P.structural_flags(s) for s in df["smiles"]])

    print(f"\n{'列':<12}{'修复前一致':>12}{'修复后一致':>12}{'需改行数':>10}")
    changed_total = 0
    for c in DERIVED:
        stored = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        truth = pd.to_numeric(flags[c], errors="coerce").to_numpy(float)
        before = int(np.isclose(stored, truth, equal_nan=True).sum())
        n_bad = len(stored) - before
        changed_total += n_bad
        print(f"{c:<12}{before:>12}{len(stored):>12}{n_bad:>10}")
        df[c] = truth

    print(f"\n[summary] 共 {changed_total} 个单元格与现算值不一致 "
          f"（去重后涉及 {int((~np.isclose(pd.to_numeric(pd.read_csv(CSV, encoding='utf-8-sig')['n_CF2'], errors='coerce').to_numpy(float), pd.to_numeric(flags['n_CF2'], errors='coerce').to_numpy(float), equal_nan=True)).sum())} 行 n_CF2）")

    if not args.apply:
        print("\n[dry run] 未写入。加 --apply 落盘。")
        return 0

    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bak = os.path.join(BACKUP_DIR, f"pfasstruct_v6.csv.bak_{stamp}")
    shutil.copy2(CSV, bak)
    print(f"\n[backup] {os.path.relpath(bak, CODE)}")

    df.to_csv(CSV, index=False, encoding="utf-8-sig")
    print(f"[write ] {os.path.relpath(CSV, CODE)}")

    # re-verify straight from disk
    chk = pd.read_csv(CSV, encoding="utf-8-sig")
    ok_all = True
    for c in DERIVED:
        a = pd.to_numeric(chk[c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(flags[c], errors="coerce").to_numpy(float)
        n = int(np.isclose(a, b, equal_nan=True).sum())
        ok_all &= (n == len(a))
        print(f"  [verify] {c:<12} {n}/{len(a)}")
    print("\n[OK] 派生列已与现算值完全一致" if ok_all else "\n[FAIL] 复核未通过")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
