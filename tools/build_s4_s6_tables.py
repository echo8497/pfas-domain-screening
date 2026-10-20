# -*- coding: utf-8 -*-
"""
Build Additional file 4 (Table S4) and Additional file 6 (Table S6).

Manuscript definitions these tables must satisfy
------------------------------------------------
Additional file 4: "Chemical-space partition counts and structural profiles for
  the panel and the inventory. Region-level counts, median perfluorinated carbon
  counts, median molecular weights and aromatic/ether/acidic fractions."
Additional file 6: "Principal-component coordinates of the descriptor space for
  both datasets."

S4 previously existed only as two loose intermediates in
code/results_alternatives/ (chemical_space_partition.csv for the inventory,
train_space_counts.csv for the panel) with no panel-side structural profile at
all, and S6 not at all. Both are produced here, as one CSV per additional file.

Every structural profile is recomputed from SMILES through
pfas_qsar.structural_flags() and screen_alternatives.classify_space() — the same
functions the analysis pipeline uses. Nothing is read from a stored derived
column; the inventory CSV's n_CF2 column has been stale before, and the audit
principle in this project is that derived columns are never trusted.

Usage
-----
    python tools/build_s4_s6_tables.py            # build, with a consistency check
    python tools/build_s4_s6_tables.py --check    # check only, do not write
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
sys.path.insert(0, CODE)

import pfas_qsar as P            # noqa: E402
import screen_alternatives as SA  # noqa: E402

DATA = os.path.join(CODE, "data")
SCREEN = os.path.join(CODE, "results_screen_v6")
ALT = os.path.join(CODE, "results_alternatives")
SUP = os.path.join(ROOT, "supplementary_tables")

INVENTORY_CSV = os.path.join(DATA, "pfasstruct_v6.csv")
PANEL_CSV = os.path.join(DATA, "pfas_chengng.csv")
PCA_SRC = os.path.join(SCREEN, "pca_coords.csv")
LEGACY_PARTITION = os.path.join(ALT, "chemical_space_partition.csv")

S4_OUT = os.path.join(SUP, "Table_S4_chemical_space_partition.csv")
S6_OUT = os.path.join(SUP, "Table_S6_PCA_coordinates.csv")

# Manuscript-English region labels (Methods 2.5 / Fig. 2 legend).
REGION_EN = {
    "A: 经典长链 PFAA(监管)": "A: classical long-chain perfluoroalkyl acids",
    "B: PFAS 替代物化学空间": "B: PFAS replacement chemical space",
    "C: 含氟芳香物(药/农/材料)": "C: fluorinated aromatics",
    "D: 其他含氟结构": "D: other fluorinated structures",
}
REGION_ORDER = list(REGION_EN)


def profiles(df: pd.DataFrame, label: str, block: str) -> pd.DataFrame:
    """Counts + structural profile per region, computed from SMILES."""
    df = SA.characterize(df.copy(), label)
    df["space"] = SA.classify_space(df)
    g = (df.groupby("space")
         .agg(n=("smiles", "size"),
              median_perfluorinated_carbons=("n_CF2", "median"),
              median_molecular_weight_Da=("mol_wt", "median"),
              aromatic_fraction=("arom", "mean"),
              ether_oxygen_fraction=("n_ether_O", lambda s: float((s > 0).mean())),
              acidic_headgroup_fraction=("headgroup", lambda s: float(s.isin([1, 2, 3]).mean())))
         .reindex(REGION_ORDER))
    g["fraction_of_dataset"] = g["n"] / len(df)
    g = g.reset_index().rename(columns={"space": "region"})
    g.insert(0, "block", block)
    g.insert(1, "dataset", "bioactivity panel" if block == "panel" else "PFASSTRUCT V6 inventory")
    g.insert(2, "dataset_n", len(df))
    g["region"] = g["region"].map(REGION_EN)
    return g


def _rounded(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    for c in ("median_perfluorinated_carbons", "median_molecular_weight_Da"):
        g[c] = g[c].round(3)
    for c in ("aromatic_fraction", "ether_oxygen_fraction",
              "acidic_headgroup_fraction", "fraction_of_dataset"):
        g[c] = g[c].round(4)
    return g


def build_s4(check_only: bool) -> None:
    print("Additional file 4: Table S4")
    inv = pd.read_csv(INVENTORY_CSV, encoding="utf-8-sig")
    pan = pd.read_csv(PANEL_CSV, encoding="utf-8-sig")
    if "smiles" not in pan.columns:
        pan = pan.rename(columns={pan.columns[0]: "smiles"})

    a = profiles(inv[["smiles"]], "V6 inventory", "inventory")
    b = profiles(pan[["smiles"]], "bioactivity panel", "panel")

    print(f"  inventory: {int(a['dataset_n'].iloc[0])} structures, regions "
          f"{a['n'].tolist()}, sum {int(a['n'].sum())}")
    print(f"  panel    : {int(b['dataset_n'].iloc[0])} structures, regions "
          f"{b['n'].tolist()}, sum {int(b['n'].sum())}")

    # Consistency against the stored partition intermediate: counts, and the
    # median perfluorinated carbon count that the stale-column episode touched.
    # Compared on the *unrounded* values so that the rounding applied for output
    # cannot show up as a discrepancy.
    if os.path.exists(LEGACY_PARTITION):
        old = pd.read_csv(LEGACY_PARTITION, encoding="utf-8-sig")
        old["region"] = old["space"].map(REGION_EN)
        m = a.merge(old[["region", "n", "median_CF2", "median_MW"]], on="region",
                    how="inner", suffixes=("_new", "_old"))
        bad = []
        for _, r in m.iterrows():
            if int(r["n_new"]) != int(r["n_old"]):
                bad.append(f"{r['region']}: n {r['n_new']} vs {r['n_old']}")
            if not np.isclose(float(r["median_perfluorinated_carbons"]),
                              float(r["median_CF2"]), rtol=0, atol=1e-9):
                bad.append(f"{r['region']}: median CF2 "
                           f"{r['median_perfluorinated_carbons']} vs {r['median_CF2']}")
            if not np.isclose(float(r["median_molecular_weight_Da"]),
                              float(r["median_MW"]), rtol=1e-12, atol=1e-9):
                bad.append(f"{r['region']}: median MW "
                           f"{r['median_molecular_weight_Da']} vs {r['median_MW']}")
        if bad:
            print("  [WARN] 与 results_alternatives/chemical_space_partition.csv 不一致：")
            for x in bad:
                print(f"         {x}")
        else:
            print("  [OK] 与 results_alternatives/chemical_space_partition.csv "
                  f"逐区一致（{len(m)} 区，计数 + 中位全氟碳数 + 中位分子量）")
    else:
        print("  [skip] 未找到 chemical_space_partition.csv，跳过一致性核对")

    if check_only:
        return
    out = pd.concat([_rounded(a), _rounded(b)], ignore_index=True)
    out.to_csv(S4_OUT, index=False, encoding="utf-8-sig")
    print(f"  [write] {os.path.relpath(S4_OUT, ROOT)}  "
          f"rows={len(out)} cols={len(out.columns)}")
    print("          NOTE: the two blocks are different statistical populations; "
          "`block` separates them and rows are not one unit of observation.")


def build_s6(check_only: bool) -> None:
    print("\nAdditional file 6: Table S6")
    if not os.path.exists(PCA_SRC):
        print(f"  [skip] 缺少 {os.path.relpath(PCA_SRC, CODE)}；先运行 screen_domain.py")
        return
    d = pd.read_csv(PCA_SRC, encoding="utf-8-sig")
    d = d.rename(columns={"split": "dataset"})
    d["dataset"] = d["dataset"].map({"train": "bioactivity panel",
                                     "V6": "PFASSTRUCT V6 inventory"}).fillna(d["dataset"])
    # order: identifier columns first, then coordinates, then annotations
    front = ["dataset", "smiles"]
    coords = [c for c in d.columns if c.startswith("PC")]
    rest = [c for c in d.columns if c not in front + coords]
    d = d[front + coords + rest]
    print(f"  {len(d)} rows × {len(d.columns)} cols; datasets: "
          f"{d['dataset'].value_counts().to_dict()}")
    if check_only:
        return
    d.to_csv(S6_OUT, index=False, encoding="utf-8-sig")
    print(f"  [write] {os.path.relpath(S6_OUT, ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="核对但不写文件")
    args = ap.parse_args()
    os.makedirs(SUP, exist_ok=True)
    print("=" * 78)
    build_s4(args.check)
    build_s6(args.check)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
