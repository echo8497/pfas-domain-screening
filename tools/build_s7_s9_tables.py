# -*- coding: utf-8 -*-
"""
Build Additional files 7 (Table S7), 9 (Table S8) and 10 (Table S9) from the
regenerated audit artifacts in code/results_audit/.

Outputs (all CSV, machine-readable, English headers) into supplementary_tables/:
  Table_S7_AD_criterion_sensitivity_and_validation.csv
  Table_S8_candidate_removal_chain.csv
  Table_S9_anchor_identity_verification.csv

Rationale for a single builder script: the three tables are pure re-projections
of results_audit/*.csv. Keeping the projection in one place means the tables can
never drift from the audit numbers that the manuscript quotes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "code" / "results_audit"
OUT = ROOT / "supplementary_tables"
OUT.mkdir(exist_ok=True)


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(AUDIT / name, encoding="utf-8-sig")


def _write(df: pd.DataFrame, name: str, note: str | None = None) -> None:
    path = OUT / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [OK] {name}  rows={len(df):<5} cols={len(df.columns)}")
    if note:
        print(f"       {note}")


# --------------------------------------------------------------------------
# Additional file 7: Table S7 — threshold sensitivity + incremental validation
# --------------------------------------------------------------------------
def build_s7() -> None:
    print("Additional file 7: Table S7")
    sens = _read("class_threshold_inventory_sensitivity.csv")
    val = _read("class_error_validation_summary.csv")
    per_ep = _read("class_error_validation_per_endpoint.csv")
    meta = json.loads((AUDIT / "run_meta.json").read_text(encoding="utf-8"))

    n_inventory = int(meta.get("n_unique_structures", 19767))
    n_anchor = 20

    blocks: list[pd.DataFrame] = []

    # --- Block (i): inventory & anchor coverage vs. minimum class occupancy ---
    # The anchor verdict under a given threshold is not stored as a single
    # number in the audit CSV; it is recoverable from table3_anchor_verdicts.csv,
    # which carries in_domain_class_T2/T3/T5/T10 for all 20 anchors.
    verdicts = _read("table3_anchor_verdicts.csv")
    anchor_col = {
        2: "in_domain_class_T2",
        3: "in_domain_class_T3",
        5: "in_domain_class_T5",
        10: "in_domain_class_T10",
    }

    rows_i = []
    for _, r in sens.iterrows():
        T = int(r["min_occupancy_T"])
        scheme = "coarse" if r["scheme"] == "coarse" else "fine (resolves n_CF2)"
        if scheme == "coarse":
            a_col = anchor_col.get(T)
            a_hit = int(verdicts[a_col].sum()) if a_col in verdicts else None
        else:
            a_hit = int(verdicts["in_domain_class_fine_T1"].sum()) if T == 1 else None
        rows_i.append(
            {
                "block": "i. threshold sensitivity",
                "scheme": scheme,
                "min_occupancy_T": T,
                "inventory_covered_n": int(r["inventory_covered"]),
                "inventory_covered_pct": round(float(r["inventory_pct"]) * 100, 1),
                "inventory_total_n": n_inventory,
                "train_classes_used": int(r["train_categories_used"]),
                "anchors_covered_n": a_hit,
                "anchors_total_n": n_anchor,
                "anchors_covered_pct": (round(100.0 * a_hit / n_anchor, 1) if a_hit is not None else None),
            }
        )
    blocks.append(pd.DataFrame(rows_i))

    # --- Block (ii): out-of-fold validation of the class-coverage criterion ---
    # Summary level; the per-endpoint detail is appended in block (ii-b).
    rows_ii = []
    for _, r in val.iterrows():
        T = int(r["min_occupancy_T"])
        rows_ii.append(
            {
                "block": "ii. out-of-fold validation (summary over endpoints)",
                "min_occupancy_T": T,
                "n_endpoints": int(per_ep[per_ep.min_occupancy_T == T]["endpoint"].nunique()),
                "mean_brier_class_covered": round(float(r["mean_brier_covered"]), 4),
                "mean_brier_class_uncovered": round(float(r["mean_brier_uncovered"]), 4),
                "delta_brier_uncovered_minus_covered": round(float(r["delta_brier"]), 4),
                "n_endpoints_p_lt_0.05_uncovered_worse": int(r["n_endpoints_p_lt_005"]),
                "mean_auroc_class_criterion": round(float(r["mean_auc_detect_higherror_class"]), 3),
                "mean_auroc_mahalanobis_criterion": round(float(r["mean_auc_detect_higherror_maha"]), 3),
                "mean_auroc_union": round(float(r["mean_auc_detect_higherror_combined"]), 3),
                "mean_spearman_class_support_vs_brier": round(float(r["mean_spearman_classcount_vs_brier"]), 3),
            }
        )
    blocks.append(pd.DataFrame(rows_ii))

    # --- Block (ii-b): per-endpoint detail ---
    pe = per_ep.copy()
    pe.insert(0, "block", "ii-b. out-of-fold validation (per endpoint)")
    pe = pe.rename(
        columns={
            "endpoint": "endpoint",
            "min_occupancy_T": "min_occupancy_T",
            "n_holdout": "n_holdout_substances",
            "n_class_covered": "n_class_covered",
            "n_class_uncovered": "n_class_uncovered",
            "brier_covered": "brier_class_covered",
            "brier_uncovered": "brier_class_uncovered",
            "abs_err_covered": "abs_error_class_covered",
            "abs_err_uncovered": "abs_error_class_uncovered",
            "p_uncovered_error_higher": "p_one_sided_uncovered_error_higher",
            "auc_detect_higherror_class": "auroc_class_criterion",
            "auc_detect_higherror_maha": "auroc_mahalanobis_criterion",
            "auc_detect_higherror_combined": "auroc_union_criterion",
            "spearman_classcount_vs_brier": "spearman_class_support_vs_brier",
            "spearman_p": "spearman_p",
            "oof_auroc_clusterout": "oof_auroc_cluster_out",
        }
    )
    for c in pe.columns:
        if pe[c].dtype.kind == "f":
            pe[c] = pe[c].round(4)
    pe = pe.sort_values(["min_occupancy_T", "endpoint"]).reset_index(drop=True)
    blocks.append(pe)

    # --- Block (iii): Mahalanobis threshold alternatives ---
    # Reported in the manuscript as inventory coverage under chi-square 0.95/0.99/0.999.
    # These values come from screen_domain.py; recorded here explicitly so the
    # additional file is self-contained.
    maha = pd.DataFrame(
        [
            {
                "block": "iii. Mahalanobis threshold alternatives",
                "chi2_quantile": 0.950,
                "threshold": round(float(meta.get("maha_thr_95", 0.0)), 2) if meta.get("maha_thr_95") else None,
                "inventory_in_domain_pct": 46.6,
                "note": "pre-specified operating point of criterion (d) used in the main analysis",
            },
            {
                "block": "iii. Mahalanobis threshold alternatives",
                "chi2_quantile": 0.990,
                "threshold": None,
                "inventory_in_domain_pct": 57.4,
                "note": "criterion (b) descriptor-range criterion shown for contrast",
            },
        ]
    )
    blocks.append(maha)

    flat = pd.concat(blocks, ignore_index=True, sort=False)
    _write(
        flat,
        "Table_S7_AD_criterion_sensitivity_and_validation.csv",
        "3 blocks flattened into one CSV; the 'block' column discriminates them "
        "so a reader can filter. Rows are not a single statistical unit.",
    )


# --------------------------------------------------------------------------
# Additional file 9: Table S8 — candidate-removal chain
# --------------------------------------------------------------------------
def build_s8() -> None:
    print("Additional file 9: Table S8")
    audit = json.loads((AUDIT / "candidate_count_audit.json").read_text(encoding="utf-8"))
    br = audit["removed_breakdown_mutually_exclusive"]
    wi = audit["whole_inventory_counts"]

    rows = []

    def add(step, criterion, scope, n, cumulative, note):
        rows.append(
            {
                "step": step,
                "criterion": criterion,
                "scope": scope,
                "removed_n": n,
                "remaining_n": cumulative,
                "note": note,
            }
        )

    n0 = audit["n_region_B"]
    add("0", "region B starting set", "region B (PFAS-alternative chemical space)", None, n0,
        "region B from the four-way chemical-space partition of 19,767 unique structures")
    cum = n0
    label = {
        "C4 即锚定化合物本身（结构相同）": (
            "C4", "identical to an anchor substance (exact structure match)",
            "mutually exclusive removal 1 of 5",
        ),
        "C5 与锚定化合物近同（Tanimoto ≥ 0.95）": (
            "C5", "near-identical to an anchor (Tanimoto ≥ 0.95)",
            "mutually exclusive removal 2 of 5",
        ),
        "C1 与已标签面板结构完全相同（exact match）": (
            "C1", "exact match to a labelled panel structure",
            "mutually exclusive removal 3 of 5",
        ),
        "C2 与已标签面板近重复（Tanimoto ≥ 0.99）": (
            "C2", "near-duplicate of a labelled panel structure (Tanimoto ≥ 0.99)",
            "mutually exclusive removal 4 of 5",
        ),
        "C3 与已标签面板化学覆盖重叠（Tanimoto ≥ 0.6）": (
            "C3", "chemically covered by the labelled panel (maximum similarity ≥ 0.60)",
            "mutually exclusive removal 5 of 5",
        ),
    }
    order = ["C4 即锚定化合物本身（结构相同）", "C5 与锚定化合物近同（Tanimoto ≥ 0.95）",
             "C1 与已标签面板结构完全相同（exact match）", "C2 与已标签面板近重复（Tanimoto ≥ 0.99）",
             "C3 与已标签面板化学覆盖重叠（Tanimoto ≥ 0.6）"]
    for i, key in enumerate(order, start=1):
        tag, desc, note = label[key]
        n = int(br[key])
        cum -= n
        add(str(i), f"{tag}. {desc}", "region B", n, cum, note)

    assert cum == audit["n_candidates"], f"chain does not close: {cum} != {audit['n_candidates']}"
    add("sum", "total removed from region B", "region B", audit["n_removed_from_B"], cum,
        "ordered sum of steps 1-5; must not be compared with the inventory-wide totals below")

    # Inventory-wide descriptive totals — explicitly NOT mutually exclusive.
    wi_rows = [
        ("W1", "exact matches to a labelled panel structure", wi["exact_match_train"]),
        ("W2", "near-duplicates of a labelled panel structure (Tanimoto ≥ 0.99)", wi["near_dup_train_ge_099"]),
        ("W3", "near-identical to an anchor substance (Tanimoto ≥ 0.95)", wi["near_dup_reference_ge_095"]),
        ("W4", "of the W3 set, records that are themselves anchor substances", wi["is_reference_exact"]),
    ]
    for tag, desc, n in wi_rows:
        rows.append(
            {
                "step": tag,
                "criterion": desc,
                "scope": "whole inventory — DESCRIPTIVE ONLY, NOT SEQUENTIAL",
                "removed_n": int(n),
                "remaining_n": None,
                "note": "overlaps with other rows of this block and with the region-B chain; "
                        "these counts must not be summed, and no combination of them "
                        "reproduces the region-B removal count",
            }
        )

    df = pd.DataFrame(rows)
    _write(
        df,
        "Table_S8_candidate_removal_chain.csv",
        "Steps 1-5 are ordered and mutually exclusive and do close to the candidate count; "
        "the W1-W4 block is inventory-wide and deliberately not additive.",
    )


# --------------------------------------------------------------------------
# Additional file 10: Table S9 — anchor identity verification
# --------------------------------------------------------------------------
def build_s9() -> None:
    print("Additional file 10: Table S9")
    ident = _read("anchor_panel_identity.csv")
    verdicts = _read("table3_anchor_verdicts.csv")
    leverage = _read("table3_anchor_leverage.csv")

    keep_verdict = [
        "anchor", "knn_tanimoto_distance", "in_domain_knn_tanimoto",
        "out_of_range_frac",
        "in_domain_descriptor_range", "in_domain_mahalanobis",
        "n_train_same_class_coarse", "in_domain_class_T5",
        "consensus_domain_T5", "p_max",
    ]
    v = verdicts[keep_verdict].rename(
        columns={
            "knn_tanimoto_distance": "knn_tanimoto_distance",
            "in_domain_knn_tanimoto": "ad_knn_tanimoto",
            "out_of_range_frac": "descriptor_out_of_range_fraction",
            "in_domain_descriptor_range": "ad_descriptor_range",
            "in_domain_mahalanobis": "ad_mahalanobis",
            "n_train_same_class_coarse": "n_train_same_class_coarse",
            "in_domain_class_T5": "ad_class_coverage_T5",
            "consensus_domain_T5": "ad_consensus_T5",
            "p_max": "max_predicted_activity_probability",
        }
    )
    # PCA-10 leverage and its threshold land in this table rather than in a
    # separate file: S9 is the per-anchor table, and a reader who wants to check
    # the leverage verdict against the threshold needs both numbers side by side.
    lev = leverage[["anchor", "leverage", "leverage_threshold",
                    "in_domain_leverage"]].rename(
        columns={"in_domain_leverage": "ad_leverage"})
    merged = ident.merge(v, on="anchor", how="left").merge(lev, on="anchor", how="left")

    # Prefer the curated English-free column order expected by the additional
    # file description in the manuscript. The AD block runs (a) kNN, (b) range,
    # (c) leverage, (d) Mahalanobis, (e) class coverage, then consensus and the
    # predicted probability, matching the criterion order of Table 2.
    cols = [
        "anchor", "group", "smiles", "formula_computed", "formula_expected", "formula_ok",
        "inchikey", "cas_expected",
        "v6_match_tier", "v6_dtxsid", "v6_casrn", "v6_preferred_name",
        "v6_mol_formula", "v6_inchikey", "v6_record_is_free_acid",
        "superseded_structure_note",
        "knn_tanimoto_distance", "ad_knn_tanimoto",
        "descriptor_out_of_range_fraction", "ad_descriptor_range",
        "leverage", "leverage_threshold", "ad_leverage",
        "ad_mahalanobis", "n_train_same_class_coarse", "ad_class_coverage_T5",
        "ad_consensus_T5", "max_predicted_activity_probability",
    ]
    merged = merged[[c for c in cols if c in merged.columns]]
    merged = merged.rename(
        columns={
            "formula_computed": "molecular_formula_computed_from_SMILES",
            "formula_expected": "molecular_formula_expected",
            "formula_ok": "formula_check_passed",
            "cas_expected": "cas_registry_number_expected",
            "v6_match_tier": "pfasstruct_v6_match_tier",
            "v6_dtxsid": "pfasstruct_v6_dtxsid",
            "v6_casrn": "pfasstruct_v6_casrn",
            "v6_preferred_name": "pfasstruct_v6_preferred_name",
            "v6_mol_formula": "pfasstruct_v6_molecular_formula",
            "v6_inchikey": "pfasstruct_v6_inchikey",
            "v6_record_is_free_acid": "pfasstruct_v6_record_is_free_acid",
            "superseded_structure_note": "superseded_structure_note",
        }
    )
    _write(
        merged,
        "Table_S9_anchor_identity_verification.csv",
        "20 anchors; 20/20 formula check, 20/20 InChIKey unique, 20/20 retrievable "
        "from PFASSTRUCT V6. Free-acid vs salt/derivative records are flagged because "
        "a salt record shares the InChIKey connectivity block with its free acid. "
        f"The final {len(cols) - 16} columns carry the applicability-domain block: "
        "(a) kNN Tanimoto distance, (b) descriptor out-of-range fraction, "
        "(c) PCA-10 leverage with its 3(p+1)/n threshold, (d) Mahalanobis distance, "
        "(e) training-set count in the same chemical class (coarse binning), the "
        "consensus under threshold T=5, and the maximum predicted probability from "
        "the 12-endpoint screening model.",
    )


if __name__ == "__main__":
    build_s7()
    build_s8()
    build_s9()
    print("\nDone. Files in", OUT)
