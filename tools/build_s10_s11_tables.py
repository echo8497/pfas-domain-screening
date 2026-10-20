#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
补充表 S10 / S11 投影脚本
=========================

从**审计与建模产物**再投影出两张投稿用补充表，不手工编表；上游产物一变，
重跑本脚本即同步。

  S10  ← results_audit/anchor_panel_membership.csv
         锚定物质 × 生物活性面板：结构身份、标签覆盖、指纹简并成因
         （对应正文 Table 3 的 "Max similarity to panel" 列与 Additional file 11）
  S11  ← results_multitask_perm/{permutation_aurocs.csv,run_meta.json}
         results_published_qmsa_perm/{permutation_aurocs.csv,run_meta.json}
         标签置换检验台账：逐次置换 AUROC + 逐 run 的次数分解与超限计数
         （对应正文 Methods 2.3 与 Results 3.1 的置换检验，Additional file 12）

输出
----
    supplementary_tables/Table_S10_anchor_panel_membership.csv
    supplementary_tables/Table_S11_label_permutation_ledger.csv

用法
----
    python tools/build_s10_s11_tables.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
AUD = os.path.join(CODE, "results_audit")
SUP = os.path.join(ROOT, "supplementary_tables")

S10_SRC = os.path.join(AUD, "anchor_panel_membership.csv")
S10_OUT = os.path.join(SUP, "Table_S10_anchor_panel_membership.csv")
S11_OUT = os.path.join(SUP, "Table_S11_label_permutation_ledger.csv")

# run 标签 → （结果目录, 人类可读配置说明）
PERM_RUNS = {
    "primary endpoint": (
        os.path.join(CODE, "results_multitask_perm"),
        "panel-level feature space (2,268 dims) and panel-level Tanimoto-Butina "
        "clustering, Tanimoto cutoff 0.7, 5-fold cluster-out CV, random forest"),
    "external QMSA dataset": (
        os.path.join(CODE, "results_published_qmsa_perm"),
        "published 17-descriptor QMSA surface representation, no structures; "
        "cluster-out by PCA in standardised descriptor space, 5-fold CV, random forest"),
}

S11_COLS = ["block", "run", "run_index", "perm_index", "permuted_AUROC",
            "observed_AUROC", "perm_mean", "perm_p95",
            "n_perm_requested", "n_perm_valid", "n_ge_real", "p_value", "config"]


def build_s10() -> int:
    if not os.path.exists(S10_SRC):
        print(f"[S10] 缺少上游产物：{S10_SRC}")
        return 0
    d = pd.read_csv(S10_SRC)
    # 列序按投稿阅读顺序整理
    order = ["anchor", "group", "inchikey", "smiles", "in_panel_by_inchikey",
             "panel_row_index", "labels_nonnull_26ep", "labels_nonnull_22ep",
             "labels_nonnull_12ep", "contributes_to_models",
             "max_sim_to_train", "sim_cause"]
    d = d[[c for c in order if c in d.columns]]
    d.to_csv(S10_OUT, index=False, encoding="utf-8-sig")
    print(f"[S10] {d.shape[0]} 行 × {d.shape[1]} 列 → "
          f"{os.path.relpath(S10_OUT, ROOT)}")
    print(f"      在面板中 {int(d['in_panel_by_inchikey'].sum())}/{len(d)}；"
          f"max_sim=1.00 共 {int((d['max_sim_to_train'] >= 1.0).sum())} 条")
    return len(d)


def build_s11() -> int:
    rows = []
    for i, (label, (dir_, cfg)) in enumerate(PERM_RUNS.items(), 1):
        meta_p = os.path.join(dir_, "run_meta.json")
        perm_p = os.path.join(dir_, "permutation_aurocs.csv")
        if not (os.path.exists(meta_p) and os.path.exists(perm_p)):
            print(f"[S11] 跳过 {label}：缺少 {os.path.relpath(perm_p, ROOT)} "
                  f"或 run_meta.json（先跑对应的建模脚本）")
            continue
        meta = json.load(open(meta_p, encoding="utf-8"))
        yr = meta.get("y_randomization") or {}
        if not yr:
            print(f"[S11] 跳过 {label}：run_meta.json 中没有 y_randomization 块")
            continue
        perm = pd.read_csv(perm_p)
        common = {
            "run": label,
            "run_index": i,
            "observed_AUROC": yr.get("real_auroc"),
            "perm_mean": yr.get("perm_mean"),
            "perm_p95": yr.get("perm_p95"),
            "n_perm_requested": yr.get("n_perm_requested"),
            "n_perm_valid": yr.get("n_perm_valid"),
            "n_ge_real": yr.get("n_ge_real"),
            "p_value": yr.get("p_value"),
            "config": cfg,
        }
        rows.append({**common, "block": "run_summary", "perm_index": None,
                     "permuted_AUROC": None})
        for _, r in perm.iterrows():
            rows.append({**common, "block": "permutation_distribution",
                         "perm_index": int(r["perm_index"]),
                         "permuted_AUROC": float(r["permuted_AUROC"])})
        print(f"[S11] {label}: 观测 {common['observed_AUROC']:.4f}，"
              f"有效置换 {common['n_perm_valid']}，"
              f"≥观测 {common['n_ge_real']}，p = {common['p_value']:.4f}")
    if not rows:
        return 0
    out = pd.DataFrame(rows)[S11_COLS]
    out.to_csv(S11_OUT, index=False, encoding="utf-8-sig")
    print(f"[S11] {len(out)} 行 → {os.path.relpath(S11_OUT, ROOT)}")
    print("      注意：block=run_summary 每 run 一行（统计单位=run）；"
          "block=permutation_distribution 每 run 每置换一行，"
          "两类行不是同一统计单位，聚合前必须按 block 过滤。")
    return len(out)


def main() -> int:
    os.makedirs(SUP, exist_ok=True)
    n10 = build_s10()
    n11 = build_s11()
    if not n11:
        print("\n[S11] 未生成——等待置换分布产物落盘后重跑本脚本。")
    return 0 if (n10 and n11) else 1


if __name__ == "__main__":
    raise SystemExit(main())
