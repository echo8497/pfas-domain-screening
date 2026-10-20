#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
补充表 S12 / Table 3「Leverage」列的可复核产物
=============================================

背景
----
Table 3 为 20 个锚定物质给出六个适用域判据的逐条判定，其中 kNN Tanimoto、
描述符范围、马氏距离、化学类别覆盖、一致域五项都有落盘产物
（`reference_pfas_predictions.csv`、`table3_anchor_verdicts.csv`），
唯独 **PCA-10 杠杆值（Williams plot）** 只出现在正文表格里，
`screen_domain.py` 从未把参考物的杠杆值写出。

本脚本补齐该缺口：在与 `screen_domain.py` **完全相同的预处理链路**上
（训练+查询合并建特征 → 稳健标准化 RobustScaler(5,95) → clip ±10 →
PCA-10）重算训练集与 20 个锚定物质的杠杆值，阈值 3(p+1)/n。

同时核验正文脚注中的子集关系断言：
「马氏域是杠杆域的严格子集（11,340 个杠杆域内结构中的 9,211 个；
马氏域内 ∧ 杠杆域外 = 0）」。该断言可直接由 `results_screen_v6/screen_all.csv`
的 `in_domain_leverage` 与 `in_maha` 两列复核。

输出
----
    results_audit/table3_anchor_leverage.csv
    results_audit/anchor_leverage.log
    results_audit/anchor_ad_full.csv   （六个判据合并的逐锚点审计表；**不是**投稿补充件）

用法
----
    python tools/audit_anchor_leverage.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
sys.path.insert(0, CODE)

import pfas_qsar as P          # noqa: E402
import screen_domain as S      # noqa: E402

DATA = os.path.join(CODE, "data")
PANEL = os.path.join(DATA, "pfas_chengng.csv")
QUERY = os.path.join(DATA, "pfasstruct_v6.csv")
SCREEN_DIR = os.path.join(CODE, "results_screen_v6")
AUD = os.path.join(CODE, "results_audit")
SUP = os.path.join(ROOT, "supplementary_tables")

OUT_CSV = os.path.join(AUD, "table3_anchor_leverage.csv")
OUT_LOG = os.path.join(AUD, "anchor_leverage.log")
OUT_FULL = os.path.join(AUD, "anchor_ad_full.csv")

MIN_POS = 30          # 筛查配置（= screen_domain.py --min-pos 默认值）
N_JOBS = 8
_lines: list[str] = []


def say(msg: str = "") -> None:
    print(msg)
    _lines.append(msg)


def main() -> int:
    say("=" * 78)
    say("Table 3 · PCA-10 杠杆值（Williams plot）逐锚点复核")
    say(f"运行时间 {datetime.now():%Y-%m-%d %H:%M:%S}")
    say("=" * 78)

    # ---- 与 screen_domain.main() 逐行对齐的数据装配 ----
    train, ep_cols = S.load_train(PANEL)
    query = pd.read_csv(QUERY, encoding="utf-8-sig")
    query.columns = [c.strip() for c in query.columns]
    if "smiles" not in query.columns:
        query = query.rename(columns={query.columns[0]: "smiles"})
    query["smiles"] = [P.standardize_smiles(s) for s in query["smiles"]]
    query = query[query["smiles"].notna()].drop_duplicates("smiles").reset_index(drop=True)
    q_smiles = query["smiles"].tolist()

    eps = S.modelable_endpoints(train, ep_cols, MIN_POS)
    ref = S.load_reference()
    ref["smiles"] = [P.standardize_smiles(s) for s in ref["smiles"]]
    say(f"\n训练面板 {len(train)} 条；查询清单 {len(q_smiles)} 条；"
        f"锚定物质 {len(ref)} 条；可建模终点 {len(eps)} 个（min_pos={MIN_POS}，筛查配置）")

    all_smiles = train["smiles"].tolist() + q_smiles + ref["smiles"].tolist()
    n_tr, n_q = len(train), len(q_smiles)
    X, desc, fps, qmeta, feat_names = S.build_features(
        all_smiles, True, N_JOBS, cache_dir=os.path.join(DATA, "cache"))
    X_tr = X[:n_tr]
    X_rf = X[n_tr + n_q:]
    n_desc_cols = desc.shape[1]
    say(f"特征矩阵 {X.shape[0]} × {X.shape[1]}；其中描述符块 {n_desc_cols} 维"
        f"（PCA/杠杆值仅在此块上计算，与正文方法一致）")

    # ---- 预处理链路：与 screen_domain.py [3c'] 完全一致 ----
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import PCA

    D_tr = np.nan_to_num(X_tr[:, :n_desc_cols], nan=0.0)
    D_rf = np.nan_to_num(X_rf[:, :n_desc_cols], nan=0.0)
    scaler = RobustScaler(quantile_range=(5, 95)).fit(D_tr)
    D_tr_s = np.clip(scaler.transform(D_tr), -10, 10)
    D_rf_s = np.clip(scaler.transform(D_rf), -10, 10)
    var_ok = D_tr_s.std(0) > 1e-8
    D_tr_s, D_rf_s = D_tr_s[:, var_ok], D_rf_s[:, var_ok]
    say(f"[scale] 稳健标准化后有效描述符 {D_tr_s.shape[1]}/{n_desc_cols} 维"
        f"（与 screen_log.txt 的同一行应为同一数字）")

    pca = PCA(n_components=10, random_state=S.RNG).fit(D_tr_s)
    Z_tr, Z_rf = pca.transform(D_tr_s), pca.transform(D_rf_s)
    lev = P.applicability_domain(Z_tr, Z_rf)
    thr = float(lev["threshold"])
    h_ref = np.asarray(lev["leverage_query"], dtype=float)
    h_tr = np.asarray(lev["leverage_train"], dtype=float)
    say(f"[leverage] p = {Z_tr.shape[1]}，n = {Z_tr.shape[0]}，"
        f"阈值 3(p+1)/n = {thr:.4f}")
    say(f"[leverage] 训练集自身域内 {np.mean(h_tr <= thr):.1%}"
        f"（正文 Table 2 报告 93.2%）")

    ref_out = pd.DataFrame({
        "anchor": ref["ref_name"].to_numpy(),
        "group": ref["group"].to_numpy() if "group" in ref.columns else "",
        "leverage": h_ref,
        "leverage_threshold": thr,
        "in_domain_leverage": h_ref <= thr,
    })
    ref_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    say(f"\n逐锚点杠杆值（> 阈值即域外）：")
    for _, r in ref_out.iterrows():
        flag = "✓ 域内" if r["in_domain_leverage"] else "✗ 域外"
        say(f"  {r['anchor']:<22}{r['leverage']:.4f}   {flag}")
    n_in = int(ref_out["in_domain_leverage"].sum())
    say(f"\n杠杆域内锚点：{n_in} / {len(ref_out)}"
        f"（正文 Table 3 与摘要报告 4 / 20）")

    # ---- 正文脚注子集关系断言的可复核性 ----
    say("\n" + "-" * 78)
    say("子集关系断言复核（源：results_screen_v6/screen_all.csv）")
    say("-" * 78)
    sap = os.path.join(SCREEN_DIR, "screen_all.csv")
    if os.path.exists(sap):
        cols = pd.read_csv(sap, nrows=0, encoding="utf-8-sig").columns
        if "in_domain_leverage" in cols and "in_maha" in cols:
            sc = pd.read_csv(sap, usecols=["in_domain_leverage", "in_maha"],
                             encoding="utf-8-sig")
            a = sc["in_domain_leverage"].astype(bool)
            b = sc["in_maha"].astype(bool)
            say(f"  清单结构数 {len(sc)}")
            say(f"  杠杆域内 {int(a.sum())}；马氏域内 {int(b.sum())}")
            say(f"  马氏域内 ∧ 杠杆域外 = {int((b & ~a).sum())}"
                f"  → 断言「马氏域 ⊂ 杠杆域」{'成立' if int((b & ~a).sum()) == 0 else '不成立'}")
            say(f"  正文脚注引用值：11,340 个杠杆域内 / 9,211 个马氏域内"
                f"  → 实测 {'一致' if int(a.sum()) == 11340 and int(b.sum()) == 9211 else '不一致，需回填实测值'}")
        else:
            say("  screen_all.csv 缺少 in_domain_leverage / in_maha 列，跳过")
    else:
        say(f"  未找到 {os.path.relpath(sap, CODE)}，跳过")

    # ---- S12：六个判据合并的逐锚点表 ----
    build_s12()

    with open(OUT_LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print(f"\n已写出 {os.path.relpath(OUT_LOG, ROOT)}")
    return 0 if n_in == 4 else 1


def build_s12() -> None:
    """把六个判据的逐锚点判定合并为一张投稿用补充表（Table S12）。"""
    lev = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
    ver = pd.read_csv(os.path.join(AUD, "table3_anchor_verdicts.csv"),
                      encoding="utf-8-sig")
    mem = pd.read_csv(os.path.join(AUD, "anchor_panel_membership.csv"),
                      encoding="utf-8-sig")
    d = lev.merge(ver, on="anchor", how="left", suffixes=("", "_v"))
    d = d.merge(mem[["anchor", "in_panel_by_inchikey", "labels_nonnull_12ep",
                     "max_sim_to_train", "sim_cause"]],
                on="anchor", how="left")
    keep = [c for c in [
        "anchor", "group", "inchikey", "cas_rn", "formula_rdkit",
        "knn_tanimoto_distance", "in_domain_knn_tanimoto",
        "out_of_range_frac", "in_domain_descriptor_range",
        "leverage", "leverage_threshold", "in_domain_leverage",
        "maha_distance", "in_domain_mahalanobis",
        "n_train_same_class_coarse", "n_train_same_class_fine",
        "in_domain_class_T5", "consensus_domain_T5",
        "p_max", "max_sim_to_train", "in_panel_by_inchikey",
        "labels_nonnull_12ep", "sim_cause",
    ] if c in d.columns]
    d = d[keep]
    d.to_csv(OUT_FULL, index=False, encoding="utf-8-sig")
    print(f"[full] {d.shape[0]} 行 × {d.shape[1]} 列 → "
          f"{os.path.relpath(OUT_FULL, ROOT)}")
    s = {
        "kNN Tanimoto": "in_domain_knn_tanimoto",
        "descriptor range": "in_domain_descriptor_range",
        "PCA-10 leverage": "in_domain_leverage",
        "PCA-10 Mahalanobis": "in_domain_mahalanobis",
        "chemical class (T=5)": "in_domain_class_T5",
        "consensus domain": "consensus_domain_T5",
    }
    print(f"      一致域内计数：" + "；".join(
        f"{k} {int(d[v].sum())}/{len(d)}" for k, v in s.items() if v in d.columns))


if __name__ == "__main__":
    raise SystemExit(main())
