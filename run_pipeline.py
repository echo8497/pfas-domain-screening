# -*- coding: utf-8 -*-
"""
run_pipeline.py — PFAS 活性预测全流程（泄漏感知验证 + QMSA 消融）

支持两种模式
------------
1. 合成数据自检（`--demo`，无需任何数据文件）
2. 真实数据（`--input`），并支持：
   - 单终点建模（`--endpoint <列名>` 或 `--label-col`）
   - 多终点建模（`--multi-task`，遍历全部标签列并汇总）

新增能力（v2）
--------------
- **QMSA 表面静电势描述符**接入（`qmsa.py`），`--no-qmsa` 可关闭
- **特征消融实验**（`--ablation`）：2D / 2D+FP / QMSA / 2D+QMSA / ALL 在同一
  簇外划分下对比，量化 QMSA 的增量贡献 —— 论文的核心证据之一
- **σ-profile 图**（`--sigma-profile`）输出代表性 PFAS 的表面静电势分布

用法
----
    python run_pipeline.py --demo --qmsa --ablation
    python run_pipeline.py --input data/pfas_dataset.csv --multi-task \
                           --qmsa --ablation --folds 5 --repeats 3 --outdir results_real
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfas_qsar as P       # noqa: E402
import qmsa as Q            # noqa: E402


# ------------------------------------------------------------ 合成 PFAS 数据
def make_synthetic_pfas(n_target: int = 700, seed: int = 0) -> pd.DataFrame:
    """系统性生成 PFAS 样结构：5 类头基 × 链长 1–14 × 醚氧位置 × 支链。
    活性由链长、头基、醚基、支链共同决定 + 噪声 —— 用于流程自检。"""
    rng = np.random.default_rng(seed)
    heads = {1: "OC(=O)", 2: "OS(=O)(=O)", 3: "NS(=O)(=O)", 4: "OP(=O)(O)O", 5: "O"}
    head_eff = {1: 0.9, 2: -0.7, 3: -0.2, 4: 0.3, 5: 0.5}

    smis, meta = [], []
    for chain in range(1, 15):
        for head, prefix in heads.items():
            for epos in [None] + list(range(2, chain + 1)):
                for branch in (0, 1):
                    body = ""
                    for i in range(1, chain + 1):
                        if epos is not None and i == epos:
                            body += "O"
                        body += "C(F)(F)"
                    if branch:
                        body = body[:-6] + "C(F)(C(F)(F)F)"
                    smis.append(prefix + body + "F")
                    meta.append({"chain": chain, "headgroup": head,
                                 "ether": int(epos is not None), "branch": branch})

    df = pd.DataFrame(meta)
    df["smiles"] = smis
    df = df.drop_duplicates("smiles").reset_index(drop=True)
    if len(df) > n_target:
        df = df.sample(n=n_target, random_state=seed).reset_index(drop=True)

    logit = (0.55 * (df["chain"] - 6.5)
             + df["headgroup"].map(head_eff).astype(float)
             + 0.6 * df["ether"] + 0.4 * df["branch"]
             + rng.normal(0, 0.6, len(df)))
    df["active"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df[["smiles", "active", "chain", "headgroup", "ether", "branch"]]


# ------------------------------------------------------------ 特征构建
def build_feature_blocks(smiles: list[str], use_qmsa: bool = True,
                         qmsa_charge: str = "auto", qmsa_unit: str = "kcal",
                         verbose: bool = True) -> dict:
    """构建 2D / 指纹 / QMSA 三个特征块，返回 dict[str, np.ndarray]。"""
    print("      · 计算 RDKit 2D 描述符 …")
    desc = P.compute_descriptors(smiles, verbose=False)
    num_cols = [c for c in desc.columns
                if c != "smiles" and pd.api.types.is_numeric_dtype(desc[c])]
    D = desc[num_cols].replace([np.inf, -np.inf], np.nan)
    D = D.fillna(D.median(numeric_only=True)).fillna(0.0)
    corr = D.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > 0.95).any()]
    D = D.drop(columns=drop)
    print(f"        2D: {len(num_cols)} → 去冗余 {D.shape[1]}")

    print("      · 计算 Morgan 指纹 …")
    FP = P.compute_fingerprints(smiles, radius=2, n_bits=2048).astype(float)

    blocks = {"2D": D.to_numpy(dtype=float), "FP": FP}

    if use_qmsa:
        print("      · 计算 QMSA 表面静电势描述符（ETKDG + MMFF94 + Shrake-Rupley）…")
        t0 = time.time()
        qdf = Q.qmsa_matrix(smiles, verbose=verbose, charge_method=qmsa_charge,
                            unit=qmsa_unit)
        ok = int(qdf.get("qmsa_ok", pd.Series([0])).fillna(0).sum())
        print(f"        成功率 {ok}/{len(smiles)}（{ok / max(len(smiles), 1):.0%}），"
              f"耗时 {time.time() - t0:.1f}s")
        QM = qdf[Q.QMSA_FEATURES].replace([np.inf, -np.inf], np.nan)
        # QMSA 计算失败的分子用列中位数填补，避免整行 NaN
        QM = QM.fillna(QM.median(numeric_only=True)).fillna(0.0)
        blocks["QMSA"] = QM.to_numpy(dtype=float)
        blocks["_qmsa_df"] = qdf
        print(f"        QMSA: {QM.shape[1]} 维"
              f"（电荷法 {qdf['qmsa_charge_method'].mode().iat[0] if len(qdf) else 'n/a'}）")
    return blocks


FEATURE_SETS = {
    "2D": ["2D"],
    "2D+FP": ["2D", "FP"],
    "QMSA": ["QMSA"],
    "2D+QMSA": ["2D", "QMSA"],
    "ALL(2D+FP+QMSA)": ["2D", "FP", "QMSA"],
}


def assemble(blocks: dict, keys: list[str]) -> np.ndarray:
    mats = [blocks[k] for k in keys if k in blocks]
    return np.hstack(mats) if mats else np.zeros((0, 0))


# ------------------------------------------------------------ 单终点评估
def evaluate_endpoint(smiles, y, blocks, models, splits_map, min_pos=10):
    """对单个终点跑三种划分策略，返回 {策略: 性能表}。"""
    res = {}
    for strat, sp in splits_map.items():
        if not sp:
            res[strat] = pd.DataFrame()
            continue
        X = assemble(blocks, ["2D", "FP", "QMSA"])
        res[strat] = P.nested_cv_summary(models, X, y, sp)
    return res


def ablation_table(blocks, y, splits, models, sets=FEATURE_SETS, mask=None) -> pd.DataFrame:
    """特征消融：同一划分下比较不同特征组合的性能。

    mask: 布尔数组。当标签稀疏（如多终点模式）时，用于把特征块裁剪到
          该终点的有效子集，保证与 y / splits 的行对齐。
    """
    rows = []
    for name, keys in sets.items():
        if not all(k in blocks for k in keys):
            continue
        X = assemble(blocks, keys)
        if mask is not None:
            X = X[mask]
        tab = P.nested_cv_summary(models, X, y, splits)
        if tab.empty:
            continue
        best = tab["AUROC"].idxmax()
        rows.append({
            "feature_set": name, "n_features": X.shape[1], "best_model": best,
            "AUROC": tab.loc[best, "AUROC"], "AUPRC": tab.loc[best, "AUPRC"],
            "MCC": tab.loc[best, "MCC"], "Brier": tab.loc[best, "Brier"],
            "AUROC_mean_all_models": tab["AUROC"].mean(),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
    base = df.loc[df.feature_set == "2D+FP", "AUROC"]
    if len(base):
        df["dAUROC_vs_2D+FP"] = df["AUROC"] - float(base.iloc[0])
    return df


# ------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="用合成数据自检全流程")
    ap.add_argument("--input", type=str, default=None, help="真实数据 CSV")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--label-col", default="active")
    ap.add_argument("--endpoint", default=None, help="指定单终点列名")
    ap.add_argument("--multi-task", action="store_true", help="遍历全部标签列分别建模")
    ap.add_argument("--max-endpoints", type=int, default=0, help="多终点模式上限（0=不限）")
    ap.add_argument("--min-pos", type=int, default=15, help="终点最少阳性数（否则跳过）")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cutoff", type=float, default=0.7, help="Butina 距离阈值")
    ap.add_argument("--qmsa", dest="qmsa", action="store_true", default=True)
    ap.add_argument("--no-qmsa", dest="qmsa", action="store_false")
    ap.add_argument("--qmsa-charge", default="auto", choices=["auto", "mmff94", "eem"])
    ap.add_argument("--qmsa-unit", default="kcal", choices=["kcal", "au", "published"],
                    help="QMSA 描述符单位体系（published = 对齐已发表 QMSA 数据集的混合单位）")
    ap.add_argument("--ablation", action="store_true", help="跑特征消融实验")
    ap.add_argument("--sigma-profile", action="store_true", help="输出 σ-profile 图")
    ap.add_argument("--models", default="LR,RF,SVM",
                    help="参与建模的模型，逗号分隔（可选 LR,RF,SVM,XGB,LGBM）")
    ap.add_argument("--n-perm", type=int, default=30, help="y-随机化置换次数")
    ap.add_argument("--perm-endpoint", default="PCBA-686978",
                    help="多终点模式下做标签置换检验的终点（缺省=主终点 PCBA-686978）")
    ap.add_argument("--shap", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()
    print("=" * 74)
    print("PFAS 活性预测 —— 泄漏感知验证 + QMSA 表面静电势描述符")
    print("=" * 74)

    # ---- 1. 数据 ----
    extra_meta_cols = []
    if args.demo or not args.input:
        print("\n[1/8] 合成 PFAS 数据自检")
        raw = make_synthetic_pfas(700)
        smiles_col, label_col = "smiles", "active"
        endpoints = ["active"]
        extra_meta_cols = ["chain", "headgroup", "ether", "branch"]
    else:
        print(f"\n[1/8] 读取真实数据：{args.input}")
        raw = pd.read_csv(args.input)
        smiles_col = args.smiles_col
        if smiles_col not in raw.columns:
            raise SystemExit(f"缺少 SMILES 列：{smiles_col}")
        meta_like = {"smiles", "name", "std_smiles", "n_perfluoro_C", "n_labels"}
        endpoints = [c for c in raw.columns
                     if c not in meta_like and pd.api.types.is_numeric_dtype(raw[c])]
        if args.endpoint:
            endpoints = [args.endpoint]
        elif not args.multi_task:
            endpoints = [args.label_col if args.label_col in raw.columns else endpoints[0]]
        if args.max_endpoints:
            endpoints = endpoints[:args.max_endpoints]
        extra_meta_cols = [c for c in ["name", "n_perfluoro_C"] if c in raw.columns]
        print(f"      终点数 {len(endpoints)}；化合物 {len(raw)}")

    # ---- 2. 结构标准化 + 去重 ----
    print("[2/8] 结构标准化与去重")
    raw["std_smiles"] = raw[smiles_col].map(P.standardize_smiles)
    n0 = len(raw)
    keep_cols = ["std_smiles"] + endpoints + extra_meta_cols
    df = raw[[c for c in keep_cols if c in raw.columns]].dropna(subset=["std_smiles"])
    df = df.drop_duplicates("std_smiles").reset_index(drop=True)
    print(f"      {n0} → {len(df)} 条")

    # 标签缺失处理：真实数据集（如 Cheng & Ng 面板）标签稀疏，单个终点常有大量缺失。
    # 单终点模式：先按该终点剔除缺失行（此时特征尚未计算，最省时）；
    # 多终点模式：保留全部行，建模时逐个终点取有效子集（见 [5/8]）。
    if len(endpoints) == 1:
        n1 = len(df)
        df = df.dropna(subset=endpoints).reset_index(drop=True)
        if len(df) < n1:
            print(f"      按终点 {endpoints[0]} 剔除标签缺失 {n1 - len(df)} 条 → {len(df)} 条")
    smiles = df["std_smiles"].tolist()

    # ---- 3. 特征 ----
    print("[3/8] 构建特征块")
    blocks = build_feature_blocks(smiles, use_qmsa=args.qmsa,
                                  qmsa_charge=args.qmsa_charge,
                                  qmsa_unit=args.qmsa_unit)
    keys_all = ["2D", "FP"] + (["QMSA"] if args.qmsa else [])
    X = assemble(blocks, keys_all)
    print(f"      特征矩阵：{X.shape[0]} × {X.shape[1]}")

    # ---- 4. 划分策略 ----
    print("[4/8] 构建划分策略（随机 / 骨架 / Tanimoto-Butina 簇外）")
    fps = P.fingerprint_bitvects(smiles, radius=2, n_bits=2048)
    cluster_labels = P.butina_cluster_labels(fps, cutoff=args.cutoff)
    n_clusters = len(np.unique(cluster_labels))
    scaffolds = np.array([P.murcko_scaffold(s) for s in smiles])
    print(f"      Butina 簇数 {n_clusters}（cutoff={args.cutoff}，平均每簇 {len(df)/n_clusters:.1f}）")
    print(f"      Murcko 骨架：空串 {float(np.mean(scaffolds == '')):.1%}，唯一骨架 {len(np.unique(scaffolds))}")
    models_all = P.get_models()
    want = [m.strip() for m in args.models.split(",") if m.strip()]
    models = {k: v for k, v in models_all.items() if k in want}
    if not models:
        models = models_all
    print(f"      参与建模：{list(models)}")

    # ---- 5. 建模 ----
    yr_meta = None          # 单终点置换成对结果，写入 run_meta.json 以便按产物核对次数
    if len(endpoints) == 1:
        ep = endpoints[0]
        print(f"\n[5/8] 单终点建模：{ep}")
        y = df[ep].astype(int).to_numpy()
        print(f"      阳性率 {y.mean():.3f}（n={len(y)}）")
        X_use, y_use, cl_use, mask_use = X, y, cluster_labels, None
        splits_map = {
            "random": P.random_splits(y, n_splits=args.folds),
            "cluster-out": P.cluster_out_splits(cluster_labels, n_splits=args.folds,
                                                 n_repeats=args.repeats),
            "scaffold": P.group_splits(scaffolds, n_splits=args.folds),
        }
        results = {}
        for strat, sp in splits_map.items():
            if not sp:
                results[strat] = pd.DataFrame()
                print(f"      - {strat}: 退化，无有效折")
                continue
            results[strat] = P.nested_cv_summary(models, X, y, sp)
            results[strat].to_csv(os.path.join(args.outdir, f"metrics_{strat}.csv"),
                                  encoding="utf-8-sig")
        for strat, tab in results.items():
            if tab.empty:
                continue
            print(f"\n      == {strat} ==")
            print(tab[["AUROC", "AUPRC", "F1", "MCC", "Brier"]].round(3).to_string())
        r, c = results.get("random", pd.DataFrame()), results.get("cluster-out", pd.DataFrame())
        if not r.empty and not c.empty:
            print(f"\n      ★ 随机划分高估 AUROC：{r['AUROC'].mean() - c['AUROC'].mean():+.3f}")
        P.plot_split_comparison(results, os.path.join(args.outdir, "Fig2_split_comparison.png"))

        # y-随机化 + 校准 + 应用域
        key = "cluster-out" if not results.get("cluster-out", pd.DataFrame()).empty else "random"
        best = results[key]["AUROC"].idxmax()
        print(f"\n      最佳模型（{key}）：{best}")
        oof = P.cv_predict(models[best], X, y, splits_map[key])
        m = ~np.isnan(oof)
        P.plot_calibration(y[m], oof[m], os.path.join(args.outdir, "Fig_calibration.png"))
        yr = P.y_randomization(models["RF"] if "RF" in models else models[best],
                               X, y, splits_map[key], n_perm=args.n_perm)
        yr_meta = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in yr.items()}
        # 置换分布落盘：图 4 面板 d 与 p 值核对均以该文件为准
        pd.DataFrame({"perm_index": np.arange(1, len(yr["perm_aurocs"]) + 1),
                      "permuted_AUROC": yr["perm_aurocs"]}).to_csv(
            os.path.join(args.outdir, "permutation_aurocs.csv"),
            index=False, encoding="utf-8-sig")
        print(f"      y-随机化：真实 {yr['real_auroc']:.3f} vs 置换 {yr['perm_mean']:.3f}，"
              f"p={yr['p_value']:.4f}"
              f"（请求 {yr['n_perm_requested']} 次，有效 {yr['n_perm_valid']} 次，"
              f"≥观测 {yr['n_ge_real']} 次）")
        d_self = P.knn_ad_distance(fps, fps, k=5, exclude_self=True)
        thr = float(np.nanpercentile(d_self, 95))
        in_dom = d_self <= thr
        print(f"      应用域：阈值 {thr:.3f}，域内 {np.mean(in_dom):.1%}")
    else:
        print(f"\n[5/8] 多终点建模：{len(endpoints)} 个终点（簇外验证）")
        rows, skipped = [], []
        for i, ep in enumerate(endpoints, 1):
            mask = df[ep].notna().to_numpy()
            if mask.sum() < 2 * args.min_pos:
                skipped.append({"endpoint": ep, "n_pos": int(df.loc[mask, ep].sum()),
                                "n_labeled": int(mask.sum()), "reason": "标签太少"})
                continue
            X_ep = X[mask]
            y = df.loc[mask, ep].astype(int).to_numpy()
            cl_ep = cluster_labels[mask]
            if y.sum() < args.min_pos or (len(y) - y.sum()) < args.min_pos:
                skipped.append({"endpoint": ep, "n_pos": int(y.sum()),
                                "n_labeled": int(mask.sum()), "reason": "阳性/阴性数不足"})
                continue
            sp_co = P.cluster_out_splits(cl_ep, n_splits=args.folds,
                                         n_repeats=args.repeats)
            sp_rd = P.random_splits(y, n_splits=args.folds)
            sp_sc = P.group_splits(scaffolds[mask], n_splits=args.folds)
            if not sp_co:
                skipped.append({"endpoint": ep, "n_pos": int(y.sum()),
                                "n_labeled": int(mask.sum()), "reason": "no folds"})
                continue
            tab_co = P.nested_cv_summary(models, X_ep, y, sp_co)
            tab_rd = P.nested_cv_summary(models, X_ep, y, sp_rd)
            tab_sc = P.nested_cv_summary(models, X_ep, y, sp_sc)
            if tab_co.empty:
                skipped.append({"endpoint": ep, "n_pos": int(y.sum()),
                                "n_labeled": int(mask.sum()), "reason": "empty"})
                continue
            bm = tab_co["AUROC"].idxmax()

            def _pick(tab, col):
                """Best model's value for `col`, or NaN when the split/column is absent."""
                if tab is None or tab.empty or col not in tab.columns or bm not in tab.index:
                    return np.nan
                return tab.loc[bm, col]

            # Full metric set for all three splitting strategies: Additional file 8
            # (Figure S1) reports per-endpoint AUPRC and MCC under each of them, so
            # every number it plots has to exist in this artefact.
            rows.append({
                "endpoint": ep, "n_labeled": int(mask.sum()),
                "n_pos": int(y.sum()), "pos_rate": float(y.mean()),
                "best_model": bm,
                "AUROC_cluster-out": tab_co.loc[bm, "AUROC"],
                "AUPRC_cluster-out": tab_co.loc[bm, "AUPRC"],
                "MCC_cluster-out": tab_co.loc[bm, "MCC"],
                "AUROC_scaffold": _pick(tab_sc, "AUROC"),
                "AUPRC_scaffold": _pick(tab_sc, "AUPRC"),
                "MCC_scaffold": _pick(tab_sc, "MCC"),
                "AUROC_random": _pick(tab_rd, "AUROC"),
                "AUPRC_random": _pick(tab_rd, "AUPRC"),
                "MCC_random": _pick(tab_rd, "MCC"),
                "AUROC_mean_cluster-out": tab_co["AUROC"].mean(),
                "AUROC_mean_scaffold": tab_sc["AUROC"].mean() if not tab_sc.empty else np.nan,
                "AUROC_mean_random": tab_rd["AUROC"].mean() if not tab_rd.empty else np.nan,
            })
            if i % 5 == 0 or i == len(endpoints):
                print(f"      进度 {i}/{len(endpoints)} …")
        mt = pd.DataFrame(rows)
        if not mt.empty:
            mt["AUROC_gap(rand-cluster)"] = mt["AUROC_random"] - mt["AUROC_cluster-out"]
            mt = mt.sort_values("AUROC_cluster-out", ascending=False).reset_index(drop=True)
            mt.to_csv(os.path.join(args.outdir, "multitask_metrics.csv"),
                      index=False, encoding="utf-8-sig")
            print(f"\n      【多终点汇总】簇外 AUROC 中位 {mt['AUROC_cluster-out'].median():.3f}，"
                  f"均值 {mt['AUROC_cluster-out'].mean():.3f}")
            print(f"      随机划分平均高估：{mt['AUROC_gap(rand-cluster)'].mean():+.3f}  ← 核心论点")
            print(mt[["endpoint", "n_pos", "best_model", "AUROC_cluster-out",
                      "AUPRC_cluster-out", "AUROC_random"]].head(15).round(3).to_string(index=False))
            if skipped:
                pd.DataFrame(skipped).to_csv(
                    os.path.join(args.outdir, "skipped_endpoints.csv"),
                    index=False, encoding="utf-8-sig")
                print(f"      {len(skipped)} 个终点因阳性数不足/无有效折被跳过")

        # 多终点下，特征消融用一个主终点代表（按该终点取有效子集）
        ep_main = mt.iloc[0]["endpoint"] if not mt.empty else endpoints[0]
        mask_main = df[ep_main].notna().to_numpy()
        X_use = X[mask_main]
        y_use = df.loc[mask_main, ep_main].astype(int).to_numpy()
        cl_use = cluster_labels[mask_main]
        mask_use = mask_main
        splits_map = {"cluster-out": P.cluster_out_splits(cl_use, n_splits=args.folds,
                                                          n_repeats=args.repeats)}
        results = {"cluster-out": P.nested_cv_summary(models, X_use, y_use,
                                                      splits_map["cluster-out"])}
        print(f"\n      （后续单终点分析以 {ep_main} 为代表终点，n={len(y_use)}）")

        # 代表终点的标签置换检验：**与 multitask_metrics.csv 同一配置**（面板级特征空间、
        # 面板级 Butina 聚类），使置换检验的观测 AUROC 与正文报告的簇外 AUROC 为同一个数。
        ep_perm = args.perm_endpoint if args.perm_endpoint in endpoints else ep_main
        m_perm = df[ep_perm].notna().to_numpy()
        X_pm = X[m_perm]
        y_pm = df.loc[m_perm, ep_perm].astype(int).to_numpy()
        cl_pm = cluster_labels[m_perm]
        best_m = "RF" if "RF" in models else sorted(models)[0]
        print(f"      y-随机化（多终点模式 · 终点 {ep_perm} · 面板级特征 "
              f"{X.shape[1]} 维 / {n_clusters} 簇）…")
        yr = P.y_randomization(models[best_m], X_pm, y_pm,
                               P.cluster_out_splits(cl_pm, n_splits=args.folds,
                                                    n_repeats=args.repeats),
                               n_perm=args.n_perm)
        yr_meta = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in yr.items()}
        # 置换分布落盘：图 4 面板 d 与 p 值核对均以该文件为准
        pd.DataFrame({"perm_index": np.arange(1, len(yr["perm_aurocs"]) + 1),
                      "permuted_AUROC": yr["perm_aurocs"]}).to_csv(
            os.path.join(args.outdir, "permutation_aurocs.csv"),
            index=False, encoding="utf-8-sig")
        yr_meta["perm_endpoint"] = ep_perm
        yr_meta["config"] = (f"panel-wide features ({X.shape[1]} dims) and panel-wide "
                             f"Butina clustering ({n_clusters} clusters) on {len(df)} substances")
        print(f"      y-随机化：真实 {yr['real_auroc']:.4f} vs 置换 {yr['perm_mean']:.3f}，"
              f"p={yr['p_value']:.4f}"
              f"（请求 {yr['n_perm_requested']} 次，有效 {yr['n_perm_valid']} 次，"
              f"≥观测 {yr['n_ge_real']} 次）")

    # ---- 6. 特征消融 ----
    if args.ablation:
        print("\n[6/8] 特征消融实验：QMSA 的增量贡献")
        sp_ab = P.cluster_out_splits(cl_use, n_splits=args.folds,
                                     n_repeats=max(1, args.repeats))
        ab = ablation_table(blocks, y_use, sp_ab, models, mask=mask_use)
        if not ab.empty:
            ab.to_csv(os.path.join(args.outdir, "ablation_qmsa.csv"),
                      index=False, encoding="utf-8-sig")
            print(ab.round(3).to_string(index=False))
            P.plot_ablation(ab, os.path.join(args.outdir, "Fig_ablation_qmsa.png"))
            print(f"      已输出：{args.outdir}/ablation_qmsa.csv")
            print(f"      已输出：{args.outdir}/Fig_ablation_qmsa.png")
        else:
            print("      消融失败（特征块缺失）")
    else:
        print("\n[6/8] 跳过消融（加 --ablation 启用）")

    # ---- 7. σ-profile ----
    if args.sigma_profile and args.qmsa:
        print("\n[7/8] σ-profile（分子表面静电势分布）")
        reps = ["OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
                "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
                "OC(=O)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)F"]
        labels = ["PFOA (C8 carboxylate)", "PFOS (C8 sulfonate)", "GenX (ether carboxylate)"]
        if P.plot_sigma_profile(reps, labels, os.path.join(args.outdir, "Fig_sigma_profile.png")):
            print(f"      已输出：{args.outdir}/Fig_sigma_profile.png")
    else:
        print("\n[7/8] 跳过 σ-profile")

    # ---- 8. 优先筛查清单 ----
    print("\n[8/8] 全量打分与优先筛查（仅单终点模式输出）")
    if len(endpoints) == 1:
        ep = endpoints[0]
        y = df[ep].astype(int).to_numpy()
        best = models["RF"] if "RF" in models else list(models.values())[0]
        best.fit(X, y)
        prob = best.predict_proba(X)[:, 1]
        d_self = P.knn_ad_distance(fps, fps, k=5, exclude_self=True)
        in_dom = d_self <= float(np.nanpercentile(d_self, 95))
        out = pd.DataFrame({"smiles": smiles, "pred_prob": prob,
                            "knn_ad_distance": d_self, "in_domain": in_dom,
                            "observed": y})
        for c in extra_meta_cols:
            if c in df.columns:
                out[c] = df[c].values
        if args.qmsa and "_qmsa_df" in blocks:
            qdf = blocks["_qmsa_df"].reset_index(drop=True)
            for c in Q.QMSA_FEATURES:
                if c in qdf.columns:
                    out[c] = qdf[c].values
        out = out.sort_values("pred_prob", ascending=False)
        out.to_csv(os.path.join(args.outdir, "predictions_all.csv"),
                   index=False, encoding="utf-8-sig")
        out[out["in_domain"]].head(30).to_csv(
            os.path.join(args.outdir, "priority_top30.csv"), index=False, encoding="utf-8-sig")
        print(f"      已输出 predictions_all.csv / priority_top30.csv（{len(out)} 条）")
        if "ether" in out.columns:
            print(f"      醚基替代物均值 {out.loc[out.ether == 1, 'pred_prob'].mean():.3f} "
                  f"vs 非醚基 {out.loc[out.ether == 0, 'pred_prob'].mean():.3f}")
    else:
        print("      多终点模式：跳过（单终点清单见各终点模型）")

    # ---- 可选 SHAP ----
    if args.shap:
        try:
            import shap  # noqa: F401
            print("\n[附] SHAP 解释")
        except Exception as e:
            print(f"\n[附] 跳过 SHAP：{e}")

    dt = time.time() - t_start
    meta = {"elapsed_sec": round(dt, 1), "n_molecules": len(df), "endpoints": endpoints,
            "n_features": int(X.shape[1]), "cutoff": args.cutoff,
            "n_clusters": int(n_clusters), "qmsa": bool(args.qmsa),
            "n_perm_requested": int(args.n_perm),
            "y_randomization": yr_meta,
            "p_value_formula": "(n_ge_real + 1) / (n_perm_valid + 1)"}
    with open(os.path.join(args.outdir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n完成，用时 {dt:.1f}s。结果目录：{os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
