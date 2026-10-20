# -*- coding: utf-8 -*-
"""
run_published_qmsa.py — 基于**已发表 PFAS QMSA 描述符数据集**的真实建模

数据
----
`data/raw/PCBA-686978.csv` + `data/raw/CF_PCBA-686978.csv`
= 1,029 个 PFAS 的 17 个 QMSA 表面静电势描述符 + 生物活性标签（0/1）。
该数据集的描述符列与本项目 `qmsa.py` 输出**一一对应**（见 `qmsa.PUBLISHED_QMSA_MAP`），
因此本脚本的结果可直接与文献数值比对。

科学问题
--------
1. **泄漏对照**：随机划分 vs **描述符化学空间簇外划分**，AUROC 高估多少？
   （无 SMILES → 无法做骨架/指纹聚类，改在标准化 QMSA 描述符空间的 PCA 上做簇外划分，
    这也正是 "chemical-space-aware splitting" 的常规做法。）
2. **哪些表面静电势描述符真正驱动活性**？置换重要性 + 分组消融。
3. **y-随机化**：确认信号非偶然。

用法
----
    python run_published_qmsa.py --outdir results_published_qmsa
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
import pfas_qsar as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUTS = ["data/raw/PCBA-686978.csv", "data/raw/CF_PCBA-686978.csv"]

# 描述符分组（按 QMSA 语义）
QMSA_GROUPS_PUB = {
    "geometry": ["Volume", "Estimateddensity"],
    "esp_extrema": ["Minimal", "Maximal"],
    "esp_area": ["Positivesurfacearea", "Negativesurfacearea",
                 "Polarsurfacearea", "Nonpolarsurfacearea"],
    "esp_intensity": ["Overallaveragevalue", "Positiveaveragevalue",
                      "Negativeaveragevalue", "Product"],
    "esp_dispersion": ["Overallvariance", "Positivevariance", "Negativevariance",
                       "Balanceofcharges"],
    "polarity": ["Molecularpolarityindex"],
}


# ------------------------------------------------------------ 划分策略
def descriptor_cluster_splits(X: np.ndarray, n_splits: int = 5, n_repeats: int = 5,
                              n_pcs: int = 5, k: int = 40, seed: int = 42):
    """在标准化描述符空间的 PCA 上做 KMeans 聚类，再整簇进训练/测试（簇外划分）。"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    Z = StandardScaler().fit_transform(X)
    Z = PCA(n_components=min(n_pcs, Z.shape[1], Z.shape[0] - 1),
            random_state=seed).fit_transform(Z)
    lab = KMeans(n_clusters=min(k, max(2, len(X) // 10)), n_init=10,
                 random_state=seed).fit_predict(Z)
    splits = P.cluster_out_splits(lab, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    return lab, splits


def perm_importance(model, X, y, n_repeats: int = 8, seed: int = 0) -> np.ndarray:
    """置换重要性（AUROC 下降量）。"""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        imp[j] = float(np.mean(drops))
    return imp


# ------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    ap.add_argument("--outdir", default="results_published_qmsa")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--clusters", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 74)
    print("已发表 PFAS QMSA 描述符数据集 —— 真实建模")
    print("=" * 74)

    # ---- 1. 读数据 ----
    frames = []
    for p in args.inputs:
        fp = p if os.path.isabs(p) else os.path.join(HERE, p)
        if not os.path.exists(fp):
            print(f"  [skip] 缺文件 {fp}")
            continue
        d = pd.read_csv(fp, encoding="utf-8-sig")
        d["source_file"] = os.path.basename(fp)
        frames.append(d)
    if not frames:
        raise SystemExit("未找到输入数据，请先运行 validate_qmsa.py 或手动放置 data/raw/PCBA-686978.csv")
    df = pd.concat(frames, ignore_index=True)
    print(f"\n[1] 原始数据：{len(df)} 条 | 来源 {df.source_file.value_counts().to_dict()}")

    feat_cols = [c for g in QMSA_GROUPS_PUB.values() for c in g if c in df.columns]
    missing = [c for g in QMSA_GROUPS_PUB.values() for c in g if c not in df.columns]
    if missing:
        print(f"      缺失描述符（已忽略）：{missing}")
    df = df.dropna(subset=["Label"]).reset_index(drop=True)
    # 缺失值填中位数
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
    y = df["Label"].astype(int).to_numpy()
    X = df[feat_cols].to_numpy(dtype=float)
    print(f"[2] 特征：{X.shape[0]} × {X.shape[1]}（QMSA {len(QMSA_GROUPS_PUB)} 组）")
    print(f"    阳性率 {y.mean():.3f}（{int(y.sum())} / {len(y)}）")

    models = P.get_models()

    # ---- 3. 划分 ----
    print(f"\n[3] 构建划分：随机分层 vs 描述符化学空间簇外（PCA+KMeans, K={args.clusters}）")
    lab, sp_cluster = descriptor_cluster_splits(X, n_splits=args.folds,
                                                n_repeats=args.repeats, k=args.clusters)
    n_cl = len(np.unique(lab))
    sizes = pd.Series(lab).value_counts()
    print(f"    簇数 {n_cl}，每簇中位 {int(sizes.median())}（范围 {sizes.min()}–{sizes.max()}）")
    sp_random = P.random_splits(y, n_splits=args.folds)
    print(f"    随机 {len(sp_random)} 折；簇外 {len(sp_cluster)} 折")

    # ---- 4. 性能对比 ----
    print("\n[4] 建模与泄漏对照")
    res = {}
    for name, sp in [("random", sp_random), ("cluster-out", sp_cluster)]:
        tab = P.nested_cv_summary(models, X, y, sp)
        res[name] = tab
        tab.to_csv(os.path.join(args.outdir, f"metrics_{name}.csv"), encoding="utf-8-sig")
    for name, tab in res.items():
        if tab.empty:
            continue
        print(f"\n    == {name} ==")
        print(tab[["AUROC", "AUPRC", "F1", "MCC", "Brier"]].round(3).to_string())
    gap = np.nan
    if not res["random"].empty and not res["cluster-out"].empty:
        gap = res["random"]["AUROC"].mean() - res["cluster-out"]["AUROC"].mean()
        print(f"\n    ★ 随机划分相对簇外划分的 AUROC 高估：{gap:+.3f}  ← 核心论点")
    P.plot_split_comparison(res, os.path.join(args.outdir, "Fig2_split_comparison.png"))

    # ---- 5. 最佳模型深入 ----
    key = "cluster-out" if not res["cluster-out"].empty else "random"
    best = res[key]["AUROC"].idxmax()
    print(f"\n[5] 深入分析最佳模型（{key}）：{best}")
    model = models[best]
    oof = P.cv_predict(model, X, y, sp_cluster if key == "cluster-out" else sp_random)
    m = ~np.isnan(oof)
    met = P.evaluate(y[m], oof[m])
    print("    OOF：" + ", ".join(f"{k}={v:.3f}" for k, v in met.items() if isinstance(v, float)))
    P.plot_calibration(y[m], oof[m], os.path.join(args.outdir, "Fig_calibration.png"))

    yr = P.y_randomization(models["RF"] if "RF" in models else model, X, y,
                           sp_cluster if sp_cluster else sp_random, n_perm=100)
    print(f"    y-随机化：真实 {yr['real_auroc']:.4f} vs 置换 {yr['perm_mean']:.3f}，"
          f"p={yr['p_value']:.4f}"
          f"（请求 {yr['n_perm_requested']} 次，有效 {yr['n_perm_valid']} 次，"
          f"≥观测 {yr['n_ge_real']} 次）")
    # 置换分布落盘：附加文件 12 与图 4 面板 d 的空分布均以该文件为准
    pd.DataFrame({"perm_index": np.arange(1, len(yr["perm_aurocs"]) + 1),
                  "permuted_AUROC": yr["perm_aurocs"]}).to_csv(
        os.path.join(args.outdir, "permutation_aurocs.csv"),
        index=False, encoding="utf-8-sig")

    # ---- 6. 描述符重要性 ----
    print("\n[6] QMSA 描述符重要性（全数据拟合 + 置换重要性）")
    model.fit(X, y)
    imp = perm_importance(model, X, y)
    imp_df = pd.DataFrame({"descriptor": feat_cols, "dAUROC_when_permuted": imp}) \
        .sort_values("dAUROC_when_permuted", ascending=False).reset_index(drop=True)
    imp_df["group"] = imp_df["descriptor"].map(
        {c: g for g, cs in QMSA_GROUPS_PUB.items() for c in cs})
    imp_df.to_csv(os.path.join(args.outdir, "descriptor_importance.csv"),
                  index=False, encoding="utf-8-sig")
    print(imp_df.round(4).to_string(index=False))

    # ---- 7. 分组消融 ----
    print("\n[7] 描述符分组消融（簇外划分）")
    rows = []
    for gname, gcols in QMSA_GROUPS_PUB.items():
        cols = [c for c in gcols if c in feat_cols]
        if not cols:
            continue
        Xi = df[cols].to_numpy(dtype=float)
        tab = P.nested_cv_summary(models, Xi, y, sp_cluster)
        if tab.empty:
            continue
        bm = tab["AUROC"].idxmax()
        rows.append({"group": gname, "n_features": len(cols), "best_model": bm,
                     "AUROC": tab.loc[bm, "AUROC"], "AUPRC": tab.loc[bm, "AUPRC"]})
    # 留一：去掉某组
    full = P.nested_cv_summary(models, X, y, sp_cluster)
    if not full.empty:
        bm = full["AUROC"].idxmax()
        rows.append({"group": "ALL", "n_features": len(feat_cols),
                     "best_model": bm, "AUROC": full.loc[bm, "AUROC"],
                     "AUPRC": full.loc[bm, "AUPRC"]})
    ab = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
    ab.to_csv(os.path.join(args.outdir, "group_ablation.csv"), index=False, encoding="utf-8-sig")
    print(ab.round(3).to_string(index=False))
    try:
        P.plot_ablation(ab.rename(columns={"group": "feature_set"}),
                        os.path.join(args.outdir, "Fig_ablation_groups.png"))
    except Exception:
        pass

    # ---- 8. 描述符与活性的方向 ----
    print("\n[8] 描述符-活性关联（点二列相关，正负方向）")
    cors = pd.DataFrame({
        "descriptor": feat_cols,
        "pearson_r_with_label": [float(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])],
    }).sort_values("pearson_r_with_label", key=lambda s: -s.abs())
    cors["group"] = cors["descriptor"].map(
        {c: g for g, cs in QMSA_GROUPS_PUB.items() for c in cs})
    cors.to_csv(os.path.join(args.outdir, "descriptor_label_correlation.csv"),
                index=False, encoding="utf-8-sig")
    print(cors.round(4).to_string(index=False))

    meta = {
        "n_molecules": int(len(df)), "n_features": int(X.shape[1]),
        "pos_rate": float(y.mean()), "n_clusters": int(n_cl),
        "auroc_gap_random_minus_cluster": None if np.isnan(gap) else float(gap),
        "best_model_cluster_out": str(best),
        "n_perm": 100,
        "perm_endpoint": "PCBA-686978 (published QMSA descriptor dataset, 17 descriptors)",
        "y_randomization": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in yr.items()},
        "p_value_formula": "(n_perm_ge_real + 1) / (n_perm_valid + 1)",
        "source_files": list(df.source_file.unique()),
        "note": ("数据来源：PCBA-686978 / CF_PCBA-686978 已发表 PFAS QMSA 描述符数据集；"
                 "无 SMILES，故簇外划分在标准化描述符空间的 PCA 上进行。"),
    }
    with open(os.path.join(args.outdir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n完成。结果目录：{os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
