# -*- coding: utf-8 -*-
"""
analyze_screen.py — PFASSTRUCT V6 域内筛查结果的「分析与发布」阶段

在 `screen_domain.py`（计算阶段，产出 screen_all.csv / pca_coords.csv）之后运行，
回答三个科学问题：

1. **模型适用域的中心在哪里？** —— 用 PCA 化学空间叠加图对比：
   训练集（有标签）· V6 结构全集 · 优先清单。
2. **优先清单有没有「化学类别富集」？** —— 计算各类别在清单中的富集倍数（odds / 富集因子），
   判断清单究竟是「毒性优先级」还是「训练集化学空间的镜像」。
3. **换一种适用域判据，结论会变吗？** —— 对比只用 kNN Tanimoto（文献常规做法）
   与四判据一致域（本研究）两种方案下的清单构成。

用法
----
    python analyze_screen.py --indir results_screen_v6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=os.path.join(HERE, "results_screen_v6"))
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or os.path.join(args.indir, "analysis")
    os.makedirs(outdir, exist_ok=True)

    a = pd.read_csv(os.path.join(args.indir, "screen_all.csv"), encoding="utf-8-sig")
    pc = pd.read_csv(os.path.join(args.indir, "pca_coords.csv"), encoding="utf-8-sig")
    cov = pd.read_csv(os.path.join(args.indir, "domain_coverage.csv"), encoding="utf-8-sig")
    ref = pd.read_csv(os.path.join(args.indir, "reference_pfas_predictions.csv"),
                      encoding="utf-8-sig")
    comp = pd.read_csv(os.path.join(args.indir, "class_composition.csv"),
                       encoding="utf-8-sig")
    print(f"[载入] 全库 {len(a)} 条 / PCA 坐标 {len(pc)} 条 / 参考物 {len(ref)} 条")

    # ============================================================ 1. 分层画像
    tiers = a["tier"].value_counts().sort_index()
    print("\n[1] 分层计数")
    print(tiers.to_string())

    def profile(df, name):
        return {
            "清单": name,
            "n": len(df),
            "含环比例": float((df["n_rings"] > 0).mean()) if len(df) else np.nan,
            "含醚氧比例": float((df["n_ether_O"] > 0).mean()) if len(df) else np.nan,
            "酸性头基比例": float(df["headgroup"].isin([1, 2, 3]).mean()) if len(df) else np.nan,
            "无环直链比例": float((df["n_rings"] == 0).mean()) if len(df) else np.nan,
            "中位n_CF2": float(df["n_CF2"].median()) if len(df) else np.nan,
            "中位预测活性": float(df["p_max"].median()) if len(df) else np.nan,
            "中位最大相似度": float(df["max_sim_to_train"].median()) if len(df) else np.nan,
        }

    ab = a[a["tier"].str.startswith(("A", "B"))]
    rows = [profile(a, "V6 结构全集"), profile(a[a["ad_consensus"]], "一致域内"),
            profile(ab, "优先清单 Tier A+B")]
    prof = pd.DataFrame(rows)
    prof.to_csv(os.path.join(outdir, "tier_profiles.csv"), index=False, encoding="utf-8-sig")
    print("\n[1b] 分层画像")
    print(prof.round(3).to_string(index=False))

    # ============================================================ 2. 类别富集
    lib_share = a["chem_class"].value_counts(normalize=True)
    lst_share = ab["chem_class"].value_counts(normalize=True)
    enr = pd.DataFrame({"库内占比": lib_share, "清单占比": lst_share}).fillna(0.0)
    enr["富集倍数"] = enr["清单占比"] / enr["库内占比"].replace(0, np.nan)
    enr["库内n"] = a["chem_class"].value_counts()
    enr["清单n"] = ab["chem_class"].value_counts().reindex(enr.index).fillna(0).astype(int)
    enr = enr.sort_values("富集倍数", ascending=False)
    enr.to_csv(os.path.join(outdir, "class_enrichment.csv"), encoding="utf-8-sig")
    print("\n[2] 化学类别富集（清单 Tier A+B vs 全库）")
    print(enr.round(3).to_string())

    # ============================================================ 3. 判据对照
    # 只用 kNN Tanimoto（文献常规） vs 四判据一致域
    conv = a["in_domain_majority"].to_numpy()
    cons = a["ad_consensus"].to_numpy()
    conv_top = a[conv].nlargest(500, "p_max")
    cons_top = a[cons & (~a["near_duplicate"])].nlargest(500, "p_max")
    cmp_tab = pd.DataFrame({
        "方案": ["仅 kNN Tanimoto 判据（常规做法）", "四判据一致域（本研究）"],
        "域内条数": [int(conv.sum()), int(cons.sum())],
        "Top500 中含环比例": [float((conv_top["n_rings"] > 0).mean()),
                        float((cons_top["n_rings"] > 0).mean())],
        "Top500 中无环直链比例": [float((conv_top["n_rings"] == 0).mean()),
                          float((cons_top["n_rings"] == 0).mean())],
        "Top500 中位 n_CF2": [float(conv_top["n_CF2"].median()),
                           float(cons_top["n_CF2"].median())],
        "Top500 中位分子量": [float(conv_top["mol_wt"].median()),
                        float(cons_top["mol_wt"].median())],
        "Top500 平均最大相似度": [float(conv_top["max_sim_to_train"].mean()),
                           float(cons_top["max_sim_to_train"].mean())],
    })
    cmp_tab.to_csv(os.path.join(outdir, "conventional_vs_consensus.csv"),
                   index=False, encoding="utf-8-sig")
    print("\n[3] 适用域判据方案对照")
    print(cmp_tab.round(3).to_string(index=False))

    # 参考物 × 判据 通过矩阵
    ref_mat = ref[["ref_name", "p_max", "in_domain_tanimoto", "in_maha", "in_range",
                   "in_class", "ad_consensus", "n_train_same_class"]].copy()
    ref_mat.to_csv(os.path.join(outdir, "reference_ad_matrix.csv"), index=False,
                   encoding="utf-8-sig")

    # ============================================================ 4. 图
    make_figures(a, pc, enr, ref, prof, outdir, ab)

    # ============================================================ 5. 汇总
    summary = {
        "n_library": int(len(a)),
        "n_train": int((pc["split"] == "train").sum()),
        "tiers": {k: int(v) for k, v in tiers.items()},
        "ad_coverage": {
            "tanimoto_only_pct": float(conv.mean()),
            "consensus_pct": float(cons.mean()),
        },
        "reference_pfas": {
            "n": int(len(ref)),
            "in_domain_tanimoto": int(ref["in_domain_tanimoto"].sum()),
            "in_domain_consensus": int(ref["ad_consensus"].sum()),
            "p_max_range": [float(ref["p_max"].min()), float(ref["p_max"].max())],
        },
        "priority_list_chemistry": {
            "n": int(len(ab)),
            "ring_pct": float((ab["n_rings"] > 0).mean()) if len(ab) else None,
            "ether_pct": float((ab["n_ether_O"] > 0).mean()) if len(ab) else None,
            "median_mw": float(ab["mol_wt"].median()) if len(ab) else None,
        },
        "library_chemistry": {
            "ring_pct": float((a["n_rings"] > 0).mean()),
            "ether_pct": float((a["n_ether_O"] > 0).mean()),
            "median_mw": float(a["mol_wt"].median()),
            "n_CF2_median": float(a["n_CF2"].median()),
        },
        "enrichment": {k: (None if pd.isna(v) else float(v))
                       for k, v in enr["富集倍数"].items()},
        "max_p_max_in_library": float(a["p_max"].max()),
        "p99_p_max": float(a["p_max"].quantile(0.99)),
        "class_composition_headline": comp.to_dict("records"),
    }
    with open(os.path.join(outdir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[5] 汇总 -> {os.path.join(outdir, 'analysis_summary.json')}")

    # 控制台速览
    print("\n" + "=" * 70)
    print("关键结论速览")
    print("=" * 70)
    print(f"  库内最高预测活性 p_max = {a['p_max'].max():.3f}（全库 {len(a)} 条）")
    print(f"  仅 Tanimoto 判据域内 {conv.mean():.1%}；四判据一致域内 {cons.mean():.1%}")
    print(f"  18 个监管/替代参考物：Tanimoto 判据下 "
          f"{int(ref['in_domain_tanimoto'].sum())}/{len(ref)} 域内，"
          f"一致域下 {int(ref['ad_consensus'].sum())}/{len(ref)} 域内")
    print(f"  优先清单含环比例 {summary['priority_list_chemistry']['ring_pct']:.1%} "
          f"（全库 {summary['library_chemistry']['ring_pct']:.1%}）")


# ==================================================================== 绘图
def make_figures(a, pc, enr, ref, prof, outdir, ab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # ---- Fig F: PCA 化学空间叠加 ----
    tr = pc[pc["split"] == "train"]
    q = pc[pc["split"] == "V6"]
    top = q[q["tier"].str.startswith(("A", "B"))]
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=170)
    ax.scatter(q["PC1"], q["PC2"], s=3, c="#d9d9d9", alpha=.55,
               label=f"PFASSTRUCT V6 结构全集 (n={len(q)})", linewidths=0)
    ax.scatter(q.loc[q["ad_consensus"], "PC1"], q.loc[q["ad_consensus"], "PC2"],
               s=3, c="#9ecae1", alpha=.55, label="其中：一致域内", linewidths=0)
    ax.scatter(tr["PC1"], tr["PC2"], s=14, c="#08519c", alpha=.85,
               label=f"有标签训练集 (n={len(tr)})", linewidths=0)
    ax.scatter(top["PC1"], top["PC2"], s=22, c="#d7301f", marker="^",
               label=f"优先清单 Tier A+B (n={len(top)})", linewidths=0)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Two PFAS chemical spaces and where the priority list lands\n"
                 "(PCA of 2D descriptor space)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FigF_chemical_space_map.png"))
    plt.close(fig)

    # ---- Fig G: 类别富集 ----
    e = enr[enr["库内n"] >= 50].sort_values("富集倍数")
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(e) + 1.8), dpi=170)
    cols = ["#d7301f" if v > 1.3 else ("#4a90c2" if v >= 0.8 else "#bdbdbd")
            for v in e["富集倍数"]]
    ax.barh(e.index.astype(str), e["富集倍数"], color=cols, height=.62)
    ax.axvline(1.0, ls="--", lw=1.1, c="k")
    for i, (v, n) in enumerate(zip(e["富集倍数"], e["清单n"])):
        ax.text(v + .03, i, f"{v:.2f}×  (清单 n={n})", va="center", fontsize=7.5)
    ax.set_xlabel("富集倍数（优先清单占比 ÷ 全库占比）")
    ax.set_title("Chemical-class enrichment in the priority list\n"
                 "(>1 = over-represented; red = enriched)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FigG_class_enrichment.png"))
    plt.close(fig)

    # ---- Fig H: 参考物 × 判据 通过矩阵 ----
    crit = ["in_domain_tanimoto", "in_maha", "in_range", "in_class", "ad_consensus"]
    labels = ["kNN\nTanimoto", "马氏\n距离", "描述符\n范围", "化学\n类别", "四判据\n一致"]
    M = ref[crit].to_numpy().astype(float)
    fig, ax = plt.subplots(figsize=(5.6, 0.34 * len(ref) + 1.8), dpi=170)
    ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(crit)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(ref)))
    ax.set_yticklabels([f"{n}  (p={p:.2f})" for n, p in
                        zip(ref["ref_name"], ref["p_max"])], fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "Y" if M[i, j] else "N", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title("Regulated PFASs under each applicability-domain criterion\n"
                 "(Y = judged inside the domain)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FigH_reference_ad_matrix.png"))
    plt.close(fig)

    # ---- Fig I: 库内预测活性分布 + 分层阈值 ----
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=170)
    ax.hist(a["p_max"], bins=80, color="#bdbdbd", label="全库")
    ax.hist(a.loc[a["ad_consensus"], "p_max"], bins=80, color="#4a90c2",
            alpha=.8, label="一致域内")
    p99 = float(a["p_max"].quantile(0.99))
    cons99 = float(a.loc[a["ad_consensus"], "p_max"].quantile(0.99))
    ax.axvline(p99, ls="--", lw=1.2, c="#333", label=f"全库 99% 分位 {p99:.3f}")
    ax.axvline(cons99, ls=":", lw=1.4, c="#d7301f",
               label=f"域内 99% 分位 {cons99:.3f}")
    ax.set_xlabel("预测免疫活性概率  p_max（12 个终点取最大）")
    ax.set_ylabel("化合物数")
    ax.set_title("Predicted activity of the whole PFAS structural universe", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FigI_prediction_distribution.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
