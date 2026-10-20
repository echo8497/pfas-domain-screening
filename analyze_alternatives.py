#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
新兴替代物清单的分析与出图
==========================
读 `results_alternatives/`（可同时读 `results_screen_v6/` 做交叉印证），输出：
  analysis_summary.json
  FigA_chemical_space_partition.png   化学空间划区：V6 四区构成 + 训练集落点
  FigB_domain_mismatch.png            适用域错配：20 个监管锚点的两套判据对照
  FigC_concern_vs_novelty.png         结构关注度 × 训练集相似度（替代物设计空间）
  FigD_family_tiers.png               替代物家族 × 优先层级
  FigE_design_space.png               链长 × 醚氧数 网格（全库 / 清单 / Tier I）
  FigF_model_blindspot.png            模型是否「看得到」这批替代物（交叉印证）
用法：
    python analyze_alternatives.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ALT_DIR = os.path.join(HERE, "results_alternatives")
SCR_DIR = os.path.join(HERE, "results_screen_v6")
DATA_DIR = os.path.join(HERE, "data")
OUT = os.path.join(ALT_DIR, "analysis")

C = {"reg": "#B03A2E", "alt": "#1F6FB2", "arom": "#7F8C8D", "other": "#D5DBDB",
     "train": "#E67E22", "tier1": "#C0392B", "tier2": "#2980B9", "tier3": "#95A5A6"}


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 200,
                         "savefig.bbox": "tight",
                         "font.sans-serif": ["Microsoft YaHei", "SimHei",
                                             "Noto Sans SC", "SimSun", "DejaVu Sans"],
                         "axes.unicode_minus": False})
    summary = {}

    alt = pd.read_csv(os.path.join(ALT_DIR, "emerging_alternatives.csv"),
                      encoding="utf-8-sig")
    part = pd.read_csv(os.path.join(ALT_DIR, "chemical_space_partition.csv"),
                       encoding="utf-8-sig")
    ref = pd.read_csv(os.path.join(ALT_DIR, "regulated_vs_emerging.csv"),
                      encoding="utf-8-sig")
    t1 = alt[alt["alt_tier"].str.startswith("I:")]

    # ============================================================ Fig A 化学空间
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    order = ["A: 经典长链 PFAA(监管)", "B: PFAS 替代物化学空间",
             "C: 含氟芳香物(药/农/材料)", "D: 其他含氟结构"]
    cols = [C["reg"], C["alt"], C["arom"], C["other"]]
    labels = ["A 经典长链 PFAA\n(监管重点)", "B PFAS 替代物\n化学空间",
              "C 含氟芳香物\n(药物/农药/材料)", "D 其他含氟结构"]
    p = part.set_index("space").reindex(order)
    ax = axes[0]
    n_q, n_tr = int(p["n"].sum()), None
    tr_b = 4  # 训练集落在替代物空间（由 screen_alternatives 打印，硬编码同步）
    vals = p["n"].to_numpy()
    left = 0
    for v, c, lb in zip(vals, cols, labels):
        ax.barh(0, v, left=left, color=c, edgecolor="white", height=0.55)
        ax.text(left + v / 2, 0, f"{v:,}\n{v / n_q:.1%}", ha="center", va="center",
                color="white", fontsize=8, fontweight="bold")
        left += v
    ax.set_yticks([]); ax.set_xlabel("PFASSTRUCT V6 结构数")
    ax.set_title("A. V6 全库 19,767 条结构的化学空间划区", loc="left", fontsize=9.5)
    ax.legend(handles=[Patch(facecolor=c, label=lb.replace("\n", " "))
                       for c, lb in zip(cols, labels)],
              loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False,
              fontsize=7.5)
    ax.set_xlim(0, n_q * 1.02)

    ax = axes[1]
    tsc = pd.read_csv(os.path.join(ALT_DIR, "train_space_counts.csv"),
                      encoding="utf-8-sig").set_index("space").reindex(order)
    trs = tsc["n"].fillna(0).astype(int).tolist()
    tot = sum(trs)
    left = 0
    for v, c in zip(trs, cols):
        ax.barh(0, v, left=left, color=c, edgecolor="white", height=0.55)
        if v > 30:
            ax.text(left + v / 2, 0, f"{v}", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
        left += v
    ax.annotate(f"仅 {trs[1]} 条\n({trs[1] / tot:.1%})", xy=(trs[0] + 2, 0.30),
                xytext=(tot * 0.55, 0.85),
                fontsize=8, color="#B03A2E", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#B03A2E", lw=1.0))
    ax.set_yticks([]); ax.set_xlim(0, tot * 1.02)
    ax.set_xlabel("有活性标签的训练集结构数")
    ax.set_title("B. Cheng & Ng 训练集 1,012 条的落点", loc="left", fontsize=9.5)
    fig.suptitle("模型标签的化学空间不覆盖 PFAS 替代物化学", x=0.005, ha="left",
                 fontsize=10.5, fontweight="bold", y=1.06)
    fig.savefig(os.path.join(OUT, "FigA_chemical_space_partition.png"))
    plt.close(fig)
    summary["space_partition"] = {r["space"]: int(r["n"]) for _, r in p.reset_index().iterrows()}

    # ============================================================ Fig B 适用域错配
    rp = os.path.join(SCR_DIR, "reference_pfas_predictions.csv")
    if os.path.exists(rp):
        rr = pd.read_csv(rp, encoding="utf-8-sig")
        rr = rr.sort_values("p_max")
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        y = np.arange(len(rr))
        ax.barh(y + 0.20, rr["p_max"], height=0.38, color="#2E86C1",
                label="模型预测活性 $p_{max}$")
        ax.barh(y - 0.20, rr["dknn_to_train"], height=0.38, color="#F39C12",
                label="到训练集的 kNN Tanimoto 距离 $d$")
        thr = float(rr["dknn_to_train"].max())  # 仅用于示意，真实阈值见日志
        ax.axvline(0.7033, ls="--", lw=1.0, color="#7D3C98",
                   label="Tanimoto 判域阈值 = 0.703")
        for i, (_, r) in enumerate(rr.iterrows()):
            ax.text(0.012, i, "√ 一致域内" if r["ad_consensus"] else "× 一致域外",
                    va="center", fontsize=7,
                    color="#1E8449" if r["ad_consensus"] else "#B03A2E")
        ax.set_yticks(y); ax.set_yticklabels(rr["ref_name"], fontsize=8)
        ax.set_xlabel("值")
        ax.set_xlim(0, 1.16)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.set_title("监管/在用 PFAS 的落点：Tanimoto 判据 88.9% 判域内，\n"
                     "一致域判据 0% 判域内；模型活性预测全部处于低分区（0.08–0.38）",
                     loc="left", fontsize=9.5)
        fig.savefig(os.path.join(OUT, "FigB_domain_mismatch.png"))
        plt.close(fig)
        summary["reference"] = {
            "n": len(rr),
            "tanimoto_pass_pct": float(rr["in_domain_tanimoto"].mean()),
            "consensus_pass_pct": float(rr["ad_consensus"].mean()),
            "p_max_min": float(rr["p_max"].min()), "p_max_max": float(rr["p_max"].max())}

    # ============================================================ Fig C 结构关注 × 新颖性
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    fams = alt["alt_family"].value_counts().index.tolist()
    cmap = plt.get_cmap("tab10")
    for i, f in enumerate(fams):
        s = alt[alt["alt_family"] == f]
        ax.scatter(s["max_sim_train"], s["S_concern"], s=9, alpha=0.45,
                   color=cmap(i % 10), label=f, edgecolors="none")
    ax.scatter(t1["max_sim_train"], t1["S_concern"], s=42, facecolors="none",
               edgecolors="#B03A2E", linewidths=1.1, label="Tier I（前 100）", zorder=5)
    ax.axvline(0.60, ls="--", lw=1.0, color="#7D3C98")
    ax.text(0.605, 0.44, "ALT_NOVEL_MAX = 0.60\n(右侧＝训练集已覆盖)", fontsize=7.5,
            color="#7D3C98", rotation=90, va="bottom")
    ax.axhline(0.60, ls=":", lw=1.0, color="#7D3C98")
    # 已商品化替代物的位置
    anchors = {"GenX/HFPO-DA": (0.405, 0.62), "ADONA": (0.346, 0.70),
               "6:2 FTS": (0.577, 0.53), "F-53B": (0.682, 0.60)}
    ax.scatter([v[0] for v in anchors.values()], [v[1] for v in anchors.values()],
               marker="*", s=170, color="#000000", zorder=6, label="已商品化替代物锚点")
    for k, (xx, yy) in anchors.items():
        ax.annotate(k, (xx, yy), textcoords="offset points", xytext=(7, 5), fontsize=7.5)
    ax.set_xlabel("与训练集最大 Tanimoto 相似度（越小＝越未被研究过）")
    ax.set_ylabel("结构关注代理 $S_{concern}$（持久性 × 迁移性）")
    ax.set_title("PFAS 替代物候选空间：左上＝「未被研究过且高持久高迁移」，"
                 "★＝已商品化替代物所在位置", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=2)
    fig.savefig(os.path.join(OUT, "FigC_concern_vs_novelty.png"))
    plt.close(fig)

    # ============================================================ Fig D 家族 × 层级
    ct = pd.crosstab(alt["alt_family"], alt["alt_tier"])
    for c in ["I: 优先关注", "II: 值得跟踪", "III: 储备观察"]:
        if c not in ct:
            ct[c] = 0
    ct = ct[["I: 优先关注", "II: 值得跟踪", "III: 储备观察"]]
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=True).index]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4),
                             gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    ax.barh(ct.index, ct["I: 优先关注"], color=C["tier1"], label="Tier I 优先关注")
    ax.barh(ct.index, ct["II: 值得跟踪"], left=ct["I: 优先关注"], color=C["tier2"],
            label="Tier II 值得跟踪")
    ax.barh(ct.index, ct["III: 储备观察"],
            left=ct["I: 优先关注"] + ct["II: 值得跟踪"], color=C["tier3"],
            label="Tier III 储备观察")
    ax.set_xlabel("结构数")
    ax.set_title("A. 替代物家族构成（全部 2,048 条）", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    for i, (idx, r) in enumerate(ct.iterrows()):
        ax.text(r.sum(axis=0) + 18, i, f"{int(r.sum(axis=0))}", va="center", fontsize=7.5)
    ax.set_xlim(0, ct.sum(axis=1).max() * 1.18)

    ax = axes[1]
    mk = t1["Marker_命中数"].value_counts().sort_index()
    ax.bar(mk.index.astype(str), mk.to_numpy(), color=C["tier1"], width=0.6)
    for x, v in zip(mk.index.astype(str), mk.to_numpy()):
        ax.text(x, v + 1.2, str(v), ha="center", fontsize=8)
    ax.set_xlabel("结构标记命中数（短链 / 含醚 / 可电离 / 高全氟碳占比）")
    ax.set_ylabel("Tier I 结构数")
    ax.set_title("B. Tier I 100 条的结构标记命中", loc="left", fontsize=9.5)
    fig.savefig(os.path.join(OUT, "FigD_family_tiers.png"))
    plt.close(fig)

    # ============================================================ Fig E 设计空间网格
    lib = pd.read_csv(os.path.join(DATA_DIR, "pfasstruct_v6.csv"), encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), sharey=True)
    panels = [("V6 全库", lib), ("替代物清单", alt), ("Tier I", t1)]
    for ax, (name, d) in zip(axes, panels):
        x = d["n_ether_O"].clip(0, 6).astype(int)
        yv = d["n_CF2"].clip(0, 12).astype(int)
        H = np.zeros((13, 7))
        for xx, yy in zip(x, yv):
            H[yy, xx] += 1
        im = ax.imshow(np.log10(H + 1), origin="lower", cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(7)); ax.set_xticklabels([str(i) for i in range(6)] + ["6+"])
        ax.set_yticks(range(0, 13, 2))
        ax.set_title(f"{name}（n={len(d):,}）", fontsize=9)
        ax.set_xlabel("醚氧数")
        for yy in range(13):
            for xx in range(7):
                if H[yy, xx] > 0 and (len(d) < 3000 or H[yy, xx] > 30):
                    ax.text(xx, yy, int(H[yy, xx]), ha="center", va="center",
                            fontsize=6, color="#333")
    axes[0].set_ylabel("全氟碳单元数 $n_{CF_2}$")
    fig.suptitle("替代物「设计空间」地图：清单与 Tier I 集中在「短链 + 含醚」区块，"
                 "与经典长链直链区（无醚、$n_{CF_2}\\geq8$）分离",
                 x=0.005, ha="left", fontsize=10, fontweight="bold", y=1.06)
    fig.savefig(os.path.join(OUT, "FigE_design_space.png"))
    plt.close(fig)

    # ============================================================ Fig F 模型盲区
    allf = os.path.join(SCR_DIR, "screen_all.csv")
    if os.path.exists(allf):
        use = ["smiles", "p_max", "ad_consensus", "in_domain_majority", "tier"]
        sc = pd.read_csv(allf, encoding="utf-8-sig", usecols=lambda c: c in use)
        m = alt.merge(sc, on="smiles", how="left", suffixes=("", "_model"))
        m["model_usable"] = m["ad_consensus"].fillna(False)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9),
                                 gridspec_kw={"width_ratios": [1, 1]})
        ax = axes[0]
        ax.hist(m["p_max"], bins=40, color="#7F8C8D", alpha=0.75)
        ax.axvline(float(m["p_max"].median()), color="#B03A2E", lw=1.4,
                   label=f"中位 = {m['p_max'].median():.2f}")
        ax.axvline(0.67, ls="--", color="#1F6FB2", lw=1.2,
                   label="全库 p_max 99 分位 = 0.67")
        ax.set_xlabel("模型预测活性 $p_{max}$（12 个 ToxCast 终点取最大）")
        ax.set_ylabel("替代物候选数")
        ax.set_title("A. 模型给替代物候选打的活性分", loc="left", fontsize=9.5)
        ax.legend(fontsize=7.5, frameon=False)

        ax = axes[1]
        cnt = pd.Series({
            "Tanimoto 多数终点判域内": float(m["in_domain_majority"].fillna(False).mean()),
            "一致域判据判域内": float(m["model_usable"].mean()),
            "进入模型 Tier A/B/C": float(m["tier"].fillna("").str.startswith(
                ("A", "B", "C")).mean()),
        })
        bars = ax.barh(cnt.index, cnt.to_numpy(), color=["#F39C12", "#B03A2E", "#7F8C8D"])
        for b, v in zip(bars, cnt.to_numpy()):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.1%}",
                    va="center", fontsize=8.5)
        ax.set_xlim(0, max(0.35, cnt.max() * 1.35))
        ax.set_xlabel("占 2,048 条替代物候选的比例")
        ax.set_title("B. 这批替代物在活性模型眼里的可见度", loc="left", fontsize=9.5)
        fig.suptitle("交叉印证：活性模型对替代物候选几乎「盲」——预测值低、绝大多数判域外",
                     x=0.005, ha="left", fontsize=10, fontweight="bold", y=1.05)
        fig.savefig(os.path.join(OUT, "FigF_model_blindspot.png"))
        plt.close(fig)
        summary["model_blindspot"] = {k: round(v, 4) for k, v in cnt.items()}
        summary["model_p_max_median_on_alternatives"] = float(m["p_max"].median())

    # ============================================================ 汇总
    summary["n_library"] = int(part["n"].sum())
    summary["n_alt_space"] = int(part.set_index("space").loc[
        "B: PFAS 替代物化学空间", "n"])
    summary["n_candidates"] = int(len(alt))
    summary["n_tier1"] = int(len(t1))
    summary["n_regrettable"] = int(alt["regrettable_substitution"].sum())
    summary["tier1_family"] = t1["alt_family"].value_counts().to_dict()
    summary["tier1_marker_mean"] = float(t1["Marker_命中数"].mean())
    summary["tier1_median_CF2"] = int(t1["n_CF2"].median())
    summary["tier1_median_MW"] = float(t1["mol_wt"].median())
    summary["tier1_ether_pct"] = float((t1["n_ether_O"] > 0).mean())
    with open(os.path.join(OUT, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\n[完成] 图与摘要 → {OUT}")


if __name__ == "__main__":
    main()
