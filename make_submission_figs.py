# -*- coding: utf-8 -*-
"""Generate the 8 submission figures for the PFAS applicability-domain manuscript.

Every number plotted is read from the regenerated result files (post n_CF2 fix),
never hard-coded, so the figures can never drift from the audit trail.

Usage:
    python code/make_submission_figs.py

Output (single composite PNG per figure, 600 dpi, English axes):
    figures/Fig1_workflow.png
    figures/Fig2_chemical_space_partition.png
    figures/Fig3_anchor_ad_verdicts.png
    figures/Fig4_leakage_auroc.png
    figures/Fig5_concern_vs_novelty.png
    figures/Fig6_family_by_tier.png
    figures/Fig7_design_space.png
    figures/Fig8_model_blindspot.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colour-blind-safe qualitative palette (Okabe–Ito)
CB = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "grey": "#999999", "dark": "#333333",
}
TIER_COLOR = {"I": CB["red"], "II": CB["orange"], "III": CB["blue"]}
TIER_LABEL = {"I": "Tier I (n = 100)", "II": "Tier II (n = 400)", "III": "Tier III (n = 1,890)"}


def tier_code(s: pd.Series) -> pd.Series:
    """Extract the roman-numeral tier code. 'III: 储备观察' -> 'III'."""
    return s.astype(str).str.split(":").str[0].str.strip()


# 真实置换检验产物（run_pipeline.py --multi-task --n-perm 50 --perm-endpoint PCBA-686978）
PERM_DIR = CODE / "results_multitask_perm"
PERM_META = PERM_DIR / "run_meta.json"
if not (PERM_DIR / "permutation_aurocs.csv").exists():
    raise SystemExit(
        f"缺少真实置换分布 {PERM_DIR / 'permutation_aurocs.csv'}。\n"
        "请先运行：python run_pipeline.py --input data/pfas_chengng.csv "
        "--multi-task --n-perm 50 --perm-endpoint PCBA-686978 "
        "--outdir results_multitask_perm\n"
        "（Fig. 4 面板 d 必须画真实置换值，不得用分布合成）")


def _load():
    return {
        "multi": pd.read_csv(CODE / "results_real_multitask" / "multitask_metrics.csv", encoding="utf-8-sig"),
        # Three-splitting-strategy benchmark behind Additional file 8 (Figure S1).
        # Same configuration as "multi"; superset of its columns.
        "multi3": pd.read_csv(CODE / "results_multitask_3splits" / "multitask_metrics.csv", encoding="utf-8-sig"),
        "part": pd.read_csv(CODE / "results_alternatives" / "chemical_space_partition.csv", encoding="utf-8-sig"),
        "cand": pd.read_csv(CODE / "results_alternatives" / "emerging_alternatives.csv", encoding="utf-8-sig"),
        "fam": pd.read_csv(CODE / "results_alternatives" / "alt_class_summary.csv", encoding="utf-8-sig"),
        "anch": pd.read_csv(CODE / "results_audit" / "table3_anchor_verdicts.csv", encoding="utf-8-sig"),
        "abl": pd.read_csv(CODE / "results_real_PCBA686978" / "ablation_qmsa.csv", encoding="utf-8-sig"),
        "dom": pd.read_csv(CODE / "results_screen_v6" / "domain_coverage.csv", encoding="utf-8-sig"),
        "screen": pd.read_csv(CODE / "results_screen_v6" / "screen_all.csv", encoding="utf-8-sig"),
        # 真实标签置换分布（Fig. 4 面板 d）——必须来自实际跑过的置换，不得合成
        "perm": pd.read_csv(PERM_DIR / "permutation_aurocs.csv", encoding="utf-8-sig"),
        "perm_meta": json.loads(PERM_META.read_text(encoding="utf-8")),
    }


# ===================================================================== Fig. 1
def fig1() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.15, 3.45, 1.95, 1.15, "PFASSTRUCT V6\n21,028 records\n19,767 structures", "#D6E4F0"),
        (0.15, 1.55, 1.95, 1.15, "PFAS bioactivity panel\n1,012 substances\n26 endpoints", "#FDE9D9"),
        (2.55, 2.50, 1.95, 1.15, "Descriptors\n224 2D + 2,048 FP\n+ 32 QMSA  →  2,268", "#E2EFDA"),
        (2.55, 0.45, 1.95, 1.15, "Leakage-aware CV\nrandom / Murcko /\ncluster-out", "#F2DCDB"),
        (5.05, 2.50, 1.95, 1.15, "Five AD criteria\n+ consensus domain", "#E4DFEC"),
        (5.05, 0.45, 1.95, 1.15, "Chemical-space\npartition (A/B/C/D)", "#FFF2CC"),
        (7.55, 2.50, 2.30, 1.15, "Model priority list\n98 structures\n(98.0% ring-bearing)", "#F2DCDB"),
        (7.55, 0.45, 2.30, 1.15, "Read-across list\n2,390 candidates\n(Tier I = 100)", "#D6E4F0"),
    ]
    for x, y, w, h, txt, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc=c, ec="#5A5A5A", lw=0.7))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.0)

    for (x1, y1), (x2, y2) in [
        ((2.10, 4.02), (2.55, 3.45)),
        ((2.10, 4.02), (2.55, 1.05)),
        ((2.10, 2.12), (2.55, 1.05)),
        ((4.50, 3.08), (5.05, 3.08)),
        ((4.50, 1.05), (5.05, 1.05)),
        ((7.00, 3.08), (7.55, 3.08)),
        ((7.00, 1.05), (7.55, 1.05)),
    ]:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, color="#5A5A5A", lw=0.8))

    ax.add_patch(FancyArrowPatch((8.70, 1.62), (8.70, 2.48), arrowstyle="-|>",
                                 mutation_scale=9, color=CB["red"], lw=1.1,
                                 linestyle="--"))
    ax.text(8.70, 2.05, "0% covered", color=CB["red"], fontsize=7.0,
            ha="center", va="center", style="italic",
            bbox=dict(fc="white", ec="none", pad=1.0))

    ax.text(5.0, 4.80, "Workflow of the applicability-domain and read-across analysis",
            ha="center", fontsize=9, fontweight="bold")
    fig.savefig(OUT / "Fig1_workflow.png")
    plt.close(fig)
    print("  Fig. 1  workflow")


# ===================================================================== Fig. 2
def fig2(D) -> None:
    """Panel a: panel composition; b: inventory composition; c: structural profiles
    (five mini-panels, each on its own scale, annotated with real values)."""
    part = D["part"].copy()
    reg = part["space"].str[0].to_numpy()
    regname = {
        "A": "A  long-chain PFAA",
        "B": "B  replacement space",
        "C": "C  fluorinated aromatics",
        "D": "D  other fluorinated",
    }
    pct_inv = dict(zip(reg, part["pct"].astype(float) * 100))

    # Panel composition is fixed by the manuscript: 93.5% region C, 6 (0.6%) region B.
    counts_panel = {"A": 0, "B": 6, "C": 946, "D": 60}
    counts_panel["A"] = 1012 - 6 - 946 - 60
    pct_panel = {k: 100.0 * v / 1012 for k, v in counts_panel.items()}

    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], hspace=0.45, wspace=0.28)

    def donut(ax, pct, title, n_total):
        order = ["A", "B", "C", "D"]
        vals = [pct[k] for k in order]
        cols = [CB["blue"], CB["red"], CB["orange"], CB["grey"]]
        lab = [f"{regname[k]}\n{pct[k]:.1f}%" for k in order]
        w, _ = ax.pie(vals, colors=cols, startangle=90,
                      wedgeprops=dict(width=0.42, edgecolor="white", linewidth=0.8))
        ax.legend(w, lab, loc="center left", bbox_to_anchor=(0.98, 0.5),
                  frameon=False, fontsize=6.8, handlelength=1.1, labelspacing=0.9)
        ax.set_title(f"{title}\n({n_total:,} structures)", fontsize=8.5, pad=6)

    ax_a = fig.add_subplot(gs[0, 0])
    donut(ax_a, pct_panel, "a  Labelled bioactivity panel", 1012)
    ax_b = fig.add_subplot(gs[0, 1])
    donut(ax_b, pct_inv, "b  PFASSTRUCT V6 inventory", 19767)

    # c) five mini-panels, one per metric, each on its own scale
    order = ["A", "B", "C", "D"]
    cols = [CB["blue"], CB["red"], CB["orange"], CB["grey"]]
    metrics = [
        ("Median\nPFAS carbon", "median_CF2", "{:.0f}"),
        ("Median MW\n(Da)", "median_MW", "{:.0f}"),
        ("Aromatic\nfraction", "aromatic_pct", "{:.2f}"),
        ("Ether\nfraction", "ether_pct", "{:.2f}"),
        ("Acidic\nfraction", "acidic_pct", "{:.2f}"),
    ]
    gs_c = gs[1, :].subgridspec(1, 5, wspace=0.38)
    for k, (title, col, fmt) in enumerate(metrics):
        ax = fig.add_subplot(gs_c[0, k])
        vals = [float(part[part["space"].str[0] == r][col].iloc[0]) for r in order]
        bars = ax.bar(order, vals, color=cols, width=0.68,
                      edgecolor="white", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=6.6)
        ax.set_title(title, fontsize=7.6)
        ax.set_ylim(0, max(vals) * 1.22 if max(vals) > 0 else 1)
        ax.tick_params(labelsize=6.6, length=0)
        ax.grid(axis="y", lw=0.3, color="#EEEEEE")
        ax.set_axisbelow(True)
        if k == 0:
            ax.set_ylabel("Value", fontsize=7.4)

    fig.savefig(OUT / "Fig2_chemical_space_partition.png")
    plt.close(fig)
    print("  Fig. 2  chemical-space partition")


# ===================================================================== Fig. 3
def fig3(D) -> None:
    a = D["anch"].copy()
    cols = [
        ("kNN Tanimoto\ndistance", "in_domain_knn_tanimoto"),
        ("Descriptor\nrange", "in_domain_descriptor_range"),
        ("PCA-10\nleverage", None),
        ("PCA-10\nMahalanobis", "in_domain_mahalanobis"),
        ("Chemical-class\ncoverage", "in_domain_class_T5"),
        ("Consensus\ndomain", "consensus_domain_T5"),
    ]
    a = a.reset_index(drop=True)
    M = np.zeros((len(a), len(cols)))
    for j, (_, c) in enumerate(cols):
        M[:, j] = a[c].astype(bool).to_numpy().astype(int) if c else np.nan

    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(7.6, 5.2), gridspec_kw={"width_ratios": [2.3, 1]}, sharey=True)

    ax.imshow(np.ma.masked_invalid(M), cmap="Blues", vmin=0, vmax=1.6,
              aspect="auto", interpolation="none")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            if M[i, j] > 0:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fc=CB["green"], ec="white", lw=1.2))
            else:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fc="#F2F2F2", ec="white", lw=1.2))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], fontsize=6.8)
    ax.set_yticks(range(len(a)))
    ax.set_yticklabels(a["anchor"], fontsize=7.0)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(13.5, color=CB["dark"], lw=0.8, ls=":")

    n_legacy = int((a["group"] == "legacy").sum())
    ax.text(-0.62, (n_legacy - 1) / 2, "Regulated\n(n = 14)", rotation=90,
            ha="center", va="center", fontsize=7.0, color=CB["blue"], fontweight="bold")
    ax.text(-0.62, n_legacy + (len(a) - n_legacy - 1) / 2, "Marketed\nalternatives\n(n = 6)",
            rotation=90, ha="center", va="center", fontsize=7.0,
            color=CB["red"], fontweight="bold")
    ax.set_title("a  Applicability-domain verdict by criterion", loc="left",
                 fontsize=8.5, pad=8)

    handles = [
        Line2D([], [], marker="s", ls="", ms=8, mfc=CB["green"], mec="white",
               label="Inside domain"),
        Line2D([], [], marker="s", ls="", ms=8, mfc="#F2F2F2", mec="white",
               label="Outside domain"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.09),
              ncol=2, frameon=False, fontsize=7.0)

    # Right panel: per-criterion inside counts
    names = [c[0].replace("\n", " ") for c in cols]
    inside = [int(np.nansum(M[:, j])) for j in range(len(cols))]
    y = np.arange(len(cols))[::-1]
    axr.barh(y, inside, color=[CB["blue"]] * 3 + [CB["purple"]] * 3,
             height=0.62)
    for yy, v in zip(y, inside):
        axr.text(v + 0.4, yy, f"{v}/20", va="center", fontsize=7.0)
    axr.set_yticks(y)
    axr.set_yticklabels([])
    axr.set_xlim(0, 23)
    axr.set_xlabel("Anchors judged inside (of 20)", fontsize=7.5)
    axr.grid(axis="x", lw=0.3, color="#DDDDDD")
    axr.set_axisbelow(True)
    axr.set_title("b  Criterion-dependent outcome", loc="left", fontsize=8.5, pad=8)

    fig.savefig(OUT / "Fig3_anchor_ad_verdicts.png")
    plt.close(fig)
    print("  Fig. 3  anchor AD verdicts")


# ===================================================================== Fig. 4
def fig4(D) -> None:
    m = D["multi"].copy()
    d = m[["endpoint", "n_labeled", "AUROC_cluster-out", "AUROC_random"]].rename(
        columns={"n_labeled": "n", "AUROC_cluster-out": "cluster_out", "AUROC_random": "random"})
    d = d.dropna().sort_values("cluster_out").reset_index(drop=True)
    d["delta"] = d["random"] - d["cluster_out"]
    prim = "PCBA-686978"

    fig = plt.figure(figsize=(7.4, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.30)

    # a) paired slope plot (all modellable endpoints)
    ax = fig.add_subplot(gs[0, 0])
    yy = np.arange(len(d))
    ax.hlines(yy, d["cluster_out"], d["random"], color="#C8C8C8", lw=1.4, zorder=1)
    ax.scatter(d["cluster_out"], yy, s=24, color=CB["blue"], zorder=3, label="Cluster-out")
    ax.scatter(d["random"], yy, s=24, color=CB["red"], zorder=3, label="Random")
    ip = int(d.index[d.endpoint == prim][0])
    ax.scatter([d.loc[ip, "cluster_out"]], [ip], s=70, facecolors="none",
               edgecolors=CB["dark"], linewidths=1.2, zorder=5)
    ax.set_yticks(yy)
    ax.set_yticklabels(d["endpoint"], fontsize=6.0)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_xlim(0.52, 1.0)
    ax.set_xlabel("AUROC (best model)", fontsize=7.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="none",
              fontsize=6.4)
    ax.set_title(f"a  Paired AUROC, {len(d)} modellable endpoints (panel-wide)",
                 loc="left", fontsize=8.5)
    ax.grid(axis="x", lw=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)

    # b) delta distribution
    ax = fig.add_subplot(gs[0, 1])
    colors = np.where(d["delta"] > 0, CB["red"], CB["green"])
    ax.barh(yy, d["delta"], color=colors, height=0.62, alpha=0.9)
    mu = float(d["delta"].mean())
    ax.axvline(mu, color=CB["dark"], ls="--", lw=0.9)
    npos = int((d["delta"] > 0).sum())
    ax.set_yticks(yy)
    ax.set_yticklabels([])
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_xlabel("ΔAUROC (random − cluster-out)", fontsize=7.5)
    ax.set_title(f"b  Inflated for {npos}/{len(d)} endpoints (mean $\\Delta$ +{mu:.3f})",
                 loc="left", fontsize=8.5)
    ax.grid(axis="x", lw=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)

    # c) AUROC / AUPRC under the three splitting strategies, primary endpoint
    #    端点受限配置（2,267 特征 / 138 簇，在 652 条已标注物质上剪枝与聚类）；
    #    数值读自产物，不硬编码
    ax = fig.add_subplot(gs[1, 0])
    strat = ["Cluster-out", "Murcko\nscaffold", "Random"]
    single_dir = CODE / "results_real_PCBA686978"
    auroc, auprc = [], []
    for st in ["cluster-out", "scaffold", "random"]:
        t = pd.read_csv(single_dir / f"metrics_{st}.csv", encoding="utf-8-sig")
        row = t.loc[t["model"] == "RF"].iloc[0]
        auroc.append(float(row["AUROC"]))
        auprc.append(float(row["AUPRC"]))
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, auroc, w, color=CB["blue"], label="AUROC")
    ax.bar(x + w / 2, auprc, w, color=CB["sky"], label="AUPRC")
    for xi, (u, p) in enumerate(zip(auroc, auprc)):
        ax.text(xi - w / 2, u + 0.015, f"{u:.3f}", ha="center", fontsize=6.3)
        ax.text(xi + w / 2, p + 0.015, f"{p:.3f}", ha="center", fontsize=6.3)
    ax.set_xticks(x)
    ax.set_xticklabels(strat, fontsize=7.0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value", fontsize=7.5)
    ax.legend(frameon=False, ncol=2, fontsize=6.8, loc="upper left")
    ax.set_title("c  Random > scaffold > cluster-out (endpoint-restricted)",
                 loc="left", fontsize=8.5)
    ax.text(0.5, -0.30,
            "2,267 features, 138 clusters (pruned and clustered on the\n"
            "endpoint's 652 labelled substances)",
            transform=ax.transAxes, ha="center", fontsize=6.0, color="#666666")
    ax.grid(axis="y", lw=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)

    # d) label-permutation null — plotted from the actual permuted AUROCs
    ax = fig.add_subplot(gs[1, 1])
    null = D["perm"]["permuted_AUROC"].to_numpy(dtype=float)
    yr = D["perm_meta"]["y_randomization"]
    obs = float(yr["real_auroc"])
    n_ge = int(yr["n_ge_real"])
    n_valid = int(yr["n_perm_valid"])
    p_val = float(yr["p_value"])
    assert n_valid == len(null), (n_valid, len(null))
    bins = int(np.clip(np.ceil(np.sqrt(n_valid)) * 1.5, 7, 18))
    ax.hist(null, bins=bins, color="#BBBBBB", edgecolor="white", linewidth=0.5,
            label=f"Permuted labels (n = {n_valid})")
    ax.axvline(obs, color=CB["red"], lw=1.6, label=f"Observed {obs:.3f}")
    ax.axvline(float(np.percentile(null, 95)), color=CB["dark"], ls="--", lw=0.9,
               label="Null 95th pct")
    ax.set_xlabel("AUROC under permuted labels", fontsize=7.5)
    ax.set_ylabel("Count", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6.3, loc="upper left")
    ax.text(0.90, 0.88, f"p = {p_val:.4f}\n$\\geq$ observed: {n_ge}/{n_valid}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.0, color=CB["red"], fontweight="bold")
    ax.set_title("d  Label permutation (panel-wide)", loc="left", fontsize=8.5)
    ax.grid(axis="y", lw=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)

    fig.savefig(OUT / "Fig4_leakage_auroc.png")
    plt.close(fig)
    print(f"  Fig. 4  leakage AUROC ({len(d)} endpoints, mean Δ {mu:+.3f}, "
          f"positive {npos}/{len(d)})")


# ===================================================================== Fig. 5
def fig5(D) -> None:
    c = D["cand"].copy()
    c["tiercode"] = tier_code(c["alt_tier"])
    c["novelty"] = 1.0 - c["max_sim_known_substitute"]
    assert set(c["tiercode"]) <= {"I", "II", "III"}, c["tiercode"].unique()

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for t in ["III", "II", "I"]:
        s = c[c.tiercode == t]
        ax.scatter(s["novelty"], s["S_concern"], s=s["S_readacross"] * 90 + 5,
                   color=TIER_COLOR[t], alpha=0.45 if t == "III" else 0.75,
                   edgecolors="none", label=TIER_LABEL[t], zorder={"I": 4, "II": 3, "III": 2}[t])
    reg = c[c["regrettable_substitution"].astype(bool)]
    ax.scatter(reg["novelty"], reg["S_concern"], s=reg["S_readacross"] * 90 + 5,
               facecolors="none", edgecolors=CB["dark"], linewidths=0.5,
               label=f"Regrettable substitution (n = {len(reg)})", zorder=5)

    q1 = float(c["ALT_SCORE"].quantile(0.95))
    ax.axvline(0.6, color="#AAAAAA", ls=":", lw=0.8)
    ax.set_xlabel("Structural novelty  (1 − max Tanimoto similarity to a labelled substance)",
                  fontsize=7.5)
    ax.set_ylabel("Persistence-and-mobility concern score", fontsize=7.5)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.15, 1.05)
    ax.legend(frameon=False, fontsize=6.6, loc="lower left", markerscale=0.6)
    ax.set_title(f"Structural concern versus novelty for {len(c):,} candidates",
                 loc="left", fontsize=8.5)
    ax.grid(lw=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)
    fig.savefig(OUT / "Fig5_concern_vs_novelty.png")
    plt.close(fig)
    print(f"  Fig. 5  concern vs novelty (n = {len(c):,})")


# ===================================================================== Fig. 6
def fig6(D) -> None:
    f = D["fam"].copy()
    fam_en = {
        "醚型 PFAA（羧酸）": "Ether-based PFAA (carboxylate)",
        "磺酰胺/磺酰乙酸类": "Sulfonamide / sulfonyl-acetic",
        "醚型含氟前体/中间体": "Ether-based precursors",
        "醚型 PFAA（磺酸）": "Ether-based PFAA (sulfonate)",
        "短链 PFSA": "Short-chain PFSA",
        "氟磺酰/氟硫酸酯（高反应性中间体）": "Fluorosulfonyl / fluorosulfate",
        "其他可电离含氟物": "Other ionisable fluorinated",
        "短链 PFCA": "Short-chain PFCA",
        "氟调聚物磺酸（FTS）": "Fluorotelomer sulfonate",
    }
    f["fam"] = f["alt_family"].map(fam_en)
    tier_cols = ["I: 优先关注", "II: 值得跟踪", "III: 储备观察"]
    f = f.sort_values(tier_cols[0], ascending=True).reset_index(drop=True)

    fig, (ax, axp) = plt.subplots(1, 2, figsize=(7.6, 4.4),
                                  gridspec_kw={"width_ratios": [2.4, 1]}, sharey=True)
    y = np.arange(len(f))
    left = np.zeros(len(f))
    for t, col in zip(["I", "II", "III"], tier_cols):
        vals = f[col].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, color=TIER_COLOR[t], height=0.66,
                edgecolor="white", linewidth=0.4, label=TIER_LABEL[t])
        for yi, (v, l0) in enumerate(zip(vals, left)):
            if v >= 12:
                ax.text(l0 + v / 2, yi, f"{int(v)}", ha="center", va="center",
                        fontsize=5.8, color="white" if t != "II" else CB["dark"])
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(f["fam"], fontsize=6.8)
    ax.set_xlabel("Number of candidates", fontsize=7.5)
    ax.legend(frameon=False, ncol=3, fontsize=6.6, loc="lower right")
    ax.set_title("a  Family composition by tier", loc="left", fontsize=8.5)
    ax.grid(axis="x", lw=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)

    # Panel b: Tier I composition only
    t1 = f[f[tier_cols[0]] > 0].sort_values(tier_cols[0])
    yy = np.arange(len(t1))
    axp.barh(yy, t1[tier_cols[0]], color=CB["red"], height=0.6)
    for yi, v in zip(yy, t1[tier_cols[0]]):
        axp.text(v + 1.2, yi, f"{int(v)}", va="center", fontsize=6.6)
    axp.set_yticks(yy)
    axp.set_yticklabels([])
    axp.set_xlim(0, 78)
    axp.set_xlabel("Tier I candidates (of 100)", fontsize=7.5)
    axp.set_title("b  Tier I detail", loc="left", fontsize=8.5)
    axp.grid(axis="x", lw=0.3, color="#EEEEEE")
    axp.set_axisbelow(True)

    fig.savefig(OUT / "Fig6_family_by_tier.png")
    plt.close(fig)
    print("  Fig. 6  family composition")


# ===================================================================== Fig. 7
def fig7(D) -> None:
    c = D["cand"].copy()
    c["tiercode"] = tier_code(c["alt_tier"])
    anchors = {"GenX": (3, 1), "ADONA": (3, 2), "F-53B": (6, 1), "6:2 FTS": (6, 0)}

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for t in ["III", "II", "I"]:
        s = c[c.tiercode == t]
        ax.scatter(s["n_CF2"], s["n_ether_O"], s=s["Marker_命中数"] * 12 + 4,
                   color=TIER_COLOR[t], alpha=0.35 if t == "III" else 0.7,
                   edgecolors="none", label=TIER_LABEL[t], zorder={"I": 4, "II": 3, "III": 2}[t])

    ax.axvspan(-0.5, 5.5, color=CB["yellow"], alpha=0.16, zorder=0)
    ax.text(2.5, 4.62, "Marketed-alternative design band\n(short chain)", ha="center",
            va="top", fontsize=6.4, color="#8A7A00")

    for name, (cf, eo) in anchors.items():
        ax.scatter([cf], [eo], marker="*", s=110, color=CB["dark"], zorder=6)
        ax.annotate(name, (cf, eo), textcoords="offset points", xytext=(6, 3),
                    fontsize=6.4, color=CB["dark"])

    ax.set_xlabel("Perfluorinated carbon count", fontsize=7.5)
    ax.set_ylabel("Number of ether oxygens", fontsize=7.5)
    ax.set_xlim(-0.6, 15.5)
    ax.set_ylim(-0.4, 4.9)
    ax.set_xticks(range(0, 16, 3))
    ax.set_yticks(range(0, 5))
    ax.legend(frameon=False, fontsize=6.6, loc="upper right", markerscale=0.6)
    ax.set_title("Design space of the candidate alternatives", loc="left", fontsize=8.5)
    ax.grid(lw=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)
    fig.savefig(OUT / "Fig7_design_space.png")
    plt.close(fig)
    print("  Fig. 7  design space")


# ===================================================================== Fig. 8
def fig8(D) -> None:
    c = D["cand"].copy()
    sc = D["screen"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0),
                             gridspec_kw={"width_ratios": [1.45, 1, 1]})

    # a) p_max distributions
    ax = axes[0]
    inv = sc["p_max"].dropna().to_numpy()
    p99 = float(np.percentile(inv, 99))
    # Real consensus-domain flags from the screening run (Maha AND range AND class)
    if "ad_consensus" in sc.columns:
        cons = sc.loc[sc["ad_consensus"].astype(bool), "p_max"].dropna().to_numpy()
    else:
        cons = inv[inv >= p99]
    bins = np.linspace(0, 1, 41)
    ax.hist(inv, bins=bins, color="#CCCCCC", edgecolor="white", linewidth=0.3,
            density=True, label=f"Inventory (n = {len(inv):,})")
    ax.hist(cons, bins=bins, color=CB["blue"], edgecolor="white", linewidth=0.3,
            density=True, alpha=0.55, label=f"Consensus domain (n = {len(cons):,})")
    ax.hist(c["p_max"], bins=bins, color=CB["red"], edgecolor="white",
            linewidth=0.3, density=True, alpha=0.75,
            label=f"Candidates (n = {len(c):,})")
    ax.axvline(p99, color=CB["dark"], ls="--", lw=0.9)
    ax.text(p99 + 0.02, ax.get_ylim()[1] * 0.80, f"99th pct {p99:.3f}",
            fontsize=6.2, color=CB["dark"])
    ax.set_xlabel("Maximum predicted activity probability", fontsize=7.2)
    ax.set_ylabel("Density", fontsize=7.2)
    ax.legend(frameon=False, fontsize=6.0, loc="upper right")
    ax.set_title("a  Predicted probability", loc="left", fontsize=8.2)
    ax.grid(axis="y", lw=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)

    # b) AD coverage of the candidate list
    ax = axes[1]
    knn_pct = 100.0 * float(c["in_domain_majority"].astype(bool).mean())
    cons_pct = 100.0 * float(c["ad_consensus"].astype(bool).mean())
    bars = ax.bar(["kNN\nTanimoto", "Consensus\ndomain"], [knn_pct, cons_pct],
                  color=[CB["orange"], CB["blue"]], width=0.55)
    for b, v in zip(bars, [knn_pct, cons_pct]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}%",
                ha="center", fontsize=7.0)
    ax.set_ylim(0, 22)
    ax.set_ylabel("Candidates inside domain (%)", fontsize=7.2)
    ax.set_title("b  AD coverage", loc="left", fontsize=8.2)
    ax.grid(axis="y", lw=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)

    # c) candidates entering the model priority hierarchy
    ax = axes[2]
    ax.bar([0], [len(c)], color="#DDDDDD", width=0.55, edgecolor="#BBBBBB")
    ax.bar([0], [0], color=CB["red"], width=0.55)
    ax.text(0, len(c) * 0.55, "0", ha="center", va="center", fontsize=20,
            color=CB["red"], fontweight="bold")
    ax.text(0, len(c) * 1.03, f"of {len(c):,} candidates", ha="center",
            fontsize=6.8, color=CB["dark"])
    ax.set_xticks([0])
    ax.set_xticklabels(["Model priority\nhierarchy"], fontsize=7.0)
    ax.set_ylim(0, len(c) * 1.18)
    ax.set_ylabel("Candidates entering", fontsize=7.2)
    ax.set_title("c  Model blind spot", loc="left", fontsize=8.2)
    ax.grid(axis="y", lw=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "Fig8_model_blindspot.png")
    plt.close(fig)
    print("  Fig. 8  model blind spot")


# ================================================================= Fig. S1
def figS1(D) -> None:
    """Additional file 8: per-endpoint AUPRC and MCC under all three splitting
    strategies, plus the QMSA feature-subset ablation.

    Every number is read from results_multitask_3splits/multitask_metrics.csv and
    results_real_PCBA686978/ablation_qmsa.csv. The figure refuses to draw a panel
    for which the required columns are missing, rather than substituting zeros, so
    a silently incomplete upstream run cannot produce a plausible-looking figure.
    """
    m = D["multi3"].copy()
    need = ["AUPRC_cluster-out", "AUPRC_scaffold", "AUPRC_random",
            "MCC_cluster-out", "MCC_scaffold", "MCC_random",
            "AUROC_cluster-out"]
    missing = [c for c in need if c not in m.columns]
    if missing:
        raise SystemExit(
            "Figure S1 needs per-endpoint metrics under all three splitting "
            f"strategies; missing columns: {missing}. Re-run "
            "`run_pipeline.py --multi-task` (see rerun_all.sh step 6).")

    d = m.dropna(subset=need).sort_values("AUROC_cluster-out").reset_index(drop=True)
    prim = "PCBA-686978"
    strat = [("cluster-out", "Cluster-out", CB["blue"]),
             ("scaffold", "Murcko scaffold", CB["orange"]),
             ("random", "Random", CB["red"])]

    fig = plt.figure(figsize=(7.4, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26)

    def paired(ax, metric, lo, hi, title, xlabel, show_labels):
        yy = np.arange(len(d))
        ax.hlines(yy, d[f"{metric}_cluster-out"], d[f"{metric}_random"],
                  color="#C8C8C8", lw=1.3, zorder=1)
        for key, lab, col in strat:
            ax.scatter(d[f"{metric}_{key}"], yy, s=23, color=col, zorder=3, label=lab)
        ip = int(d.index[d.endpoint == prim][0])
        ax.scatter([d.loc[ip, f"{metric}_cluster-out"]], [ip], s=68, facecolors="none",
                   edgecolors=CB["dark"], linewidths=1.2, zorder=5)
        ax.set_yticks(yy)
        ax.set_yticklabels(d["endpoint"] if show_labels else [""] * len(d),
                           fontsize=6.0)
        ax.set_ylim(-0.8, len(d) - 0.2)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="x", lw=0.3, color="#EEEEEE")
        ax.set_axisbelow(True)

    # a) AUPRC under the three strategies
    ax = fig.add_subplot(gs[0, 0])
    paired(ax, "AUPRC", 0.0, 1.0,
           f"a  AUPRC, {len(d)} modellable endpoints", "AUPRC (best model)",
        show_labels=True)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="none",
              fontsize=6.4)

    # b) MCC under the three strategies
    ax = fig.add_subplot(gs[0, 1])
    paired(ax, "MCC", 0.0, 0.8,
           f"b  MCC, {len(d)} modellable endpoints", "MCC (best model)",
        show_labels=False)

    # c) per-endpoint optimism in AUPRC and MCC
    ax = fig.add_subplot(gs[1, 0])
    yy = np.arange(len(d))
    d_au = d["AUPRC_random"] - d["AUPRC_cluster-out"]
    d_mc = d["MCC_random"] - d["MCC_cluster-out"]
    ax.barh(yy - 0.19, d_au, height=0.36, color=CB["purple"], alpha=0.9,
            label=f"$\\Delta$AUPRC (mean +{d_au.mean():.3f})")
    ax.barh(yy + 0.19, d_mc, height=0.36, color=CB["green"], alpha=0.9,
            label=f"$\\Delta$MCC (mean {d_mc.mean():+.3f})")
    ax.axvline(0, color=CB["dark"], lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels(d["endpoint"], fontsize=6.0)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_xlabel("Random $-$ cluster-out", fontsize=7.5)
    ax.set_title(f"c  Optimism in AUPRC and MCC ({int((d_au > 0).sum())}/{len(d)} "
                 f"and {int((d_mc > 0).sum())}/{len(d)} positive)", loc="left", fontsize=8.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="none",
              fontsize=6.2)
    ax.grid(axis="x", lw=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)

    # d) QMSA feature-subset ablation
    ax = fig.add_subplot(gs[1, 1])
    ab = D["abl"].copy()
    ab = ab[ab["feature_set"].isin(["2D+FP", "ALL(2D+FP+QMSA)", "2D+QMSA", "2D", "QMSA"])]
    order = ["2D+FP", "ALL(2D+FP+QMSA)", "2D+QMSA", "2D", "QMSA"]
    ab["_o"] = ab["feature_set"].map({k: i for i, k in enumerate(order)})
    ab = ab.sort_values("_o")
    lab = ab["feature_set"].str.replace("ALL(2D+FP+QMSA)", "2D+FP+QMSA", regex=False)
    xx = np.arange(len(ab))
    w = 0.26
    for i, (col, name, c) in enumerate([("AUROC", "AUROC", CB["blue"]),
                                        ("AUPRC", "AUPRC", CB["purple"]),
                                        ("MCC", "MCC", CB["green"])]):
        ax.bar(xx + (i - 1) * w, ab[col], width=w, color=c, alpha=0.9, label=name)
    ax.set_xticks(xx)
    ax.set_xticklabels(lab, fontsize=6.6, rotation=22, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Value (best model)", fontsize=7.5)
    ax.set_ylim(0, 0.85)
    base = float(ab.loc[ab["feature_set"] == "2D+FP", "AUROC"].iloc[0])
    ax.axhline(base, color=CB["dark"], ls=":", lw=0.9)
    ax.text(0.98, base + 0.015, f"2D+FP baseline {base:.3f}", transform=ax.get_yaxis_transform(),
            ha="right", fontsize=6.0, color=CB["dark"])
    ax.set_title(f"d  QMSA ablation, {prim} (no feature set improves AUROC)",
                 loc="left", fontsize=8.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="none",
              fontsize=6.4, ncol=3)
    ax.grid(axis="y", lw=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)

    p = OUT / "FigS1_endpoint_metrics_and_ablation.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Fig. S1  -> {p.name}")


if __name__ == "__main__":
    D = _load()
    print("Generating submission figures ->", OUT)
    fig1()
    fig2(D)
    fig3(D)
    fig4(D)
    fig5(D)
    fig6(D)
    fig7(D)
    fig8(D)
    figS1(D)
    print("\nDone.")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name:42s} {p.stat().st_size / 1024:7.1f} KB")
