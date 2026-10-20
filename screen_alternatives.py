#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PFASSTRUCT V6 → 「真正的 PFAS 新兴替代物」候选清单
=====================================================

为什么不能用活性模型来做
------------------------
Cheng & Ng (ES&T 2019) 的 1,012 条有标签 PFAS 里：
  · 95.6% 含芳环（多为含氟农药/药物骨架）
  · 只有 8 条（0.8%）落在「PFAS 替代物」化学空间
    （酸性头基 + 非芳环 + 全氟碳数 ≥3）
而 PFASSTRUCT V6 里有 4,389 条落在该空间。
→ 用这批标签训练出的模型，其「高优先」= 「长得像训练集里的含氟芳香物」，
  实测 Tier A 名单 100% 含环、以氟喹诺酮类酰胺 / 三氟丙醇胺类为主，
  而 GenX / F-53B / 6:2 FTS / ADONA 等真替代物全部被判域外且 p_max 仅 0.13–0.38。
  结论：活性模型无法生成替代物清单。这是适用域问题，不是调参问题。

本模块路线
----------
化学信息学 + read-across（读跨/同类推断），不依赖活性标签：
  ① 化学空间划区（经典长链 PFAA / 替代物空间 / 含氟药农 / 其他）
  ② 剔除训练集成员、近重复、以及已知在用替代物本身
  ③ 归类到替代物家族（短链 PFCA/PFSA、醚型 PFAA、氟调聚物、磺酰胺…）
  ④ 透明打分：新颖性 × 环境关注度代理（持久性 × 迁移性）
  ⑤ 附上模型 p_max 并强制标注「适用域外·不可信」，两路结果并置对照

输出
----
results_alternatives/
  emerging_alternatives.csv        真清单（按 ALT_SCORE 排序）
  emerging_top50.csv               前 50（人工复核用）
  alt_class_summary.csv            家族构成
  chemical_space_partition.csv     全库划区统计
  regulated_vs_emerging.csv        监管物 vs 替代物 化学坐标对照
  domain_mismatch_evidence.csv     适用域错配的量化证据
  run_meta.json

用法
----
    python screen_alternatives.py --query data/pfasstruct_v6.csv --outdir results_alternatives
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
RDLogger.DisableLog("rdApp.*")

import pfas_qsar as P  # noqa: E402
import pfas_anchors as A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# ---------------------------------------------------------------- 结构参数
MIN_CF2 = 2              # 至少要 2 个全氟碳单元才算 PFAS 骨架
SHORT_CHAIN_MAX = 5      # 短链分界（OECD 口径：CF2 ≤5 为短链）
LEGACY_MIN = 6           # 经典长链起点
NEAR_DUP_TRAIN = 0.99    # 与训练集判为近重复
NEAR_DUP_REF = 0.95      # 与已知监管/在用物判为同一物（骨架等同）
ALT_NOVEL_MAX = 0.60     # 与训练集相似度低于此才算「未被研究过」
TIER1_N = 100            # Tier I 规模：可逐条人工复核
TIER2_N = 400            # Tier II 规模

# 已知监管物 / 已商品化替代物（read-across 锚点）
# 唯一真源：pfas_anchors.ANCHORS（20 条 = 14 受监管/前体 PFAA + 6 替代物）。
# 早期版本在此硬编码 20 条，但 ADONA / F-53B / 6:2 FTS / 4:2 FTS 结构有误，
# 且与 screen_domain.load_reference() 的 18 条版本不一致，故改为统一导入。
REFERENCE = A.reference_tuples()
# 其中「已经商品化的替代物」——不能算 emerging（同样取自单一真源）
KNOWN_SUBSTITUTES = set(A.KNOWN_SUBSTITUTES)

_GEN = GetMorganGenerator(radius=2, fpSize=2048)


def mol_of(smi):
    return Chem.MolFromSmiles(smi)


def fp_of(mol):
    return _GEN.GetFingerprint(mol) if mol is not None else None


def tanimoto_to_set(fp, fps):
    if fp is None or not fps:
        return 0.0
    return float(max(DataStructs.BulkTanimotoSimilarity(fp, fps)))


# ---------------------------------------------------------------- 结构表征
def characterize(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """补齐结构表征列（一律从 SMILES 现算，保证口径一致）。"""
    smells = df["smiles"].tolist()
    flags = pd.DataFrame([P.structural_flags(s) for s in smells])
    for c in ("n_CF2", "n_ether_O", "n_rings", "n_C", "n_F", "F_per_C", "headgroup"):
        df[c] = flags[c].to_numpy() if c in flags else np.nan
    mols = [mol_of(s) for s in smells]
    df["_mol"] = mols
    df["_fp"] = [fp_of(m) for m in mols]
    df["fg"] = [refine_fg(m) for m in mols]
    df["arom"] = [bool(m is not None and any(a.GetIsAromatic() for a in m.GetAtoms()))
                  for m in mols]
    df["mol_wt"] = [float(Descriptors.MolWt(m)) if m is not None else np.nan for m in mols]
    df["tpsa"] = [float(Descriptors.TPSA(m)) if m is not None else np.nan for m in mols]
    df["hbd"] = [rdMolDescriptors.CalcNumHBD(m) if m is not None else np.nan for m in mols]
    df["rotb"] = [rdMolDescriptors.CalcNumRotatableBonds(m) if m is not None else np.nan
                  for m in mols]
    # 全氟化碳占比（持久性代理：全氟碳单元碳原子占全部碳原子的比例）
    df["pf_carbon_frac"] = (2.0 * df["n_CF2"] / df["n_C"].replace(0, np.nan)).clip(0, 1)
    print(f"  [{tag}] {len(df)} 条，含芳环 {df['arom'].mean():.1%}，"
          f"中位 n_CF2 {df['n_CF2'].median():.0f}")
    return df


# ---------------------------------------------------------------- 化学空间划区
def classify_space(df: pd.DataFrame) -> pd.Series:
    """把全库划成 4 个区（顺序敏感：先判 A，再判 B，再判 C，其余兜底 D）。

    A 经典长链 PFAA（监管重点）——直链、无醚、无环的全氟羧/磺酸
    B PFAS 替代物化学空间——可电离头基（羧/磺/磺酰胺）、非芳环、无环、CF2≥2
      含氟调聚物磺酸/羧酸（6:2 FTS 类）也落在这里，它们本身就是替代物
    C 含氟芳香物——训练集的 95.6% 都在此区（含氟药物/农药/材料）
    D 其他含氟结构
    """
    cf2, eth, ring = df["n_CF2"], df["n_ether_O"], df["n_rings"]
    acid = df["headgroup"].isin([1, 2, 3])
    arom = df["arom"]

    legacy = ((cf2 >= LEGACY_MIN) & (eth == 0) & (ring == 0)
              & (df["headgroup"].isin([1, 2, 3])))
    alt = acid & (~arom) & (cf2 >= MIN_CF2) & (ring == 0)

    cls = pd.Series("D: 其他含氟结构", index=df.index)
    cls[arom & (cf2 >= 1)] = "C: 含氟芳香物(药/农/材料)"
    cls[alt] = "B: PFAS 替代物化学空间"
    cls[legacy] = "A: 经典长链 PFAA(监管)"
    return cls


# ---------------------------------------------------------------- 替代物家族
# ---------------------------------------------------------------- 精细官能团
# pfas_qsar.structural_flags 的 headgroup 只做「有/无」判断且羧基优先，会把
# 酯类和磺酰乙酸误判成羧酸。替代物分类需要更细的口径，故在此单列。
_FG_SMARTS = {
    "cooh": "[CX3](=[OX1])[OX2H1]",                       # 羧酸
    "so3h": "[SX4](=[OX1])(=[OX1])[OX2H1]",               # 磺酸
    "so2n": "[SX4](=[OX1])(=[OX1])[NX3]",                 # 磺酰胺
    "so2c": "[SX4](=[OX1])(=[OX1])[#6]",                  # 磺酰碳（磺酰乙酸/砜）
    "so2f": "[SX4](=[OX1])(=[OX1])F",                     # 氟磺酰（磺酰氟）
    "oso2f": "[OX2][SX4](=[OX1])(=[OX1])F",               # 氟硫酸酯
    "ester": "[CX3](=[OX1])[OX2][#6]",                    # 酯
    "amide": "[CX3](=[OX1])[NX3]",                        # 酰胺
    "oh": "[OX2H1]",                                      # 醇/酚
    "alkyl_h": "[CX4]([H])([H])",                          # 非全氟碳（调聚物间隔）
    "cf3": "[CX4](F)(F)(F)",                              # 端基 CF3
}
_FG_MOLS = {k: Chem.MolFromSmarts(v) for k, v in _FG_SMARTS.items()}


def refine_fg(mol) -> str:
    """返回精细官能团标签（用于替代物家族归类）。"""
    if mol is None:
        return "unknown"
    has = {k: mol.HasSubstructMatch(v) for k, v in _FG_MOLS.items()}
    if has["so2f"] or has["oso2f"]:
        return "氟磺酰基中间体"
    if has["so2n"]:
        return "磺酰胺"
    if has["so3h"]:
        return "磺酸"
    if has["cooh"]:
        return "羧酸"
    if has["so2c"]:
        return "磺酰/砜"
    if has["ester"]:
        return "酯"
    if has["amide"]:
        return "酰胺"
    if has["oh"]:
        return "醇"
    return "其他"


def alt_family(row) -> str:
    """替代物家族归类。依赖 fg（精细官能团）与结构标志。"""
    fg = row["fg"]
    eth, ring, cf2 = row["n_ether_O"], row["n_rings"], row["n_CF2"]
    hg = row["headgroup"]

    if fg == "氟磺酰基中间体":
        return "氟磺酰/氟硫酸酯（高反应性中间体）"
    if fg == "磺酰胺":
        return "磺酰胺/磺酰乙酸类"
    if fg in ("羧酸", "磺酸"):
        if eth >= 1:
            return "醚型 PFAA（羧酸）" if fg == "羧酸" else "醚型 PFAA（磺酸）"
        if ring > 0:
            return "环状替代物"
        is_telomer = row["_mol"].HasSubstructMatch(_FG_MOLS["alkyl_h"]) and cf2 < row["n_F"] / 2.0
        if fg == "羧酸":
            return "氟调聚物羧酸（FTCA）" if is_telomer else "短链 PFCA"
        return "氟调聚物磺酸（FTS）" if is_telomer else "短链 PFSA"
    if eth >= 1:
        return "醚型含氟前体/中间体"
    if hg in (1, 2, 3):
        return "其他可电离含氟物"
    return "含氟中间体"


# ---------------------------------------------------------------- 打分
def score(df: pd.DataFrame) -> pd.DataFrame:
    # 新颖性：与训练集最高相似度越低越新
    novel = 1.0 - df["max_sim_train"]
    # 与监管/在用物的 read-across 强度（0.3–0.95 区间线性映射：太像=已知物，太不像=另类）
    ra = ((df["max_sim_reference"] - 0.30) / 0.65).clip(0, 1)
    ra = np.where(df["max_sim_reference"] > NEAR_DUP_REF, 0.0, ra)
    # 持久性代理：全氟碳占比
    persist = df["pf_carbon_frac"].fillna(0.0)
    # 迁移性代理：链越短、醚氧越多 → 越易迁移（Koc 低、水溶性强）
    mobility = np.clip(1.0 - (df["n_CF2"] - 2) / 10.0, 0.0, 1.0)
    mobility = np.clip(mobility + 0.15 * np.minimum(df["n_ether_O"], 2), 0.0, 1.0)
    df["S_novelty"] = novel
    df["S_readacross"] = ra
    df["S_persistence"] = persist
    df["S_mobility"] = mobility
    df["S_concern"] = 0.5 * persist + 0.5 * mobility
    df["ALT_SCORE"] = (0.35 * novel + 0.20 * ra +
                       0.45 * df["S_concern"])

    # 四个「持久性/迁移性相关结构标记」——全部为布尔，便于逐条核对。
    # 命名纪律：这些是 structural markers，**不是** PMT 评估。本工作不含毒性项，
    # 因此管道、表头与产物一律不得出现 PMT 字样（母本已全面改名）。
    df["Marker_短链(CF2<=5)"] = df["n_CF2"] <= SHORT_CHAIN_MAX
    df["Marker_含醚键"] = df["n_ether_O"] >= 1
    df["Marker_可电离"] = df["fg"].isin(["羧酸", "磺酸", "磺酰胺", "氟磺酰基中间体"])
    df["Marker_高全氟碳占比(>=0.5)"] = df["pf_carbon_frac"] >= 0.5
    df["Marker_命中数"] = df[["Marker_短链(CF2<=5)", "Marker_含醚键", "Marker_可电离",
                            "Marker_高全氟碳占比(>=0.5)"]].sum(1)
    return df


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="PFASSTRUCT V6 新兴替代物清单（化学信息学路线）")
    ap.add_argument("--query", default=os.path.join(DATA_DIR, "pfasstruct_v6.csv"))
    ap.add_argument("--train", default=os.path.join(DATA_DIR, "pfas_chengng.csv"))
    ap.add_argument("--score-table",
                    default=os.path.join(HERE, "results_screen_v6", "screen_all.csv"),
                    help="可选：域内筛查输出，用于并置模型 p_max（适用域外时标注不可信）")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results_alternatives"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    log = []

    def say(s):
        print(s)
        log.append(str(s))

    say("=" * 78)
    say("PFASSTRUCT V6 新兴替代物清单（化学信息学 + read-across）")
    say("=" * 78)

    # ---------------------------------------------------------- 1. 数据
    q = pd.read_csv(args.query, encoding="utf-8-sig")
    q.columns = [c.strip() for c in q.columns]
    if "smiles" not in q.columns:
        q = q.rename(columns={q.columns[0]: "smiles"})
    q["smiles"] = [P.standardize_smiles(s) for s in q["smiles"]]
    q = q[q["smiles"].notna()].drop_duplicates("smiles").reset_index(drop=True)

    tr = pd.read_csv(args.train, encoding="utf-8-sig")
    tr.columns = [c.strip() for c in tr.columns]
    tr = tr.rename(columns={tr.columns[0]: "smiles"})
    tr["smiles"] = [P.standardize_smiles(s) for s in tr["smiles"]]
    tr = tr[tr["smiles"].notna()]

    ref = pd.DataFrame({"ref_name": [r[0] for r in REFERENCE],
                        "note": [r[1] for r in REFERENCE],
                        "smiles": [P.standardize_smiles(r[2]) for r in REFERENCE]})
    say(f"[数据] 结构全集 {len(q)} 条 / 有标签训练集 {len(tr)} 条 / 参考锚点 {len(ref)} 条")

    say("\n[1] 结构表征")
    q = characterize(q, "V6")
    tr = characterize(tr, "训练集")
    ref = characterize(ref, "参考锚点")

    # ---------------------------------------------------------- 2. 划区
    say("\n[2] 化学空间划区")
    q["space"] = classify_space(q)
    for k, n in q["space"].value_counts().sort_index().items():
        say(f"    {k:<28s} {n:6d}  ({n / len(q):6.2%})")
    tr_space = classify_space(tr)
    tr_counts = tr_space.value_counts()
    pd.DataFrame({
        "space": ["A: 经典长链 PFAA(监管)", "B: PFAS 替代物化学空间",
                  "C: 含氟芳香物(药/农/材料)", "D: 其他含氟结构"],
        "n": [int(tr_counts.get(k, 0)) for k in
              ["A: 经典长链 PFAA(监管)", "B: PFAS 替代物化学空间",
               "C: 含氟芳香物(药/农/材料)", "D: 其他含氟结构"]],
    }).to_csv(os.path.join(args.outdir, "train_space_counts.csv"),
              index=False, encoding="utf-8-sig")
    say(f"    训练集落在「替代物化学空间」的：{int((tr_space == 'B: PFAS 替代物化学空间').sum())} 条"
        f" / {len(tr)}")

    # ---------------------------------------------------------- 3. 相似度
    say("\n[3] 相似度：与训练集 / 与监管锚点")
    fps_tr = tr["_fp"].tolist()
    fps_ref = ref["_fp"].tolist()
    q["max_sim_train"] = [tanimoto_to_set(f, fps_tr) for f in q["_fp"]]
    q["max_sim_reference"] = [tanimoto_to_set(f, fps_ref) for f in q["_fp"]]
    tr_smi = set(tr["smiles"])
    ref_smi = set(ref["smiles"])
    q["in_training"] = q["smiles"].isin(tr_smi)
    q["is_reference"] = q["smiles"].isin(ref_smi)
    q["near_dup_train"] = q["max_sim_train"] >= NEAR_DUP_TRAIN
    q["near_dup_reference"] = q["max_sim_reference"] >= NEAR_DUP_REF
    say(f"    与训练集近重复(≥{NEAR_DUP_TRAIN})：{int(q['near_dup_train'].sum())} 条；"
        f"完全同构：{int(q['in_training'].sum())} 条")
    say(f"    与监管锚点近同(≥{NEAR_DUP_REF})：{int(q['near_dup_reference'].sum())} 条；"
        f"即锚点本身：{int(q['is_reference'].sum())} 条")

    # ---------------------------------------------------------- 4. 替代物清单
    say("\n[4] 生成新兴替代物候选")
    q["alt_family"] = q.apply(alt_family, axis=1)
    alt_space = q["space"] == "B: PFAS 替代物化学空间"
    eligible = (alt_space & (~q["in_training"]) & (~q["near_dup_train"])
                & (~q["near_dup_reference"]) & (~q["is_reference"])
                & (q["max_sim_train"] < ALT_NOVEL_MAX))
    say(f"    替代物化学空间内 {int(alt_space.sum())} 条")
    say(f"    剔除训练集成员 / 近重复 / 已知在用物 / 与训练集相似度≥{ALT_NOVEL_MAX} 后 "
        f"→ {int(eligible.sum())} 条")
    cand = q[eligible].copy()
    if len(cand) == 0:
        raise SystemExit("候选为空：检查化学空间定义")

    cand = score(cand)
    cand = cand.sort_values("ALT_SCORE", ascending=False).reset_index(drop=True)

    # 与模型预测并置（如果域内筛查已完成）
    if os.path.exists(args.score_table):
        sc = pd.read_csv(args.score_table, encoding="utf-8-sig",
                         usecols=lambda c: c in ("smiles", "p_max", "ad_consensus",
                                                 "in_domain_majority", "tier"))
        cand = cand.merge(sc, on="smiles", how="left")
        cand["model_p_max"] = cand.get("p_max")
        cand["model_usable"] = cand.get("ad_consensus").fillna(False)
        say(f"    已并置模型预测列（其中适用域内仅 {int(cand['model_usable'].sum())} 条，"
            f"其余模型值不可用）")
    else:
        say("    [warn] 未找到域内筛选输出，跳过模型列并置")

    # ★ 遗憾替代（regrettable substitution）标记：
    #   与已商品化替代物（GenX/ADONA/F-53B/6:2 FTS/PFMOAA）结构相近，说明它属于
    #   「正在被投入使用的替代路线」；同时与训练集不相似 → 说明尚无毒性数据。
    #   这两条同时成立 = 最典型的下一个 GenX。
    known_sub_fps = ref.loc[ref["ref_name"].isin(KNOWN_SUBSTITUTES), "_fp"].tolist()
    cand["max_sim_known_substitute"] = [
        tanimoto_to_set(f, known_sub_fps) for f in cand["_fp"]]
    cand["regrettable_substitution"] = (
        (cand["max_sim_known_substitute"] >= 0.45)
        & (cand["max_sim_train"] < ALT_NOVEL_MAX)
        & (cand["S_concern"] >= 0.60))
    say(f"    遗憾替代标记命中：{int(cand['regrettable_substitution'].sum())} 条"
        f"（与已商品化替代物相似度≥0.45 且无训练标签）")

    # 分层：ALT_SCORE 降序，固定规模（Tier I 100 条 = 可逐条人工复核的量级）
    tier = pd.Series("III: 储备观察", index=cand.index)
    tier.iloc[TIER1_N:TIER1_N + TIER2_N] = "II: 值得跟踪"
    tier.iloc[:TIER1_N] = "I: 优先关注"
    cand["alt_tier"] = tier

    drop_cols = [c for c in ("_mol", "_fp") if c in cand.columns]
    out = cand.drop(columns=drop_cols)
    out.to_csv(os.path.join(args.outdir, "emerging_alternatives.csv"),
               index=False, encoding="utf-8-sig")
    out.head(50).to_csv(os.path.join(args.outdir, "emerging_top50.csv"),
                        index=False, encoding="utf-8-sig")
    out[out["alt_tier"].str.startswith("I:")].to_csv(
        os.path.join(args.outdir, "emerging_tier1.csv"), index=False, encoding="utf-8-sig")
    out[out["regrettable_substitution"]].to_csv(
        os.path.join(args.outdir, "emerging_regrettable.csv"), index=False,
        encoding="utf-8-sig")
    say(f"    清单已输出：{len(out)} 条（Tier I {int((tier == 'I: 优先关注').sum())} 条）")

    # 家族构成
    fam = pd.crosstab(out["alt_family"], out["alt_tier"])
    fam["n"] = fam.sum(axis=1)
    fam = fam.sort_values("I: 优先关注", ascending=False)
    fam.to_csv(os.path.join(args.outdir, "alt_class_summary.csv"), encoding="utf-8-sig")
    fi = out[out["alt_tier"].str.startswith("I:")]
    say(f"\n[5] 家族构成（全部 {len(out)} 条；其中 Tier I {len(fi)} 条）")
    say(fam.to_string())
    say("    —— Tier I 名单的家族 ——")
    say(fi["alt_family"].value_counts().to_string())

    # ---------------------------------------------------------- 全库划区统计
    part = (q.groupby("space")
            .agg(n=("smiles", "size"),
                 median_CF2=("n_CF2", "median"),
                 median_MW=("mol_wt", "median"),
                 aromatic_pct=("arom", "mean"),
                 ether_pct=("n_ether_O", lambda s: float((s > 0).mean())),
                 acidic_pct=("headgroup", lambda s: float(s.isin([1, 2, 3]).mean()))))
    part["pct"] = part["n"] / len(q)
    part.to_csv(os.path.join(args.outdir, "chemical_space_partition.csv"),
                encoding="utf-8-sig")

    # ------------------------------------------------ 监管物 vs 替代物 对照
    ref_out = ref.drop(columns=drop_cols).copy()
    ref_out["max_sim_train"] = [tanimoto_to_set(f, fps_tr) for f in ref["_fp"]]
    ref_out["in_training"] = ref_out["smiles"].isin(tr_smi)
    ref_out["space"] = classify_space(ref)
    ref_out["known_substitute"] = ref_out["ref_name"].isin(KNOWN_SUBSTITUTES)
    ref_out.to_csv(os.path.join(args.outdir, "regulated_vs_emerging.csv"),
                   index=False, encoding="utf-8-sig")

    # ------------------------------------------------ 适用域错配证据
    ev = []
    ev.append({"证据": "训练集含芳环比例", "值": f"{tr['arom'].mean():.1%}",
               "说明": "Cheng & Ng 1012 条，以含氟农药/药物骨架为主"})
    ev.append({"证据": "训练集落在替代物化学空间", "值": f"{int((tr_space == 'B: PFAS 替代物化学空间').sum())}/{len(tr)}"
               f" ({(tr_space == 'B: PFAS 替代物化学空间').mean():.1%})",
               "说明": "酸性头基+非芳环+CF2≥2+无环"})
    ev.append({"证据": "V6 落在替代物化学空间",
               "值": f"{int(alt_space.sum())}/{len(q)} ({alt_space.mean():.1%})",
               "说明": "训练集标签几乎不覆盖该空间"})
    ev.append({"证据": "监管锚点在训练集内的同类样本数",
               "值": f"中位 {int(np.median(ref_out['max_sim_train'].values) * 100)}% 相似度",
               "说明": "无一条形成类别级覆盖"})
    ev.append({"证据": "已知在用替代物的训练集相似度",
               "值": ", ".join(f"{r['ref_name']}={r['max_sim_train']:.2f}"
                              for _, r in ref_out.iterrows()
                              if r["known_substitute"]),
               "说明": "锚点本身多不在训练集，模型无对应标签"})
    pd.DataFrame(ev).to_csv(os.path.join(args.outdir, "domain_mismatch_evidence.csv"),
                            index=False, encoding="utf-8-sig")

    meta = {
        "n_query": len(q), "n_train": len(tr),
        "n_alt_space": int(alt_space.sum()),
        "n_alt_space_in_train": int((tr_space == 'B: PFAS 替代物化学空间').sum()),
        "n_emerging": len(out),
        "n_tier1": int((tier == "I: 优先关注").sum()),
        "n_eligible": int(eligible.sum()),
        "elapsed_s": round(time.time() - t0, 1),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.outdir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "screen_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    say(f"\n[完成] {meta['elapsed_s']}s → {args.outdir}")


if __name__ == "__main__":
    main()
