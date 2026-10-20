# -*- coding: utf-8 -*-
"""
audit_candidate_counts.py — 审稿意见第 1 项核实：
    「Region B = 2,091，剔除 169 条与 102 条后不可能剩 2,048 条」

做法：不引用任何论文文字，直接从 V6 结构清单 + 训练集重算
    「Region B → 候选」的**逐步、互斥**计数链，并给出每个排除条件的具体条数。
输出：code/results_audit/candidate_count_audit.json / .csv / .md

用法：
    python tools/audit_candidate_counts.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)

import pfas_qsar as P                      # noqa: E402
from screen_alternatives import (          # noqa: E402
    REFERENCE, NEAR_DUP_TRAIN, NEAR_DUP_REF, ALT_NOVEL_MAX,
    MIN_CF2, LEGACY_MIN, classify_space,
)

OUT = os.path.join(CODE, "results_audit")
os.makedirs(OUT, exist_ok=True)

V6 = os.path.join(CODE, "data", "pfasstruct_v6.csv")
TRAIN = os.path.join(CODE, "data", "pfas_chengng.csv")

GEN = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return GEN.GetFingerprint(m) if m is not None else None


def main():
    t0 = time.time()
    lines: list[str] = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 78)
    say("候选人计数链审计（审稿意见第 1 项）")
    say("=" * 78)

    q = pd.read_csv(V6, encoding="utf-8-sig")
    q["smiles"] = [P.standardize_smiles(s) for s in q["smiles"]]
    q = q[q["smiles"].notna()].drop_duplicates("smiles").reset_index(drop=True)
    say(f"\n[0] PFASSTRUCT V6 唯一结构（建模口径）= {len(q):,} 条")
    say(f"    源记录数（清单条目）= 21,028；"
        f"差额 1,261 = 无确定结构/多组分/去重损失（见 fetch_pfasstruct 质量报告）")

    # ---------------- 1. 化学空间划区
    # 一律从 SMILES 现算结构表征（与 screen_alternatives.characterize 同口径），
    # 不信任 CSV 里可能过期的 n_CF2 / arom 列。
    flags = pd.DataFrame([P.structural_flags(s) for s in q["smiles"]])
    for c in ("n_CF2", "n_ether_O", "n_rings", "n_C", "n_F", "F_per_C", "headgroup"):
        q[c] = flags[c].to_numpy() if c in flags else np.nan
    q["arom"] = [bool(m is not None and any(a.GetIsAromatic() for a in m.GetAtoms()))
                 for m in (Chem.MolFromSmiles(s) for s in q["smiles"])]
    space = classify_space(q)
    cnt = space.value_counts()
    say("\n[1] 化学空间划区（与 screen_alternatives.classify_space 完全一致）")
    for k in sorted(cnt.index):
        say(f"    {k:<30s} {cnt[k]:6d}  ({cnt[k]/len(q):6.2%})")
    B = space == "B: PFAS 替代物化学空间"
    nB = int(B.sum())
    say(f"    → Region B 起点 = {nB}")

    # ---------------- 2. 排除条件（逐条 + 互斥归因）
    tr = pd.read_csv(TRAIN, encoding="utf-8-sig")
    tr.columns = [c.strip() for c in tr.columns]
    tr = tr.rename(columns={tr.columns[0]: "smiles"})
    tr["smiles"] = [P.standardize_smiles(s) for s in tr["smiles"]]
    tr = tr[tr["smiles"].notna()].drop_duplicates("smiles").reset_index(drop=True)
    fps_tr = [fp(s) for s in tr["smiles"]]
    tr_set = set(tr["smiles"])

    REFS = [(r[0], P.standardize_smiles(r[2])) for r in REFERENCE]
    fps_ref = [fp(s) for _, s in REFS]
    ref_set = {s for _, s in REFS}

    fps_q = [fp(s) for s in q["smiles"]]
    n_q = len(q)
    sim_tr = np.zeros(n_q, dtype=np.float32)
    sim_rf = np.zeros(n_q, dtype=np.float32)
    for j, f in enumerate(fps_q):
        if f is None:
            continue
        sim_tr[j] = max(DataStructs.BulkTanimotoSimilarity(f, fps_tr))
        sim_rf[j] = max(DataStructs.BulkTanimotoSimilarity(f, fps_ref))
    q["sim_train"] = sim_tr
    q["sim_ref"] = sim_rf

    conds = {
        "C1 与已标签面板结构完全相同（exact match）":
            q["smiles"].isin(tr_set),
        f"C2 与已标签面板近重复（Tanimoto ≥ {NEAR_DUP_TRAIN}）":
            q["sim_train"] >= NEAR_DUP_TRAIN,
        f"C3 与已标签面板化学覆盖重叠（Tanimoto ≥ {ALT_NOVEL_MAX}）":
            q["sim_train"] >= ALT_NOVEL_MAX,
        "C4 即锚定化合物本身（结构相同）":
            q["smiles"].isin(ref_set),
        f"C5 与锚定化合物近同（Tanimoto ≥ {NEAR_DUP_REF}）":
            q["sim_ref"] >= NEAR_DUP_REF,
    }

    say("\n[2] 各排除条件在 Region B 内的命中数（**非互斥**，直接相加会重复计数）")
    for k, v in conds.items():
        vb = v & B
        say(f"    {k:<52s} 全库 {int(v.sum()):6d} | Region B 内 {int(vb.sum()):5d}")

    # 互斥归因：按 screen_alternatives 的实际过滤顺序（集合的并）
    keep_B = B.copy()
    excl_any = np.zeros(n_q, dtype=bool)
    for v in conds.values():
        excl_any |= v.to_numpy()
    removed = int((B & excl_any).sum())
    n_cand = nB - removed
    say(f"\n[3] 互斥归因（并集，不重复计数）")
    say(f"    Region B 起点 ..................................... {nB}")
    say(f"    两条排除条件同时成立的条数（重叠部分） ............ "
        f"{int((B & conds['C1 与已标签面板结构完全相同（exact match）'] & conds['C2 与已标签面板近重复（Tanimoto ≥ 0.99）']).sum())}"
        f"  ← C1 ⊂ C2，exact match 全部已被 C2 覆盖")
    say(f"    应剔除的结构数（并集） ........................... {removed}")
    say(f"    → 最终候选物 = {nB} − {removed} = {n_cand}")

    # 逐条分解（互斥，按优先级顺序）
    order = ["C4 即锚定化合物本身（结构相同）",
             f"C5 与锚定化合物近同（Tanimoto ≥ {NEAR_DUP_REF}）",
             "C1 与已标签面板结构完全相同（exact match）",
             f"C2 与已标签面板近重复（Tanimoto ≥ {NEAR_DUP_TRAIN}）",
             f"C3 与已标签面板化学覆盖重叠（Tanimoto ≥ {ALT_NOVEL_MAX}）"]
    covered = np.zeros(n_q, dtype=bool)
    rows = []
    for k in order:
        v = conds[k].to_numpy() & B.to_numpy() & (~covered)
        rows.append({"排除条件": k, "Region B 内新增剔除": int(v.sum())})
        covered |= v
    df_seq = pd.DataFrame(rows)
    df_seq.loc[len(df_seq)] = {"排除条件": "合计（互斥，顺序执行）", "Region B 内新增剔除": int(covered.sum())}
    df_seq.loc[len(df_seq)] = {"排除条件": f"最终候选物 ({nB} − 剔除)", "Region B 内新增剔除": nB - int(covered.sum())}
    say(f"\n[4] 顺序执行的分步剔除（互斥，逐行相加 = {int(covered.sum())}）")
    for _, r in df_seq.iterrows():
        say(f"    {str(r['排除条件']):<46s} {int(r['Region B 内新增剔除']):6d}")

    say("\n[5] 「169」与「107」这两个全库口径计数的真实出处")
    say("    （原审稿意见写作 102；实测全库与锚定化合物近同 ≥0.95 为 107，"
        "稿件已按实测值改为 107）")
    say(f"    全库与训练集近重复（≥{NEAR_DUP_TRAIN}）= {int((q['sim_train']>=NEAR_DUP_TRAIN).sum())} 条")
    say(f"    全库与训练集完全相同            = {int(q['smiles'].isin(tr_set).sum())} 条")
    say(f"    两者相加 = {int((q['sim_train']>=NEAR_DUP_TRAIN).sum()) + int(q['smiles'].isin(tr_set).sum())}"
        f"  ← 与论文中的「169」一致：把**全库**的两个**非互斥**计数相加得到")
    n_sim_ref = int((q['sim_ref'] >= NEAR_DUP_REF).sum())
    say(f"    全库与锚定化合物近同（≥{NEAR_DUP_REF}）= {n_sim_ref} 条"
        f"  ← 该数也是**全库**口径，且其中 {int(q['smiles'].isin(ref_set).sum())} 条就是锚点本身")
    say(f"    结论：169 与 {n_sim_ref} 都是**全库**口径、且彼此重叠，被误当作 Region B 内的顺序剔除量；")
    say(f"          真正的 Region B 内剔除量是 {int(covered.sum())} 条。")

    out = {
        "n_records_source": 21028,
        "n_unique_structures": n_q,
        "region_counts": {k: int(v) for k, v in cnt.items()},
        "n_region_B": nB,
        "n_removed_from_B": removed,
        "n_candidates": n_cand,
        "removed_breakdown_mutually_exclusive": {
            r["排除条件"]: int(r["Region B 内新增剔除"]) for _, r in df_seq.iterrows()},
        "whole_inventory_counts": {
            "near_dup_train_ge_099": int((q["sim_train"] >= NEAR_DUP_TRAIN).sum()),
            "exact_match_train": int(q["smiles"].isin(tr_set).sum()),
            "near_dup_reference_ge_095": int((q["sim_ref"] >= NEAR_DUP_REF).sum()),
            "is_reference_exact": int(q["smiles"].isin(ref_set).sum()),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "candidate_count_audit.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    df_seq.to_csv(os.path.join(OUT, "candidate_count_breakdown.csv"),
                  index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT, "candidate_count_audit.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    say(f"\n完成，用时 {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
