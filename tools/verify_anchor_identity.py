"""
锚点面板身份核验（审稿意见第 2 项）。

对 `pfas_anchors.ANCHORS` 的 20 条锚点逐条核验：
  1. 由 SMILES 计算的分子式是否等于该物质的公认分子式；
  2. InChIKey 是否唯一、是否能在 PFASSTRUCT V6 中定位到同一条结构；
  3. 能否取得该物质的 DSSTox 标识符（DTXSID）与 CAS 登记号。

匹配必须分级，因为 InChIKey 的第一段（连通性块）对游离酸与其盐/衍生物
是相同的：只按连通性块匹配会命中"全氟辛酸钾"之类的记录，从而把盐的
CAS 号误当作母体酸的 CAS 号。因此按严格度分级报告：

  T1 full_inchikey          三段全同 —— 游离酸（或同一电离态）本尊
  T2 neutral_formula       连通性块同 + V6 分子式等于期望分子式
  T3 connectivity_only     仅连通性块同（通常为盐/衍生物记录）
  T4 smiles_exact          标准化 SMILES 全同
  T5 absent                清单中不存在

输出：results_audit/anchor_panel_identity.csv
      results_audit/anchor_identity_report.log
"""

import os
import sys
import json
import time

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)

import pfas_anchors as A                      # noqa: E402
import pfas_qsar as P                          # noqa: E402

OUT = os.path.join(CODE, "results_audit")
os.makedirs(OUT, exist_ok=True)
V6 = os.path.join(CODE, "data", "pfasstruct_v6.csv")

lines = []


def say(s=""):
    print(s)
    lines.append(s)


def main():
    t0 = time.time()
    say("=" * 78)
    say("锚点面板身份核验（20 条 = 14 legacy + 6 alternative）")
    say("=" * 78)

    v6 = pd.read_csv(V6, encoding="utf-8-sig")
    say(f"\n[0] PFASSTRUCT V6：{len(v6)} 条记录，"
        f"{v6['smiles'].nunique()} 个唯一 SMILES")

    v6["ik"] = v6["inchikey"].astype(str).str.strip()
    v6["ik_conn"] = v6["ik"].str.split("-").str[0]
    v6["formula_norm"] = v6["mol_formula"].astype(str).str.strip()
    v6["smiles_std"] = [P.standardize_smiles(s) for s in v6["smiles"]]

    by_full, by_conn, by_smiles = {}, {}, {}
    for i, r in enumerate(v6.itertuples(index=False)):
        by_full.setdefault(r.ik, i)
        by_conn.setdefault(r.ik_conn, []).append(i)
        if r.smiles_std:
            by_smiles.setdefault(r.smiles_std, i)

    # ------------------------------------------------------------- 逐条核验
    rows = []
    for name, group, smi, exp_formula, exp_cas, note in A.ANCHORS:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, name
        formula = rdMolDescriptors.CalcMolFormula(mol)
        ik = Chem.MolToInchiKey(mol)
        conn = ik.split("-")[0]
        smi_std = P.standardize_smiles(smi)

        tier, idx = "T5 absent", None
        if ik in by_full:
            tier, idx = "T1 full_inchikey", by_full[ik]
        elif smi_std and smi_std in by_smiles:
            tier, idx = "T4 smiles_exact", by_smiles[smi_std]
        else:
            cand = by_conn.get(conn, [])
            neutral = [j for j in cand
                       if v6["formula_norm"].iloc[j] == exp_formula]
            if neutral:
                tier, idx = "T2 neutral_formula", neutral[0]
            elif cand:
                tier, idx = "T3 connectivity_only", cand[0]

        rec = {"anchor": name, "group": group, "smiles": smi,
               "formula_computed": formula, "formula_expected": exp_formula,
               "formula_ok": formula == exp_formula,
               "inchikey": ik, "cas_expected": exp_cas,
               "v6_match_tier": tier,
               "v6_dtxsid": "", "v6_casrn": "", "v6_preferred_name": "",
               "v6_mol_formula": "", "v6_inchikey": "",
               "v6_record_is_free_acid": ""}
        if idx is not None:
            r = v6.iloc[idx]
            free = (r["formula_norm"] == exp_formula)
            rec.update({"v6_dtxsid": r["dtxsid"], "v6_casrn": r["casrn"],
                        "v6_preferred_name": r["preferred_name"],
                        "v6_mol_formula": r["formula_norm"],
                        "v6_inchikey": r["ik"],
                        "v6_record_is_free_acid": bool(free)})
        rec["superseded_structure_note"] = A.SUPERSEDED.get(name, "")
        rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "anchor_panel_identity.csv"),
             index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------- 报告
    say("\n[1] 分子式核对（由 SMILES 计算 vs 公认分子式）")
    for _, r in d.iterrows():
        mark = "OK " if r["formula_ok"] else "MISMATCH"
        say(f"    {r['anchor']:<24s} {r['formula_computed']:<14s} {mark}")
    say(f"    → 通过 {int(d['formula_ok'].sum())}/{len(d)}")

    say("\n[2] 在 V6 清单中的定位（分级匹配）")
    for _, r in d.iterrows():
        line = (f"    {r['anchor']:<24s} {r['v6_match_tier']:<20s}")
        if r["v6_dtxsid"]:
            free = "游离酸" if r["v6_record_is_free_acid"] else "盐/衍生物记录"
            line += f" {r['v6_dtxsid']:<18s} CAS {str(r['v6_casrn']):<14s} {free}"
        say(line)
        if r["superseded_structure_note"]:
            say(f"        ↳ 已修正早期版本中的结构：{r['superseded_structure_note']}")

    say("\n[3] 汇总")
    vc = d["v6_match_tier"].value_counts()
    for k, v in vc.items():
        say(f"    {k:<22s} {v}")
    n_free = int((d["v6_record_is_free_acid"] == True).sum())    # noqa: E712
    say(f"    定位到的 V6 记录为游离酸本体 .............. {n_free}/{len(d)}")
    say(f"    CAS 号与 V6 记录 CAS 号一致 .............. "
        f"{int((d['cas_expected'].astype(str) == d['v6_casrn'].astype(str)).sum())}"
        f"/{len(d)}")
    say(f"    分子式核对通过 .......................... "
        f"{int(d['formula_ok'].sum())}/{len(d)}")
    say(f"    在 V6 中可定位（任意分级）................ "
        f"{int((d['v6_match_tier'] != 'T5 absent').sum())}/{len(d)}")

    say("\n[4] 相对早期 18 条面板的差异")
    say(f"    新增锚点：{', '.join(A.ADDED_VS_EARLIER_PANEL)}")
    say(f"    结构已修正：{', '.join(A.SUPERSEDED)}")
    say("    → 早期面板 16/18（88.9%）与 0/18 的计数因此不再成立，"
        "统一面板下的重算结果见 table3_anchor_verdicts.csv")

    with open(os.path.join(OUT, "anchor_identity_report.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(OUT, "anchor_identity_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "n_anchors": len(d),
                   "n_legacy": int((d["group"] == "legacy").sum()),
                   "n_alternative": int((d["group"] == "alternative").sum()),
                   "tier_counts": {k: int(v) for k, v in vc.items()},
                   "elapsed_s": round(time.time() - t0, 1)},
                  f, ensure_ascii=False, indent=2)
    say(f"\n完成，用时 {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
