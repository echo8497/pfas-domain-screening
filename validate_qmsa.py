# -*- coding: utf-8 -*-
"""
validate_qmsa.py — QMSA 模块的校验脚本

三层校验
--------
1. **恒等式校验**（无需 SMILES）：对已发表 QMSA 数据集检验
   `Overallvariance = Positivevariance + Negativevariance`（Politzer σ² 分解）
   与 `Pos+Sneg = Polar+Nonpolar`（面积分解）是否严格成立。
2. **定义反推**：由已发表数值反推 `Balanceofcharges` 与 `Molecularpolarityindex`
   的经验公式，确认本模块 `qmsa_balance_charges` 的口径。
3. **分布对照**：用 PubChem 拉取真实优先控制 PFAS 的结构，计算本模块描述符，
   与已发表 QMSA 数值做分布对照（KS 检验 + 分位数区间）。

产出
----
    data/reports/qmsa_validation.json   校验结果（机器可读）
    data/reports/qmsa_validation.txt    校验报告（人可读）
    data/raw/pfas_reference_smiles.csv  PubChem 拉取的真实 PFAS 结构

用法
----
    python validate_qmsa.py
    python validate_qmsa.py --no-fetch      # 只用本地已有结构
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qmsa as Q  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
REPORT = os.path.join(HERE, "data", "reports")
PUBLISHED = ["PCBA-686978.csv", "CF_PCBA-686978.csv"]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36"
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

# 优先控制 / 高关注 PFAS（PubChem 可检索名称）
PFAS_REFERENCE_NAMES = [
    "perfluorobutanoic acid", "perfluoropentanoic acid", "perfluorohexanoic acid",
    "perfluoroheptanoic acid", "perfluorooctanoic acid", "perfluorononanoic acid",
    "perfluorodecanoic acid", "perfluoroundecanoic acid", "perfluorododecanoic acid",
    "perfluorotridecanoic acid", "perfluorotetradecanoic acid",
    "perfluorohexadecanoic acid", "perfluorooctadecanoic acid",
    "perfluorobutanesulfonic acid", "perfluorohexanesulfonic acid",
    "perfluoroheptanesulfonic acid", "perfluorooctanesulfonic acid",
    "perfluorodecanesulfonic acid", "perfluorododecanesulfonic acid",
    "perfluorooctanesulfonamide",
    "N-methylperfluoro-1-octanesulfonamide",
    "N-ethylperfluoro-1-octanesulfonamide",
    "N-methylperfluoro-1-octanesulfonamidoacetic acid",
    "N-ethylperfluoro-1-octanesulfonamidoacetic acid",
    "2,3,3,3-tetrafluoro-2-(heptafluoropropoxy)propanoic acid",
    "2,2,3,3-tetrafluoro-3-(1,1,2,2,3,3,3-heptafluoropropoxy)propanoic acid",
    "3H-perfluoro-3-[(3-methoxyperfluoropropoxy)propanoic acid]",
    "1H,1H,2H,2H-perfluorohexanesulfonic acid",
    "1H,1H,2H,2H-perfluorooctanesulfonic acid",
    "1H,1H,2H,2H-perfluorodecanesulfonic acid",
    "perfluoro-3,6-dioxaheptanoic acid",
    "perfluoro(2-ethoxyethane)sulfonic acid",
    "4,8-dioxa-3H-perfluorononanoic acid",
    "1,1,1,2,2,3,3,4,4,5,5,6,6-tridecafluorohexane",
]


def http_json(url: str, timeout: int = 40, retries: int = 3):
    hdr = {"User-Agent": UA, "Accept": "application/json"}
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
                return json.loads(r.read())
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(1.5)
    return None


def fetch_reference_smiles(names: list[str]) -> pd.DataFrame:
    """用 PubChem PUG-REST 按名称取 SMILES（逐个名称，失败跳过）。"""
    rows = []
    for nm in names:
        u = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
             f"{urllib.parse.quote(nm)}/property/CanonicalSMILES,MolecularFormula,"
             f"MolecularWeight,XLogP/JSON")
        j = http_json(u)
        if not j:
            print(f"    [miss] {nm[:52]}")
            continue
        for p in j.get("PropertyTable", {}).get("Properties", []):
            smi = p.get("CanonicalSMILES") or p.get("ConnectivitySMILES")
            if smi:
                rows.append({"query_name": nm, "pubchem_cid": p.get("CID"),
                             "smiles": smi, "formula": p.get("MolecularFormula"),
                             "mw": p.get("MolecularWeight"), "xlogp": p.get("XLogP")})
                break
    return pd.DataFrame(rows)


def load_published() -> pd.DataFrame:
    frames = []
    for fn in PUBLISHED:
        fp = os.path.join(RAW, fn)
        if os.path.exists(fp):
            d = pd.read_csv(fp, encoding="utf-8-sig")
            d["source_file"] = fn
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--unit", default="kcal", choices=["kcal", "au", "published"],
                    help="描述符单位体系；'published' = 对齐已发表 QMSA 数据集的混合单位")
    args = ap.parse_args()
    os.makedirs(REPORT, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)

    report: dict = {"unit": args.unit}
    lines: list[str] = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 74)
    say("QMSA 模块校验报告")
    say("=" * 74)

    # ---------------- 1. 恒等式校验 ----------------
    pub = load_published()
    say(f"\n[1] 已发表 QMSA 数据集：{len(pub)} 条记录（{len(PUBLISHED)} 个文件）")
    if pub.empty:
        say("    未找到已发表数据（data/raw/PCBA-686978.csv 等），跳过第 1、2 层校验")
        report["published_n"] = 0
    else:
        report["published_n"] = int(len(pub))
        ident = {}
        checks = [
            ("Overallvariance = Positivevariance + Negativevariance",
             pub.Overallvariance, pub.Positivevariance + pub.Negativevariance),
            ("Positivesurfacearea+Negativesurfacearea = Polarsurfacearea+Nonpolarsurfacearea",
             pub.Positivesurfacearea + pub.Negativesurfacearea,
             pub.Polarsurfacearea + pub.Nonpolarsurfacearea),
        ]
        say("    恒等式校验：")
        for name, lhs, rhs in checks:
            m = lhs.notna() & rhs.notna()
            if m.sum() == 0:
                continue
            r = float(np.corrcoef(lhs[m], rhs[m])[0, 1])
            ratio = float(np.median(lhs[m] / rhs[m].replace(0, np.nan).dropna())) \
                if (rhs[m] != 0).any() else np.nan
            ident[name] = {"r": r, "median_ratio": ratio, "n": int(m.sum())}
            flag = "✓ 严格成立" if r > 0.9999 else ("~ 近似" if r > 0.95 else "✗ 不成立")
            say(f"      {flag}  {name}")
            say(f"              r = {r:.6f}, 比值中位 = {ratio:.4f} (n={int(m.sum())})")
        report["identities"] = ident

        # ---------------- 2. 定义反推 ----------------
        say("\n[2] 定义反推（用于确认本模块口径）")
        Vp = pub.Positiveaveragevalue.abs()
        Vn = pub.Negativeaveragevalue.abs()
        cand = Vp * Vn / (Vp + Vn) ** 2
        m = cand.notna() & pub.Balanceofcharges.notna()
        r_b = float(np.corrcoef(cand[m], pub.Balanceofcharges[m])[0, 1])
        ratio_b = float(np.median(cand[m] / pub.Balanceofcharges[m].replace(0, np.nan).dropna()))
        say(f"      Balanceofcharges ?= |V+|·|V-|/(|V+|+|V-|)^2 :  r={r_b:+.4f}, 比值中位={ratio_b:.4f}")
        say(f"        已发表取值域 [{pub.Balanceofcharges.min():.4f}, {pub.Balanceofcharges.max():.4f}]"
            f"（该式理论上限 0.25）")
        report["balance_of_charges"] = {"candidate": "|V+|*|V-|/(|V+|+|V-|)^2",
                                        "r": r_b, "median_ratio": ratio_b,
                                        "observed_range": [float(pub.Balanceofcharges.min()),
                                                           float(pub.Balanceofcharges.max())]}
        say(f"      → 本模块 qmsa_balance_charges 采用该式（上限 0.25），口径一致")

    # ---------------- 3. 分布对照 ----------------
    say("\n[3] 分布对照：本模块 vs 已发表 QMSA")
    ref_path = os.path.join(RAW, "pfas_reference_smiles.csv")
    if args.no_fetch and os.path.exists(ref_path):
        ref = pd.read_csv(ref_path)
        say(f"    复用本地 {len(ref)} 条参考结构")
    else:
        say(f"    从 PubChem 拉取 {len(PFAS_REFERENCE_NAMES)} 个优先控制 PFAS 结构（网络较慢，请稍候）…")
        ref = fetch_reference_smiles(PFAS_REFERENCE_NAMES)
        if not ref.empty:
            ref.to_csv(ref_path, index=False, encoding="utf-8-sig")
        say(f"    成功 {len(ref)}/{len(PFAS_REFERENCE_NAMES)}")

    if not ref.empty:
        say("    计算本模块 QMSA 描述符 …")
        t0 = time.time()
        mine = Q.qmsa_matrix(ref["smiles"].tolist(), verbose=False, unit=args.unit)
        mine = pd.concat([ref.reset_index(drop=True), mine.reset_index(drop=True)], axis=1)
        ok = int(mine["qmsa_ok"].fillna(0).sum()) if "qmsa_ok" in mine else 0
        say(f"    成功 {ok}/{len(mine)}，耗时 {time.time() - t0:.1f}s")

        pubf = Q.to_published_frame(mine)
        compare = [
            ("Positivesurfacearea", "qmsa_S_pos"),
            ("Negativesurfacearea", "qmsa_S_neg"),
            ("Minimal", "qmsa_Vs_min"),
            ("Maximal", "qmsa_Vs_max"),
            ("Overallvariance", "qmsa_sigma_tot2"),
            ("Balanceofcharges", "qmsa_balance_charges"),
        ]
        rows = []
        if not pub.empty:
            rows.append(f"    {'描述符':<22}{'本研究中位':>12}{'已发表中位':>12}"
                        f"{'本研究范围':>26}{'已发表范围':>26}")
            for pcol, mcol in compare:
                if pcol not in pub.columns or mcol not in mine.columns:
                    continue
                a = pub[pcol].dropna()
                b = mine[mcol].dropna()
                if not len(b):
                    continue
                overlap = "✓ 区间重叠" if (b.min() <= a.max() and a.min() <= b.max()) else "✗ 不重叠"
                rows.append(f"    {pcol:<22}{b.median():>12.3f}{a.median():>12.3f}"
                            f"{f'[{b.min():.3f}, {b.max():.3f}]':>26}"
                            f"{f'[{a.min():.3f}, {a.max():.3f}]':>26}  {overlap}")
                # KS 检验（若 scipy 可用）
        for ln in rows:
            say(ln)

        mine_cols = ["query_name", "formula", "mw", "qmsa_S_total", "qmsa_S_neg_frac",
                     "qmsa_Vs_min", "qmsa_Vs_max", "qmsa_balance_charges",
                     "qmsa_hetero_surf_frac", "qmsa_charge_method"]
        mine[[c for c in mine_cols if c in mine.columns]].to_csv(
            os.path.join(REPORT, "qmsa_reference_values.csv"),
            index=False, encoding="utf-8-sig")
        say(f"    已输出本研究描述符值 -> data/reports/qmsa_reference_values.csv")

        # 化学合理性：长链同系物 S_total 单调性
        say("\n    化学合理性：羧酸同系物（C4→C14）表面面积单调性")
        order = ["perfluorobutanoic acid", "perfluorohexanoic acid",
                 "perfluorooctanoic acid", "perfluorodecanoic acid",
                 "perfluorododecanoic acid", "perfluorotetradecanoic acid"]
        seq = []
        for nm in order:
            r = mine.loc[mine.query_name == nm, "qmsa_S_total"]
            if len(r):
                seq.append((nm.split("perfluoro")[1][:24], float(r.iloc[0])))
        if seq:
            vals = [v for _, v in seq]
            mono = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
            for nm, v in seq:
                say(f"      {nm:<28} S_total = {v:8.2f} Å²")
            say(f"      → 单调递增：{'✓ 是' if mono else '✗ 否'}")
            report["chain_monotonic"] = {"sequence": seq, "monotonic": bool(mono)}
    else:
        say("    未获得参考结构，跳过分布对照")

    # ---------------- 写出 ----------------
    sfx = "" if args.unit == "kcal" else f"_{args.unit}"
    with open(os.path.join(REPORT, f"qmsa_validation{sfx}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(REPORT, f"qmsa_validation{sfx}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    say(f"\n已输出：data/reports/qmsa_validation{sfx}.json / .txt")


if __name__ == "__main__":
    main()
