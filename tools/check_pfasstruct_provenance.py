# -*- coding: utf-8 -*-
"""
check_pfasstruct_provenance.py — PFASSTRUCT V6 结构"来源与可验证性"核查

回答的问题：
  这 21,028 条结构，是真实数据库记录，还是模型生成/理论枚举的结构？
  其中有多少已被正式登记（CAS）、多少被实验测过（ToxCast）？
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "data", "pfasstruct_v6.csv")

df = pd.read_csv(CSV, dtype=str, low_memory=False)
n = len(df)
print("=" * 62)
print(f"PFASSTRUCT V6 本地记录数: {n}")
print(f"DTXSID 唯一数: {df['dtxsid'].nunique()}")
print(f"SMILES 唯一数(去重后结构): {df['smiles'].nunique()}")
print("=" * 62)

# 1) CAS 登记情况
cas = df["casrn"].fillna("")
has_cas = cas.str.match(r"^\d{2,7}-\d{2}-\d$")
nocas = cas.str.startswith("NOCAS_")
print("\n[1] 身份标识 —— 是否有正式 CAS 登记号")
print(f"    有真实 CAS 号        : {has_cas.sum():>6}  ({has_cas.mean()*100:.1f}%)")
print(f"    NOCAS_ 占位（未登记）: {nocas.sum():>6}  ({nocas.mean()*100:.1f}%)")
print(f"    CAS 列全空           : {(cas == '').sum():>6}")

# 2) 结构标识符
#    百分比取两位小数：InChIKey 覆盖率是 19,763/19,767 = 99.98%，一位小数会显示成
#    "100.0%"，从而把 4 条缺失记录掩盖掉 —— 而那 4 条正是审稿时容易被追问的地方。
for col, label in [("inchikey", "InChIKey"), ("pubchem_cid", "PubChem CID"),
                   ("wikipedia_article", "Wikipedia 条目")]:
    v = df[col].fillna("")
    n_ok = int((v != "").sum())
    print(f"\n[2] 有 {label:<15}: {n_ok:>6}  ({n_ok / len(df) * 100:.2f}%)")
    if 0 < len(df) - n_ok <= 10:
        miss = df.loc[v == "", "dtxsid"].tolist()
        print(f"    缺 {label} 的记录（{len(miss)} 条）: {', '.join(map(str, miss))}")

# 3) 是否被实验测过（ToxCast）
ta = pd.to_numeric(df["total_assays"], errors="coerce")
aa = pd.to_numeric(df["active_assays"], errors="coerce")
print("\n[3] 是否经过实验测定（ToxCast / Tox21 体外测定）")
print(f"    有测定记录 (total_assays 非空): {ta.notna().sum():>6}  ({ta.notna().mean()*100:.1f}%)")
print(f"    做过 >=15 个测定              : {(ta >= 15).sum():>6}")
print(f"    至少一个终点阳性 (active>0)   : {(aa > 0).sum():>6}")
print(f"    从未被测过 (total_assays 空)  : {ta.isna().sum():>6}  ({ta.isna().mean()*100:.1f}%)")

# 4) 化学空间分区（复用筛查脚本的分区口径）
def classify(row):
    head = str(row.get("headgroup", "") or "")
    rings = pd.to_numeric(pd.Series([row.get("n_rings")]), errors="coerce").iloc[0]
    ncf2 = pd.to_numeric(pd.Series([row.get("n_CF2")]), errors="coerce").iloc[0]
    ether = pd.to_numeric(pd.Series([row.get("n_ether_O")]), errors="coerce").iloc[0]
    has_ring = (rings or 0) > 0
    if has_ring:
        return "C 含氟芳香/杂环"
    if pd.notna(ncf2) and ncf2 >= 6 and head in ("PFCAs", "PFSAs", "PFASs"):
        return "A 经典长链 PFAA"
    if pd.notna(ncf2) and ncf2 >= 1:
        return "B 替代物化学空间"
    return "D 其他含氟结构"


df["_zone"] = df.apply(classify, axis=1)
print("\n[4] 化学空间分区（本地口径）")
for z, c in df["_zone"].value_counts().items():
    print(f"    {z:<22}: {c:>6}  ({c/n*100:.1f}%)")

# 5) 每个分区里有多少被实验测过
print("\n[5] 分区 × 是否被测过（交叉）")
ct = df.assign(measured=ta.notna()).groupby("_zone")["measured"].agg(["sum", "count"])
for z, r in ct.iterrows():
    print(f"    {z:<22}: 被测过 {int(r['sum']):>5} / {int(r['count']):>5}  ({r['sum']/r['count']*100:.1f}%)")
print("=" * 62)
