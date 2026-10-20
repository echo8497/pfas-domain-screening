# -*- coding: utf-8 -*-
"""
check_tier1_provenance.py — 核查"新兴替代物清单"里每一条的"身份证据强度"

清单本身来自 EPA CompTox PFASSTRUCTV6，本脚本回答：
  这些候选物里，有多少有正式 CAS 登记、有多少被实验测过？
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(HERE, "results_alternatives")


def summarise(fname, label):
    path = os.path.join(RD, fname)
    if not os.path.exists(path):
        print(f"[跳过] 未找到 {fname}")
        return
    df = pd.read_csv(path, dtype=str, low_memory=False)
    n = len(df)
    cas = df["casrn"].fillna("")
    has_cas = cas.str.match(r"^\d{2,7}-\d{2}-\d$")
    ta = pd.to_numeric(df["total_assays"], errors="coerce")
    aa = pd.to_numeric(df["active_assays"], errors="coerce")
    sel = df["toxcast_select"].fillna("") != ""
    print(f"\n--- {label}  (n = {n}) ---")
    print(f"  有正式 CAS 登记号        : {has_cas.sum():>5}  ({has_cas.mean()*100:.1f}%)")
    print(f"  无 CAS（NOCAS_ 占位）    : {(~has_cas).sum():>5}  ({(~has_cas).mean()*100:.1f}%)")
    print(f"  有 ToxCast 体外测定记录  : {ta.notna().sum():>5}  ({ta.notna().mean()*100:.1f}%)")
    print(f"  至少一个终点被测出阳性   : {(aa > 0).sum():>5}")
    print(f"  被 ToxCast 计划选中      : {sel.sum():>5}")


summarise("emerging_tier1.csv", "Tier I 优先关注")
summarise("emerging_regrettable.csv", "遗憾替代")
summarise("emerging_alternatives.csv", "全部候选（Tier I+II+III）")
