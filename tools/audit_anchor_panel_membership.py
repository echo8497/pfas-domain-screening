#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
锚定物质在生物活性面板中的结构身份与标签覆盖审计
==================================================

背景
----
Table 3 的 "Max similarity to panel" 一列对 PFOS / PFHxS / PFDS 均为 1.00。
该列存在两种截然不同的成因，仅凭数值无法区分，会直接影响适用域分析的解读：

  (1) **真实自匹配**：该锚定物质本身就在面板中（结构完全相同，InChIKey 一致）。
  (2) **指纹简并**：该锚定物质不在面板中，但其二进制 Morgan(radius=2, 2048 bit)
      指纹与某个面板成员**位级完全相同**，故 Tanimoto = 1.00。

成因 (2) 的机制：二进制 Morgan 指纹只记录"子结构环境是否出现"，不记录出现次数。
全氟烷基同系物（如 C6 / C8 / C10 磺酸盐）的**环境集合**相同，仅**计数**不同，
因此位向量相同。计数版 Morgan 或含更长半径/更多位的表示可以区分。

本脚本对全部 20 个锚定物质逐条判定，输出：
  - 是否在面板中（按 InChIKey 精确匹配）
  - 在面板中的标签数（26 个终点中的非空标签数）
  - 是否在 22 终点集 / 12 终点集内拥有标签（即是否真的参与建模）
  - max_sim_to_train = 1.00 的成因归属

输出：results_audit/anchor_panel_membership.csv
      results_audit/anchor_panel_membership.log

用法
----
    python tools/audit_anchor_panel_membership.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)

import pfas_qsar as P          # noqa: E402
import screen_domain as S      # noqa: E402

PANEL = os.path.join(CODE, "data", "pfas_chengng.csv")
REF = os.path.join(CODE, "results_screen_v6", "reference_pfas_predictions.csv")
OUTDIR = os.path.join(CODE, "results_audit")
OUT_CSV = os.path.join(OUTDIR, "anchor_panel_membership.csv")
OUT_LOG = os.path.join(OUTDIR, "anchor_panel_membership.log")

_lines: list[str] = []

# 锚定物质分组（参考表本身没有 group 列，这里按正文定义补齐，避免输出空列）
REGULATED = {"PFOA", "PFNA", "PFDA", "PFHxA", "PFHpA", "PFPeA", "PFBA",
             "PFOS", "PFHxS", "PFBS", "PFDS", "PFOSA", "N-EtFOSAA", "PFHxSA"}
ALTERNATIVE = {"GenX / HFPO-DA", "ADONA", "F-53B (6:2 Cl-PFESA)",
               "6:2 FTS", "4:2 FTS", "PFMOAA"}


def group_of(name: str) -> str:
    if name in REGULATED:
        return "Regulated PFAA"
    if name in ALTERNATIVE:
        return "Marketed alternative"
    return "—"


def say(msg: str = "") -> None:
    print(msg)
    _lines.append(msg)


def main() -> int:
    say("=" * 78)
    say("锚定物质 × 生物活性面板：结构身份与标签覆盖审计")
    say(f"运行时间 {datetime.now():%Y-%m-%d %H:%M:%S}")
    say("=" * 78)

    # ---------------------------------------------------------------- 面板
    train, ep_cols = S.load_train(PANEL)
    ep22 = S.modelable_endpoints(train, ep_cols, 15, min_neg=15)
    ep12 = S.modelable_endpoints(train, ep_cols, 30, min_neg=30)
    say(f"\n面板：{len(train)} 条结构，{len(ep_cols)} 个 PCBA 终点")
    say(f"  22 终点集（min_pos=15）：{len(ep22)}")
    say(f"  12 终点集（min_pos=30）：{len(ep12)}")

    # 面板 InChIKey 索引（从 SMILES 现算，不信任任何缓存列）
    pan_ik: dict[str, int] = {}
    for i, smi in enumerate(train["smiles"]):
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            pan_ik.setdefault(Chem.MolToInchiKey(m), i)
    say(f"  面板唯一 InChIKey：{len(pan_ik)}")

    # 面板指纹（建模同款配置）
    fps_tr = P.fingerprint_bitvects(list(train["smiles"]), radius=2, n_bits=2048)

    # ---------------------------------------------------------------- 锚定物质
    ref = pd.read_csv(REF)
    say(f"\n锚定物质：{len(ref)} 条\n")

    hdr = (f"{'substance':<22}{'InChIKey':<29}{'in_panel':<9}"
           f"{'labels':<8}{'in22':<6}{'in12':<6}{'sim':<7}{'cause'}")
    say(hdr)
    say("-" * len(hdr))

    rows = []
    for _, r in ref.iterrows():
        name = r["ref_name"]
        smi_raw = r["smiles"]
        m = Chem.MolFromSmiles(smi_raw) if isinstance(smi_raw, str) else None
        ik = Chem.MolToInchiKey(m) if m is not None else ""
        std = P.standardize_smiles(smi_raw)

        in_panel = ik in pan_ik
        n_labels = n_lab22 = n_lab12 = 0
        if in_panel:
            row = train.iloc[pan_ik[ik]]
            n_labels = int(row[ep_cols].notna().sum())
            n_lab22 = int(row[ep22].notna().sum())
            n_lab12 = int(row[ep12].notna().sum())

        sim = float(r["max_sim_to_train"])

        # 成因判定
        if sim >= 1.0 and in_panel:
            cause = "true self-match (in panel)"
        elif sim >= 1.0 and not in_panel:
            # 找出位级完全相同的面板成员
            fp = P.fingerprint_bitvects([std], radius=2, n_bits=2048)[0]
            s = np.array(DataStructs.BulkTanimotoSimilarity(fp, fps_tr))
            j = int(np.argmax(s))
            cause = (f"fingerprint degeneracy (identical bits to "
                     f"panel row {j}, Tanimoto {s[j]:.4f})")
        else:
            cause = "—"

        say(f"{name:<22}{ik:<29}{str(in_panel):<9}{n_labels:<8}"
            f"{n_lab22:<6}{n_lab12:<6}{sim:<7.4f}{cause}")

        rows.append({
            "anchor": name,
            "group": group_of(name),
            "inchikey": ik,
            "smiles": smi_raw,
            "in_panel_by_inchikey": in_panel,
            "panel_row_index": int(pan_ik[ik]) if in_panel else -1,
            "labels_nonnull_26ep": n_labels,
            "labels_nonnull_22ep": n_lab22,
            "labels_nonnull_12ep": n_lab12,
            "contributes_to_models": bool(n_lab12 > 0),
            "max_sim_to_train": sim,
            "sim_cause": cause,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    say(f"\n已写出 {os.path.relpath(OUT_CSV, CODE)}")

    # ---------------------------------------------------------------- 指纹简并
    say("\n" + "-" * 78)
    say("指纹分辨率对照：二进制 vs 计数版 Morgan(radius=2, 2048 bit)")
    say("-" * 78)
    demo = {
        "PFBS C4":  "O=S(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "PFHxS C6": "O=S(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "PFOS C8":  "O=S(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "PFDS C10": "O=S(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)"
                    "C(F)(F)C(F)(F)F",
    }
    from rdkit.Chem import rdMolDescriptors  # noqa: E402

    keys = list(demo)
    say(f"{'':<12}" + "".join(f"{k:>12}" for k in keys) + "   (binary / count)")
    for a in keys:
        ma = Chem.MolFromSmiles(demo[a])
        fpa_b = P.fingerprint_bitvects([demo[a]], radius=2, n_bits=2048)[0]
        fpa_c = rdMolDescriptors.GetHashedMorganFingerprint(ma, 2, nBits=2048)
        cells = []
        for b in keys:
            mb = Chem.MolFromSmiles(demo[b])
            fpb_b = P.fingerprint_bitvects([demo[b]], radius=2, n_bits=2048)[0]
            fpb_c = rdMolDescriptors.GetHashedMorganFingerprint(mb, 2, nBits=2048)
            cells.append(f"{DataStructs.TanimotoSimilarity(fpa_b, fpb_b):.3f}/"
                         f"{DataStructs.TanimotoSimilarity(fpa_c, fpb_c):.3f}")
        say(f"{a:<12}" + "".join(f"{c:>12}" for c in cells))

    say("\n结论：二进制 Morgan 位向量对 C6/C8/C10 全氟磺酸盐不可区分"
        "（Tanimoto 恒为 1.000），计数版可区分（0.79/0.83/0.65）。")

    # ------------------------------------------------- 锚定物之间的位级合并
    say("\n" + "-" * 78)
    say("锚定物质之间的位级同一性（二进制 Morgan：同一组内互为 Tanimoto 1.000）")
    say("-" * 78)
    names = list(out["anchor"])
    aps = P.fingerprint_bitvects([P.standardize_smiles(s) for s in out["smiles"]],
                                 radius=2, n_bits=2048)
    seen: set[int] = set()
    n_clusters = 0
    for i in range(len(names)):
        if i in seen:
            continue
        grp = [names[j] for j in range(len(names))
               if DataStructs.TanimotoSimilarity(aps[i], aps[j]) >= 1.0]
        if len(grp) > 1:
            n_clusters += 1
            seen.update(names.index(g) for g in grp)
            say(f"  组 {n_clusters}（n={len(grp)}）：" + "、".join(grp))
    if n_clusters == 0:
        say("  无（所有锚定物质两两可分）")
    say("  说明：这些分组的成员在**整数位向量**层面不可区分，"
        "但它们的经典结构式（含各自链长）并不相同——")
    say("  这正是 Table 3「最大相似度」一列在解释时必须结合成因的原因。")

    # ---------------------------------------------------------------- 汇总
    n_in = int(out["in_panel_by_inchikey"].sum())
    n_self = int(((out["max_sim_to_train"] >= 1.0) & out["in_panel_by_inchikey"]).sum())
    n_degen = int(((out["max_sim_to_train"] >= 1.0) & ~out["in_panel_by_inchikey"]).sum())
    say("\n" + "=" * 78)
    say(f"在面板中（InChIKey 精确匹配）：{n_in} / {len(out)}")
    say(f"max_sim = 1.00 且为真实自匹配：{n_self}")
    say(f"max_sim = 1.00 但为指纹简并　：{n_degen}")
    say("=" * 78)

    with open(OUT_LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print(f"\n已写出 {os.path.relpath(OUT_LOG, CODE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
