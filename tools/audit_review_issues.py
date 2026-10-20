# -*- coding: utf-8 -*-
"""
audit_review_issues.py — 回应审稿意见第 2、3 项

第 2 项 锚定化合物数量不一致（Methods 列 20 个 / Results 与 Table 3 用 18 个）
  · 统一为**一份**锚点面板（14 个受监管/前体 PFAA + 6 个已商品化替代物 = 20）
  · 每个锚点给出 RDKit 计算的分子式与 InChIKey，并与 PFASSTRUCT V6 记录比对，
    取回 DTXSID / CASRN / 官方首选名 作为身份核验证据
  · 在统一面板上重算全部 AD 判据 → Table 3 的每一个 n/20 比例

第 3 项 化学类别覆盖判据需验证
  · 最小类别样本量阈值敏感度：T = 2 / 3 / 5 / 10（另加精确 n_CF2 分组版）
  · 在**留出化合物**上检验该判据能否识别更高的预测误差：
      Tanimoto–Butina 簇外 5 折 × 2 次重复，逐折只用训练折标定类别支持度与马氏距离，
      再看留出集上「类别未覆盖」是否伴随更高的 Brier / 绝对误差，
      并与距离型判据（马氏）比较增量价值。

输出：code/results_audit/
用法：python tools/audit_review_issues.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from scipy.stats import chi2, mannwhitneyu, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)

import pfas_qsar as P                       # noqa: E402
import screen_domain as S                   # noqa: E402

OUT = os.path.join(CODE, "results_audit")
os.makedirs(OUT, exist_ok=True)
RNG = 42
CLASS_TS = [2, 3, 5, 10]

# ==================================================================== 锚点面板
# 结构一律用重复单元**程序化构造**，避免手写 CF2 个数出错。
_CF2 = "C(F)(F)"


def _pfca(n):
    """全氟羧酸 Cn：COOH + (n-1) 个 CF 单元。"""
    return "OC(=O)" + _CF2 * (n - 1) + "F"


def _pfsa(n):
    """全氟磺酸 Cn。磺酰基不含碳，故需 n 个 CF 单元。"""
    return "OS(=O)(=O)" + _CF2 * n + "F"


def _samide(n):
    """全氟磺酰胺 Cn。"""
    return "NS(=O)(=O)" + _CF2 * n + "F"


def _fts(m):
    """F(CF2)mCH2CH2SO3H，m = 全氟碳数。"""
    return "OS(=O)(=O)CC" + _CF2 * m + "F"


_ADONA = "OC(=O)C(F)(F)C(OC(F)(F)C(F)(F)C(F)(F)OC(F)(F)F)F"
_F53B = "OS(=O)(=O)" + _CF2 * 2 + "O" + _CF2 * 6 + "Cl"
_GENX = "OC(=O)C(F)(C(F)(F)F)OC(F)(F)C(F)(F)C(F)(F)F"
_PFMOAA = "OC(=O)C(F)(OC(F)(F)F)F"
_NETFOSAA = "CCN(CC(=O)O)S(=O)(=O)" + _CF2 * 8 + "F"

# 14 个受监管 / 前体 PFAA（"legacy"）+ 6 个已商品化替代物（"alternative"）
ANCHORS = [
    # name, group, smiles, expected_formula, CAS RN
    ("PFOA", "legacy", _pfca(8), "C8HF15O2", "335-67-1"),
    ("PFNA", "legacy", _pfca(9), "C9HF17O2", "375-95-1"),
    ("PFDA", "legacy", _pfca(10), "C10HF19O2", "335-76-2"),
    ("PFHxA", "legacy", _pfca(6), "C6HF11O2", "307-24-4"),
    ("PFHpA", "legacy", _pfca(7), "C7HF13O2", "375-85-9"),
    ("PFBA", "legacy", _pfca(4), "C4HF7O2", "375-22-4"),
    ("PFPeA", "legacy", _pfca(5), "C5HF9O2", "2706-90-3"),
    ("PFOS", "legacy", _pfsa(8), "C8HF17O3S", "1763-23-1"),
    ("PFHxS", "legacy", _pfsa(6), "C6HF13O3S", "355-46-4"),
    ("PFBS", "legacy", _pfsa(4), "C4HF9O3S", "375-73-5"),
    ("PFDS", "legacy", _pfsa(10), "C10HF21O3S", "335-77-3"),
    ("PFOSA", "legacy", _samide(8), "C8H2F17NO2S", "754-91-6"),
    ("N-EtFOSAA", "legacy", _NETFOSAA, "C12H8F17NO4S", "2991-50-6"),
    ("PFHxSA", "legacy", _samide(6), "C6H2F13NO2S", "41997-13-1"),
    # 已商品化替代物（结构按官方名称的分子式核对；括号内为旧脚本中被改正的错误结构）
    ("GenX / HFPO-DA", "alternative", _GENX, "C6HF11O3", "13252-13-6"),
    ("ADONA", "alternative", _ADONA, "C7H2F12O4", "919005-14-4"),
    ("F-53B (6:2 Cl-PFESA)", "alternative", _F53B, "C8HClF16O4S", "73606-19-6"),
    ("6:2 FTS", "alternative", _fts(6), "C8H5F13O3S", "27619-97-2"),
    ("4:2 FTS", "alternative", _fts(4), "C6H5F9O3S", "757124-72-4"),
    ("PFMOAA", "alternative", _PFMOAA, "C3HF5O3", "674-13-5"),
]

# 旧脚本中被证实写错的结构（用于在报告中列出「已修正」项）
SUPERSEDED = {
    "ADONA": "O=C(O)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)C(F)(F)C(F)(F)F  (C7HF13O4，多 1 个 F、少 1 个 H)",
    "F-53B (6:2 Cl-PFESA)": "O=S(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)Cl  (漏掉醚氧，实为全氟氯代磺酸)",
    "6:2 FTS": "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CC  (CH2CH2 错接在全氟链末端，实为 C8H6F12O3S)",
    "4:2 FTS": "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CCCC  (碳数 9，非 4:2 FTS)",
}


def _class_key(cf2, head, arom, ether, fine=False):
    """screen_domain._class_key 的复刻；fine=True 时用精确 n_CF2 替代分箱。"""
    if fine:
        cb = f"C{int(cf2)}"
    elif cf2 <= 1:
        cb = "C0-1"
    elif cf2 <= 3:
        cb = "C2-3"
    elif cf2 <= 7:
        cb = "C4-7"
    else:
        cb = "C8+"
    return f"h{int(head) if pd.notna(head) else -1}|{'ar' if arom else 'al'}|" \
           f"{'e' if ether else '-'}|{cb}"


def main():
    t0 = time.time()
    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 80)
    say("审稿意见第 2、3 项核实：锚点面板统一 + 化学类别判据敏感性/留出验证")
    say("=" * 80)

    # ---------------------------------------------------------------- 0. 锚点身份核验
    say("\n[0] 锚点面板身份核验（20 个 = 14 legacy + 6 alternative）")
    v6 = pd.read_csv(os.path.join(CODE, "data", "pfasstruct_v6.csv"), encoding="utf-8-sig")
    # ① 精确结构匹配（与流水线同一口径：standardize_smiles 后逐字相等）
    by_smiles = {}
    for i, s in enumerate(v6["smiles"]):
        st = P.standardize_smiles(s)
        if st and st not in by_smiles:
            by_smiles[st] = i
    # ② 骨架匹配（InChIKey 第一段），用于捕捉同位素标记同系物等
    skel = defaultdict(list)
    for i, k in enumerate(v6["inchikey"]):
        if isinstance(k, str) and len(k) > 14:
            skel[k.split("-")[0]].append(i)

    rows = []
    for name, group, smi, exp_formula, exp_cas in ANCHORS:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, name
        formula = rdMolDescriptors.CalcMolFormula(mol)
        ik = Chem.MolToInchiKey(mol)
        i = by_smiles.get(P.standardize_smiles(smi))
        if i is not None:
            mt = "exact_structure"
            rec = v6.iloc[i]
        else:
            hits = skel.get(ik.split("-")[0], [])
            mt = "skeleton_only" if hits else "absent"
            rec = v6.iloc[hits[0]] if hits else None
        rows.append({
            "anchor": name, "group": group, "smiles": smi,
            "formula_rdkit": formula, "formula_expected": exp_formula,
            "formula_ok": formula == exp_formula,
            "inchikey": ik,
            "cas_rn": exp_cas,
            "v6_match_type": mt,
            "V6_dtxsid": rec["dtxsid"] if rec is not None else "",
            "V6_casrn": rec["casrn"] if rec is not None else "",
            "V6_preferred_name": rec["preferred_name"] if rec is not None else "",
            "V6_mol_formula": rec["mol_formula"] if rec is not None else "",
            "V6_inchikey": rec["inchikey"] if rec is not None else "",
            "superseded_structure_note": SUPERSEDED.get(name, ""),
        })
    anc = pd.DataFrame(rows)
    anc.to_csv(os.path.join(OUT, "anchor_panel_verified.csv"), index=False, encoding="utf-8-sig")
    for _, r in anc.iterrows():
        say(f"    {r['anchor']:<22s} {r['formula_rdkit']:<13s} "
            f"式核={'OK ' if r['formula_ok'] else 'X  '} "
            f"V6={r['v6_match_type']:<15s} CAS={r['cas_rn']}")
        if r["superseded_structure_note"]:
            say(f"        ↳ 已改正旧脚本结构：{r['superseded_structure_note']}")
    say(f"    分子式核对通过 {int(anc['formula_ok'].sum())}/20；"
        f"在 V6 中精确结构匹配 {int((anc['v6_match_type'] == 'exact_structure').sum())}/20，"
        f"仅骨架匹配 {int((anc['v6_match_type'] == 'skeleton_only').sum())}/20，"
        f"完全不在 V6 内 {int((anc['v6_match_type'] == 'absent').sum())}/20")

    # ---------------------------------------------------------------- 1. 特征与标定
    say("\n[1] 特征计算（训练集 1,012 + 锚点 20）")
    train, ep_cols = S.load_train(os.path.join(CODE, "data", "pfas_chengng.csv"))
    eps = S.modelable_endpoints(train, ep_cols, 30)
    say(f"    可建模终点 {len(eps)} 个：{', '.join(eps)}")

    smis = train["smiles"].tolist() + anc["smiles"].tolist()
    n_tr, n_a = len(train), len(anc)
    X, desc, fps, _, _ = S.build_features(
        smis, True, os.cpu_count() or 4, cache_dir=os.path.join(CODE, "data", "cache_audit"))
    X_tr, X_a = X[:n_tr], X[n_tr:]
    fps_tr, fps_a = fps[:n_tr], fps[n_tr:]

    say("\n[2] 应用域阈值标定（与 screen_domain.py 同一协议，仅用训练集）")
    S_self = S.similarity_block(fps_tr, fps_tr, "train×train")
    np.fill_diagonal(S_self, 0.0)
    d_self = np.array([S.knn_distance_from_sim(S_self[:, i]) for i in range(n_tr)])
    ad_thr = float(np.percentile(d_self, 95.0))
    say(f"    (a) kNN Tanimoto 距离阈值 = {ad_thr:.3f}")

    n_desc = desc.shape[1]
    D_tr = np.nan_to_num(X_tr[:, :n_desc], nan=0.0)
    D_a = np.nan_to_num(X_a[:, :n_desc], nan=0.0)
    lo, hi = D_tr.min(0), D_tr.max(0)
    of_a = ((D_a < lo) | (D_a > hi)).mean(1)
    in_range_a = of_a <= 0.05
    say(f"    (b) 描述符范围包络：锚点越界比例中位 {np.median(of_a):.1%}")

    scaler = RobustScaler(quantile_range=(5, 95)).fit(D_tr)
    D_tr_s = np.clip(scaler.transform(D_tr), -10, 10)
    D_a_s = np.clip(scaler.transform(D_a), -10, 10)
    var_ok = D_tr_s.std(0) > 1e-8
    D_tr_s, D_a_s = D_tr_s[:, var_ok], D_a_s[:, var_ok]

    pca = PCA(n_components=10, random_state=RNG).fit(D_tr_s)
    Z_tr, Z_a = pca.transform(D_tr_s), pca.transform(D_a_s)
    mu = Z_tr.mean(0)
    cov = np.cov(Z_tr, rowvar=False) + np.eye(Z_tr.shape[1]) * 1e-6
    cov_inv = np.linalg.pinv(cov)
    maha_a = np.einsum("ij,jk,ik->i", Z_a - mu, cov_inv, Z_a - mu)
    chi_thr = float(chi2.ppf(0.99, df=10))
    in_maha_a = maha_a <= chi_thr
    say(f"    (d) PCA-10 马氏卡方阈值(99%,df=10) = {chi_thr:.2f}")

    # 类别键（训练集）
    tr_flags = pd.DataFrame([P.structural_flags(s) for s in train["smiles"]])
    arom_tr = np.array([bool(any(a.GetIsAromatic()
                                 for a in Chem.MolFromSmiles(s).GetAtoms()))
                        for s in train["smiles"]])
    keys_coarse = [_class_key(tr_flags["n_CF2"].iloc[i], tr_flags["headgroup"].iloc[i],
                              arom_tr[i], tr_flags["n_ether_O"].iloc[i] > 0)
                   for i in range(n_tr)]
    keys_fine = [_class_key(tr_flags["n_CF2"].iloc[i], tr_flags["headgroup"].iloc[i],
                            arom_tr[i], tr_flags["n_ether_O"].iloc[i] > 0, fine=True)
                 for i in range(n_tr)]
    cnt_coarse = Counter(keys_coarse)
    cnt_fine = Counter(keys_fine)
    say(f"    (e) 训练集覆盖类别数：粗分箱 {len(cnt_coarse)} 类，精确 n_CF2 {len(cnt_fine)} 类")

    a_flags = [P.structural_flags(s) for s in anc["smiles"]]
    a_arom = [bool(any(a.GetIsAromatic()
                       for a in Chem.MolFromSmiles(s).GetAtoms()))
              for s in anc["smiles"]]
    a_keys_c = [_class_key(a_flags[i]["n_CF2"], a_flags[i]["headgroup"], a_arom[i],
                           a_flags[i]["n_ether_O"] > 0) for i in range(n_a)]
    a_keys_f = [_class_key(a_flags[i]["n_CF2"], a_flags[i]["headgroup"], a_arom[i],
                           a_flags[i]["n_ether_O"] > 0, fine=True) for i in range(n_a)]

    # 锚点 kNN 距离
    S_ta = S.similarity_block(fps_a, fps_tr, "anchor×train")   # (n_tr, n_a)
    dknn_a = np.array([S.knn_distance_from_sim(S_ta[:, j]) for j in range(n_a)])
    in_tani_a = dknn_a <= ad_thr

    # 逐终点建模 → 锚点预测
    say("\n[3] 逐终点 RF 建模并预测 20 个锚点")
    pred_a = np.full((n_a, len(eps)), np.nan)
    for k, ep in enumerate(eps):
        v = train[ep]
        m = v.notna().to_numpy()
        pred_a[:, k] = S.train_and_predict(X_tr[m], v[m].astype(int).to_numpy(), X_a)
    p_max_a = np.nanmax(pred_a, axis=1)

    # ---------------------------------------------------------------- Table 3 重算
    say("\n[4] Table 3 重算（统一 20 锚点面板）")
    t3 = anc[["anchor", "group", "cas_rn", "V6_dtxsid", "v6_match_type",
              "formula_rdkit", "inchikey"]].copy()
    t3["knn_tanimoto_distance"] = dknn_a
    t3["in_domain_knn_tanimoto"] = in_tani_a
    t3["out_of_range_frac"] = of_a
    t3["in_domain_descriptor_range"] = in_range_a
    t3["maha_distance"] = maha_a
    t3["in_domain_mahalanobis"] = in_maha_a
    t3["n_train_same_class_coarse"] = [cnt_coarse.get(k, 0) for k in a_keys_c]
    t3["n_train_same_class_fine"] = [cnt_fine.get(k, 0) for k in a_keys_f]
    t3["p_max"] = p_max_a
    for T in CLASS_TS:
        t3[f"in_domain_class_T{T}"] = \
            (np.array([cnt_coarse.get(k, 0) for k in a_keys_c]) >= T)
    t3["in_domain_class_fine_T1"] = (np.array([cnt_fine.get(k, 0) for k in a_keys_f]) >= 1)
    for T in CLASS_TS:
        t3[f"consensus_T{T}"] = (in_maha_a & in_range_a &
                                 (np.array([cnt_coarse.get(k, 0) for k in a_keys_c]) >= T))
    # 主报告口径 = T 5
    t3["consensus_domain_T5"] = t3["consensus_T5"]
    t3.to_csv(os.path.join(OUT, "table3_anchor_verdicts.csv"), index=False, encoding="utf-8-sig")

    def rate(mask, label, sub=None):
        m = mask if sub is None else (mask & sub)
        n = int(np.sum(sub)) if sub is not None else len(mask)
        return f"{int(np.sum(m))}/{n} ({np.sum(m)/max(n,1):.1%})"

    sub_leg = (t3["group"] == "legacy").to_numpy()
    sub_alt = (t3["group"] == "alternative").to_numpy()
    say(f"    全部 20 个锚点：")
    say(f"      kNN Tanimoto 域内 ......... {rate(np.asarray(in_tani_a), 'knn')}")
    say(f"      描述符范围域内 ........... {rate(in_range_a, 'range')}")
    say(f"      马氏距离域内 ............. {rate(in_maha_a, 'maha')}")
    for T in CLASS_TS:
        say(f"      类别覆盖域内 (T={T:<2d}) ...... {rate(t3[f'in_domain_class_T{T}'].to_numpy(), 'class')}")
        say(f"      一致域内 (T={T:<2d}) ......... {rate(t3[f'consensus_T{T}'].to_numpy(), 'cons')}")
    say(f"    legacy 14 个：kNN {rate(np.asarray(in_tani_a), '', sub_leg)}，"
        f"一致域 T5 {rate(t3['consensus_T5'].to_numpy(), '', sub_leg)}")
    say(f"    alternative 6 个：kNN {rate(np.asarray(in_tani_a), '', sub_alt)}，"
        f"一致域 T5 {rate(t3['consensus_T5'].to_numpy(), '', sub_alt)}")
    say(f"    旧稿「16/18（88.9%）」与「0/18」在 20 锚点统一面板下被取代；"
        f"差异来源：新增 PFPeA / PFMOAA 两个锚点，且 ADONA、F-53B、6:2 FTS、4:2 FTS 的结构已修正。")

    # ---------------------------------------------------------------- 敏感性：清单覆盖
    # 结构标志一律从 SMILES 现算，与 screen_domain.py 主口径完全一致。
    # ⚠️ 不得读取 pfasstruct_v6.csv 里的 n_CF2 派生列：该列为上游缓存，
    #    与 P.structural_flags() 现算值在 78.7% 的行上不一致（普遍高估），
    #    曾导致本表的 T=5 覆盖率报成 45.63%，与 Table 2 / screen_all.csv 的 53.20% 冲突。
    say("\n[5] 化学类别阈值敏感度（清单侧覆盖率，结构标志从 SMILES 现算）")
    q = v6
    _qflags = pd.DataFrame([P.structural_flags(s) for s in q["smiles"]])
    _cf2 = _qflags["n_CF2"].to_numpy()
    _head = _qflags["headgroup"].to_numpy()
    _arom = np.array([bool(any(a.GetIsAromatic()
                               for a in Chem.MolFromSmiles(s).GetAtoms()))
                      for s in q["smiles"]])
    _eth = (_qflags["n_ether_O"] > 0).to_numpy()
    q_cc = [_class_key(_cf2[i], _head[i], _arom[i], _eth[i]) for i in range(len(q))]
    q_cf = [_class_key(_cf2[i], _head[i], _arom[i], _eth[i], fine=True)
            for i in range(len(q))]
    tr_cc = [_class_key(tr_flags["n_CF2"].iloc[i], tr_flags["headgroup"].iloc[i],
                        arom_tr[i], tr_flags["n_ether_O"].iloc[i] > 0) for i in range(n_tr)]
    tr_cf = [_class_key(tr_flags["n_CF2"].iloc[i], tr_flags["headgroup"].iloc[i],
                        arom_tr[i], tr_flags["n_ether_O"].iloc[i] > 0, fine=True)
             for i in range(n_tr)]
    ct_c, ct_f = Counter(tr_cc), Counter(tr_cf)
    sens = []
    for T in CLASS_TS:
        for tag, keys, cts in (("coarse", q_cc, ct_c), ("fine_nCF2", q_cf, ct_f)):
            cov = np.array([cts.get(k, 0) >= T for k in keys])
            sens.append({"scheme": tag, "min_occupancy_T": T,
                         "inventory_covered": int(cov.sum()),
                         "inventory_pct": float(cov.mean()),
                         "train_categories_used": int(sum(1 for v in cts.values() if v >= T))})
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(os.path.join(OUT, "class_threshold_inventory_sensitivity.csv"),
                   index=False, encoding="utf-8-sig")
    for _, r in sens_df.iterrows():
        say(f"    {r['scheme']:<11s} T={r['min_occupancy_T']:<2d}  "
            f"清单域内 {r['inventory_covered']:6d} ({r['inventory_pct']:5.1%})  "
            f"可用类别 {r['train_categories_used']:3d}")

    # ---------------------------------------------------------------- 留出误差验证
    say("\n[6] 留出化合物检验：类别覆盖能否识别更高的预测误差")
    say("    协议：Tanimoto–Butina 簇外 5 折 × 2 次重复；每折仅用训练折标定")
    say("          类别支持度 / RobustScaler+PCA10 / 马氏卡方阈值，再评估留出集。")
    recs = []
    ep0 = "PCBA-686978"
    for ep in eps:
        v = train[ep]
        m = v.notna().to_numpy()
        idx = np.where(m)[0]
        y = v[m].astype(int).to_numpy()
        Xe = X_tr[m]
        keys = [keys_coarse[i] for i in idx]
        keysf = [keys_fine[i] for i in idx]
        fps_e = [fps_tr[i] for i in idx]
        labels = P.butina_cluster_labels(fps_e, cutoff=0.7)
        splits = P.cluster_out_splits(labels, n_splits=5, n_repeats=2, seed=RNG)
        p_acc = defaultdict(list)
        cc_acc = defaultdict(list)
        cf_acc = defaultdict(list)
        mh_acc = defaultdict(list)
        for tr_i, te_i in splits:
            clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                         n_jobs=-1, random_state=RNG)
            clf.fit(Xe[tr_i], y[tr_i])
            p = clf.predict_proba(Xe[te_i])[:, 1]
            ctc = Counter([keys[i] for i in tr_i])
            ctf = Counter([keysf[i] for i in tr_i])
            Df = np.nan_to_num(Xe[tr_i][:, :n_desc], nan=0.0)
            sc = RobustScaler(quantile_range=(5, 95)).fit(Df)
            vo = np.clip(sc.transform(Df), -10, 10).std(0) > 1e-8
            A = np.clip(sc.transform(Df), -10, 10)[:, vo]
            pc = PCA(n_components=10, random_state=RNG).fit(A)
            Ztr = pc.transform(A)
            mu_f = Ztr.mean(0)
            cv_f = np.cov(Ztr, rowvar=False) + np.eye(10) * 1e-6
            ci_f = np.linalg.pinv(cv_f)
            Bm = np.nan_to_num(Xe[te_i][:, :n_desc], nan=0.0)
            Bs = np.clip(sc.transform(Bm), -10, 10)[:, vo]
            Zte = pc.transform(Bs)
            d = Zte - mu_f
            mh = np.einsum("ij,jk,ik->i", d, ci_f, d)
            for j, gi in enumerate(te_i):
                p_acc[gi].append(p[j])
                cc_acc[gi].append(ctc.get(keys[gi], 0))
                cf_acc[gi].append(ctf.get(keysf[gi], 0))
                mh_acc[gi].append(mh[j])
        gi_sorted = sorted(p_acc)
        Pv = np.array([np.mean(p_acc[i]) for i in gi_sorted])
        yv = np.array([y[i] for i in gi_sorted])
        CC = np.array([np.mean(cc_acc[i]) for i in gi_sorted])
        CF = np.array([np.mean(cf_acc[i]) for i in gi_sorted])
        MH = np.array([np.mean(mh_acc[i]) for i in gi_sorted])
        brier = (Pv - yv) ** 2
        absr = np.abs(Pv - yv)
        hi_err = brier >= np.quantile(brier, 0.75)     # 该终点内误差最大的 25%
        for T in CLASS_TS:
            inc = CC >= T
            n_in, n_out = int(inc.sum()), int((~inc).sum())
            rec = {"endpoint": ep, "min_occupancy_T": T,
                   "n_holdout": len(Pv), "n_class_covered": n_in, "n_class_uncovered": n_out}
            if n_in >= 5 and n_out >= 5:
                rec["brier_covered"] = float(brier[inc].mean())
                rec["brier_uncovered"] = float(brier[~inc].mean())
                rec["abs_err_covered"] = float(absr[inc].mean())
                rec["abs_err_uncovered"] = float(absr[~inc].mean())
                # 单侧检验：未覆盖组的误差更大
                u = mannwhitneyu(brier[~inc], brier[inc], alternative="greater")
                rec["p_uncovered_error_higher"] = float(u.pvalue)
                # 识别「高误差」的能力：AUC(高误差 ~ 未覆盖)
                rec["auc_detect_higherror_class"] = float(
                    roc_auc_score(hi_err, (~inc).astype(float))) if 0 < hi_err.sum() < len(hi_err) else np.nan
                rec["auc_detect_higherror_maha"] = float(roc_auc_score(hi_err, MH))
                rec["auc_detect_higherror_combined"] = float(
                    roc_auc_score(hi_err, ((~inc) | (MH > chi_thr)).astype(float)))
            else:
                for kk in ("brier_covered", "brier_uncovered", "abs_err_covered",
                           "abs_err_uncovered"):
                    rec[kk] = np.nan
                rec["p_uncovered_error_higher"] = np.nan
                rec["auc_detect_higherror_class"] = np.nan
                rec["auc_detect_higherror_maha"] = np.nan
                rec["auc_detect_higherror_combined"] = np.nan
            rh = spearmanr(np.log1p(CC), brier)
            rec["spearman_classcount_vs_brier"] = float(rh.statistic)
            rec["spearman_p"] = float(rh.pvalue)
            recs.append(rec)
        # 与旧报告口径对照：簇外 AUROC
        try:
            a = roc_auc_score(yv, Pv)
        except ValueError:
            a = np.nan
        for r in recs:
            if r["endpoint"] == ep:
                r["oof_auroc_clusterout"] = float(a)
        r5 = [r for r in recs if r["endpoint"] == ep and r["min_occupancy_T"] == 5][0]
        say(f"    {ep:<14s} 留出 AUROC={a:.3f}  "
            f"Brier 未覆盖−已覆盖 (T=5) = {r5['brier_uncovered'] - r5['brier_covered']:+.4f}")

    ev = pd.DataFrame(recs)
    ev.to_csv(os.path.join(OUT, "class_error_validation_per_endpoint.csv"),
              index=False, encoding="utf-8-sig")

    say("\n[7] 汇总（12 终点等权平均）")
    agg = []
    for T in CLASS_TS:
        s = ev[ev["min_occupancy_T"] == T]
        agg.append({
            "min_occupancy_T": T,
            "mean_brier_covered": float(s["brier_covered"].mean()),
            "mean_brier_uncovered": float(s["brier_uncovered"].mean()),
            "delta_brier": float((s["brier_uncovered"] - s["brier_covered"]).mean()),
            "n_endpoints_p_lt_005": int((s["p_uncovered_error_higher"] < 0.05).sum()),
            "mean_auc_detect_higherror_class": float(s["auc_detect_higherror_class"].mean()),
            "mean_auc_detect_higherror_maha": float(s["auc_detect_higherror_maha"].mean()),
            "mean_auc_detect_higherror_combined": float(s["auc_detect_higherror_combined"].mean()),
            "mean_spearman_classcount_vs_brier": float(s["spearman_classcount_vs_brier"].mean()),
        })
    aggd = pd.DataFrame(agg)
    aggd.to_csv(os.path.join(OUT, "class_error_validation_summary.csv"),
                index=False, encoding="utf-8-sig")
    for _, r in aggd.iterrows():
        say(f"    T={int(r['min_occupancy_T']):<2d}  Brier 覆盖 {r['mean_brier_covered']:.4f} vs "
            f"未覆盖 {r['mean_brier_uncovered']:.4f}  (Δ={r['delta_brier']:+.4f})  "
            f"12 终点中 p<0.05 的 {int(r['n_endpoints_p_lt_005'])} 个")
        say(f"          识别高误差 AUC：类别 {r['mean_auc_detect_higherror_class']:.3f} | "
            f"马氏 {r['mean_auc_detect_higherror_maha']:.3f} | "
            f"合并 {r['mean_auc_detect_higherror_combined']:.3f}")

    # ---------------------------------------------------------------- 输出
    with open(os.path.join(OUT, "review_issues_audit.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_anchors": int(len(anc)),
            "n_endpoints": len(eps),
            "class_thresholds": CLASS_TS,
            "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    say(f"\n完成，用时 {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
