# -*- coding: utf-8 -*-
"""
screen_domain.py — 对 PFASSTRUCT V6（21,028 个 PFAS 结构全集）做**适用域内**优先筛查，
生成真正的「新兴替代物」候选清单 —— 而不是训练集自评。

科学逻辑
--------
1. **训练集**：Cheng & Ng 面板（1,012 条有生物活性标签的 PFAS，26 个 PubChem BioAssay 终点）。
   可建模终点 12 个（阳性数 ≥ 30）。主终点 PCBA-686978。
2. **查询集**：EPA CompTox PFASSTRUCT V6 结构全集 21,028 条（只有结构、没有活性标签）。
   —— 这是「无标签的化学空间」，与训练集的「有标签子集」是两个空间。
3. 每个终点训练 RF → 对 21,028 条预测活性概率。
4. **应用域（AD）三重判定**（互补，任一给出一致性证据）：
   a. kNN Tanimoto 距离（k=5）：阈值 = 训练集自身留一 kNN 距离的 95 分位；
   b. 描述符范围包络：各维落在训练集 min–max 之外的比例；
   c. PCA-10 杠杆值（Williams plot），阈值 3(p+1)/n。
5. 剔除与训练集重叠/近重复（标准 SMILES 相同，或最大 Tanimoto ≥ 0.99）。
6. 综合打分 → 优先清单：
   PRI = 0.45·活性 + 0.25·适用域 + 0.20·新兴替代物结构 + 0.10·新颖性

⚠️ 声明：本清单是**优先排序（prioritization）工具**，不是危害/风险评估结论。
   域外化合物的预测不做任何声明。

用法
----
    python screen_domain.py --query data/pfasstruct_v6.csv --min-pos 30 --qmsa \
        --outdir results_screen_v6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import DataStructs

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
sys.path.insert(0, HERE)

import pfas_qsar as P  # noqa: E402
import pfas_anchors as A  # noqa: E402
from rdkit.Chem import Descriptors as _D  # noqa: E402

RNG = 42
AD_PCTL = 95.0        # kNN 距离阈值分位
NEAR_DUP_SIM = 0.99   # 与训练集近重复的相似度门槛
KNN_K = 5
RANGE_TOL = 0.05      # 描述符范围越界比例容忍（>5% 视为域外）
CLASS_MIN_N = 5       # 化学类别覆盖：训练集中同类样本数下限


# ==================================================================== 输入
def load_reference() -> pd.DataFrame:
    """监管语境下的经典 PFAA 与已知替代物，用于演示适用域错配。

    清单唯一真源为 `pfas_anchors.ANCHORS`（20 条 = 14 受监管/前体 PFAA +
    6 已商品化替代物）。早期版本在本函数内硬编码 18 条，且 ADONA / F-53B /
    6:2 FTS / 4:2 FTS 四条结构有误，导致与 screen_alternatives.REFERENCE
    的"18 条 vs 20 条"计数矛盾。现统一为导入，不得再各自维护。
    """
    return A.load_reference()


def load_train(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "smiles"})
    ep_cols = [c for c in df.columns if c != "smiles"]
    df["smiles"] = [P.standardize_smiles(s) for s in df["smiles"]]
    df = df[df["smiles"].notna()].drop_duplicates("smiles").reset_index(drop=True)
    for c in ep_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, ep_cols


def modelable_endpoints(train: pd.DataFrame, ep_cols: list[str], min_pos: int,
                        min_neg: int = 30) -> list[str]:
    out = []
    for c in ep_cols:
        v = train[c].dropna()
        npos = int((v == 1).sum())
        nneg = int((v == 0).sum())
        if npos >= min_pos and nneg >= min_neg:
            out.append(c)
    return out


# ==================================================================== 特征
_MP_POOL = None


def _qmsa_worker(smi: str):
    try:
        import qmsa as Q
        d = Q.qmsa_descriptors(smi)
        return d
    except Exception:
        return None


def qmsa_matrix_parallel(smiles_list, n_jobs: int = 8, chunksize: int = 100) -> pd.DataFrame:
    """QMSA 多进程批量计算（21k 分子单核约 7 min，多核 1–2 min）。"""
    import multiprocessing as mp
    t0 = time.time()
    n = len(smiles_list)
    with mp.Pool(processes=n_jobs) as pool:
        rows = list(pool.imap(_qmsa_worker, smiles_list, chunksize=chunksize))
    import qmsa as Q
    fixed = []
    for smi, r in zip(smiles_list, rows):
        if not isinstance(r, dict):
            r = {k: np.nan for k in Q.QMSA_FEATURES}
            r.update({"smiles": smi, "qmsa_ok": 0, "qmsa_charge_method": ""})
        fixed.append(r)
    df = pd.DataFrame(fixed)
    print(f"    QMSA {n} 分子完成，用时 {time.time() - t0:.0f}s，"
          f"成功率 {df['qmsa_ok'].mean():.1%}")
    return df


def build_features(smiles_all: list[str], use_qmsa: bool, n_jobs: int,
                   cache_dir: str | None = None) -> tuple:
    """返回 (X, 描述符表, 指纹 BitVect 列表, qmeta, 特征名)。
    描述符在「训练+查询」合并集上一次性计算，保证两边列名与顺序完全一致。
    cache_dir 给定时缓存 X / 描述符表 / QMSA（指纹不可序列化，每次重算，约 20s）。"""
    import hashlib
    key = hashlib.md5(("|".join(smiles_all) + f"|qmsa={use_qmsa}").encode()
                      ).hexdigest()[:16]
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        xp = os.path.join(cache_dir, f"featX_{key}.npy")
        dp = os.path.join(cache_dir, f"featdesc_{key}.pkl")
        qp = os.path.join(cache_dir, f"featqmsa_{key}.pkl")
        if os.path.exists(xp) and os.path.exists(dp):
            print(f"  [缓存] 命中 {os.path.basename(xp)}")
            X = np.load(xp)
            desc = pd.read_pickle(dp)
            fps = P.fingerprint_bitvects(smiles_all)
            qmeta = pd.read_pickle(qp) if os.path.exists(qp) else None
            if qmeta is not None and use_qmsa:
                qcols = [c for c in qmeta.columns
                         if c not in ("smiles", "qmsa_ok", "qmsa_charge_method")]
                qm = qmeta[qcols].astype(np.float32).to_numpy()
                qm = np.nan_to_num(qm, nan=0.0)
                names = list(desc.columns) + \
                    [f"morgan_{i}" for i in range(X.shape[1] - desc.shape[1] - qm.shape[1])] \
                    + qcols
            else:
                names = list(desc.columns) + \
                    [f"morgan_{i}" for i in range(X.shape[1] - desc.shape[1])]
            return X, desc, fps, qmeta, np.array(names, dtype=object)

    print("  [特征] RDKit 2D 描述符 + 结构标志 …")
    t0 = time.time()
    desc = P.compute_descriptors(smiles_all)
    desc = desc.drop(columns=[c for c in ("smiles",) if c in desc.columns])
    print(f"         完成 {desc.shape[1]} 维，{time.time() - t0:.0f}s")

    print("  [特征] Morgan 指纹 (r=2, 2048) …")
    fp_mat = P.compute_fingerprints(smiles_all).astype(np.uint8)
    fps = P.fingerprint_bitvects(smiles_all)

    mats = [desc.to_numpy(dtype=np.float32), fp_mat.astype(np.float32)]
    names = list(desc.columns) + [f"morgan_{i}" for i in range(fp_mat.shape[1])]

    qmeta = None
    if use_qmsa:
        print("  [特征] QMSA 分子表面静电势 …")
        qd = qmsa_matrix_parallel(smiles_all, n_jobs=n_jobs)
        qcols = [c for c in qd.columns
                 if c not in ("smiles", "qmsa_ok", "qmsa_charge_method")]
        qcols = [c for c in qcols if pd.api.types.is_numeric_dtype(qd[c])]
        qm = qd[qcols].astype(np.float32).to_numpy()
        qm = np.nan_to_num(qm, nan=0.0)
        mats.append(qm)
        names += qcols
        qmeta = qd[["qmsa_ok", "qmsa_charge_method"]].reset_index(drop=True)

    X = np.hstack(mats)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  [特征] 合计 {X.shape[0]} × {X.shape[1]}")

    if cache_dir:
        np.save(xp, X)
        desc.to_pickle(dp)
        if qmeta is not None:
            qmeta.to_pickle(qp)
        print(f"  [缓存] 已写入 {cache_dir}")
    return X, desc.reset_index(drop=True), fps, qmeta, np.array(names, dtype=object)


# ==================================================================== 相似度
def similarity_block(query_fps: list, train_fps: list, tag: str = "") -> np.ndarray:
    """返回 (n_train, n_query) 的 Tanimoto 相似度矩阵。"""
    n_tr, n_q = len(train_fps), len(query_fps)
    S = np.zeros((n_tr, n_q), dtype=np.float32)
    t0 = time.time()
    for j, q in enumerate(query_fps):
        if q is None:
            continue
        S[:, j] = DataStructs.BulkTanimotoSimilarity(q, train_fps)
        if (j + 1) % 5000 == 0:
            print(f"      {tag} {j + 1}/{n_q}  ({time.time() - t0:.0f}s)")
    print(f"    {tag} 相似度矩阵 {S.shape} 用时 {time.time() - t0:.0f}s")
    return S


def knn_distance_from_sim(sim_col: np.ndarray, k: int = KNN_K) -> float:
    """从一列相似度算出 kNN 距离（1 - 前 k 大相似度的均值）。"""
    if sim_col.size == 0:
        return np.nan
    kk = min(k, sim_col.size)
    top = np.partition(sim_col, -kk)[-kk:]
    return float(1.0 - top.mean())


# ==================================================================== 建模
def train_and_predict(X_tr: np.ndarray, y: np.ndarray, X_q: np.ndarray, seed: int = RNG):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                 n_jobs=-1, random_state=seed)
    clf.fit(X_tr, y)
    return clf.predict_proba(X_q)[:, 1]


def oof_auroc(X: np.ndarray, y: np.ndarray, labels_cluster: np.ndarray,
              n_splits: int = 5, repeats: int = 2, seed: int = RNG) -> float:
    """簇外划分的 out-of-fold AUROC（用于给筛查结果配一个诚实的性能锚点）。"""
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    splits = P.cluster_out_splits(labels_cluster, n_splits=n_splits,
                                  n_repeats=repeats, seed=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in splits:
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     n_jobs=-1, random_state=seed)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        oof[te] = np.where(np.isnan(oof[te]), p, (oof[te] + p) / 2.0)
    m = ~np.isnan(oof)
    return float(roc_auc_score(y[m], oof[m]))


# ==================================================================== 主流程
def main():
    ap = argparse.ArgumentParser(description="PFASSTRUCT V6 域内优先筛查")
    ap.add_argument("--train", default=os.path.join(HERE, "data", "pfas_chengng.csv"))
    ap.add_argument("--query", default=os.path.join(HERE, "data", "pfasstruct_v6.csv"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "results_screen_v6"))
    ap.add_argument("--min-pos", type=int, default=30)
    ap.add_argument("--qmsa", action="store_true", default=True)
    ap.add_argument("--no-qmsa", dest="qmsa", action="store_false")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--ad-pctl", type=float, default=AD_PCTL)
    ap.add_argument("--max-query", type=int, default=0, help=">0 时只取前 N 条做冒烟测试")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()
    log = []

    def say(s):
        print(s)
        log.append(str(s))

    # ---------------------------------------------------------- 1. 数据
    say("=" * 78)
    say("PFASSTRUCT V6 域内优先筛查")
    say("=" * 78)
    train, ep_cols = load_train(args.train)
    query = pd.read_csv(args.query, encoding="utf-8-sig")
    query.columns = [c.strip() for c in query.columns]
    if "smiles" not in query.columns:
        query = query.rename(columns={query.columns[0]: "smiles"})
    query["smiles"] = [P.standardize_smiles(s) for s in query["smiles"]]
    query = query[query["smiles"].notna()].drop_duplicates("smiles").reset_index(drop=True)
    if args.max_query:
        query = query.head(args.max_query).copy()
    # 结构标志一律从 SMILES 现算，保证与训练集口径完全一致（不依赖上游 CSV 列）
    flags = pd.DataFrame([P.structural_flags(s) for s in query["smiles"]])
    for c in flags.columns:
        query[c] = flags[c].to_numpy()
    from rdkit.Chem import Descriptors as _D
    _mw, _arom = [], []
    for s in query["smiles"]:
        m = P.Chem.MolFromSmiles(s)
        _mw.append(float(_D.MolWt(m)) if m is not None else np.nan)
        _arom.append(bool(m is not None and any(a.GetIsAromatic() for a in m.GetAtoms())))
    query["mol_wt"] = _mw
    query["has_aromatic_ring"] = _arom
    q_smiles = query["smiles"].tolist()
    eps = modelable_endpoints(train, ep_cols, args.min_pos)
    say(f"[数据] 训练集 {len(train)} 条 / 终点 {len(ep_cols)} 个 / 可建模终点 {len(eps)} 个")
    say(f"[数据] 查询集 PFASSTRUCT V6 {len(query)} 条")
    say(f"[数据] 可建模终点: {', '.join(eps)}")

    ref = load_reference()
    ref["smiles"] = [P.standardize_smiles(s) for s in ref["smiles"]]

    # ---------------------------------------------------------- 2. 特征
    say("\n[2] 特征计算（训练+查询合并，保证列一致）")
    all_smiles = train["smiles"].tolist() + q_smiles + ref["smiles"].tolist()
    n_tr, n_q, n_rf = len(train), len(q_smiles), len(ref)
    X, desc, fps, qmeta, feat_names = build_features(
        all_smiles, args.qmsa, args.n_jobs, cache_dir=os.path.join(DATA_DIR, "cache"))

    X_tr = X[:n_tr]
    X_q = X[n_tr:n_tr + n_q]
    X_rf = X[n_tr + n_q:]
    fps_tr = fps[:n_tr]
    fps_q = fps[n_tr:n_tr + n_q]
    fps_rf = fps[n_tr + n_q:]

    # ---------------------------------------------------------- 3. 应用域阈值
    say("\n[3] 应用域阈值标定")
    # 3a. 训练集自身留一 kNN 距离分位（排除自身）
    S_self = similarity_block(fps_tr, fps_tr, "train×train")
    np.fill_diagonal(S_self, 0.0)  # 排除自身
    d_self = np.array([knn_distance_from_sim(S_self[:, i]) for i in range(S_self.shape[1])])
    ad_thr = float(np.percentile(d_self, args.ad_pctl))
    say(f"    (a) kNN Tanimoto 距离（k={KNN_K}）阈值 = {ad_thr:.3f} "
        f"（训练集自身 {args.ad_pctl:g} 分位；中位 {np.median(d_self):.3f}）")

    # 3b. 描述符范围包络（用 2D 描述符部分，指纹不适合做范围判定）
    # （阈值在上一步已得；此处仅计算越界比例）

    n_desc_cols = desc.shape[1]
    D_tr = np.nan_to_num(X_tr[:, :n_desc_cols], nan=0.0)
    D_q = np.nan_to_num(X_q[:, :n_desc_cols], nan=0.0)
    lo, hi = D_tr.min(0), D_tr.max(0)
    out_frac = ((D_q < lo) | (D_q > hi)).mean(1)
    in_range = out_frac <= RANGE_TOL
    say(f"    (b) 描述符范围越界比例：中位 {np.median(out_frac):.1%}，"
        f"≤{RANGE_TOL:.0%} 的比例 {in_range.mean():.1%}")

    # 3c'. 稳健标准化：RDKit 里 Ipc/分子量等描述符跨 12+ 个数量级，
    #      不标准化会让 PCA/马氏距离被单个描述符绑架（实测 PC1 量纲达 1e37）。
    #      只影响 3c/3d 的几何判据；(b) 的范围包络仍用原始刻度，保持可解释。
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler(quantile_range=(5, 95)).fit(D_tr)
    D_tr_s = np.clip(scaler.transform(D_tr), -10, 10)
    D_q_s = np.clip(scaler.transform(D_q), -10, 10)
    var_ok = D_tr_s.std(0) > 1e-8
    D_tr_s, D_q_s = D_tr_s[:, var_ok], D_q_s[:, var_ok]
    say(f"    [scale] 稳健标准化后有效描述符 {D_tr_s.shape[1]}/{n_desc_cols} 维")

    # 3c. PCA-10 杠杆值
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10, random_state=RNG).fit(D_tr_s)
    Z_tr, Z_q = pca.transform(D_tr_s), pca.transform(D_q_s)
    lev = P.applicability_domain(Z_tr, Z_q)
    lev_thr = lev["threshold"]
    in_lev = lev["in_domain"]
    say(f"    (c) PCA-10 杠杆值阈值 = {lev_thr:.3f}，域内比例 {in_lev.mean():.1%}")

    # 3d. PCA-10 Mahalanobis 距离 + 卡方阈值（比杠杆值稳健，是描述符空间 AD 的主判据）
    from scipy.stats import chi2
    mu = Z_tr.mean(0)
    cov = np.cov(Z_tr, rowvar=False) + np.eye(Z_tr.shape[1]) * 1e-6
    cov_inv = np.linalg.pinv(cov)

    def _maha(Z):
        d = Z - mu
        return np.einsum("ij,jk,ik->i", d, cov_inv, d)

    m_tr, m_q = _maha(Z_tr), _maha(Z_q)
    chi_thr = float(chi2.ppf(0.99, df=Z_tr.shape[1]))
    in_maha = m_q <= chi_thr
    say(f"    (d) PCA-10 马氏距离卡方阈值(99%,df=10) = {chi_thr:.2f}，"
        f"域内比例 {in_maha.mean():.1%}（训练集自身 {float((m_tr <= chi_thr).mean()):.1%}）")

    # 3e. 化学类别覆盖 —— PFAS 专属 AD 补充判据
    # 理由：PFAS 全空间内 Tanimoto 相似度普遍偏高，指纹距离 AD 会「过度宽容」
    # （实测长链 PFOA/PFOS 都能被判为域内）。因此额外要求查询结构的「化学类别」
    # 在训练集中有足够样本支撑：类别 = 头基 × 芳/脂 × 醚氧 × 全氟碳数分箱。
    def _class_key(cf2, head, arom, ether):
        if cf2 <= 1:
            cb = "C0-1"
        elif cf2 <= 3:
            cb = "C2-3"
        elif cf2 <= 7:
            cb = "C4-7"
        else:
            cb = "C8+"
        return f"h{int(head) if pd.notna(head) else -1}|{'ar' if arom else 'al'}|" \
               f"{'e' if ether else '-'}|{cb}"

    tr_flags = pd.DataFrame([P.structural_flags(s) for s in train["smiles"]])
    arom_tr = np.array([bool(any(a.GetIsAromatic() for a in P.Chem.MolFromSmiles(s).GetAtoms()))
                        for s in train["smiles"]])
    tr_keys = [_class_key(tr_flags["n_CF2"].iloc[i], tr_flags["headgroup"].iloc[i],
                          arom_tr[i], tr_flags["n_ether_O"].iloc[i] > 0)
               for i in range(len(train))]
    tr_key_counts = pd.Series(tr_keys).value_counts().to_dict()

    q_keys = [_class_key(query["n_CF2"].iloc[i], query["headgroup"].iloc[i],
                         query["has_aromatic_ring"].iloc[i], query["n_ether_O"].iloc[i] > 0)
              for i in range(n_q)]
    class_count = np.array([tr_key_counts.get(k, 0) for k in q_keys])
    in_class = class_count >= CLASS_MIN_N
    say(f"    (e) 化学类别覆盖（训练集中同类样本 ≥{CLASS_MIN_N} 条）："
        f"域内比例 {in_class.mean():.1%}；"
        f"训练集覆盖 {len(tr_key_counts)} 个类别，查询集命中 "
        f"{int(pd.Series(q_keys).isin(tr_key_counts).sum())} 条")

    # 一致性判定：描述符空间（马氏）∧ 范围包络 ∧ 类别覆盖
    ad_consensus = in_maha & in_range & in_class
    say(f"    ★ 一致域内（马氏 ∧ 范围 ∧ 类别）：{ad_consensus.mean():.1%}")

    # ---------------------------------------------------------- 4. 相似度与重叠
    say("\n[4] 与训练集的相似度（新颖性/重叠判定）")
    S_tq = similarity_block(fps_q, fps_tr, "V6×train")     # (n_tr, n_q)
    max_sim = S_tq.max(0)
    nn_idx = S_tq.argmax(0)
    say(f"    最大 Tanimoto 相似度：中位 {np.median(max_sim):.3f}，"
        f"≥{NEAR_DUP_SIM} 的 {int((max_sim >= NEAR_DUP_SIM).sum())} 条，"
        f"<0.5 的 {int((max_sim < 0.5).sum())} 条")

    train_smiles = set(train["smiles"])
    exact_dup = np.array([s in train_smiles for s in q_smiles])
    say(f"    与训练集标准 SMILES 完全相同的：{int(exact_dup.sum())} 条")

    # 参考物相似度
    S_rq = similarity_block(fps_rf, fps_tr, "ref×train")   # (n_tr, n_ref)
    ref_max_sim = S_rq.max(0)
    ref_nn = [train["smiles"].iloc[i] for i in S_rq.argmax(0)]

    # ---------------------------------------------------------- 5. 分终点建模与预测
    say(f"\n[5] 逐终点建模（RF，全部有标签样本训练）并预测 {n_q} 条查询结构")
    pred_q = np.full((n_q, len(eps)), np.nan, dtype=np.float32)
    dom_q = np.zeros((n_q, len(eps)), dtype=bool)
    dknn_q = np.full((n_q, len(eps)), np.nan, dtype=np.float32)
    thr_eps: dict[str, float] = {}
    n_lab: dict[str, int] = {}

    for k, ep in enumerate(eps):
        v = train[ep]
        m = v.notna().to_numpy()
        y = v[m].astype(int).to_numpy()
        n_lab[ep] = int(m.sum())
        # 该终点的 AD 阈值（用该终点自身有标签子集的留一分位）
        idx = np.where(m)[0]
        sub = S_self[np.ix_(idx, idx)].copy()
        np.fill_diagonal(sub, 0.0)
        d_sub = np.array([knn_distance_from_sim(sub[:, i]) for i in range(sub.shape[1])])
        thr_ep = max(float(np.percentile(d_sub, args.ad_pctl)), 1e-6)
        thr_eps[ep] = thr_ep

        pred_q[:, k] = train_and_predict(X_tr[m], y, X_q)

        # 该终点的 kNN 距离（从全局 S_tq 里切片）
        Sm = S_tq[idx, :]
        dk = np.array([knn_distance_from_sim(Sm[:, j]) for j in range(n_q)])
        dknn_q[:, k] = dk
        dom_q[:, k] = dk <= thr_ep

        say(f"    {ep:<14s} n={m.sum():4d} 阳性={int(y.sum()):4d} "
            f"AD阈值={thr_ep:.3f} 域内={dom_q[:, k].mean():6.1%} "
            f"预测中位={np.median(pred_q[:, k]):.3f}")

    # 主终点簇外性能锚点
    say("\n[5b] 主终点簇外 AUROC（诚实性能锚点）")
    ep_main = "PCBA-686978" if "PCBA-686978" in eps else eps[0]
    v = train[ep_main]
    m = v.notna().to_numpy()
    y_main = v[m].astype(int).to_numpy()
    fp_tr_main = [fps_tr[i] for i in np.where(m)[0]]
    lab_cl = P.butina_cluster_labels(fp_tr_main, cutoff=0.7)
    auroc_main = oof_auroc(X_tr[m], y_main, lab_cl)
    say(f"    {ep_main}: 簇外 5 折×2 次重复 AUROC = {auroc_main:.3f}  "
        f"（随机划分高估 ≈ +0.10，见第一版结果报告）")

    # ---------------------------------------------------------- 6. 综合打分
    say("\n[6] 综合打分与优先清单")
    p_max = np.nanmax(pred_q, axis=1)
    p_mean = np.nanmean(pred_q, axis=1)
    n_active = (pred_q >= 0.5).sum(1)
    n_dom = dom_q.sum(1)
    dom_frac = n_dom / len(eps)

    # 适用域软得分：四类判据的均值（都归一到 [0,1]）
    #   Tanimoto（逐终点）、马氏距离、范围包络、类别覆盖
    soft_tani = np.nanmean(np.vstack(
        [np.clip(1.0 - dknn_q[:, k] / thr_eps[ep], 0.0, 1.0) for k, ep in enumerate(eps)]),
        axis=0)
    soft_maha = np.clip(1.0 - m_q / chi_thr, 0.0, 1.0)
    soft_range = 1.0 - np.clip(out_frac / (RANGE_TOL * 4), 0.0, 1.0)
    soft_class = np.clip(class_count / CLASS_MIN_N, 0.0, 1.0)
    S_domain = (soft_tani + soft_maha + soft_range + soft_class) / 4.0

    # 新兴替代物结构得分（4 项）
    cf2 = query["n_CF2"].to_numpy(float)
    ether_o = query["n_ether_O"].to_numpy(float)
    rings = query["n_rings"].to_numpy(float)
    head = query["headgroup"].to_numpy(float)
    s_short = np.where(cf2 <= 8, 1.0, 0.0)
    s_ether = np.where(ether_o >= 1, 1.0, 0.0)
    s_head = np.where(np.isin(head, [1, 2, 3]), 1.0, 0.0)
    s_novelstruct = np.where((rings >= 1) | (ether_o >= 1) | (cf2 <= 6), 1.0, 0.0)
    S_emerging = (s_short + s_ether + s_head + s_novelstruct) / 4.0

    S_activity = p_max
    S_novelty = 1.0 - max_sim

    PRI = (0.45 * S_activity + 0.25 * S_domain +
           0.20 * S_emerging + 0.10 * S_novelty)

    out = query.copy()
    out["p_max"] = p_max
    out["p_mean"] = p_mean
    out["n_active_ep"] = n_active
    out["n_domain_ep"] = n_dom
    out["domain_frac"] = dom_frac
    out["S_domain"] = S_domain
    out["S_emerging"] = S_emerging
    out["S_novelty"] = S_novelty
    out["max_sim_to_train"] = max_sim
    out["nearest_train_smiles"] = [train["smiles"].iloc[i] for i in nn_idx]
    out["exact_match_train"] = exact_dup
    out["in_domain_any"] = n_dom > 0
    out["in_domain_majority"] = dom_frac >= 0.5
    out["in_domain_strict"] = dom_frac >= 1.0
    out["near_duplicate"] = max_sim >= NEAR_DUP_SIM
    out["out_of_range_frac"] = out_frac
    out["in_range"] = in_range
    out["maha_dist"] = m_q
    out["in_maha"] = in_maha
    out["class_key"] = q_keys
    out["n_train_same_class"] = class_count
    out["in_class"] = in_class
    out["ad_consensus"] = ad_consensus
    out["soft_tanimoto"] = soft_tani
    out["soft_maha"] = soft_maha
    out["soft_range"] = soft_range
    out["soft_class"] = soft_class
    out["leverage"] = lev["leverage_query"]
    out["in_domain_leverage"] = in_lev
    out["PRI"] = PRI
    for k, ep in enumerate(eps):
        out[f"p_{ep}"] = pred_q[:, k]
        out[f"dom_{ep}"] = dom_q[:, k]
        out[f"dknn_{ep}"] = dknn_q[:, k]

    out = out.sort_values("PRI", ascending=False).reset_index(drop=True)
    out.to_csv(os.path.join(args.outdir, "screen_all.csv"), index=False,
               encoding="utf-8-sig")

    # ---- 阈值由数据分布决定（不写死 0.8 这种拍脑袋的门槛）----
    # 注意：分层必须在排序「之前」完成，否则 tier 与 pca_coords 的顺序会对不上。
    p99 = float(out["p_max"].quantile(0.99))
    p995 = float(out["p_max"].quantile(0.995))
    cons = out[out["ad_consensus"]]
    p_cons99 = float(cons["p_max"].quantile(0.99)) if len(cons) else p99
    say(f"    p_max 分位：99% = {p99:.3f}，99.5% = {p995:.3f}；"
        f"一致域内子集 99% = {p_cons99:.3f}，最大值 = {out['p_max'].max():.3f}")

    # Tier A/B/C 分层（Tier A = 最高优先，三层逐级放宽）
    base = out["ad_consensus"] & (~out["near_duplicate"])
    tier = pd.Series("D: 域外或近重复", index=out.index)
    tier[out["ad_consensus"] & (out["p_max"] >= p99)] = "C: 域内·活性前1%"
    tier[base & (out["p_max"] >= p_cons99) & (out["max_sim_to_train"] < 0.5)] = \
        "B: 域内·结构新颖·活性前1%"
    tier[base & (out["p_max"] >= p_cons99) & (out["max_sim_to_train"] < 0.5)
         & (out["n_active_ep"] >= 2)] = "A: 域内·新颖·多终点一致"
    out["tier"] = tier.to_numpy()
    say("    分层计数：")
    for t, n in pd.Series(out["tier"]).value_counts().sort_index().items():
        say(f"      {t}: {n}")

    cand = out[out["tier"].str.startswith(("A", "B", "C"))].copy()
    for tname, fname in (("A", "priority_tierA.csv"), ("B", "priority_tierB.csv")):
        sub = out[out["tier"].str.startswith(tname)].copy()
        sub.to_csv(os.path.join(args.outdir, fname), index=False, encoding="utf-8-sig")

    # 「新兴替代物」严格档：域内 + 新颖 + 高分 + 结构符合替代物特征
    strict = out[base & (out["max_sim_to_train"] < 0.5) & (out["S_emerging"] >= 0.75)
                 & (out["p_max"] >= p_cons99)].copy()
    say(f"    候选（域内+非近重复，分层 A/B/C）：{len(cand)} 条")
    say(f"    严格档（域内+新颖+替代物结构≥0.75+p≥{p_cons99:.3f}）：{len(strict)} 条")

    # 化学空间坐标（供 PCA 叠加图用；train 与 query 同一次 PCA 变换）
    # Store all ten components that the geometric AD criteria are computed in
    # (leverage and Mahalanobis both use the full PCA-10 space). Additional file
    # 6 is described as "principal-component coordinates of the descriptor space",
    # so a reader has to be able to reproduce those two criteria from the file;
    # storing only the first six would leave them unreproducible.
    pcs = [f"PC{i + 1}" for i in range(Z_q.shape[1])]
    coords = pd.DataFrame(Z_q, columns=pcs)
    coords["split"] = "V6"
    coords["smiles"] = q_smiles
    coords["dtxsid"] = query["dtxsid"].to_numpy() if "dtxsid" in query else ""
    coords["chem_class"] = query["chem_class"].to_numpy() if "chem_class" in query else ""
    coords["p_max"] = p_max
    coords["ad_consensus"] = ad_consensus
    coords["PRI"] = PRI
    coords["tier"] = np.asarray(out["tier"].to_numpy())
    coords_tr = pd.DataFrame(Z_tr, columns=pcs)
    coords_tr["split"] = "train"
    coords_tr["smiles"] = train["smiles"].to_numpy()
    pd.concat([coords_tr, coords], ignore_index=True).to_csv(
        os.path.join(args.outdir, "pca_coords.csv"), index=False, encoding="utf-8-sig")
    del coords_tr

    # 排序后落盘（排序会重置索引，所有掩码必须重算）
    out = out.sort_values("PRI", ascending=False).reset_index(drop=True)
    out.to_csv(os.path.join(args.outdir, "screen_all.csv"), index=False,
               encoding="utf-8-sig")
    cand = out[out["tier"].str.startswith(("A", "B", "C"))].copy()
    cand.to_csv(os.path.join(args.outdir, "priority_candidates.csv"), index=False,
                encoding="utf-8-sig")
    strict_f = (out["ad_consensus"] & (~out["near_duplicate"])
                & (out["max_sim_to_train"] < 0.5) & (out["S_emerging"] >= 0.75)
                & (out["p_max"] >= p_cons99))
    out[strict_f].to_csv(os.path.join(args.outdir, "emerging_alternatives_strict.csv"),
                         index=False, encoding="utf-8-sig")
    out.head(100).to_csv(os.path.join(args.outdir, "priority_top100.csv"), index=False,
                         encoding="utf-8-sig")

    # ---------------------------------------------------------- 7. 参考物落点
    say("\n[7] 经典 PFAA / 已知替代物的落点（适用域错配演示）")
    ref_pred = np.full((n_rf, len(eps)), np.nan, dtype=np.float32)
    for k, ep in enumerate(eps):
        v = train[ep]
        m = v.notna().to_numpy()
        ref_pred[:, k] = train_and_predict(X_tr[m], v[m].astype(int).to_numpy(), X_rf)
    ref_out = pd.DataFrame({
        "ref_name": ref["ref_name"],
        "smiles": ref["smiles"],
        "in_PFASSTRUCT_V6": [s in train_smiles or s in set(q_smiles) for s in ref["smiles"]],
        "max_sim_to_train": ref_max_sim,
        "nearest_train_smiles": ref_nn,
        "p_max": np.nanmax(ref_pred, axis=1),
        "p_mean": np.nanmean(ref_pred, axis=1),
    })
    # 参考物是否域内：用与训练集的 kNN 距离判断
    ref_dk, ref_maha, ref_range, ref_class = [], [], [], []
    Drf_s = np.clip(scaler.transform(
        np.nan_to_num(X_rf[:, :n_desc_cols], nan=0.0)), -10, 10)[:, var_ok]
    Z_rf = pca.transform(Drf_s)
    Drf_raw = np.nan_to_num(X_rf[:, :n_desc_cols], nan=0.0)
    m_rf = _maha(Z_rf)
    of_rf = ((Drf_raw < lo) | (Drf_raw > hi)).mean(1)
    for j in range(n_rf):
        col = np.array(DataStructs.BulkTanimotoSimilarity(fps_rf[j], fps_tr))
        ref_dk.append(knn_distance_from_sim(col))
    for j, s in enumerate(ref["smiles"]):
        m = P.Chem.MolFromSmiles(s)
        fl = P.structural_flags(s)
        arom = bool(m is not None and any(a.GetIsAromatic() for a in m.GetAtoms()))
        ck = _class_key(fl.get("n_CF2", 0), fl.get("headgroup", 0), arom,
                        fl.get("n_ether_O", 0) > 0)
        ref_class.append(tr_key_counts.get(ck, 0))
    ref_maha = m_rf
    ref_range = of_rf
    del Drf_s
    ref_out["dknn_to_train"] = ref_dk
    ref_out["maha_dist"] = ref_maha
    ref_out["in_maha"] = ref_maha <= chi_thr
    ref_out["out_of_range_frac"] = ref_range
    ref_out["in_range"] = ref_range <= RANGE_TOL
    ref_out["n_train_same_class"] = ref_class
    ref_out["in_class"] = np.array(ref_class) >= CLASS_MIN_N
    ref_out["in_domain_tanimoto"] = ref_out["dknn_to_train"] <= ad_thr
    ref_out["ad_consensus"] = (ref_out["in_maha"] & ref_out["in_range"]
                               & ref_out["in_class"])
    ref_out["in_domain"] = ref_out["ad_consensus"]
    ref_out.to_csv(os.path.join(args.outdir, "reference_pfas_predictions.csv"),
                   index=False, encoding="utf-8-sig")
    for _, r in ref_out.iterrows():
        say(f"    {r['ref_name']:<22s} p_max={r['p_max']:.3f} "
            f"kNN={r['dknn_to_train']:.3f} 类别n={int(r['n_train_same_class']):3d} "
            f"一致域内={'Y' if r['ad_consensus'] else 'N'} "
            f"(tanimoto={'Y' if r['in_domain_tanimoto'] else 'N'}/"
            f"maha={'Y' if r['in_maha'] else 'N'}/"
            f"range={'Y' if r['in_range'] else 'N'}/"
            f"class={'Y' if r['in_class'] else 'N'})")
    say(f"    一致域内参考物占比：{ref_out['ad_consensus'].mean():.1%}；"
        f"仅按 Tanimoto 判据则为 {ref_out['in_domain_tanimoto'].mean():.1%}"
        f"（阈值 {ad_thr:.3f}）")

    # ---------------------------------------------------------- 8. 覆盖统计
    cov = pd.DataFrame({
        "endpoint": eps,
        "n_labeled": [n_lab[e] for e in eps],
        "ad_threshold": [thr_eps[e] for e in eps],
        "v6_in_domain_pct": [dom_q[:, k].mean() for k in range(len(eps))],
    })
    cov.to_csv(os.path.join(args.outdir, "domain_coverage.csv"), index=False,
               encoding="utf-8-sig")
    cov_all = {
        "knn_ad_threshold_global": ad_thr,
        "v6_in_domain_tanimoto_majority_pct": float((dom_frac >= 0.5).mean()),
        "v6_in_maha_pct": float(in_maha.mean()),
        "v6_in_range_pct": float(in_range.mean()),
        "v6_in_class_pct": float(in_class.mean()),
        "v6_ad_consensus_pct": float(ad_consensus.mean()),
        "v6_in_domain_leverage_pct": float(in_lev.mean()),
        "train_in_domain_leverage_pct": float((lev["leverage_train"] <= lev_thr).mean()),
        "legacy_reference_consensus_pct": float(ref_out["ad_consensus"].mean()),
        "legacy_reference_tanimoto_pct": float(ref_out["in_domain_tanimoto"].mean()),
    }
    say("\n[8] 适用域覆盖率")
    for k_, v_ in cov_all.items():
        say(f"    {k_}: {v_:.4f}" if isinstance(v_, float) else f"    {k_}: {v_}")

    # 化学类别构成对比（两边都用同一套 SMILES 现算标志；tr_flags/arom_tr 在 3e 已算）
    tr_mw = np.array([_D.MolWt(P.Chem.MolFromSmiles(s)) for s in train["smiles"]])
    comp = pd.DataFrame({
        "化学空间": ["训练集(有标签)", "PFASSTRUCT V6(结构全集)"],
        "n": [n_tr, n_q],
        "含芳环比例": [float(arom_tr.mean()), float(query["has_aromatic_ring"].mean())],
        "中位全氟碳数(n_CF2)": [float(tr_flags["n_CF2"].median()),
                          float(query["n_CF2"].median())],
        "含醚氧比例": [float(tr_flags["n_ether_O"].gt(0).mean()),
                   float(query["n_ether_O"].gt(0).mean())],
        "酸性头基比例": [float(tr_flags["headgroup"].isin([1, 2, 3]).mean()),
                    float(query["headgroup"].isin([1, 2, 3]).mean())],
        "无环直链比例(n_rings==0)": [float(tr_flags["n_rings"].eq(0).mean()),
                               float(query["n_rings"].eq(0).mean())],
        "中位分子量": [float(np.median(tr_mw)), float(query["mol_wt"].median())],
        "中位F/C比": [float(tr_flags["F_per_C"].median()),
                   float(query["F_per_C"].median())],
    })
    comp.to_csv(os.path.join(args.outdir, "class_composition.csv"), index=False,
                encoding="utf-8-sig")
    say("\n[9] 化学空间构成对比")
    say(comp.to_string(index=False))

    # ---------------------------------------------------------- 10. 图
    try:
        make_figures(out, d_self, max_sim, ad_thr, cov, comp, ref_out, args.outdir,
                     n_lab, eps)
        say("\n[10] 图已输出")
    except Exception as e:
        say(f"\n[10] 绘图失败：{str(e)[:200]}")

    meta = {
        "n_train": n_tr, "n_query": n_q, "endpoints": eps,
        "n_features": int(X.shape[1]),
        "qmsa": bool(args.qmsa),
        "ad_threshold": ad_thr,
        "coverage": cov_all,
        "main_endpoint": ep_main,
        "main_clusterout_auroc": auroc_main,
        "n_priority_candidates": int(len(cand)),
        "n_emerging_strict": int(len(strict)),
        "elapsed_s": round(time.time() - t_start, 1),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.outdir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "screen_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    say(f"\n完成，用时 {meta['elapsed_s']}s -> {args.outdir}")


# ==================================================================== 绘图
def make_figures(out, d_self, max_sim, ad_thr, cov, comp, ref_out, outdir, n_lab, eps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # Fig A: kNN 距离分布（V6 vs 训练集自身）
    dq = out[[f"dknn_{e}" for e in eps]].min(axis=1).to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=170)
    bins = np.linspace(0, 1, 60)
    ax.hist(d_self, bins=bins, alpha=.55, label=f"训练集自身 (n={len(d_self)})",
            color="#2c7fb8", density=True)
    ax.hist(dq, bins=bins, alpha=.55, label=f"PFASSTRUCT V6 (n={len(dq)})",
            color="#d95f02", density=True)
    ax.axvline(ad_thr, color="k", ls="--", lw=1.2, label=f"AD 阈值 {ad_thr:.3f}")
    ax.set_xlabel("kNN Tanimoto distance to training set")
    ax.set_ylabel("density")
    ax.set_title("Applicability-domain distance: query universe vs training set", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FigA_ad_distance.png"))
    plt.close(fig)

    # Fig B: 预测活性 vs 新颖性
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=170)
    ind = out["in_domain_majority"].to_numpy()
    ax.scatter(1 - max_sim[~ind], out["p_max"].to_numpy()[~ind], s=5, alpha=.25,
               c="#bdbdbd", label="域外 / 与训练集高度相似")
    ax.scatter(1 - max_sim[ind], out["p_max"].to_numpy()[ind], s=6, alpha=.45,
               c="#d7301f", label="域内（多数终点）")
    ax.axhline(0.5, ls="--", lw=1, c="gray")
    ax.axvline(1 - 0.5, ls=":", lw=1, c="gray")
    ax.set_xlabel("结构新颖性  (1 − max Tanimoto to training set)")
    ax.set_ylabel("预测免疫活性概率  (max over endpoints)")
    ax.set_title("Predicted activity vs structural novelty (PFASSTRUCT V6)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FigB_activity_vs_novelty.png"))
    plt.close(fig)

    # Fig C: 每终点域内覆盖率
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=170)
    c = cov.sort_values("v6_in_domain_pct")
    ax.barh(c["endpoint"], c["v6_in_domain_pct"] * 100, color="#4582b4", height=.62)
    for i, (v, n) in enumerate(zip(c["v6_in_domain_pct"], c["n_labeled"])):
        ax.text(v * 100 + .6, i, f"{v:.1%}  (n={n})", va="center", fontsize=7.5)
    ax.set_xlabel("PFASSTRUCT V6 落在该终点适用域内的比例 (%)")
    ax.set_title("Endpoint-wise applicability-domain coverage of the PFAS structural universe",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FigC_domain_coverage.png"))
    plt.close(fig)

    # Fig D: 类别构成对比
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=170)
    labels = ["含芳环比例", "含醚氧比例", "酸性头基比例"]
    x = np.arange(len(labels))
    w = .36
    tv = [comp.iloc[0]["含芳环比例"], comp.iloc[0]["含醚氧比例"], comp.iloc[0]["酸性头基比例"]]
    qv = [comp.iloc[1]["含芳环比例"], comp.iloc[1]["含醚氧比例"], comp.iloc[1]["酸性头基比例"]]
    ax.bar(x - w / 2, np.array(tv) * 100, w, label="训练集（有标签）", color="#2c7fb8")
    ax.bar(x + w / 2, np.array(qv) * 100, w, label="PFASSTRUCT V6（结构全集）", color="#d95f02")
    for xi, (a, b) in enumerate(zip(tv, qv)):
        ax.text(xi - w / 2, a * 100 + 1.5, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, b * 100 + 1.5, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("比例 (%)")
    ax.set_title("Two PFAS chemical spaces: labeled training set vs structural universe",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FigD_chemical_space.png"))
    plt.close(fig)

    # Fig E: 参考物落点
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
    r = ref_out.sort_values("dknn_to_train")
    cols = ["#d7301f" if v else "#969696" for v in r["in_domain"]]
    ax.barh(r["ref_name"], r["dknn_to_train"], color=cols, height=.62)
    ax.axvline(ad_thr, color="k", ls="--", lw=1.2, label=f"AD 阈值 {ad_thr:.3f}")
    ax.set_xlabel("kNN Tanimoto distance to training set")
    ax.set_title("Where do regulated PFASs fall?  (red = inside model domain)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FigE_reference_pfas_domain.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
