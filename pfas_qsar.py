# -*- coding: utf-8 -*-
"""
pfas_qsar.py — PFAS 机器学习/QSAR 核心函数库

覆盖：结构标准化 → 描述符/指纹 → Tanimoto-Butina 簇外划分 → 建模 → 验证 → 应用域
依赖：numpy pandas scikit-learn rdkit matplotlib（xgboost/lightgbm/shap 可选）
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

RNG_SEED = 42


# ---------------------------------------------------------------- 结构处理
def standardize_smiles(smi: str):
    """去盐、中和电荷、规范化；返回 canonical SMILES 或 None。"""
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def murcko_scaffold(smi: str) -> str:
    """Murcko 骨架。PFAS 多为无环直链 → 绝大多数返回空字符串（这正是骨架划分失效的原因）。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


# ---------------------------------------------------------------- 描述符
_RDKIT_DESC = {name: fn for name, fn in Descriptors.descList}


def structural_flags(smi: str) -> dict:
    """PFAS 类别结构标志：全氟链长、醚氧数、环数、是否含非氟片段。

    注意 n_CF2（全氟碳数）的口径。早期实现用
    `len(mol.GetSubstructMatches(Smarts("[CX4](F)(F)")))`，RDKit 对同一碳原子
    会枚举所有 F 的两两组合，因此 CF3 端基被计 3 次、CF2 被计 1 次，
    使全氟碳数系统性偏大（实测 19,767 条中仅 4,214 条与真值一致，
    均值偏大 2.3，多 CF3 结构最多偏大 12）。这会直接改变化学空间分区
    （A 区定义"≥6 个全氟碳"、B 区"≥2 个全氟碳"）与类别覆盖判据的分档，
    故改为逐原子判定：全氟碳 = 该碳上至少 2 个取代基为 F。
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    # 全氟碳数：碳原子上至少 2 个 F 取代（CF2/CF3/CF2- 均计 1 次）
    n_cf = sum(
        1 for a in mol.GetAtoms()
        if a.GetSymbol() == "C"
        and sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "F") >= 2
    )
    ether_o = mol.GetSubstructMatches(Chem.MolFromSmarts("[OD2]([#6])[#6]"))
    rings = rdMolDescriptors.CalcNumRings(mol)
    n_c = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    n_f = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "F")
    head = 0
    for smarts, tag in (("C(=O)O", 1), ("S(=O)(=O)O", 2), ("S(=O)(=O)N", 3)):
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            head = tag
            break
    return {
        "n_CF2": n_cf,                # 全氟碳数（逐原子判定，CF3 不计 3 次）
        "n_ether_O": len(ether_o),    # 醚氧数 → 醚基替代物特征
        "n_rings": rings,
        "n_C": n_c,
        "n_F": n_f,
        "F_per_C": n_f / n_c if n_c else np.nan,
        "headgroup": head,            # 1=羧酸 2=磺酸 3=磺酰胺 0=其他
    }


def compute_descriptors(smiles: list[str], verbose: bool = False) -> pd.DataFrame:
    """计算 RDKit 2D 描述符 + 结构标志。目标物少时可改接 Mordred（1800+ 描述符）。"""
    rows = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        rec = {"smiles": smi}
        if mol is None:
            rows.append(rec)
            continue
        for name, fn in _RDKIT_DESC.items():
            try:
                rec[name] = fn(mol)
            except Exception:
                rec[name] = np.nan
        rec.update(structural_flags(smi))
        rows.append(rec)
        if verbose and (i + 1) % 500 == 0:
            print(f"  descriptors: {i + 1}/{len(smiles)}")
    df = pd.DataFrame(rows)
    return df.dropna(axis=1, how="all")


def compute_fingerprints(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Morgan/ECFP 指纹矩阵。注意：二元指纹会把化学上不同的长链 PFAS 压成相同向量，
    因此指纹只能作基线，不能作为唯一特征（见 J Chem Inf Model 2026）。"""
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.uint8))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.vstack(fps)


def fingerprint_bitvects(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> list:
    """返回 RDKit BitVect 列表，用于相似度计算。"""
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        out.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                   if mol is not None else None)
    return out


# ---------------------------------------------------------------- 划分策略
def butina_cluster_labels(fps: list, cutoff: float = 0.6) -> np.ndarray:
    """Tanimoto 距离 + Butina 聚类。cutoff 为距离阈值（1 - 相似度）。"""
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    labels = np.full(n, -1, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            labels[m] = cid
    return labels


def cluster_out_splits(labels: np.ndarray, n_splits: int = 5, n_repeats: int = 5, seed: int = RNG_SEED):
    """Tanimoto-Butina 簇外交叉验证：整簇进训练或进测试，杜绝相似物泄漏。"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(labels)
    sizes = {c: int((labels == c).sum()) for c in uniq}
    idx = np.arange(len(labels))
    splits = []
    for _ in range(n_repeats):
        order = list(rng.permutation(uniq))
        order.sort(key=lambda c: -sizes[c])          # 大簇优先，贪心平衡
        fold_size = np.zeros(n_splits)
        assign = {}
        for c in order:
            f = int(np.argmin(fold_size))
            assign[c] = f
            fold_size[f] += sizes[c]
        for f in range(n_splits):
            te = np.array([i for i in idx if assign[labels[i]] == f])
            tr = np.array([i for i in idx if assign[labels[i]] != f])
            if len(te) and len(tr):
                splits.append((tr, te))
    return splits


def group_splits(groups: np.ndarray, n_splits: int = 5, seed: int = RNG_SEED):
    """通用分组划分（骨架划分、来源划分等）。空串骨架会退化为平凡折——这正是问题所在。"""
    return cluster_out_splits(np.asarray(groups), n_splits=n_splits, n_repeats=1, seed=seed)


def random_splits(y: np.ndarray, n_splits: int = 5, seed: int = RNG_SEED):
    """随机分层划分（对照用，会高估性能）。"""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y)]


# ---------------------------------------------------------------- 建模与评价
def get_models(seed: int = RNG_SEED) -> dict:
    """层次化模型库：可解释基线 → 树模型 → 高级模型（可选）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV

    models = {
        "LR": Pipeline([("sc", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                   random_state=seed))]),
        "RF": RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                     n_jobs=-1, random_state=seed),
        # SVC(probability=True) 在 sklearn>=1.9 已弃用，改用显式校准包装
        "SVM": Pipeline([("sc", StandardScaler()),
                         ("clf", CalibratedClassifierCV(
                             SVC(class_weight="balanced", random_state=seed),
                             ensemble=False, cv=3))]),
    }
    try:
        from xgboost import XGBClassifier
        models["XGB"] = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                                      subsample=0.8, colsample_bytree=0.8,
                                      eval_metric="logloss", random_state=seed, n_jobs=-1)
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LGBM"] = LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                        class_weight="balanced", verbose=-1,
                                        random_state=seed, n_jobs=-1)
    except Exception:
        pass
    return models


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    """类别不平衡场景的标准指标集：AUROC + AUPRC + Brier + 校准。"""
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                 matthews_corrcoef, brier_score_loss, confusion_matrix)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)
    out = {"n": len(y_true), "n_pos": int(y_true.sum()), "pos_rate": float(y_true.mean())}
    for key, fn in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score),
                    ("Brier", brier_score_loss)):
        try:
            out[key] = float(fn(y_true, y_prob))
        except Exception:
            out[key] = np.nan
    try:
        out["F1"] = float(f1_score(y_true, y_pred, zero_division=0))
        out["MCC"] = float(matthews_corrcoef(y_true, y_pred))
        tn, fp, fn_, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out["Sens"] = tp / (tp + fn_) if (tp + fn_) else np.nan
        out["Spec"] = tn / (tn + fp) if (tn + fp) else np.nan
    except Exception:
        out.update({"F1": np.nan, "MCC": np.nan, "Sens": np.nan, "Spec": np.nan})
    return out


def cv_predict(model, X: np.ndarray, y: np.ndarray, splits: list) -> np.ndarray:
    """按给定划分做交叉验证预测，返回每个样本的 out-of-fold 概率。"""
    oof = np.full(len(y), np.nan, dtype=float)
    for tr, te in splits:
        m = model
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof[te] = np.where(np.isnan(oof[te]), p, (oof[te] + p) / 2.0)
    return oof


def nested_cv_summary(models: dict, X: np.ndarray, y: np.ndarray, splits: list) -> pd.DataFrame:
    """多模型 × 指定划分策略的性能汇总表。无有效折时返回空表（由调用方解释原因）。"""
    cols = ["n", "n_pos", "pos_rate", "AUROC", "AUPRC", "Brier", "F1", "MCC", "Sens", "Spec"]
    if not splits:
        return pd.DataFrame(columns=cols)
    rows = []
    for name, model in models.items():
        oof = cv_predict(model, X, y, splits)
        mask = ~np.isnan(oof)
        if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
            continue
        met = evaluate(y[mask], oof[mask])
        met["model"] = name
        rows.append(met)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).set_index("model").sort_values("AUROC", ascending=False)


def y_randomization(model, X: np.ndarray, y: np.ndarray, splits: list,
                    n_perm: int = 100, seed: int = RNG_SEED) -> dict:
    """标签置换检验：确认真实模型显著优于随机。返回真实 AUROC 与置换分布。

    返回值同时给出**置换次数的完整分解**（n_perm / n_perm_valid / n_ge_real），
    使 p 值可由产物本身核对，而不必从 p 的数值反推分母：
        p = (n_ge_real + 1) / (n_perm_valid + 1)
    """
    rng = np.random.default_rng(seed)
    oof_real = cv_predict(model, X, y, splits)
    mask_real = ~np.isnan(oof_real)
    real = evaluate(y[mask_real], oof_real[mask_real])["AUROC"]
    perm = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        oof = cv_predict(model, X, yp, splits)
        mask = ~np.isnan(oof)
        if mask.sum():
            perm.append(evaluate(yp[mask], oof[mask])["AUROC"])
    perm = np.array(perm, dtype=float)
    n_valid = int(len(perm))
    n_ge = int(np.nansum(perm >= real))
    return {"real_auroc": real, "perm_mean": float(np.nanmean(perm)),
            "perm_p95": float(np.nanpercentile(perm, 95)),
            "n_perm_requested": int(n_perm), "n_perm_valid": n_valid,
            "n_ge_real": n_ge,
            # 完整的置换分布本身一并返回：图 4 面板 d 的空分布必须画真实置换值，
            # 不能用正态分布合成；同时便于第三方逐个核对 n_ge_real。
            "perm_aurocs": perm.tolist(),
            "p_value": float((n_ge + 1) / (n_valid + 1)),
            "p_value_formula": "(n_ge_real + 1) / (n_perm_valid + 1)"}


# ---------------------------------------------------------------- 应用域
def applicability_domain(X_train: np.ndarray, X_query: np.ndarray,
                         y_train: np.ndarray = None, hat_threshold: float = None) -> dict:
    """Williams plot 杠杆值法：h > 3(p+1)/n 视为域外。返回杠杆值与布尔掩码。"""
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    n, p = X_train.shape
    XtX_inv = np.linalg.pinv(X_train.T @ X_train)
    h_train = np.einsum("ij,jk,ik->i", X_train, XtX_inv, X_train)
    h_query = np.einsum("ij,jk,ik->i", X_query, XtX_inv, X_query)
    thr = hat_threshold if hat_threshold is not None else 3.0 * (p + 1) / n
    return {"leverage_train": h_train, "leverage_query": h_query,
            "threshold": thr, "in_domain": h_query <= thr}


def knn_ad_distance(fp_train: list, fp_query: list, k: int = 5,
                    exclude_self: bool = False) -> np.ndarray:
    """kNN 距离法：查询化合物到训练集最近 k 个的平均 Tanimoto 距离（越大越域外）。
    exclude_self=True 时剔除相似度=1 的项 —— 自筛（query 与 train 同一集合）时必须开启，
    否则会把分子自身当作最近邻，得到一个恒为 0 的虚假距离。"""
    out = []
    for q in fp_query:
        if q is None:
            out.append(np.nan)
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(q, fp_train), dtype=float)
        order = np.argsort(sims)[::-1]
        if exclude_self:
            order = order[sims[order] < 1.0 - 1e-12]
        order = order[:k]
        if len(order) == 0:
            out.append(np.nan)
            continue
        out.append(1.0 - float(np.mean(sims[order])))
    return np.array(out)


# ---------------------------------------------------------------- 绘图
def plot_split_comparison(results: dict, out_png: str):
    """Fig.2 核心图：不同划分策略下的性能对比（随机会明显高估）。
    自动跳过无有效折的策略（如 PFAS 的骨架划分退化）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = {k: v for k, v in results.items() if v is not None and len(v) > 0}
    skipped = [k for k in results if k not in usable]
    if not usable:
        return False

    models = sorted({m for v in usable.values() for m in v.index})
    x = np.arange(len(models))
    w = 0.8 / max(len(usable), 1)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    for i, (s, tab) in enumerate(usable.items()):
        vals = [tab.loc[m, "AUROC"] if m in tab.index else np.nan for m in models]
        ax.bar(x + i * w, vals, w, label=s)
    ax.set_xticks(x + w * (len(usable) - 1) / 2)
    ax.set_xticklabels(models)
    ax.set_ylabel("AUROC (out-of-fold)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, ls="--", lw=1, c="gray")
    ax.legend(frameon=False, fontsize=8)
    title = "Leakage-aware validation: random vs cluster-out"
    if skipped:
        title += f"  ({', '.join(skipped)}: degenerate, no valid folds)"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def plot_ablation(ab: pd.DataFrame, out_png: str):
    """特征消融图：不同特征组合的簇外 AUROC 对比，直观展示 QMSA 的增量贡献。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ab is None or len(ab) == 0:
        return False
    d = ab.sort_values("AUROC", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(d) + 1.6), dpi=160)
    colors = ["#c0392b" if "QMSA" in s else "#7f8c8d" for s in d["feature_set"]]
    bars = ax.barh(d["feature_set"], d["AUROC"], color=colors, height=0.62)
    for b, (v, nf) in zip(bars, zip(d["AUROC"], d["n_features"])):
        ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.3f}  ({nf} dims)",
                va="center", fontsize=8)
    ax.set_xlabel("AUROC (out-of-fold, cluster-out)")
    ax.set_xlim(0.4, max(1.0, float(d["AUROC"].max()) + 0.09))
    ax.axvline(0.5, ls="--", lw=1, c="gray")
    ax.set_title("QMSA ablation: surface-ESP descriptors vs 2D/fingerprint features", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def plot_sigma_profile(smiles_list, labels, out_png: str, bins: int = 60):
    """σ-profile 图：代表性 PFAS 的分子表面静电势面积加权分布。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import qmsa as Q

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    ok = False
    for smi, lab in zip(smiles_list, labels):
        g, h = Q.esp_profile(smi, bins=bins)
        if g is None:
            continue
        ax.plot(g, h, lw=1.8, label=lab)
        ok = True
    if not ok:
        plt.close(fig)
        return False
    ax.axvline(0, ls="--", lw=1, c="gray")
    ax.set_xlabel(r"Molecular surface ESP  $\sigma$  (kcal$\cdot$mol$^{-1}\cdot$e$^{-1}$)")
    ax.set_ylabel(r"Area-weighted $\sigma$-profile  ($\AA^2$)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(r"Surface ESP ($\sigma$) profiles of representative PFASs", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, out_png: str):
    """校准曲线：不平衡数据下必须报。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=160)
    try:
        frac, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", label="model")
    except Exception:
        pass
    ax.plot([0, 1], [0, 1], "--", c="gray", label="ideal")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
