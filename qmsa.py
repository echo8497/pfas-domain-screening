# -*- coding: utf-8 -*-
"""
qmsa.py — 定量分子表面分析 (Quantitative Molecular Surface Analysis, QMSA) 描述符

设计目标
--------
在 **不依赖 pyscf/半经验量化** 的前提下，用"部分电荷 + 表面采样"重建分子表面静电势
(ESP) 场，并导出与 *Water Research* 2025 一类工作可比拟的表面静电势描述符块。

科学定位
--------
PFAS 是"头基极性 + 全氟烷基链疏水"的两性分子；其界面行为（吸附、膜分配、与受体
结合）主要由**分子表面**的静电势分布决定，而非体相性质。2D 描述符只给"分子有多少
极性"，QMSA 给"极性长在表面的什么位置、占多大面积、有多强"——这正是长链 PFAS
同系物之间 2D 描述符无法区分的部分。

描述符分组
----------
A. 表面几何   : S_total / Rg / 形状各向异性 / 球度
B. ESP 面积谱 : 正/负/近中性 ESP 表面面积及其占比
C. ESP 强度谱 : 平均/极值/范围、正负分侧均值
D. ESP 离散度 : σ_tot²、σ_±²、Politzer 平衡参数 η、极性指数 Π、σ-矩 M1–M4
E. 头部取向   : 杂原子贡献表面占比（头基暴露度）、烷基链贡献表面占比
F. 梯度场     : 表面 ESP 梯度模均值（局部极性起伏）

方法学出处
----------
- Politzer / Murray 表面静电势统计框架（σ_tot²、Π、balance）:
  Politzer P. et al. J. Mol. Struct. THEOCHEM 2007, 804, 43; Murray & Politzer,
  WIREs Comput. Mol. Sci. 2011, 1, 153.
- 定量分子表面分析 (QMSA):
  Xiao X. et al. (Chen group) 关于分子表面描述符的工作; 见 *Water Research* 2025
  同类表面静电势建模范式。
- Shrake–Rupley 表面采样: Shrake A., Rupley J.A., J. Mol. Biol. 1973, 79, 351.
- MMFF94 部分电荷: Halgren T.A., J. Comput. Chem. 1996, 17, 490.

单位
----
ESP 默认输出 kcal·mol⁻¹·e⁻¹（换算常数 332.0637，即 1 e/Å）；面积单位 Å²。
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem

# ------------------------------------------------------------------ 常数
ESP_KCAL = 332.0637          # e/Å -> kcal·mol⁻¹·e⁻¹
PROBE_DEFAULT = 0.0          # 0 = vdW 分子表面；1.4 = 溶剂可及表面 (SAS)
POINTS_PER_ATOM = 240        # 每原子表面采样点数（Shrake–Rupley 密度）
MIN_POINTS_PER_ATOM = 48

# 单位换算：已发表 QMSA 数据集使用**原子单位**（bohr / Hartree），本模块默认 kcal·Å
BOHR_PER_ANG = 1.8897259886          # 1 Å = 1.8897 bohr
AU_AREA = BOHR_PER_ANG ** 2          # 3.5711 bohr² / Å²
KCAL_PER_HARTREE = 627.509474        # 1 Hartree = 627.5095 kcal/mol
HARTREE_PER_KCAL = 1.0 / KCAL_PER_HARTREE
AU_GRAD = HARTREE_PER_KCAL / BOHR_PER_ANG   # (Hartree/e)/bohr per (kcal/mol/e)/Å

# 各描述符在换算为原子单位时的幂次（ESP 的 n 次方）
_AREA_KEYS = ["qmsa_S_total", "qmsa_S_pos", "qmsa_S_neg", "qmsa_S_neutral"]
_LEN_KEYS = ["qmsa_Rg"]
_ESP_POW1 = ["qmsa_Vs_mean", "qmsa_Vs_pos_mean", "qmsa_Vs_neg_mean", "qmsa_Vs_min",
             "qmsa_Vs_max", "qmsa_Vs_range", "qmsa_M1", "qmsa_head_neighbor_esp"]
_ESP_POW1_VAR = ["qmsa_sigma_tot2", "qmsa_sigma_pos2", "qmsa_sigma_neg2", "qmsa_M2"]
_ESP_POW3 = ["qmsa_M3"]
_ESP_POW4 = ["qmsa_M4"]
_GRAD_KEYS = ["qmsa_grad_mean", "qmsa_grad_max"]

# Bondi 范德华半径 (Å) + 常见扩展
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "S": 1.80,
    "P": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98, "Si": 2.10, "B": 1.92,
    "Na": 2.27, "K": 2.75, "Li": 1.82, "Ca": 2.31, "Mg": 1.73, "Fe": 2.00,
}
VDW_DEFAULT = 1.70

# 各描述符组的特征名（供消融实验按组筛选）
QMSA_GROUPS = {
    "geom": ["qmsa_S_total", "qmsa_Rg", "qmsa_asphericity", "qmsa_anisotropy",
             "qmsa_surf_to_rg2"],
    "esp_area": ["qmsa_S_pos", "qmsa_S_neg", "qmsa_S_neutral",
                 "qmsa_S_pos_frac", "qmsa_S_neg_frac", "qmsa_S_neutral_frac"],
    "esp_int": ["qmsa_Vs_mean", "qmsa_Vs_pos_mean", "qmsa_Vs_neg_mean",
                "qmsa_Vs_min", "qmsa_Vs_max", "qmsa_Vs_range"],
    "esp_disp": ["qmsa_sigma_tot2", "qmsa_sigma_pos2", "qmsa_sigma_neg2",
                 "qmsa_balance_eta", "qmsa_balance_charges", "qmsa_polarity_Pi",
                 "qmsa_M1", "qmsa_M2", "qmsa_M3", "qmsa_M4"],
    "head": ["qmsa_hetero_surf_frac", "qmsa_carbon_surf_frac",
             "qmsa_head_neighbor_esp"],
    "grad": ["qmsa_grad_mean", "qmsa_grad_max"],
}
QMSA_FEATURES = [f for g in QMSA_GROUPS.values() for f in g]


# ------------------------------------------------------------------ 几何工具
def fibonacci_sphere(n: int) -> np.ndarray:
    """斐波那契格点在单位球面上近似均匀分布，返回 (n,3)。"""
    n = int(max(n, 1))
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)          # 极角，等面积
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i       # 黄金角
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def _vdw(atom) -> float:
    return VDW_RADII.get(atom.GetSymbol(), VDW_DEFAULT)


def embed_conformer(mol: Chem.Mol, seed: int = 0xF00D, n_confs: int = 1,
                    optimize: bool = True) -> Chem.Mol:
    """生成 3D 构象（ETKDGv3 + 可选 MMFF94 优化）。失败则返回 None。"""
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.maxIterations = 400
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=ps)
    if len(cids) == 0:
        ps.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=ps)
        if len(cids) == 0:
            return None
    if optimize:
        try:
            res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=600)
            # 取能量最低构象
            energies = [e for _, e in res]
            best = int(np.argmin(energies))
        except Exception:
            best = 0
    else:
        best = 0
    mol = Chem.Mol(mol, confId=int(cids[best]) if best < len(cids) else int(cids[0]))
    return mol


# ------------------------------------------------------------------ 电荷
def mmff_charges(mol: Chem.Mol) -> np.ndarray | None:
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        if props is None:
            return None
        return np.array([props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())],
                        dtype=float)
    except Exception:
        return None


def gasteiger_charges(mol: Chem.Mol) -> np.ndarray:
    mol = Chem.Mol(mol)
    AllChem.ComputeGasteigerCharges(mol)
    out = np.zeros(mol.GetNumAtoms(), dtype=float)
    for i, a in enumerate(mol.GetAtoms()):
        try:
            out[i] = float(a.GetProp("_GasteigerCharge"))
        except Exception:
            out[i] = 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def eem_charges(mol: Chem.Mol, positions: np.ndarray) -> np.ndarray | None:
    """Electronegativity Equalization Method（简单参数化），作 MMFF 缺失时的兜底。
    参数：Pauling 电负性 + 硬度（Rahmani 类近似），仅覆盖有机主族元素。"""
    CHI = {"H": 7.18, "C": 5.34, "N": 7.30, "O": 8.50, "F": 10.41, "S": 6.28,
           "P": 5.62, "Cl": 8.30, "Br": 7.59, "I": 6.76, "Si": 4.77, "B": 3.35}
    ETA = {"H": 6.43, "C": 5.10, "N": 6.90, "O": 7.90, "F": 9.90, "S": 4.90,
           "P": 5.00, "Cl": 6.10, "Br": 5.90, "I": 5.20, "Si": 3.90, "B": 4.00}
    atoms = list(mol.GetAtoms())
    for a in atoms:
        if a.GetSymbol() not in CHI:
            return None
    n = len(atoms)
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    for i, a in enumerate(atoms):
        A[i, i] = ETA[a.GetSymbol()]
        A[i, n] = -1.0
        A[n, i] = 1.0
        b[i] = -CHI[a.GetSymbol()]
    b[n] = -float(sum(a.GetFormalCharge() for a in atoms))
    A[n, n] = 0.0
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return sol[:n]


def assign_charges(mol: Chem.Mol, positions: np.ndarray,
                   method: str = "auto") -> tuple[np.ndarray, str]:
    """返回 (电荷数组, 实际使用的方法名)。"""
    if method in ("auto", "mmff94"):
        q = mmff_charges(mol)
        if q is not None:
            return q, "MMFF94"
        if method == "mmff94":
            return gasteiger_charges(mol), "Gasteiger(fallback)"
    if method in ("auto", "eem"):
        q = eem_charges(mol, positions)
        if q is not None:
            return q, "EEM"
    return gasteiger_charges(mol), "Gasteiger"


# ------------------------------------------------------------------ 表面采样
def surface_points(mol: Chem.Mol, positions: np.ndarray, probe: float = PROBE_DEFAULT,
                   points_per_atom: int = POINTS_PER_ATOM):
    """Shrake–Rupley 表面采样。

    返回
    ----
    pts      : (M,3) 表面点坐标
    owner    : (M,)  每个点归属的原子索引
    area     : (M,)  每个点代表的面积 (Å²)
    """
    radii = np.array([_vdw(a) for a in mol.GetAtoms()], dtype=float)
    n_atoms = len(radii)
    if n_atoms == 0:
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0)

    expanded = radii + probe
    unit = fibonacci_sphere(points_per_atom)

    all_pts, all_owner, all_area = [], [], []
    for i in range(n_atoms):
        # 该原子球面上的候选点
        cand = positions[i] + expanded[i] * unit                     # (P,3)
        d2 = cand[:, None, :] - positions[None, :, :]                # (P,N,3)
        dist2 = np.einsum("ijk,ijk->ij", d2, d2)                     # (P,N)
        # 若落在其它原子的扩展球内 -> 被埋没
        inside = dist2 < (expanded ** 2)[None, :]
        inside[:, i] = False
        keep = ~inside.any(axis=1)
        if not keep.any():
            continue
        p = cand[keep]
        # 面积按每个原子的球面积均分（只统计暴露点）
        a = 4.0 * np.pi * expanded[i] ** 2 / points_per_atom
        all_pts.append(p)
        all_owner.append(np.full(len(p), i, dtype=int))
        all_area.append(np.full(len(p), a, dtype=float))

    if not all_pts:
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0)
    return (np.vstack(all_pts), np.concatenate(all_owner),
            np.concatenate(all_area))


def esp_on_surface(pts: np.ndarray, positions: np.ndarray,
                   charges: np.ndarray) -> np.ndarray:
    """表面点上的库仑静电势 V(r) = Σ q_j / |r - R_j|，单位 kcal·mol⁻¹·e⁻¹。

    对近核点做软化（avoid 除零放大），软化半径 0.8 Å。
    """
    if len(pts) == 0:
        return np.zeros(0)
    diff = pts[:, None, :] - positions[None, :, :]            # (M,N,3)
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))      # (M,N)
    dist = np.maximum(dist, 0.8)
    return ESP_KCAL * (charges[None, :] / dist).sum(axis=1)


def esp_gradient_magnitude(pts: np.ndarray, positions: np.ndarray,
                           charges: np.ndarray, h: float = 0.05) -> np.ndarray:
    """表面 ESP 的梯度模 |∇V|（中心差分）。描述局部极性起伏。"""
    if len(pts) == 0:
        return np.zeros(0)
    grads = np.zeros((len(pts), 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        vp = esp_on_surface(pts + e, positions, charges)
        vm = esp_on_surface(pts - e, positions, charges)
        grads[:, k] = (vp - vm) / (2 * h)
    return np.sqrt((grads ** 2).sum(axis=1))


# ------------------------------------------------------------------ 描述符主函数
def apply_unit_system(d: dict, unit: str = "kcal") -> dict:
    """把描述符从默认的 kcal·mol⁻¹·Å 体系换算到其他体系。

    unit 取值
    ---------
    'kcal'      : kcal·mol⁻¹·e⁻¹ + Å（本模块默认）
    'au'        : Hartree/e + bohr（全原子单位）
    'published' : **对齐已发表 QMSA 数据集的混合单位** ——
                  ESP 幅度（Minimal/Maximal/平均势）用 kcal·mol⁻¹·e⁻¹，
                  ESP 方差用 Hartree²（= au），面积用 bohr²。
                  依据：由 1,029 条已发表记录反推，其 Overallvariance ∈ [0, 0.003]
                  恰与 kcal²→Hartree² 换算一致，而 Minimal 中位数 −53 则与 kcal 量级一致。
    """
    if unit == "kcal":
        d["qmsa_unit"] = "kcal_mol_A"
        return d

    # 面积 / 长度
    area_scale = AU_AREA if unit in ("au", "published") else 1.0
    for k in _AREA_KEYS:
        if k in d and np.isfinite(d[k]):
            d[k] = d[k] * area_scale
    for k in _LEN_KEYS:
        if k in d and np.isfinite(d[k]):
            d[k] = d[k] * BOHR_PER_ANG

    # ESP 强度
    if unit == "au":
        for k in _ESP_POW1:
            if k in d and np.isfinite(d[k]):
                d[k] = d[k] * HARTREE_PER_KCAL
        for k in _GRAD_KEYS:
            if k in d and np.isfinite(d[k]):
                d[k] = d[k] * AU_GRAD
    # published: ESP 幅度保持 kcal 不变（即不换算 _ESP_POW1 / _GRAD_KEYS）

    # ESP 方差（两种体系都用 Hartree²）
    for k in _ESP_POW1_VAR:
        if k in d and np.isfinite(d[k]):
            d[k] = d[k] * HARTREE_PER_KCAL ** 2
    # 高阶矩保持原量级，避免下溢
    d["qmsa_unit"] = unit
    return d


def qmsa_descriptors(smiles: str, probe: float = PROBE_DEFAULT,
                     charge_method: str = "auto", conf_seed: int = 0xF00D,
                     with_gradient: bool = True, unit: str = "kcal") -> dict:
    """单分子 QMSA 描述符。失败时返回全 NaN（调用方按列中位数填补）。
    unit='kcal'（默认，kcal·mol⁻¹·e⁻¹ 与 Å）或 'au'（Hartree 与 bohr，对齐已发表 QMSA 口径）。"""
    out = {k: np.nan for k in QMSA_FEATURES}
    out["smiles"] = smiles
    out["qmsa_ok"] = 0
    out["qmsa_charge_method"] = ""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out
    mol3d = embed_conformer(mol, seed=conf_seed)
    if mol3d is None:
        return out

    conf = mol3d.GetConformer()
    positions = conf.GetPositions().astype(float)
    charges, cmethod = assign_charges(mol3d, positions, method=charge_method)
    if not np.isfinite(charges).all() or np.allclose(charges, 0.0):
        return out

    pts, owner, area = surface_points(mol3d, positions, probe=probe)
    if len(pts) < 10:
        return out

    esp = esp_on_surface(pts, positions, charges)
    S_total = float(area.sum())
    if S_total <= 0:
        return out

    # ---- A. 表面几何 ------------------------------------------------
    centroid = np.average(pts, axis=0, weights=area)
    rel = pts - centroid
    rg2 = float(np.average((rel ** 2).sum(axis=1), weights=area))
    Rg = float(np.sqrt(max(rg2, 1e-9)))
    gyr = (rel[:, :, None] * rel[:, None, :] * area[:, None, None]).sum(axis=0) / S_total
    ev = np.sort(np.linalg.eigvalsh(gyr))[::-1]          # λ1 ≥ λ2 ≥ λ3
    ev = np.maximum(ev, 0.0)
    s1, s2, s3 = ev
    tr = s1 + s2 + s3 + 1e-12
    asphericity = float(s1 - 0.5 * (s2 + s3))
    anisotropy = float(((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s1 - s3) ** 2) / (2 * tr ** 2))
    out.update({
        "qmsa_S_total": S_total,
        "qmsa_Rg": Rg,
        "qmsa_asphericity": asphericity,
        "qmsa_anisotropy": anisotropy,
        "qmsa_surf_to_rg2": S_total / (Rg ** 2),
    })

    # ---- B. ESP 面积谱 ----------------------------------------------
    pos_m = esp > 0
    neg_m = esp < 0
    S_pos = float(area[pos_m].sum())
    S_neg = float(area[neg_m].sum())
    S_neu = float(S_total - S_pos - S_neg)
    out.update({
        "qmsa_S_pos": S_pos, "qmsa_S_neg": S_neg, "qmsa_S_neutral": S_neu,
        "qmsa_S_pos_frac": S_pos / S_total, "qmsa_S_neg_frac": S_neg / S_total,
        "qmsa_S_neutral_frac": S_neu / S_total,
    })

    # ---- C. ESP 强度谱 ----------------------------------------------
    Vs_mean = float(np.average(esp, weights=area))
    Vs_pos = float(np.average(esp[pos_m], weights=area[pos_m])) if S_pos > 0 else 0.0
    Vs_neg = float(np.average(esp[neg_m], weights=area[neg_m])) if S_neg > 0 else 0.0
    out.update({
        "qmsa_Vs_mean": Vs_mean,
        "qmsa_Vs_pos_mean": Vs_pos,
        "qmsa_Vs_neg_mean": Vs_neg,
        "qmsa_Vs_min": float(esp.min()),
        "qmsa_Vs_max": float(esp.max()),
        "qmsa_Vs_range": float(esp.max() - esp.min()),
    })

    # ---- D. ESP 离散度 + σ-矩 ----------------------------------------
    w = area / S_total
    d = esp - Vs_mean
    M2 = float((w * d ** 2).sum())
    M3 = float((w * d ** 3).sum())
    M4 = float((w * d ** 4).sum())
    sigma_tot2 = M2
    sigma_pos2 = float(np.average((esp[pos_m] - Vs_pos) ** 2, weights=area[pos_m])) if S_pos > 0 else 0.0
    sigma_neg2 = float(np.average((esp[neg_m] - Vs_neg) ** 2, weights=area[neg_m])) if S_neg > 0 else 0.0
    # Politzer 平衡参数：η ∈ [-1,1]，>0 = 电子缺陷（亲电/Lewis 酸），<0 = 富电子（亲核/Lewis 碱）
    denom = abs(Vs_pos) + abs(Vs_neg)
    eta = float((abs(Vs_pos) - abs(Vs_neg)) / denom) if denom > 1e-12 else 0.0
    # 文献 "Balance of charges"：B = |V+|·|V-| / (|V+|+|V-|)² ∈ [0, 0.25]，0.25 = 正负势完全均衡
    # （已用 1,029 条已发表 QMSA 记录反推确认为该式，见 data/raw 与 README）
    bcharges = float(abs(Vs_pos) * abs(Vs_neg) / denom ** 2) if denom > 1e-12 else 0.0
    out.update({
        "qmsa_sigma_tot2": sigma_tot2,
        "qmsa_sigma_pos2": sigma_pos2,
        "qmsa_sigma_neg2": sigma_neg2,
        "qmsa_balance_eta": eta,
        "qmsa_balance_charges": bcharges,
        "qmsa_polarity_Pi": S_neg / S_total * 100.0,
        "qmsa_M1": Vs_mean, "qmsa_M2": M2, "qmsa_M3": M3, "qmsa_M4": M4,
    })

    # ---- E. 头部取向（杂原子 vs 碳骨架的表面贡献）--------------------
    sym = np.array([a.GetSymbol() for a in mol3d.GetAtoms()])
    carbon_owner = np.isin(sym[owner], ["C"])
    hetero_owner = np.isin(sym[owner], ["O", "N", "S", "P"])
    S_carb = float(area[carbon_owner].sum())
    S_het = float(area[hetero_owner].sum())
    head_esp = float(np.average(esp[hetero_owner], weights=area[hetero_owner])) if S_het > 0 else 0.0
    out.update({
        "qmsa_hetero_surf_frac": S_het / S_total,
        "qmsa_carbon_surf_frac": S_carb / S_total,
        "qmsa_head_neighbor_esp": head_esp,
    })

    # ---- F. 梯度场 ---------------------------------------------------
    if with_gradient:
        g = esp_gradient_magnitude(pts, positions, charges)
        out.update({"qmsa_grad_mean": float(np.average(g, weights=area)),
                    "qmsa_grad_max": float(g.max())})

    out["qmsa_ok"] = 1
    out["qmsa_charge_method"] = cmethod
    return apply_unit_system(out, unit)


# ------------------------------------------------------------------ 批量
def qmsa_matrix(smiles_list, verbose: bool = True, **kw) -> pd.DataFrame:
    """批量计算，返回 DataFrame（含 smiles / qmsa_ok）。"""
    rows = []
    n = len(smiles_list)
    for i, smi in enumerate(smiles_list):
        try:
            rows.append(qmsa_descriptors(smi, **kw))
        except Exception:
            r = {k: np.nan for k in QMSA_FEATURES}
            r.update({"smiles": smi, "qmsa_ok": 0, "qmsa_charge_method": ""})
            rows.append(r)
        if verbose and (i + 1) % 100 == 0:
            ok = sum(r.get("qmsa_ok", 0) for r in rows)
            print(f"  QMSA: {i + 1}/{n}  (成功 {ok}, {ok / (i + 1):.0%})")
    return pd.DataFrame(rows)


def esp_profile(smiles: str, bins: int = 60, span: float = 120.0, unit: str = "kcal", **kw):
    """返回 (σ 网格, 面积加权 σ-profile)，用于 Fig 的 σ-profile 图。
    unit='au' 时横轴为 Hartree/e、纵轴为 bohr²，与已发表 QMSA 口径一致。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    mol3d = embed_conformer(mol)
    if mol3d is None:
        return None, None
    positions = mol3d.GetConformer().GetPositions().astype(float)
    charges, _ = assign_charges(mol3d, positions, kw.get("charge_method", "auto"))
    pts, _, area = surface_points(mol3d, positions, probe=kw.get("probe", PROBE_DEFAULT))
    if len(pts) < 10:
        return None, None
    esp = esp_on_surface(pts, positions, charges)
    if unit == "au":
        esp = esp * HARTREE_PER_KCAL
        area = area * AU_AREA
        span = span * HARTREE_PER_KCAL
    grid = np.linspace(-span, span, bins)
    hist, _ = np.histogram(esp, bins=grid, weights=area)
    return grid[:-1] + (grid[1] - grid[0]) / 2.0, hist


# ------------------------------------------------------------------ 与已发表 QMSA 口径对照
# 已发表 QMSA 描述符名 → 本模块字段。来源：PCBA-686978 数据集的 17 个 QMSA 列
# （见 data/raw/，由 validate_qmsa.py 反推核对）。用于与文献数值直接比对。
PUBLISHED_QMSA_MAP = {
    "Positivesurfacearea": "qmsa_S_pos",
    "Negativesurfacearea": "qmsa_S_neg",
    "Minimal": "qmsa_Vs_min",
    "Maximal": "qmsa_Vs_max",
    "Overallaveragevalue": "qmsa_Vs_mean",
    "Positiveaveragevalue": "qmsa_Vs_pos_mean",
    "Negativeaveragevalue": "qmsa_Vs_neg_mean",
    "Overallvariance": "qmsa_sigma_tot2",
    "Positivevariance": "qmsa_sigma_pos2",
    "Negativevariance": "qmsa_sigma_neg2",
    "Balanceofcharges": "qmsa_balance_charges",
    "Molecularpolarityindex": "qmsa_polarity_Pi",
}
PUBLISHED_ONLY = ["Volume", "Estimateddensity", "Product",
                  "Nonpolarsurfacearea", "Polarsurfacearea"]


def to_published_frame(df: pd.DataFrame) -> pd.DataFrame:
    """把本模块输出的描述符表改写为**已发表 QMSA 列名**，便于与文献数据集直接比对。"""
    ren = {v: k for k, v in PUBLISHED_QMSA_MAP.items()}
    cols = [c for c in df.columns if c in ren]
    out = df[cols].rename(columns=ren)
    return out
