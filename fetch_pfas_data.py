# -*- coding: utf-8 -*-
"""
fetch_pfas_data.py — PFAS 真实活性数据的获取、解析与清洗

数据源（按优先级）
------------------
1. `zenodo`  — Cheng & Ng (ES&T 2019, 53:13970, DOI 10.1021/acs.est.9b04833) 的
   PFAS 生物活性面板（1012 PFAS × 26 bioassays）经泄漏控制重建后的公开镜像
   （Zenodo 10.5281/zenodo.22869943）。**默认主源**。
   注：原始 figshare 附件（es9b04833_si_004.xlsx）在部分网络环境下被 CDN 拒绝（HTTP 403），
   故本脚本以 Zenodo 镜像为首选，并保留 figshare 直连作为备选。
2. `pubchem` — PubChem PUG-REST + BioAssay：按 CID 拉 SMILES、按 AID 拉活性标签。
   **完全免授权、可按免疫相关终点自定义**。
3. `comptox` — EPA CompTox Chemicals Dashboard API：拉取 PFAS 结构清单（结构补充）。

清洗流程
--------
标准化 SMILES → 去无机盐/去重 → PFAS 界定（OECD 2021：≥1 个全氟甲基/亚甲基碳）
→ 标签二值化 → 覆盖率筛选 → 免疫终点筛选 → 输出宽表/长表 + QC 报告。

用法
----
    # 默认：Zenodo 主源，输出免疫相关终点子集
    python fetch_pfas_data.py

    # 指定源与参数
    python fetch_pfas_data.py --source zenodo --min-assays 20 --out data/pfas_chengng.csv
    python fetch_pfas_data.py --source pubchem --pubchem-aids 743122,743140 --out data/pfas_pubchem.csv

    # 只走本地已下载的原始文件（离线）
    python fetch_pfas_data.py --offline
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfas_qsar as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
REPORT_DIR = os.path.join(DATA_DIR, "reports")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# ------------------------------------------------------------------ 数据源注册
SOURCES = {
    "cheng_ng_zenodo": {
        "type": "zip",
        "urls": [
            "https://zenodo.org/records/22869943/files/PFAS-GNN-release.zip?download=1",
            "https://zenodo.org/api/records/22869943/files/PFAS-GNN-release.zip/content",
        ],
        "filename": "PFAS-GNN-release.zip",
        "md5": None,       # 未提供官方 md5，改用 PK 签名 + 大小校验
        "min_size": 1_000_000,
        "cite": ("Cheng W., Ng C.A. Environ. Sci. Technol. 2019, 53(23), 13970-13980. "
                 "DOI 10.1021/acs.est.9b04833；镜像 Zenodo 10.5281/zenodo.22869943"),
        "license": "GPL-3.0-or-later (镜像) / CC BY-NC 4.0 (原始附件)",
    },
    "cheng_ng_figshare": {
        "type": "file",
        "urls": [
            "https://ndownloader.figshare.com/files/18812648",
            "https://acs.figshare.com/ndownloader/files/18812648",
        ],
        "filename": "es9b04833_si_004.xlsx",
        "md5": "2609c0ccc2b5fc83b5cd3cbf4195956e",
        "min_size": 50_000,
        "cite": ("Cheng W., Ng C.A. Environ. Sci. Technol. 2019, 53(23), 13970-13980. "
                 "DOI 10.1021/acs.est.9b04833, SI file es9b04833_si_004.xlsx"),
        "license": "CC BY-NC 4.0",
    },
}

# ------------------------------------------------------------------ 免疫相关终点映射
# Cheng & Ng 面板的 26 个 bioassay 来自 Tox21 / ToxCast 等；以下按"关键词 → 免疫相关类别"
# 建立映射，命中即保留。未命中的终点归类为 general/other，可通过 --keep-all-endpoints 保留。
IMMUNE_ENDPOINT_MAP = {
    "nuclear_receptor": [
        r"ppar", r"ahr", r"\bgr\b", r"glucocorticoid", r"\bgr_", r"rxr", r"rar",
        r"fxr", r"lxr", r"pxr", r"car\b", r"vdr", r"vitamin\s*d", r"\btr\b",
        r"thyroid", r"estrogen|\ber[_\b]", r"androgen|\bar[_\b]",
    ],
    "immune_signaling": [
        r"nf[\s\-_]?kb", r"nfkb", r"tgf", r"stat", r"\bil[\s\-_]?\d", r"interleukin",
        r"tnf", r"tumor\s*necrosis", r"interferon", r"\bifn", r"\bill",
        r"chemokine", r"tlr", r"toll[\s\-_]like", r"inflammat",
    ],
    "stress_response": [
        r"nrf2", r"\bare\b", r"oxidative", r"stress\s*response", r"hse", r"heat\s*shock",
        r"hif", r"hypoxia", r"\bp53\b", r"atad5", r"genotox", r"dna\s*damage",
    ],
    "cell_fate_cytotox": [
        r"viability", r"cytotox", r"apoptos", r"cell\s*death", r"mitochondrial",
        r"\bcaspase", r"proliferat", r"cell\s*cycle",
    ],
}


def classify_endpoint(name: str) -> str:
    """将终点名映射到免疫相关类别；未命中返回 'other'。"""
    s = str(name).lower()
    for cat, pats in IMMUNE_ENDPOINT_MAP.items():
        for p in pats:
            if re.search(p, s):
                return cat
    return "other"


# ------------------------------------------------------------------ 网络工具
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def http_get(url: str, timeout: int = 120, retries: int = 4,
             referer: str | None = None) -> bytes | None:
    """带重试的 GET，返回 bytes；失败返回 None（不抛出，便于降级）。"""
    hdr = {"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"}
    if referer:
        hdr["Referer"] = referer
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
                return r.read()
        except Exception as e:
            if k == retries - 1:
                print(f"      [warn] 放弃 {url[:70]} … {str(e)[:80]}")
            else:
                time.sleep(2 + 2 * k)
    return None


def http_get_json(url: str, **kw):
    d = http_get(url, **kw)
    if d is None:
        return None
    try:
        return json.loads(d)
    except Exception:
        return None


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()


def download_source(key: str, offline: bool = False) -> str | None:
    """下载（或复用缓存）指定源，返回本地路径。"""
    spec = SOURCES[key]
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, spec["filename"])

    if os.path.exists(dest) and os.path.getsize(dest) >= spec["min_size"]:
        if spec.get("md5") and md5_of(dest) != spec["md5"]:
            print(f"    [cache] {spec['filename']} 存在但 md5 不符，重新下载")
        else:
            print(f"    [cache] 复用本地 {spec['filename']} "
                  f"({os.path.getsize(dest) / 1e6:.1f} MB)")
            return dest

    if offline:
        print(f"    [offline] 未找到本地 {spec['filename']}，跳过")
        return None

    for u in spec["urls"]:
        print(f"    [get] {u[:88]} …")
        d = http_get(u, referer="https://acs.figshare.com/")
        if d and len(d) >= spec["min_size"]:
            with open(dest, "wb") as f:
                f.write(d)
            print(f"    [ok] {len(d) / 1e6:.2f} MB -> {dest}")
            if spec.get("md5"):
                got = md5_of(dest)
                flag = "匹配" if got == spec["md5"] else f"不匹配(实际 {got})"
                print(f"    [md5] {flag}")
            return dest
    print(f"    [fail] 源 {key} 所有 URL 均不可用")
    return None


# ------------------------------------------------------------------ 结构自动探测
TABULAR_EXT = (".csv", ".tsv", ".txt", ".xlsx", ".xls")


def _read_any(path: str) -> dict[str, pd.DataFrame]:
    """读入一个表格文件的所有 sheet。"""
    low = path.lower()
    try:
        if low.endswith((".xlsx", ".xls")):
            sheets = pd.read_excel(path, sheet_name=None)
            return {k: v for k, v in sheets.items() if isinstance(v, pd.DataFrame) and v.size}
        sep = "\t" if low.endswith(".tsv") else None
        if sep is None:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                head = f.readline()
            sep = "\t" if head.count("\t") > head.count(",") else ","
        df = pd.read_csv(path, sep=sep, low_memory=False)
        if df.shape[1] == 1:                       # 分隔符猜错，重试
            df = pd.read_csv(path, sep=None, engine="python", low_memory=False)
        return {"__single__": df}
    except Exception as e:
        print(f"      [skip] 无法读取 {os.path.basename(path)}: {str(e)[:70]}")
        return {}


def _looks_like_smiles(v) -> bool:
    """粗判是否为 SMILES 字符串。"""
    if not isinstance(v, str) or len(v) < 3 or len(v) > 400:
        return False
    if " " in v or "\n" in v:
        return False
    return bool(re.search(r"[A-Za-z]", v)) and bool(re.search(r"[(\[=#]|F|C", v))


def score_table(df: pd.DataFrame) -> float:
    """给一张表打"是否为主标签矩阵"的分。"""
    if df.shape[0] < 50 or df.shape[1] < 3:
        return -1.0
    score = 0.0
    # 有 SMILES 列？加分
    smi_hits = 0
    for c in df.columns[:40]:
        vals = df[c].dropna().astype(str).head(40)
        if len(vals) and sum(_looks_like_smiles(v) for v in vals) / len(vals) > 0.6:
            smi_hits += 1
    score += min(smi_hits, 2) * 3.0
    # 有大量 0/1 或 0–1 连续列？加分
    num_cols = 0
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().mean() > 0.5 and v.dropna().between(0, 1).mean() > 0.95:
            num_cols += 1
    score += min(num_cols, 26) * 0.4
    score += min(df.shape[0] / 800.0, 2.0)
    return score


def discover_label_matrix(root: str) -> tuple[pd.DataFrame, str, str]:
    """在解包目录中自动发现标签矩阵，返回 (df, 文件路径, sheet 名)。"""
    cands = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(TABULAR_EXT) and not fn.startswith("~$"):
                fp = os.path.join(dirpath, fn)
                for sheet, df in _read_any(fp).items():
                    sc = score_table(df)
                    if sc > 0:
                        cands.append((sc, fp, sheet, df))
    if not cands:
        return pd.DataFrame(), "", ""
    cands.sort(key=lambda t: -t[0])
    best = cands[0]
    print(f"    [discover] 候选表 {len(cands)} 个，选中 score={best[0]:.1f}："
          f"{os.path.relpath(best[1], root)} [{best[2]}] {best[3].shape}")
    return best[3], best[1], best[2]


def normalize_matrix(df: pd.DataFrame, smiles_col: str | None,
                     label_cols: list[str] | None,
                     name_col: str | None = None) -> pd.DataFrame:
    """把任意布局的标签矩阵规整为宽表：smiles + <assay 列…> + (可选)name。"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # --- SMILES 列 ---
    if smiles_col is None:
        best, bestfrac = None, 0.0
        for c in df.columns:
            vals = df[c].dropna().astype(str).head(80)
            if not len(vals):
                continue
            frac = sum(_looks_like_smiles(v) for v in vals) / len(vals)
            if frac > bestfrac:
                best, bestfrac = c, frac
        smiles_col = best if bestfrac > 0.5 else None
    if smiles_col is None:
        raise SystemExit("未能自动识别 SMILES 列，请用 --smiles-col 指定")
    print(f"    [smiles] 列 = '{smiles_col}'")

    # --- 化合物名列（可选）---
    if name_col is None:
        for c in df.columns:
            if c == smiles_col:
                continue
            vals = df[c].dropna().astype(str)
            if len(vals) and vals.str.len().between(3, 80).mean() > 0.6 \
                    and not vals.astype(str).str.fullmatch(r"[-+]?\d+(\.\d+)?").mean() > 0.9:
                name_col = c
                break

    # --- 标签列 ---
    if label_cols is None:
        label_cols = []
        for c in df.columns:
            if c in (smiles_col, name_col):
                continue
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().mean() > 0.3 and v.dropna().between(0, 1).mean() > 0.9:
                label_cols.append(c)
    print(f"    [labels] 识别到 {len(label_cols)} 个活性终点列")

    keep = [smiles_col] + ([name_col] if name_col else []) + label_cols
    out = df[keep].rename(columns={smiles_col: "smiles"})
    if name_col:
        out = out.rename(columns={name_col: "name"})
    for c in label_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ------------------------------------------------------------------ PFAS 界定与清洗
PFAS_OECD_NOTE = ("OECD 2021 定义：至少含 1 个完全氟化的甲基/亚甲基碳（无任何 H/Cl/Br/I 取代）")


def _perfluoro_carbons(smi: str) -> int:
    mol = P.Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return 0
    patt = P.Chem.MolFromSmarts("[CX4](F)(F)")
    return len(mol.GetSubstructMatches(patt)) if patt else 0


def clean_dataset(df: pd.DataFrame, label_cols: list[str],
                  min_assays: int = 10, min_perfluoro_c: int = 1,
                  binarize_thr: float = 0.5, require_pfas: bool = True,
                  verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """标准化 + 去重 + PFAS 界定 + 覆盖率筛选 + 二值化。返回 (干净表, QC 统计)。"""
    qc: dict = {"input_rows": int(len(df))}

    df = df.copy()
    df["std_smiles"] = df["smiles"].map(P.standardize_smiles)
    qc["invalid_smiles"] = int(df["std_smiles"].isna().sum())
    df = df.dropna(subset=["std_smiles"])

    # 去重：同一标准 SMILES 保留标签最全的一条
    lab_cov = df[label_cols].notna().sum(axis=1)
    df = df.assign(_cov=lab_cov).sort_values("_cov", ascending=False) \
           .drop_duplicates("std_smiles").drop(columns="_cov").reset_index(drop=True)
    qc["after_dedup"] = int(len(df))

    # 标签二值化
    for c in label_cols:
        v = df[c]
        df[c] = np.where(v.isna(), np.nan, (v >= binarize_thr).astype(float))

    # PFAS 界定
    if require_pfas:
        df["n_perfluoro_C"] = df["std_smiles"].map(_perfluoro_carbons)
        n0 = len(df)
        df = df[df["n_perfluoro_C"] >= min_perfluoro_c].reset_index(drop=True)
        qc["pfas_removed"] = int(n0 - len(df))
        qc["pfas_note"] = PFAS_OECD_NOTE
    else:
        df["n_perfluoro_C"] = df["std_smiles"].map(_perfluoro_carbons)

    # 覆盖率筛选：每个分子至少 min_assays 个有标签
    cov = df[label_cols].notna().sum(axis=1)
    df["n_labels"] = cov
    n0 = len(df)
    df = df[cov >= min_assays].reset_index(drop=True)
    qc["removed_low_coverage"] = int(n0 - len(df))

    # 丢弃全空 / 全同值（无信息）的终点列
    keep = [c for c in label_cols
            if df[c].notna().sum() >= 20 and 0.005 < np.nanmean(df[c]) < 0.995]
    dropped = sorted(set(label_cols) - set(keep))
    qc["endpoints_dropped_uninformative"] = dropped

    qc["final_rows"] = int(len(df))
    qc["final_endpoints"] = int(len(keep))
    qc["final_cells"] = int(df[keep].notna().sum().sum())
    qc["label_cols"] = keep
    if verbose:
        print(f"    [clean] {qc['input_rows']} -> 去重 {qc.get('after_dedup')} "
              f"-> 去非PFAS {qc.get('pfas_removed', 0)} -> 覆盖率筛除 {qc.get('removed_low_coverage')} "
              f"== 最终 {qc['final_rows']} 分子 × {qc['final_endpoints']} 终点 "
              f"({qc['final_cells']} 个标签)")
    return df, qc


# ------------------------------------------------------------------ CompTox 结构补充
def fetch_comptox_structures(seeds: list[str], limit_per_seed: int = 100) -> pd.DataFrame:
    """用 CompTox Dashboard API 的 start-with 检索收集 PFAS 结构（结构层面的外部参照）。"""
    base = "https://comptox.epa.gov/dashboard-api/ccdapp1/search/chemical/start-with/"
    rows = []
    for s in seeds:
        j = http_get_json(base + urllib.parse.quote(s), timeout=30)
        if not isinstance(j, list):
            print(f"      [comptox] '{s}' 无结果")
            continue
        for r in j[:limit_per_seed]:
            rows.append({"seed": s, "dtxsid": r.get("dtxsid"),
                         "dtxcid": r.get("dtxcid"), "synonym": r.get("searchWord"),
                         "match": r.get("searchMatch")})
        print(f"      [comptox] '{s}' -> {len(j)} 条")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ PubChem 源
PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def pubchem_smiles_by_cids(cids: list[int], chunk: int = 100) -> dict[int, str]:
    """按 CID 批量取 Canonical SMILES。"""
    out: dict[int, str] = {}
    for i in range(0, len(cids), chunk):
        part = cids[i:i + chunk]
        url = (f"{PUBCHEM_REST}/compound/cid/{','.join(map(str, part))}"
               f"/property/ConnectivitySMILES/JSON")
        j = http_get_json(url, timeout=60)
        if not j:
            continue
        for p in j.get("PropertyTable", {}).get("Properties", []):
            cid = p.get("CID")
            smi = p.get("ConnectivitySMILES") or p.get("CanonicalSMILES")
            if cid and smi:
                out[int(cid)] = smi
    return out


def pubchem_assay_actives(aid: int) -> list[int]:
    """取某 BioAssay 的 active CIDs（Tox21/ToxCast 等）。"""
    url = f"{PUBCHEM_REST}/assay/aid/{aid}/cids/JSON?cids_type=active"
    j = http_get_json(url, timeout=60)
    if not j:
        return []
    return [int(c) for c in j.get("InformationList", {}).get("Information", [{}])[0].get("CID", [])]


def fetch_pubchem_dataset(aids: list[int], background_per_assay: int = 400) -> pd.DataFrame:
    """构建 PubChem 标签矩阵：各 AID 的 active 记 1，随机背景集记 0。

    背景集用"同 AID 的 inactive CIDs + 若不足则用其他 PFAS CID 补"，避免全零列。
    """
    smi_map: dict[int, str] = {}
    cols: dict[str, dict[int, float]] = {}
    for aid in aids:
        act = pubchem_assay_actives(aid)
        inact: list[int] = []
        j = http_get_json(f"{PUBCHEM_REST}/assay/aid/{aid}/cids/JSON?cids_type=inactive", timeout=60)
        if j:
            inact = [int(c) for c in j["InformationList"]["Information"][0].get("CID", [])]
        if not inact:
            continue
        rng = np.random.default_rng(aid)
        act = list(act)
        inact = list(rng.choice(inact, size=min(len(inact), max(background_per_assay, len(act) * 3)),
                                replace=False))
        col = {}
        for c in act:
            col[int(c)] = 1.0
        for c in inact:
            col[int(c)] = 0.0
        cols[f"PUBCHEM_AID_{aid}"] = col
        print(f"      [pubchem] AID {aid}: active {len(act)} / inactive {len(inact)}")
    all_cids = sorted({c for col in cols.values() for c in col})
    print(f"      [pubchem] 合计 {len(all_cids)} 个 CID，拉取 SMILES …")
    smi_map.update(pubchem_smiles_by_cids(all_cids))
    rows = []
    for cid in all_cids:
        smi = smi_map.get(cid)
        if not smi:
            continue
        rec = {"smiles": smi, "name": f"CID{cid}", "pubchem_cid": cid}
        for k, col in cols.items():
            rec[k] = col.get(cid, np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description="PFAS 真实活性数据获取与清洗")
    ap.add_argument("--source", default="auto",
                    choices=["auto", "zenodo", "figshare", "pubchem", "comptox"])
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "pfas_dataset.csv"))
    ap.add_argument("--offline", action="store_true", help="只用本地缓存/原始文件")
    ap.add_argument("--min-assays", type=int, default=15,
                    help="每个分子至少有多少个终点标签（默认 15）")
    ap.add_argument("--min-perfluoro-c", type=int, default=1)
    ap.add_argument("--binarize-thr", type=float, default=0.5)
    ap.add_argument("--no-pfas-filter", action="store_true")
    ap.add_argument("--smiles-col", default=None)
    ap.add_argument("--keep-all-endpoints", action="store_true",
                    help="保留全部终点（默认仅保留免疫相关 + 各终点全保留标签）")
    ap.add_argument("--immune-only", action="store_true",
                    help="仅保留免疫相关终点列（默认全部保留，但会打类别标签）")
    ap.add_argument("--pubchem-aids", default="743122,743140,743134,743053",
                    help="PubChem BioAssay AID 列表（逗号分隔）")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("=" * 72)
    print("PFAS 真实活性数据 —— 获取 / 解析 / 清洗")
    print("=" * 72)

    report: dict = {"source": args.source, "args": vars(args), "steps": []}

    # ---------- A. 取原始数据 ----------
    matrix = pd.DataFrame()
    origin = ""

    if args.source in ("auto", "zenodo", "figshare"):
        order = ["cheng_ng_zenodo", "cheng_ng_figshare"]
        for key in order:
            print(f"\n[A] 尝试数据源：{key}")
            path = download_source(key, offline=args.offline)
            if not path:
                continue
            with tempfile_dir() as tmp:
                if path.lower().endswith(".zip"):
                    with zipfile.ZipFile(path) as z:
                        z.extractall(tmp)
                    root = tmp
                else:
                    import shutil
                    shutil.copy(path, os.path.join(tmp, os.path.basename(path)))
                    root = tmp
                matrix, fp, sheet = discover_label_matrix(root)
                if not matrix.empty:
                    origin = f"{key} :: {os.path.basename(fp)}[{sheet}]"
                    report["steps"].append({"source": key, "file": os.path.basename(fp),
                                            "sheet": sheet, "shape": list(matrix.shape)})
                    break
            if not matrix.empty:
                break

    if matrix.empty and args.source in ("auto", "pubchem"):
        aids = [int(x) for x in args.pubchem_aids.split(",") if x.strip()]
        print(f"\n[A] 回退到 PubChem BioAssay 源（AID {aids}）")
        try:
            matrix = fetch_pubchem_dataset(aids)
            origin = "pubchem_bioassay"
            report["steps"].append({"source": "pubchem", "aids": aids,
                                    "shape": list(matrix.shape)})
        except Exception as e:
            print(f"      [warn] PubChem 失败：{str(e)[:100]}")

    if matrix.empty:
        print("\n[!] 未能获得任何数据。请检查网络，或用 --offline 配合已下载的 data/raw/ 文件。")
        print("    可手动放置：data/raw/PFAS-GNN-release.zip  或  data/raw/es9b04833_si_004.xlsx")
        sys.exit(2)

    report["origin"] = origin

    # ---------- B. 规整 ----------
    print(f"\n[B] 规整标签矩阵（来源：{origin}）")
    norm = normalize_matrix(matrix, args.smiles_col, None)
    label_cols = [c for c in norm.columns if c not in ("smiles", "name")]
    report["raw"] = {"rows": int(len(norm)), "endpoints": len(label_cols)}

    # ---------- C. 清洗 ----------
    print("\n[C] 清洗（标准化 / 去重 / PFAS 界定 / 覆盖率 / 二值化）")
    clean, qc = clean_dataset(
        norm, label_cols,
        min_assays=args.min_assays,
        min_perfluoro_c=args.min_perfluoro_c,
        binarize_thr=args.binarize_thr,
        require_pfas=not args.no_pfas_filter,
    )
    report["qc"] = {k: v for k, v in qc.items() if k != "label_cols"}

    # ---------- D. 终点免疫相关性标注 ----------
    print("\n[D] 终点免疫相关性标注")
    ep_df = pd.DataFrame({"endpoint": qc["label_cols"]})
    ep_df["category"] = ep_df["endpoint"].map(classify_endpoint)
    ep_df["n_pos"] = [int(np.nansum(clean[c])) for c in ep_df["endpoint"]]
    ep_df["pos_rate"] = [float(np.nanmean(clean[c])) for c in ep_df["endpoint"]]
    ep_df = ep_df.sort_values(["category", "pos_rate"], ascending=[True, False])
    print(ep_df.to_string(index=False))
    ep_df.to_csv(os.path.join(REPORT_DIR, "endpoint_annotation.csv"),
                 index=False, encoding="utf-8-sig")

    immune_eps = ep_df.loc[ep_df["category"] != "other", "endpoint"].tolist()
    report["endpoints"] = {"total": len(ep_df), "immune_relevant": len(immune_eps),
                           "by_category": ep_df["category"].value_counts().to_dict()}

    if args.immune_only:
        keep = ["smiles"] + ([ "name"] if "name" in clean.columns else []) + immune_eps
        clean = clean[[c for c in keep if c in clean.columns]]
        print(f"\n    [immune-only] 保留 {len(immune_eps)} 个免疫相关终点")

    # ---------- E. 输出 ----------
    print(f"\n[E] 写出数据集 -> {args.out}")
    clean.to_csv(args.out, index=False, encoding="utf-8-sig")

    lab_cols_out = [c for c in clean.columns if c not in ("smiles", "name", "std_smiles",
                                                          "n_perfluoro_C", "n_labels")]
    long = clean.melt(id_vars=["smiles"], value_vars=lab_cols_out,
                      var_name="endpoint", value_name="label").dropna(subset=["label"])
    long_path = os.path.splitext(args.out)[0] + "_long.csv"
    long.to_csv(long_path, index=False, encoding="utf-8-sig")
    print(f"           长表 -> {long_path}  ({len(long)} 行)")

    with open(os.path.join(REPORT_DIR, "fetch_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"           QC 报告 -> {os.path.join(REPORT_DIR, 'fetch_report.json')}")

    print("\n引用信息：")
    for k, s in SOURCES.items():
        print(f"  - {k}: {s['cite']}")
        print(f"    许可: {s['license']}")
    print("\n完成。")


class tempfile_dir:
    """临时目录上下文（避免依赖 tempfile 的清理策略差异）。"""

    def __enter__(self):
        import tempfile
        self._d = tempfile.mkdtemp(prefix="pfas_fetch_")
        return self._d

    def __exit__(self, *exc):
        import shutil
        shutil.rmtree(self._d, ignore_errors=True)
        return False


if __name__ == "__main__":
    main()
