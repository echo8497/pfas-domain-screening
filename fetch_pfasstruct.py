# -*- coding: utf-8 -*-
"""
fetch_pfasstruct.py — 获取并解析 EPA CompTox PFASSTRUCT V6 结构全集（21,028 个 PFAS 结构）

背景
----
CompTox Chemicals Dashboard 的 PFAS 结构清单（PFASSTRUCT）逐版迭代：
    V1 2018-03 / V2 2019-11 / V3 2020-08 / V4 2021-08 / V5 2022-08 / **V6 2026-01（最新，21,028 条）**
官方页面 https://comptox.epa.gov/dashboard/chemical-lists/PFASSTRUCTV6 是 Nuxt SSR，
21,028 条记录以「参数化函数 + 短变量名引用」的形式内联在 HTML 中（约 21.6 MB）。
官方未提供稳定的 REST 批量导出端点（已实测 `/ccdapp1|/ccdapp2/chemical-lists/download/*` 全部 404/空），
因此本脚本直接抓取该页面，再交给 `tools/nuxt_payload.js` 在 vm 沙箱中求值解析。

产出
----
    data/raw/pfasstruct_v6.html     原始页面（断点续传，21.6 MB）
    data/raw/pfasstruct_v6.json     解析后的记录数组
    data/pfasstruct_v6.csv          规范化结构表（含标准 SMILES、结构标志）
    data/reports/pfasstruct_v6_qc.json  QC 报告

用法
----
    python fetch_pfasstruct.py                      # 全流程
    python fetch_pfasstruct.py --seed <部分文件>     # 用已有部分下载做续传种子
    python fetch_pfasstruct.py --skip-download      # 已有 HTML，只解析
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "raw")
REPORTS = os.path.join(DATA, "reports")
TOOLS = os.path.join(HERE, "tools")

URL = "https://comptox.epa.gov/dashboard/chemical-lists/PFASSTRUCTV6"
HTML_PATH = os.path.join(RAW, "pfasstruct_v6.html")
JSON_PATH = os.path.join(RAW, "pfasstruct_v6.json")
CSV_PATH = os.path.join(DATA, "pfasstruct_v6.csv")
QC_PATH = os.path.join(REPORTS, "pfasstruct_v6_qc.json")
EXPECTED_N = 21028  # 官方页面标注的 V6 条数

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# 某些环境 schannel 吊销检查失败 → 统一用不校验证书的上下文（本机实测需要）
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

# Node.js is used only to evaluate the site's Nuxt payload (see tools/nuxt_payload.js).
# Resolved in this order so that the script stays platform independent:
#   1. the PFAS_NODE environment variable, if set (use this to pin a specific binary);
#   2. `node` on PATH.
NODE_ENV_VAR = "PFAS_NODE"


def find_node() -> str:
    override = os.environ.get(NODE_ENV_VAR, "").strip()
    if override:
        if os.path.exists(override) or shutil.which(override):
            return override
        raise RuntimeError(
            f"{NODE_ENV_VAR}={override!r} was set but is not an executable; "
            "unset it to fall back to `node` on PATH")
    found = shutil.which("node")
    if found:
        return found
    raise RuntimeError(
        "Node.js was not found on PATH. Install Node.js >= 18, or point "
        f"{NODE_ENV_VAR} at the node executable. Node is required only to "
        "recover the inventory from the published Nuxt payload; if you already "
        "have data/raw/pfasstruct_v6.json, this step is skipped.")


# ------------------------------------------------------------------ 断点续传下载
def remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": UA, "Accept": "*/*"})
        with urllib.request.urlopen(req, timeout=60, context=_CTX) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception as e:
        print(f"    [warn] HEAD 失败：{str(e)[:80]}", flush=True)
        return None


def download(url: str, dest: str, max_rounds: int = 20, seed: str | None = None) -> str:
    """下载 PFASSTRUCT V6 页面。

    实测（2026-10-09）：
      * 服务端 **不支持 Range**（带 Range 头仍返回 200 + 全量），因此无法断点续传；
        改为「每轮从头下载 + 失败重试」。
      * 服务端 **支持 gzip**（显式带 Accept-Encoding: gzip 时返回 Content-Encoding: gzip），
        21.6 MB 的 HTML 压缩后显著变小，是可行性的关键。
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if seed and os.path.exists(seed):
        print(f"    [seed] 忽略种子 {os.path.getsize(seed) / 1e6:.2f} MB（服务端不支持续传）",
              flush=True)

    total = remote_size(url)
    print(f"    [info] 远端 Content-Length（未压缩）= "
          f"{total if total else '未知'}", flush=True)

    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"    [cache] 已存在完整文件（{os.path.getsize(dest) / 1e6:.2f} MB），跳过",
              flush=True)
        return dest

    tmp = dest + ".gz"
    for rnd in range(1, max_rounds + 1):
        try:
            hdr = {"User-Agent": UA, "Accept": "*/*", "Accept-Encoding": "gzip"}
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=300, context=_CTX) as r:
                ce = (r.headers.get("Content-Encoding") or "").lower()
                cl = r.headers.get("Content-Length")
                print(f"    [round {rnd}] encoding={ce or 'identity'} "
                      f"content-length={cl}", flush=True)
                with open(tmp, "wb") as f:
                    got = 0
                    while True:
                        chunk = r.read(1 << 18)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        if got % (2 << 20) < (1 << 18):
                            print(f"      … {got / 1e6:.2f} MB", flush=True)
            raw = open(tmp, "rb").read()
            if raw[:2] == b"\x1f\x8b":
                import gzip as _gz
                html = _gz.decompress(raw)
                print(f"    [gzip] {len(raw) / 1e6:.2f} MB -> {len(html) / 1e6:.2f} MB",
                      flush=True)
            else:
                html = raw
            if total and len(html) < total * 0.9:
                raise RuntimeError(f"解压后 {len(html)} 与预期 {total} 差距过大")
            with open(dest, "wb") as f:
                f.write(html)
            print(f"    [ok] {os.path.getsize(dest) / 1e6:.2f} MB -> {dest}", flush=True)
            try:
                os.remove(tmp)
            except OSError:
                pass
            return dest
        except Exception as e:
            print(f"    [round {rnd}] 失败：{str(e)[:100]}；5s 后重试", flush=True)
            time.sleep(5)
    raise RuntimeError("下载 PFASSTRUCT V6 失败（已达最大重试次数）")


# ------------------------------------------------------------------ 解析
def run_node_extract(html: str, out_json: str, key: str = "chemicals") -> list[dict]:
    node = find_node()
    js = os.path.join(TOOLS, "nuxt_payload.js")
    print(f"    [node] {os.path.basename(node)} {js}")
    p = subprocess.run([node, js, html, out_json, "--key", key],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.stderr:
        print("    " + p.stderr.strip()[:600])
    if p.returncode != 0:
        raise RuntimeError(f"node 解析失败：{p.stdout[-500:]} {p.stderr[-500:]}")
    with open(out_json, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ 规范化
COL_MAP = {
    "dtxsid": "dtxsid",
    "dtxcid": "dtxcid",
    "casrn": "casrn",
    "preferredName": "preferred_name",
    "iupacName": "iupac_name",
    "smiles": "smiles_raw",
    "qsarReadySmiles": "smiles_qsar_ready",
    "msReadySmiles": "smiles_ms_ready",
    "molFormula": "mol_formula",
    "averageMass": "average_mass",
    "monoisotopicMass": "monoisotopic_mass",
    "pubchemCid": "pubchem_cid",
    "compoundId": "compound_id",
    "genericSubstanceId": "generic_substance_id",
    "inchikey": "inchikey",
    "activeAssays": "active_assays",
    "totalAssays": "total_assays",
    "toxcastSelect": "toxcast_select",
    "hasStructureImage": "has_structure_image",
    "multicomponent": "multicomponent",
    "wikipediaArticle": "wikipedia_article",
}


def to_records(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        d = {}
        for k_src, k_dst in COL_MAP.items():
            v = r.get(k_src)
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            d[k_dst] = v
        recs.append(d)
    df = pd.DataFrame(recs)
    # 布尔/数值清洗
    for c in ("average_mass", "monoisotopic_mass", "pubchem_cid", "compound_id",
              "generic_substance_id", "active_assays", "total_assays"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("toxcast_select", "has_structure_image", "multicomponent"):
        if c in df.columns:
            df[c] = df[c].apply(
                lambda v: True if v is True else (False if v is False else pd.NA))
    return df


def main():
    ap = argparse.ArgumentParser(description="获取并解析 PFASSTRUCT V6")
    ap.add_argument("--seed", default=None, help="已有部分下载文件，用作续传种子")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force", action="store_true", help="忽略缓存重新解析")
    args = ap.parse_args()

    for d in (RAW, REPORTS):
        os.makedirs(d, exist_ok=True)

    print("[1/4] 下载 PFASSTRUCT V6 页面")
    if args.skip_download:
        if not os.path.exists(HTML_PATH):
            sys.exit(f"缺少 {HTML_PATH}，无法 --skip-download")
        print(f"    [skip] 使用已有 {os.path.getsize(HTML_PATH) / 1e6:.2f} MB")
    else:
        download(URL, HTML_PATH, seed=args.seed)

    print("[2/4] 解析 Nuxt 载荷")
    if os.path.exists(JSON_PATH) and not args.force:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            rows = json.load(f)
        print(f"    [cache] 复用 {JSON_PATH}（{len(rows)} 条）")
    else:
        rows = run_node_extract(HTML_PATH, JSON_PATH)
    print(f"    [ok] 解析出 {len(rows)} 条记录")

    print("[3/4] 规范化 + RDKit 标准化")
    df = to_records(rows)
    n_raw = len(df)

    sys.path.insert(0, HERE)
    from pfas_qsar import standardize_smiles, structural_flags

    # 优先用 qsarReadySmiles（已做盐/电荷处理的 QSAR 就绪结构），退回 smiles
    def pick_smiles(r):
        for c in ("smiles_qsar_ready", "smiles_raw", "smiles_ms_ready"):
            v = r.get(c)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    df["smiles_input"] = [pick_smiles(r) for r in df.to_dict("records")]
    df["smiles"] = [standardize_smiles(s) if s else None for s in df["smiles_input"]]
    n_bad = int(df["smiles"].isna().sum())

    n_before = len(df)
    df = df[df["smiles"].notna()].copy()
    n_dup = int(df["smiles"].duplicated().sum())
    df = df.drop_duplicates(subset=["smiles"], keep="first").reset_index(drop=True)

    flags = pd.DataFrame([structural_flags(s) for s in df["smiles"]])
    df = pd.concat([df, flags], axis=1)

    # 化学类别粗标签（用于报告分层）
    def cls(row):
        if row.get("headgroup") == 1:
            return "PFCAs" if row.get("n_rings", 0) == 0 else "含环羧酸"
        if row.get("headgroup") == 2:
            return "PFSAs" if row.get("n_rings", 0) == 0 else "含环磺酸"
        if row.get("headgroup") == 3:
            return "磺酰胺/磺酰类"
        if (row.get("n_ether_O") or 0) > 0:
            return "醚类(未检出酸性头基)"
        if (row.get("n_F") or 0) > 0:
            return "其他含氟"
        return "未分类"

    df["chem_class"] = df.apply(cls, axis=1)

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    qc = {
        "source_url": URL,
        "list_version": "PFASSTRUCTV6 (January 2026)",
        "expected_n_official": EXPECTED_N,
        "n_parsed": n_raw,
        "n_smiles_missing": n_bad,
        "n_duplicate_smiles": n_dup,
        "n_final": int(len(df)),
        "html_bytes": os.path.getsize(HTML_PATH),
        "csv": os.path.relpath(CSV_PATH, HERE),
        "fields_available": sorted(set().union(*[set(r.keys()) for r in rows[:200]])),
    }
    with open(QC_PATH, "w", encoding="utf-8") as f:
        json.dump(qc, f, ensure_ascii=False, indent=2)

    print("[4/4] QC")
    for k, v in qc.items():
        if k != "fields_available":
            print(f"    {k}: {v}")
    print(f"\n完成 -> {CSV_PATH}（{len(df)} 条）")


if __name__ == "__main__":
    main()
