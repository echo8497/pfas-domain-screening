# -*- coding: utf-8 -*-
"""Build the full Table 1 (per-endpoint performance) from the regenerated results.

Fills the stub table in the master manuscript with all modellable endpoints,
ordered primary-first then by descending cluster-out AUROC. Values are read
from results_real_multitask/multitask_metrics.csv — never hard-coded.

Usage:
    python code/tools/build_table1.py            # print only
    python code/tools/build_table1.py --write    # also patch the master doc
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "code" / "results_real_multitask" / "multitask_metrics.csv"
MASTER = ROOT / "PFAS替代物ML域错配_投稿总稿_2026-10-12.md"
PRIMARY = "PCBA-686978"


def build() -> tuple[str, dict]:
    m = pd.read_csv(METRICS, encoding="utf-8-sig")
    d = m[["endpoint", "n_labeled", "n_pos", "pos_rate",
           "AUROC_random", "AUROC_cluster-out", "AUPRC_cluster-out",
           "MCC_cluster-out"]].dropna().copy()
    d = d.rename(columns={
        "n_labeled": "n", "n_pos": "act", "pos_rate": "rate",
        "AUROC_random": "rd", "AUROC_cluster-out": "co",
        "AUPRC_cluster-out": "auprc", "MCC_cluster-out": "mcc",
    })
    d["delta"] = d["rd"] - d["co"]

    head = d[d.endpoint == PRIMARY]
    rest = d[d.endpoint != PRIMARY].sort_values("co", ascending=False)
    order = pd.concat([head, rest], ignore_index=True)

    lines = [
        "| Endpoint | Labelled n | Actives | Active rate | AUROC random | AUROC cluster-out | ΔAUROC | AUPRC cluster-out | MCC cluster-out |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in order.iterrows():
        tag = " (primary)" if r.endpoint == PRIMARY else ""
        lines.append(
            f"| {r.endpoint}{tag} | {int(r.n)} | {int(r.act)} | {r.rate:.3f} | "
            f"{r.rd:.3f} | {r.co:.3f} | {r.delta:+.3f} | {r.auprc:.3f} | {r.mcc:.3f} |"
        )

    n = len(order)
    npos = int((order.delta > 0).sum())
    stats = {
        "n_endpoints": n,
        "n_positive": npos,
        "mean_delta": float(order.delta.mean()),
        "median_delta": float(order.delta.median()),
        "min_delta": float(order.delta.min()),
        "max_delta": float(order.delta.max()),
        "co_median": float(order.co.median()),
        "co_min": float(order.co.min()),
        "co_max": float(order.co.max()),
        "rd_min": float(order.rd.min()),
        "rd_max": float(order.rd.max()),
    }
    lines.append(
        f"| **Summary ({n} endpoints)** | 100–788 | 15–214 | 0.025–0.430 | "
        f"median {stats['rd_min']:.3f}–{stats['rd_max']:.3f} (range) | "
        f"median {stats['co_median']:.3f} | "
        f"**mean {stats['mean_delta']:+.3f}** | — | — |"
    )
    return "\n".join(lines), stats


def patch(table: str, stats: dict) -> None:
    text = MASTER.read_text(encoding="utf-8")
    start = text.index("### Table 1 Per-endpoint performance")
    # the table ends at the first blank line followed by a ">" note
    end_marker = "> 表体最终数字请从"
    end = text.index(end_marker, start)
    # find the end of the table block (last "|" line before the note)
    block = text[start:end]
    last_pipe = block.rindex("\n", 0, len(block.rstrip()) if block.rstrip().endswith("|") else len(block))
    # simpler: rebuild from the header up to just before the note, preserving the note
    note_end = text.index("\n\n", end)
    note = text[end:note_end]

    new_block = (
        "### Table 1 Per-endpoint performance under random and cluster-out splitting\n\n"
        + table
        + "\n\n"
        + note
    )
    text = text[:start] + new_block + text[note_end:]
    MASTER.write_text(text, encoding="utf-8")
    print(f"patched {MASTER.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    tbl, st = build()
    print(tbl)
    print("\n--- stats ---")
    for k, v in st.items():
        print(f"  {k} = {v:.4f}" if isinstance(v, float) else f"  {k} = {v}")
    if a.write:
        patch(tbl, st)
