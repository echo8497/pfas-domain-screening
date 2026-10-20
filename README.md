# pfas-domain-screening

Code and derived tables for the manuscript

> **Criterion-dependent applicability-domain assessment of PFAS bioactivity machine learning reveals gaps in replacement-chemical space**

The study screens all 19,767 unique structures of the United States Environmental
Protection Agency **PFASSTRUCT V6** suspect list against a 26-endpoint PFAS
bioactivity panel (1,012 substances), measures how far the panel's applicability
domain (AD) extends over replacement chemistry under five different criteria, and
releases a tiered read-across priority list of candidate replacement-space chemicals.

This repository is the software artefact referenced under *Availability of data
and materials* in the manuscript. It contains the analysis pipeline, the derived
tables and the figure-generation script. It does **not** contain a copy of the
third-party input datasets; those are retrieved from their published sources (see
`DATA_SOURCES.md`).

---

## 1. What the repository establishes

| Claim in the manuscript | Reproduced by |
|---|---|
| 19,767 unique structures from 21,028 PFASSTRUCT V6 records | `fetch_pfasstruct.py` |
| Random splitting overestimates performance for 21 of 22 modellable endpoints (mean ΔAUROC +0.069) | `run_pipeline.py --multi-task` |
| Label permutation confirms genuine signal (primary endpoint, p = 0.0196) | `run_pipeline.py --multi-task` (permutation block) |
| AD verdicts are criterion-dependent: kNN 17/20, range 20/20, leverage 4/20, Mahalanobis 3/20, class coverage 0/20, consensus 0/20 | `screen_domain.py` + `analyze_screen.py` |
| Only 6 of 1,012 panel substances (0.6%) occupy replacement chemical space, versus 2,468 of 19,767 inventory structures (12.5%) | `screen_alternatives.py` |
| 2,390 candidate alternatives (Tier I 100; 239 regrettable-substitution flags) | `screen_alternatives.py` + `analyze_alternatives.py` |
| External QMSA descriptor dataset reproduces the effect (Δ = +0.049, p = 0.0099) | `run_published_qmsa.py` |

## 2. Requirements

* **Python ≥ 3.13** with the pinned packages in `requirements.txt`.
* **Node.js ≥ 18** — optional, and needed only to evaluate the published Nuxt
  payload when re-retrieving the inventory from the EPA dashboard. Set
  `PFAS_NODE=/path/to/node` to pin a specific binary; otherwise `node` is taken
  from `PATH`. If `data/pfasstruct_v6.csv` is already present, no Node.js step
  runs.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verified on Linux and Windows. No step requires registration, login or
acceptance of any licence terms beyond those of the dependencies.

## 3. Repository layout

```
.
├── fetch_pfasstruct.py         retrieve and normalise the PFASSTRUCT V6 inventory
├── screen_domain.py            five-criterion AD screening of the inventory
├── screen_alternatives.py      four-region chemical-space partition + read-across list
├── analyze_screen.py           screening tables and figures
├── analyze_alternatives.py     candidate-list statistics and family composition
├── run_pipeline.py             leakage-aware modelling benchmark (single/multi endpoint)
├── run_published_qmsa.py       external validation on a published descriptor dataset
├── pfas_qsar.py                core library: fingerprints, splits, models, AD, metrics
├── qmsa.py                     quantitative molecular surface analysis descriptors
├── pfas_anchors.py             the 20 regulated/marketed anchor substances
├── make_submission_figs.py     builds the eight manuscript figures
├── rerun_all.sh                end-to-end reproduction in one command
├── data/
│   ├── pfasstruct_v6.csv       normalised inventory (public EPA list; see DATA_SOURCES.md)
│   ├── pfas_chengng.csv        bioactivity panel — NOT shipped; see DATA_SOURCES.md
│   └── raw/                    published QMSA descriptors — NOT shipped
├── results_screen_v6/          screening outputs, incl. per-structure AD verdicts
├── results_alternatives/       chemical-space partition + the 2,390-candidate list
├── results_real_multitask/     multi-endpoint leakage benchmark
├── results_real_PCBA686978/    single-endpoint benchmark + QMSA ablation
├── results_multitask_3splits/  multi-endpoint benchmark under all three splits
├── results_published_qmsa*/    external-dataset validation
├── results_audit/              provenance and consistency audits
├── tools/                      table builders, provenance checkers, payload parser
├── figures/                    Fig. 1–8 (600 dpi, composite single files)
└── supplementary_tables/       Additional files 1–12 as shipped to the journal
```

## 4. Quick start

Two settings differ between the screening and the modelling analyses, and both are
deliberate:

* **Screening and AD analysis** use `min_pos = 30`, giving 12 modellable endpoints.
* **Modelling benchmark and leakage analysis** use `min_pos = 15`, giving 22
  modellable endpoints.

Reporting a number from one configuration as if it came from the other is the
single easiest mistake to make with this code, so every run writes its
configuration to `run_meta.json` next to its outputs.

### 4.1 Inventory and applicability-domain screening

```bash
python fetch_pfasstruct.py                  # -> data/pfasstruct_v6.csv
python screen_domain.py                     # -> results_screen_v6/
python analyze_screen.py                    # -> screening tables and figures
```

`screen_domain.py` deliberately computes all structural flags from SMILES rather
than reading them from the inventory CSV, so the panel and the inventory are
always characterised identically. See `pfas_qsar.structural_flags()` for the
perfluorinated-carbon-count definition and why it matters.

### 4.2 Chemical-space partition and candidate list

```bash
python screen_alternatives.py               # -> results_alternatives/
python analyze_alternatives.py
```

### 4.3 Leakage benchmark and external validation

```bash
python run_pipeline.py --input data/pfas_chengng.csv \
    --multi-task --n-perm 50 --perm-endpoint PCBA-686978 \
    --outdir results_multitask_3splits

python run_pipeline.py --input data/pfas_chengng.csv \
    --endpoint PCBA-686978 --qmsa --ablation --sigma-profile \
    --models LR,RF --folds 5 --repeats 3 --n-perm 50 \
    --outdir results_real_PCBA686978

python run_published_qmsa.py --outdir results_published_qmsa
```

### 4.4 Additional files and figures

```bash
python tools/make_supplementary_tables.py   # Additional files 1-3
python tools/build_s7_s9_tables.py          # Additional files 7, 9, 10
python tools/build_s10_s11_tables.py        # Additional files 11, 12
python tools/build_s4_s6_tables.py          # Additional files 4, 6
python make_submission_figs.py              # figures/Fig1..Fig8
```

### 4.5 Everything at once

```bash
bash rerun_all.sh
```

## 5. Third-party data

Two input datasets are not covered by this repository's licence and are therefore
not shipped: `data/pfas_chengng.csv` (the bioactivity panel) and the two QMSA
descriptor files under `data/raw/`. `DATA_SOURCES.md` records what each one is,
where it came from, the terms it carries, and how to place it so that the commands
above work unchanged.

## 6. Licence

The source code in this repository is released under the **MIT License**
(`LICENSE`). The licence covers the code and the derived tables; it does not
extend to the third-party inputs described in `DATA_SOURCES.md`.

## 7. Citation

See `CITATION.cff`. If you use the candidate list, please cite the manuscript and
the archived release (DOI in `CITATION.cff`).
