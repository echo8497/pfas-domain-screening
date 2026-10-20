# Reproduction guide

End-to-end reproduction of every number in the manuscript. `bash rerun_all.sh`
runs the whole sequence; this file records what each step produces, how long it
takes, and the two settings that are easy to get wrong.

---

## 0. Before you start

Place the two third-party inputs as described in `DATA_SOURCES.md`:

```
data/pfas_chengng.csv
data/raw/PCBA-686978.csv
data/raw/CF_PCBA-686978.csv
```

The inventory at `data/pfasstruct_v6.csv` is shipped with the repository. To
re-retrieve it instead, run `python fetch_pfasstruct.py` (needs Node.js).

---

## 1. The two configurations, and why they must not be mixed

| Configuration | Setting | Modellable endpoints | Where it is used |
|---|---|---|---|
| Screening | `min_pos = 30` | 12 | AD screening, the five AD criteria, anchor verdicts, Table 2, Table 3 |
| Modelling | `min_pos = 15` | 22 | leakage benchmark, label permutation, Table 1, Fig. 4 |

A threshold of `min_pos` means "an endpoint is modellable only if it has at least
this many positives and at least this many negatives". Raising it from 15 to 30
drops ten endpoints, so a quantity computed under one setting is not
interchangeable with the same quantity computed under the other. Every run writes
its own `run_meta.json`, which is the authoritative record of which
configuration produced a given directory.

---

## 2. Steps

| # | Command | Output | Approx. runtime |
|---|---|---|---|
| 1 | `python fetch_pfasstruct.py` | `data/pfasstruct_v6.csv` (19,767 rows) | 2–5 min (network) |
| 2 | `python screen_domain.py` | `results_screen_v6/` — `screen_all.csv` (per-structure AD verdicts for all 19,767), `domain_coverage.csv`, `reference_pfas_predictions.csv`, `pca_coords.csv` | ~8 min |
| 3 | `python analyze_screen.py` | screening tables and figures | < 1 min |
| 4 | `python screen_alternatives.py` | `results_alternatives/` — chemical-space partition, the 2,390-candidate list, Tier I, regrettable substitutions | < 1 min |
| 5 | `python analyze_alternatives.py` | candidate-list statistics, family composition | < 1 min |
| 6 | `python run_pipeline.py --input data/pfas_chengng.csv --multi-task --n-perm 50 --perm-endpoint PCBA-686978 --outdir results_multitask_3splits` | `multitask_metrics.csv` (22 endpoints × 3 splitting strategies), `permutation_aurocs.csv`, `run_meta.json` | ~20 min |
| 7 | `python run_pipeline.py --input data/pfas_chengng.csv --endpoint PCBA-686978 --qmsa --ablation --sigma-profile --models LR,RF --folds 5 --repeats 3 --n-perm 50 --outdir results_real_PCBA686978` | single-endpoint metrics under all three splits, QMSA ablation table, σ-profile output | ~7 min |
| 8 | `python run_published_qmsa.py --outdir results_published_qmsa` | external-dataset validation, incl. its own permutation ledger | ~15 min |
| 9 | `python tools/make_supplementary_tables.py` | Additional files 1–3 (Tier I, regrettable, full candidate list) | < 1 min |
| 10 | `python tools/build_s7_s9_tables.py` | Additional files 7, 9, 10 | < 1 min |
| 11 | `python tools/build_s10_s11_tables.py` | Additional files 11, 12 | < 1 min |
| 12 | `python tools/build_s4_s6_tables.py` | Additional files 4, 6 | ~2 min |
| 13 | `python make_submission_figs.py` | `figures/Fig1`–`Fig8` (600 dpi, composite single files) | ~1 min |
| 14 | `python tools/fix_pfasstruct_derived_columns.py --apply` | repairs the inventory's derived columns (idempotent; a no-op on a corrected file) | ~2 min |

Steps 9–12 are pure re-projections of artefacts written by steps 2–8, so they cost
seconds and never need re-running unless an upstream artefact changes.

---

## 3. Checks worth running

Two scripts exist specifically to make a result falsifiable rather than merely
reported.

```bash
python tools/check_pfasstruct_provenance.py
```
Recomputes the inventory identity statistics quoted in the manuscript: DTXSID
coverage 100%, InChIKey 100%, CASRN 83.8%, ToxCast/Tox21 coverage 0.9%. These three
numbers are the anchors a reader can compare against the manuscript text without
running the modelling.

```bash
python tools/check_tier1_provenance.py
```
Verifies the identity of every Tier I candidate against the inventory record it was
drawn from.

```bash
python tools/audit_candidate_counts.py
```
Re-derives the removal chain 2,468 − (9 + 21 + 2 + 0 + 46) = 2,390 and prints the
per-step counts. The steps are an ordered, mutually exclusive set difference, so
summing the individual condition counts across the whole inventory gives a
different — and wrong — total.

```bash
python tools/fix_pfasstruct_derived_columns.py
```
Dry run. Reports stored-versus-recomputed agreement for the seven derived structural
columns of the inventory; useful as a general check that the shipped CSV has not
drifted.

---

## 4. Determinism

Seed handling is centralised: `pfas_qsar.RNG_SEED` fixes the random splits, the
Butina clustering and the label permutations; `screen_domain.py` uses its own
constant for the PCA. Two runs of the same command on the same inputs and the same
package versions reproduce the reported metrics to machine precision — verified by
re-running the multi-endpoint benchmark and the external-dataset validation and
comparing every numeric column (maximum absolute difference 0).

Across package versions, expect small differences in tree-model probabilities.
`requirements.txt` pins the versions used for the reported numbers.
