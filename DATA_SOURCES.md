# Data sources, terms and provenance

This repository does not redistribute its two third-party input datasets. Both are
publicly available and are retrieved from the sources below. Nothing in the
analysis requires a licence beyond the ones stated here.

---

## 1. PFASSTRUCT V6 chemical list — United States Environmental Protection Agency

| | |
|---|---|
| Source | CompTox Chemicals Dashboard, PFASSTRUCTV6 chemical list |
| URL | https://comptox.epa.gov/dashboard/chemical-lists/PFASSTRUCTV6 |
| List version | V6, released January 2026 |
| Accessed | 12 October 2026 |
| Records | 21,028 |
| Normalised output | `data/pfasstruct_v6.csv` — 19,767 unique structures after standardisation and de-duplication |
| Terms | US federal government work; no registration, login or licence acceptance required to retrieve or redistribute |

Retrieved by `fetch_pfasstruct.py`. The dashboard is a single-page application, so
the record list is recovered from the published Nuxt payload with
`tools/nuxt_payload.js` rather than by HTML scraping; `fetch_pfasstruct.py` also
accepts the site's own CSV export for a script-free cross-check.

**Provenance of the shipped copy.** `data/pfasstruct_v6.csv` carries seven derived
structural columns (`n_CF2`, `n_ether_O`, `n_rings`, `n_C`, `n_F`, `F_per_C`,
`headgroup`) alongside the retrieved fields. All seven are recomputable from
`smiles`, and the analysis code recomputes them on load rather than trusting the
stored values. The stored `n_CF2` column was nonetheless stale in the copy fetched
on 9 October 2026: an early implementation counted `[CX4](F)(F)` substructure
matches, which enumerates pairs of fluorine on the same carbon and therefore counts
a CF3 terminus three times. On all 19,767 rows the stored column had a median of 6
and a maximum of 96, against a true median of 3 and maximum of 60, agreeing on only
4,214 rows (21.3%).

The column was repaired on 13 October 2026 by
`tools/fix_pfasstruct_derived_columns.py`, which recomputes all seven columns from
SMILES; all seven now agree on 19,767 / 19,767 rows. **No reported result changes**,
because every consumer of these columns — `screen_domain.py`,
`screen_alternatives.characterize()` — already recomputed them from SMILES. The
repair matters for third parties reading the CSV directly, who would otherwise pick
up the stale value.

An auditor can confirm the diagnosis without any archived copy: re-fetch the
inventory with `fetch_pfasstruct.py`, then run
`python tools/fix_pfasstruct_derived_columns.py` (the default is a dry run). It
reports the per-column agreement between the stored values and a fresh computation,
and on a newly fetched copy it reproduces the 4,214 / 19,767 figure for `n_CF2` and
19,767 / 19,767 for the other six columns.

---

## 2. PFAS bioactivity label matrix — via its archived reconstruction

| | |
|---|---|
| Panel | 1,012 unique PFAS structures × 26 PubChem BioAssay (PCBA) endpoints |
| Source publication | Cheng W, Ng CA. *J. Chem. Inf. Model.* 2019 |
| Analysed archive | https://doi.org/10.5281/zenodo.22869943 |
| Archive contents | Decontaminated, leakage-controlled reconstruction of the panel, with baseline models |
| Archived by | Yang Z (Qinghai University), September 2026 — **not** the original authors' own deposit |
| Terms | GPL-3.0-or-later, as declared on the archive record |
| Expected local path | `data/pfas_chengng.csv` |

**Why an archive rather than the original deposit.** The archive exists precisely
because the original panel has a documented structure–label contamination problem,
which is one of the phenomena the accompanying manuscript quantifies. Using the
reconstruction keeps the analysis on a version whose provenance can be cited, and
attributing it to its actual depositor avoids the misattribution that would follow
from citing the original authors for a file they did not publish.

To place it:

```bash
# download the label matrix for PCBA-686978 from the Zenodo record above, then:
cp /path/to/downloaded/pfas_chengng.csv data/pfas_chengng.csv
```

---

## 3. QMSA descriptor dataset — external validation

| | |
|---|---|
| Source publication | Yin Z, Zhang M, Liu R, Cai Y. *Water Res.* 2025;280:123500 |
| Contents | Two structure-free descriptor files: quantitative molecular surface descriptors, an anonymous molecule identifier, and the binary label |
| Size | 1,029 molecules × 17 published descriptors |
| Expected local path | `data/raw/PCBA-686978.csv`, `data/raw/CF_PCBA-686978.csv` |

Used by `run_published_qmsa.py` for the secondary external-dataset validation. The
files carry no SMILES, so the cluster-out split is computed in the standardised
descriptor space rather than on fingerprints; this is stated in the manuscript's
Methods.

---

## 4. What this repository does redistribute

Everything under `figures/`, `results_*/` and `supplementary_tables/` is derived
output produced by the scripts here, and is covered by this repository's MIT
licence. `data/pfasstruct_v6.csv` is a normalised rendering of a US federal
government list (Section 1). No third-party data file is redistributed.
