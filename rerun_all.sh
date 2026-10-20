#!/usr/bin/env bash
# End-to-end reproduction of the results in the accompanying manuscript.
#
#   bash rerun_all.sh
#   PYTHON=/usr/bin/python3.13 bash rerun_all.sh      # pin the interpreter
#
# Two configurations are kept apart on purpose (see REPRODUCE.md):
#   * AD screening and the anchor verdicts   -> min_pos = 30, 12 modellable endpoints
#   * modelling benchmark and permutation    -> min_pos = 15, 22 modellable endpoints
# Do not merge the two output trees; every run records its own configuration in
# run_meta.json.
set -u

PY="${PYTHON:-python}"
cd "$(dirname "$0")" || exit 1

PANEL="data/pfas_chengng.csv"
if [ ! -f "$PANEL" ]; then
  echo "ERROR: $PANEL not found."
  echo "       That dataset is not redistributed with this repository."
  echo "       See DATA_SOURCES.md for where to obtain it."
  exit 1
fi

rc_total=0
step () {                       # step <n> <label> <logfile> <command...>
  local n="$1"; shift
  local label="$1"; shift
  local log="$1"; shift
  echo "########## $n  $label ##########"
  "$@" > "$log" 2>&1
  local rc=$?
  echo "           exit=$rc  ->  $log"
  if [ $rc -ne 0 ]; then rc_total=$((rc_total + 1)); fi
}

# ---- inventory (skipped if already present; pass --force to re-retrieve) -----
if [ -f data/pfasstruct_v6.csv ]; then
  echo "########## 1  inventory already present, skipping fetch ##########"
else
  step 1 "retrieve PFASSTRUCT V6 inventory" logs_fetch_v6.txt \
       "$PY" fetch_pfasstruct.py
fi

# ---- AD screening (min_pos = 30 -> 12 endpoints) -----------------------------
step 2 "apply five-criterion AD screening"        logs_screen_v6.txt \
     "$PY" screen_domain.py
step 3 "screening tables and figures"             logs_screen_v6.txt \
     "$PY" analyze_screen.py

# ---- chemical-space partition and candidate list ----------------------------
step 4 "chemical-space partition + read-across"   logs_alternatives.txt \
     "$PY" screen_alternatives.py
step 5 "candidate-list statistics"                logs_alternatives.txt \
     "$PY" analyze_alternatives.py

# ---- leakage benchmark (min_pos = 15 -> 22 endpoints) -----------------------
step 6 "multi-endpoint benchmark, three splits"   logs_multitask_3splits.txt \
     "$PY" run_pipeline.py --input "$PANEL" --multi-task \
     --n-perm 50 --perm-endpoint PCBA-686978 --outdir results_multitask_3splits
step 7 "single-endpoint benchmark + QMSA ablation" logs_main_PCBA686978.txt \
     "$PY" run_pipeline.py --input "$PANEL" --endpoint PCBA-686978 \
     --qmsa --ablation --sigma-profile --models LR,RF \
     --folds 5 --repeats 3 --n-perm 50 --outdir results_real_PCBA686978

# ---- external-dataset validation --------------------------------------------
step 8 "external QMSA dataset validation"         logs_published_qmsa.txt \
     "$PY" run_published_qmsa.py --outdir results_published_qmsa

# ---- additional files (pure re-projections; cheap, run last) ----------------
step 9  "additional files 1-3"                    logs_supp_tables.txt \
      "$PY" tools/make_supplementary_tables.py
step 10 "additional files 4, 6"                   logs_supp_tables.txt \
      "$PY" tools/build_s4_s6_tables.py
step 11 "additional files 7, 9, 10"               logs_supp_tables.txt \
      "$PY" tools/build_s7_s9_tables.py
step 12 "additional files 11, 12"                 logs_supp_tables.txt \
      "$PY" tools/build_s10_s11_tables.py

# ---- figures -----------------------------------------------------------------
step 13 "manuscript figures 1-8"                  logs_figures.txt \
      "$PY" make_submission_figs.py

# ---- consistency audit -------------------------------------------------------
step 14 "candidate-count audit"                   logs_audit.txt \
      "$PY" tools/audit_candidate_counts.py
step 15 "inventory provenance check"              logs_audit.txt \
      "$PY" tools/check_pfasstruct_provenance.py

echo "################################################################"
if [ $rc_total -eq 0 ]; then
  echo "ALL STEPS COMPLETED"
else
  echo "$rc_total STEP(S) FAILED - inspect the logs listed above"
fi
exit $rc_total
