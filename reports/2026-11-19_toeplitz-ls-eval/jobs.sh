#!/bin/zsh
# P50 runner: gate -> timing -> gate -> profile assigned -> gate -> profile full-year.
# A watcher logs any other session's heavy job seen while one of ours runs (contamination flag).
S=${0:A:h}; W=/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/p27-validation-scoping-4c1c1b
R=$W/reports/2026-11-19_toeplitz-ls-eval/data; PY=/opt/anaconda3/envs/ss_openblas/bin/python
pat="scorecard.py|stage2_harness|screen.sh|pytest"
IDS=smarts-tess-v1.0-000732,smarts-tess-v1.0-004513,smarts-tess-v1.0-017292,smarts-tess-v1.0-066625,smarts-tess-v1.0-201481
cd $W
step() {  # name, command...
  local name=$1; shift
  $S/gate.sh >> $S/jobs.log; echo "START $name $(date)" >> $S/jobs.log
  ( while true; do pgrep -f "$pat" >/dev/null && echo "CONTAMINATED $name $(date)" >> $S/jobs.log; sleep 10; done ) &
  local wpid=$!
  nice -n 15 "$@" >> $S/$name.log 2>&1; local rc=$?
  kill $wpid; echo "END $name rc=$rc $(date)" >> $S/jobs.log; return $rc
}
step timing $PY reports/2026-11-19_toeplitz-ls-eval/timing.py --generic $S/src/Toeplitz-LS/toeplitz-ls --native $S/src/Toeplitz-LS-native/toeplitz-ls || exit 1
step prof_as $PY benchmarks/profile_pipeline.py --manifest datasets/dev_small_clean.csv --data-root data/smarts_dev --star-ids $IDS --repeats 3 --out $R/profile_assigned.csv || exit 1
step prof_fy $PY benchmarks/profile_pipeline.py --manifest datasets/dev_small_clean.csv --data-root data/smarts_dev --star-ids $IDS --ignore-manifest-sectors --repeats 3 --out $R/profile_fullyear.csv || exit 1
echo ALLDONE >> $S/jobs.log
