#!/bin/zsh
# P43 screening queue: waits for the reference scorecards, then the new test file, then the harness
# (dev_medium full-year, then assigned; Stage-1 cache built on the first pass), 2-min gaps, nice 15, 3 workers.
cd "$(dirname $0)/../.."
PY=/opt/anaconda3/envs/ss_openblas/bin/python
REP=reports/2026-11-15_p43-decayed-tail
until grep -q "ALL DONE" runs/p43/queue_ref.log 2>/dev/null; do sleep 20; done
sleep 120
echo "== $(date +%H:%M) pytest p43 | $(memory_pressure -Q | tail -1)"
nice -n 15 $PY -m pytest -q tests/test_p43_decayed_tail.py 2>&1 | tail -15
VS=local_tol,local_tol_pos,local_tol_tail,local_tol_norej,local_tol_norej_tail,local_tol_norej_pos,local_tol_tail_rh,local_tol_norej_rh
for mode in fullyear assigned; do
  out=runs/p43/harness/dev_medium_$mode.pkl
  [[ -f $out ]] && { echo "skip $out"; continue; }
  echo "== $(date +%H:%M) harness $mode | $(memory_pressure -Q | tail -1) | $(pmset -g therm | grep -c 'No thermal warning') therm-ok"
  nice -n 15 $PY $REP/run_harness.py $mode $out $VS 3 2>&1 | grep -v Warning
  echo "== $(date +%H:%M) done harness $mode; cooling 120 s"
  sleep 120
done
echo "HARNESS DONE $(date +%H:%M)"
