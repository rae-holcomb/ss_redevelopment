#!/bin/zsh
# T32: official dev_medium scorecards (full-year + assigned) for each comb variant given as an argument,
# one heavy job at a time (nice 15, 3 workers, 2-min gaps, --resume), check_goals vs the _v7 baselines,
# and a commit + push of the check_goals/metrics files after each variant.
# Usage: zsh reports/2026-11-13_t32-local-drift/run_scorecards.sh local_track local_scatter ...
cd "$(dirname $0)/../.."
PY=/opt/anaconda3/envs/ss_openblas/bin/python
REP=reports/2026-11-13_t32-local-drift
mkdir -p runs/t32/sc $REP/data/check_goals $REP/data/metrics
for v in "$@"; do
  for mode in fullyear assigned; do
    if [[ $mode == fullyear ]]; then tag=dmfy; extra=(--ignore-manifest-sectors); else tag=dmas; extra=(); fi
    out=runs/t32/sc/${v}_${tag}
    if [[ -f $out/metrics.json ]]; then echo "skip $out (done)"; continue; fi
    echo "== $(date +%H:%M) $v $mode | $(memory_pressure -Q | tail -1) | $(pmset -g therm | grep -c 'warning level has been recorded') no-warning lines"
    SS_COMB_VARIANT=$v nice -n 15 $PY benchmarks/scorecard.py --manifest datasets/dev_medium.csv --data-root data/smarts_dev \
      $extra --n-workers 3 --outdir $out --resume > $out.log 2>&1 || { echo "FAILED $v $mode"; tail -5 $out.log; continue; }
    $PY benchmarks/check_goals.py $out/metrics.json --baseline benchmarks/baselines/dev_medium_${mode}_v7.json \
      > $REP/data/check_goals/${v}_${mode}.txt 2>&1
    cp $out/metrics.json $REP/data/metrics/${v}_${tag}.json
    grep -E "final.overall.success|OVERALL" $REP/data/check_goals/${v}_${mode}.txt
    echo "== $(date +%H:%M) done $v $mode; cooling 120 s"
    sleep 120
  done
  git add $REP/data/check_goals $REP/data/metrics $REP/PROGRESS.md 2>/dev/null
  git commit -qm "T32: scored $v on dev_medium (full-year + assigned)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
done
echo "ALL DONE $(date +%H:%M)"
