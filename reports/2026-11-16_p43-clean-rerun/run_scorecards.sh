#!/bin/zsh
# P43 clean rerun: scorecards for each comb variant given as an argument, one heavy job at a time
# (nice 15, 3 workers, 2-min gaps, --resume), on dev_medium_clean full-year + assigned and synth200 seed 0;
# check_goals vs the clean _v7 baselines; commit + push after each variant.
# Usage: zsh reports/2026-11-16_p43-clean-rerun/run_scorecards.sh local_tol variantB ...
# MODES env var (default "fullyear assigned synth") selects which runs.
cd "$(dirname $0)/../.."
PY=/opt/anaconda3/envs/ss_openblas/bin/python
REP=reports/2026-11-16_p43-clean-rerun
mkdir -p runs/p43_clean/sc $REP/data/check_goals $REP/data/metrics $REP/data/per_star
modes=(${=MODES:-fullyear assigned synth})
for v in "$@"; do
  for mode in $modes; do
    case $mode in
      fullyear) tag=dmfy; args=(--manifest datasets/dev_medium_clean.csv --data-root data/smarts_dev --ignore-manifest-sectors); base=dev_medium_clean_fullyear_v7;;
      assigned) tag=dmas; args=(--manifest datasets/dev_medium_clean.csv --data-root data/smarts_dev); base=dev_medium_clean_assigned_v7;;
      synth) tag=syn; args=(--synthetic 200 --seed 0); base=synth200_s0_v7;;
    esac
    out=runs/p43_clean/sc/${v}_${tag}
    if [[ -f $out/metrics.json ]]; then echo "skip $out (done)"; continue; fi
    echo "== $(date +%H:%M) $v $mode | $(memory_pressure -Q | tail -1) | therm-warn-lines: $(pmset -g therm | grep -vc 'No ')"
    mkdir -p $out
    SS_COMB_VARIANT=$v nice -n 15 $PY benchmarks/scorecard.py $args --n-workers 3 --outdir $out --resume > $out.log 2>&1 || { echo "FAILED $v $mode"; tail -5 $out.log; continue; }
    $PY benchmarks/check_goals.py $out/metrics.json --baseline benchmarks/baselines/${base}.json > $REP/data/check_goals/${v}_${mode}.txt 2>&1
    cp $out/metrics.json $REP/data/metrics/${v}_${tag}.json
    cp $out/per_star.csv $REP/data/per_star/${v}_${tag}.csv 2>/dev/null
    grep -iE "success|OVERALL" $REP/data/check_goals/${v}_${mode}.txt | head -4
    echo "== $(date +%H:%M) done $v $mode; cooling 120 s"
    sleep 120
  done
  git add $REP 2>/dev/null
  git commit -qm "P43 clean rerun: scored $v (${modes})

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q -u origin HEAD 2>&1 | tail -1
done
echo "ALL DONE $(date +%H:%M)"
