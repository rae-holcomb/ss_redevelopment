#!/bin/zsh
# P59: official scorecards of a comb variant (default local_tol_trim_l0x, cutoff from its spec) on
# dev_medium_clean (judging set) and dev_medium_b_clean (tuning set), full-year + assigned, and synth200 seed 0;
# check_goals --baseline vs the _v9 files. One heavy job at a time (nice 15, 3 workers, 2-min cooling, --resume).
# Usage: zsh reports/2026-11-20_p59-lag0-gates/run_scorecards.sh [variant]; MODES env selects runs.
cd "$(dirname $0)/../.."
PY=/opt/anaconda3/envs/ss_openblas/bin/python
REP=reports/2026-11-20_p59-lag0-gates
v=${1:-local_tol_trim_l0x2}
mkdir -p runs/p59/sc $REP/data/check_goals $REP/data/metrics $REP/data/per_star
modes=(${=MODES:-dmfy dmas bfy bas syn})
for tag in $modes; do
  case $tag in
    dmfy) args=(--manifest datasets/dev_medium_clean.csv --data-root data/smarts_dev --ignore-manifest-sectors); base=dev_medium_clean_fullyear_v9;;
    dmas) args=(--manifest datasets/dev_medium_clean.csv --data-root data/smarts_dev); base=dev_medium_clean_assigned_v9;;
    bfy) args=(--manifest datasets/dev_medium_b_clean.csv --data-root data/smarts_dev --ignore-manifest-sectors); base=dev_medium_b_clean_fullyear_v9;;
    bas) args=(--manifest datasets/dev_medium_b_clean.csv --data-root data/smarts_dev); base=dev_medium_b_clean_assigned_v9;;
    syn) args=(--synthetic 200 --seed 0); base=synth200_s0_v9;;
  esac
  out=runs/p59/sc/${v}_${tag}
  if [[ -f $out/metrics.json ]]; then echo "skip $out (done)"; continue; fi
  echo "== $(date +%H:%M) $v $tag | $(memory_pressure -Q | tail -1) | therm-warn-lines: $(pmset -g therm | grep -vc 'No ')"
  mkdir -p $out
  SS_COMB_VARIANT=$v nice -n 15 $PY benchmarks/scorecard.py $args --n-workers 3 --outdir $out --resume > $out.log 2>&1 || { echo "FAILED $v $tag"; tail -5 $out.log; continue; }
  $PY benchmarks/check_goals.py $out/metrics.json --baseline benchmarks/baselines/${base}.json > $REP/data/check_goals/${v}_${tag}.txt 2>&1
  cp $out/metrics.json $REP/data/metrics/${v}_${tag}.json
  gzip -9 -c $out/per_star.csv > $REP/data/per_star/${v}_${tag}.csv.gz 2>/dev/null
  grep -E "overall.success|OVERALL" $REP/data/check_goals/${v}_${tag}.txt | head -3
  echo "== $(date +%H:%M) done $v $tag; cooling 120 s"
  sleep 120
done
echo "ALL DONE $(date +%H:%M)"
