#!/bin/zsh
# P43: serial per-stage timing (benchmarks/profile_pipeline.py, --features --repeats 3) on the fixed
# benchmarks/profile_stars.txt set, dev_medium full-year, each variant; rounds interleaved across variants.
# Usage: zsh reports/2026-11-15_p43-decayed-tail/run_profiles.sh <rounds> <variant> ...
cd "$(dirname $0)/../.."
PY=/opt/anaconda3/envs/ss_openblas/bin/python
mkdir -p runs/p43/prof
rounds=$1; shift
for r in $(seq 1 $rounds); do
  for v in "$@"; do
    out=runs/p43/prof/prof_${v}_r${r}.csv
    [[ -f $out ]] && continue
    SS_COMB_VARIANT=$v nice -n 15 $PY benchmarks/profile_pipeline.py --manifest datasets/dev_medium.csv \
      --data-root data/smarts_dev --ignore-manifest-sectors --star-ids-file benchmarks/profile_stars.txt \
      --features --repeats 3 --out $out > runs/p43/prof/prof_${v}_r${r}.log 2>&1 || echo "FAILED $v r$r"
  done
  echo "round $r done $(date +%H:%M)"
done
