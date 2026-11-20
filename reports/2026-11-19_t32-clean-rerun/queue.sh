#!/bin/zsh
# usage: queue.sh variant:mode ...   mode in fy|as|syn
PY=/opt/anaconda3/envs/ss_openblas/bin/python
cd /Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/t32-clean-rerun-2c0779
for job in "$@"; do
  v=${job%%:*}; m=${job##*:}
  memory_pressure -Q | tail -1; pmset -g therm | grep -i speed
  out=runs/t32_clean/sc/${v}_${m}
  case $m in
    fy) args=(--manifest datasets/dev_medium_clean.csv --data-root data/smarts_dev --ignore-manifest-sectors);;
    as) args=(--manifest datasets/dev_medium_clean.csv --data-root data/smarts_dev);;
    syn) args=(--synthetic 200 --seed 0);;
  esac
  echo "=== $job start $(date +%T)"
  SS_COMB_VARIANT=$v nice -n 15 $PY benchmarks/scorecard.py $args --n-workers 3 --outdir $out --resume > $out.log 2>&1
  echo "=== $job done $(date +%T) rc=$?"; tail -2 $out.log
  sleep 120
done
