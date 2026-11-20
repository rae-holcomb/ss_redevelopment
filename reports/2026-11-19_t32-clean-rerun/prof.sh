#!/bin/zsh
PY=/opt/anaconda3/envs/ss_openblas/bin/python
cd /Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/t32-clean-rerun-2c0779
mkdir -p runs/t32_clean/prof
for v in "$@"; do
  memory_pressure -Q | tail -1
  echo "=== $v start $(date +%T)"
  SS_COMB_VARIANT=$v nice -n 15 $PY benchmarks/profile_pipeline.py --manifest datasets/dev_medium_clean.csv --data-root data/smarts_dev \
    --ignore-manifest-sectors --star-ids-file benchmarks/profile_stars.txt --features --repeats 3 \
    --out runs/t32_clean/prof/$v.csv --summary-json runs/t32_clean/prof/$v.json > runs/t32_clean/prof/$v.log 2>&1
  echo "=== $v done $(date +%T) rc=$?"
  sleep 60
done
