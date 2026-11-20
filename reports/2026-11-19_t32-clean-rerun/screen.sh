#!/bin/zsh
# Stage-2 screening of every comb variant on the cached clean dev_medium inputs (both modes), one job at a time.
PY=/opt/anaconda3/envs/ss_openblas/bin/python
cd /Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/t32-clean-rerun-2c0779
for v in "$@"; do
  for m in fy as; do
    [[ -f runs/t32_clean/s2/${v}_${m}.csv ]] && continue
    nice -n 15 $PY tools/stage2_harness.py fit --cache runs/t32_clean/cache_$m --variant $v --out runs/t32_clean/s2/${v}_${m} | sed "s/^/$m /"
  done
done
