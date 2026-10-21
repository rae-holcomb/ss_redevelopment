#!/bin/zsh
# T2 baselines: dev_small and dev_medium, each with the manifest's assigned coverage and full-year
# (--ignore-manifest-sectors). Local data only (data/smarts_dev); never scores the holdout.
# Resumable: rerun to continue (scorecard --resume). Outputs in runs/t2_<set>_<mode>/.
set -e
cd "${0:A:h}/.."
PY=/opt/anaconda3/envs/ss_openblas/bin/python
for set in dev_small dev_medium; do
  $PY benchmarks/scorecard.py --manifest datasets/$set.csv --data-root data/smarts_dev \
      --n-workers 4 --resume --outdir runs/t2_${set}_assigned --label "${set} (assigned coverage)"
  $PY benchmarks/scorecard.py --manifest datasets/$set.csv --data-root data/smarts_dev \
      --ignore-manifest-sectors --n-workers 4 --resume --outdir runs/t2_${set}_fullyear \
      --label "${set} (full year)"
done
