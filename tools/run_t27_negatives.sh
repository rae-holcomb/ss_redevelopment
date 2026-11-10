#!/bin/zsh
# T27: build the non-rotator negative feature tables (drive), then the FP analysis and ranker runs.
# Resumable: rerun the same command after an unplug. ~80 min at 4 workers on the i5.
#   zsh tools/run_t27_negatives.sh            # everything
#   N_WORKERS=2 zsh tools/run_t27_negatives.sh
set -e
cd "${0:A:h}/.."
PY=/opt/anaconda3/envs/ss_openblas/bin/python
ROOT=${SMARTS_DATA_ROOT:-/Volumes/Data_Vault/smarts_data}
[[ -d $ROOT/extracted/galaxy_stack ]] || { echo "Plug in the Data_Vault drive ($ROOT missing)"; exit 1; }
NW=${N_WORKERS:-4}
OUT=$ROOT/extracted/t27_negatives
[[ -f datasets/negatives_v0.csv ]] || $PY tools/build_negatives_manifest.py --data-root $ROOT   # ~15 min (slow drive reads)
$PY tools/audit/negatives_fairness.py --data-root $ROOT
for SET in dev_medium train; do
  $PY tools/build_feature_tables.py --nonrotator --manifest datasets/negatives_v0.csv --only-set $SET \
      --data-root $ROOT --outdir $OUT/$SET --n-workers $NW --resume
done
$PY tools/audit/negatives_false_positives.py --neg-train $OUT/train --neg-dev $OUT/dev_medium
$PY tools/train_ranker_negatives.py --neg-train $OUT/train --neg-dev $OUT/dev_medium
