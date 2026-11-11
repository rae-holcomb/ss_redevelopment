#!/bin/zsh
# T5 short-period recovery: after-fix scorecards (dev_medium full-year + assigned), then the
# pool-comparison runs (rotators both modes, dev_medium non-rotator negatives on the drive).
# Resumable: rerun the same command after an interruption. ~4 h at 4 workers on the i5.
set -e
cd "${0:A:h}/.."
PY=/opt/anaconda3/envs/ss_openblas/bin/python
NW=${N_WORKERS:-4}
DM="--manifest datasets/dev_medium.csv --data-root data/smarts_dev"
$PY benchmarks/scorecard.py ${=DM} --ignore-manifest-sectors --outdir runs/t5_after_fullyear --n-workers $NW --resume
$PY benchmarks/scorecard.py ${=DM} --outdir runs/t5_after_assigned --n-workers $NW --resume
ROT=base,base+highpass,base+acf_short,base+ls_short
$PY tools/t5_pool_compare.py ${=DM} --ignore-manifest-sectors --outdir runs/t5_pool/rot_fy --configs $ROT --n-workers $NW --resume
$PY tools/t5_pool_compare.py ${=DM} --outdir runs/t5_pool/rot_asg --configs $ROT --n-workers $NW --resume
ROOT=${SMARTS_DATA_ROOT:-/Volumes/Data_Vault/smarts_data}
$PY tools/t5_pool_compare.py --manifest datasets/negatives_v0.csv --only-set dev_medium --data-root $ROOT \
    --outdir runs/t5_pool/neg --configs base_old,$ROT --n-workers $NW --resume
