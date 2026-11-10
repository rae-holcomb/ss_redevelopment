#!/bin/zsh
# T10: build the ranker feature table for the full 3,000-star training sample (datasets/train_v0.csv).
# Needs the Data_Vault drive (inputs in extracted/train_v0/, outputs in extracted/t10_tables/).
# Resumable: rerun the same command after an unplug; finished stars are skipped.
# ~1.2-1.5 s/star of CPU per worker -> roughly 20-25 min per 1,000 stars at 4 workers, so ~65-75 min
# for all 3,000 (the 1,000-star nested_1k subset is already built if you reuse its shards, see below).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${SMARTS_DATA_ROOT:-/Volumes/Data_Vault/smarts_data}
[[ -d "$ROOT/tess_retry" ]] || { echo "Plug in the Data_Vault drive (or set SMARTS_DATA_ROOT)"; exit 1; }
OUT="$ROOT/extracted/t10_tables/train_3k"
# Reuse the 1k shards so only the other 2,000 stars run:
if [[ -d "$ROOT/extracted/t10_tables/train_1k/shards" && ! -d "$OUT/shards" ]]; then
  mkdir -p "$OUT/shards" && cp -n "$ROOT/extracted/t10_tables/train_1k/shards/"*.pkl "$OUT/shards/"
fi
/opt/anaconda3/envs/ss_openblas/bin/python tools/build_feature_tables.py \
  --manifest datasets/train_v0.csv --data-root "$ROOT" --outdir "$OUT" --n-workers 4 --resume
