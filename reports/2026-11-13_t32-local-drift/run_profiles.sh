#!/bin/zsh
# T32: serial per-stage timing (benchmarks/profile_pipeline.py) on T12's 15 dev_medium stars, every comb
# variant, both modes, rounds interleaved across variants so slow-machine periods hit all variants alike.
# Usage: zsh reports/2026-11-13_t32-local-drift/run_profiles.sh <rounds> <variant> ...
cd "$(dirname $0)/../.."
PY=/opt/anaconda3/envs/ss_openblas/bin/python
IDS=$(for s in 008705 014308 009444 091212 151481 098694 029013 049025 030879 001810 004433 002102 218521 995392 311204; do printf "smarts-tess-v1.0-%s," $s; done)
IDS=${IDS%,}
mkdir -p runs/t32/prof
rounds=$1; shift
for r in $(seq 1 $rounds); do
  for v in "$@"; do
    for mode in fullyear assigned; do
      out=runs/t32/prof/prof_${v}_${mode}_r${r}.csv
      [[ -f $out ]] && continue
      if [[ $mode == fullyear ]]; then extra=(--ignore-manifest-sectors); else extra=(); fi
      SS_COMB_VARIANT=$v nice -n 15 $PY benchmarks/profile_pipeline.py --manifest datasets/dev_medium.csv \
        --data-root data/smarts_dev $extra --star-ids $IDS --out $out > /dev/null 2>&1 || echo "FAILED $v $mode r$r"
    done
  done
  echo "round $r done $(date +%H:%M)"
done
