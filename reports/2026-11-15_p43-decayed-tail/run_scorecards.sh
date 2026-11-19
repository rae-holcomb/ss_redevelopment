#!/bin/zsh
# P43: scorecards for each comb variant given as an argument, from a frozen snapshot of the committed HEAD
# (so code edits during a run can't leak in). One heavy job at a time: nice 15, 3 workers, 2-min gaps,
# --resume with stable folders runs/p43/sc/<variant>_<tag>. check_goals vs the _v7 baselines AND vs the
# local_tol reference runs (runs/p43/sc/local_tol_*); commit + push after each variant.
# Usage: zsh reports/2026-11-15_p43-decayed-tail/run_scorecards.sh [--modes "dmfy dmas syn"] variant ...
cd "$(dirname $0)/../.."
WT=$PWD
PY=/opt/anaconda3/envs/ss_openblas/bin/python
REP=reports/2026-11-15_p43-decayed-tail
modes=(dmfy dmas)
if [[ $1 == --modes ]]; then modes=(${=2}); shift 2; fi
SHA=$(git rev-parse --short HEAD)
SRC=$WT/runs/p43/src_$SHA
if [[ ! -d $SRC ]]; then
  mkdir -p $SRC && git archive HEAD | tar -x -C $SRC
  mkdir -p $SRC/data && ln -s /Users/raeholcomb/Documents/GitHub/ss_redevelopment/data/smarts_dev $SRC/data/smarts_dev
fi
mkdir -p runs/p43/sc $REP/data/check_goals $REP/data/metrics
for v in "$@"; do
  for tag in $modes; do
    case $tag in
      dmfy) src=(--manifest datasets/dev_medium.csv --data-root data/smarts_dev --ignore-manifest-sectors); base=dev_medium_fullyear_v7;;
      dmas) src=(--manifest datasets/dev_medium.csv --data-root data/smarts_dev); base=dev_medium_assigned_v7;;
      syn)  src=(--synthetic 200 --seed 0); base=synth200_s0_v7;;
    esac
    out=$WT/runs/p43/sc/${v}_${tag}
    if [[ ! -f $out/metrics.json ]]; then
      echo "== $(date +%H:%M) $v $tag @ $SHA | $(memory_pressure -Q | tail -1) | therm: $(pmset -g therm | grep -c 'No thermal warning') ok"
      mkdir -p $out
      (cd $SRC && SS_COMB_VARIANT=$v nice -n 15 $PY benchmarks/scorecard.py $src --n-workers 3 --outdir $out --resume) \
        > $out.log 2>&1 || { echo "FAILED $v $tag"; tail -5 $out.log; continue; }
      echo "== $(date +%H:%M) done $v $tag; cooling 120 s"
      sleep 120
    else
      echo "skip $out (done)"
    fi
    $PY benchmarks/check_goals.py $out/metrics.json --baseline benchmarks/baselines/$base.json \
      > $REP/data/check_goals/${v}_${tag}_vs_v7.txt 2>&1
    if [[ $v != local_tol && -f runs/p43/sc/local_tol_${tag}/metrics.json ]]; then
      $PY benchmarks/check_goals.py $out/metrics.json --baseline runs/p43/sc/local_tol_${tag}/metrics.json \
        > $REP/data/check_goals/${v}_${tag}_vs_local_tol.txt 2>&1
    fi
    cp $out/metrics.json $REP/data/metrics/${v}_${tag}.json
    grep -E "final.overall.success|OVERALL" $REP/data/check_goals/${v}_${tag}_vs_v7.txt
  done
  git add $REP/data/check_goals $REP/data/metrics $REP/PROGRESS.md 2>/dev/null
  git commit -qm "P43: scored $v ($modes)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q -u origin HEAD 2>&1 | tail -1
done
echo "ALL DONE $(date +%H:%M)"
