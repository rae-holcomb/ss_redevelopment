#!/bin/zsh
# wait until no heavy job from other sessions, then 120 s idle, then report memory/thermal
pat="scorecard.py|stage2_harness|screen.sh|pytest"
while true; do
  while pgrep -f "$pat" >/dev/null; do sleep 30; done
  sleep 120
  pgrep -f "$pat" >/dev/null || break
done
date; memory_pressure -Q | tail -1; pmset -g therm | grep -i "limit\|warning" 
