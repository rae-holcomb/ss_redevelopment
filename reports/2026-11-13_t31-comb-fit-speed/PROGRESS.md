# T31 progress log (running; crash-recovery notes)

- 02:20 restart after the 01:59 WindowServer hang killed run 1 (nothing committed). AC power, 56% charging.
  Keep-awake request denied by auto-mode classifier; relying on the app keeping the machine awake during the turn.
  Safety rules: `taskpolicy -b` for heavy jobs, 3 workers, memory/thermal check before each run, 2-min gaps,
  commit+push after Part A and after each scored variant.
- 02:20 load avg ~216 right after re-login (Spotlight/WindowServer storm, not ours); waiting for it to settle.
