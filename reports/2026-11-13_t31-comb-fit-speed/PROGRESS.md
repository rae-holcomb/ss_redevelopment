# T31 progress log (running; crash-recovery notes)

- 02:20 restart after the 01:59 WindowServer hang killed run 1 (nothing committed). AC power, 56% charging.
  Keep-awake request denied by auto-mode classifier; relying on the app keeping the machine awake during the turn.
  Safety rules: `taskpolicy -b` for heavy jobs, 3 workers, memory/thermal check before each run, 2-min gaps,
  commit+push after Part A and after each scored variant.
- 02:20 load avg ~216 right after re-login (Spotlight/WindowServer storm, not ours); waiting for it to settle.
- 02:30 `taskpolicy -b` throttles to ~0.5 core even on an idle machine (pytest >10 min instead of 75 s); switched to
  `nice -n 15` (the brief's alternative).
- 02:40 dev_small memory check: peak RSS ~245 MB per worker, ~0.9 GB total at 3 workers -> 3 workers OK.
- 03:05 Part A needed two fixes: (1) Morlet kernel as one complex exp (in-process np.exp flips, T12's finding);
  (2) |c|^2 as re*re+im*im and a libm exp for the Gaussian peak fits: np.abs(complex) flipped in 4/12 fresh
  pytest processes. Fixed code stable in 12/12 processes. First queue (old fix) stopped at main_dmfy 896/1000.
- 03:41 Part A dev_medium done (runs in scratch): main vs Part A differ only in methods.wavelet.top1
  (fy 884 / 1,007,054 values; as 818 / 863,445), metrics.json 0 differ in both modes. Part A runs ~13.7 min (fy) /
  8.3 min (as) at 3 workers, nice 15.
- 03:45 Part B variants committed (bac443e). Running profile round 1, in-process repro before/after, and a Part A
  rerun pair (determinism proof), then the variant scorecards.
