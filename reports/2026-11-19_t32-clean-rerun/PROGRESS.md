# T32 clean rerun — progress log

- 2026-11-19: Stage-2 harness `tools/stage2_harness.py` (cache Stage 1, refit per variant). Reproduces
  `_v7` (local_screen2) per star on dev_medium_clean, both modes: 0/874 outcome or decline-gate diffs, max |dP|/P 1.6e-16.
- `local` via harness: fy success 0.264 / wrong 0.038 (v7 0.254 / 0.038); assigned 0.152 / 0.038 (v7 0.146 / 0.029).
  Transitions match T31's changed_stars filtered to clean stars exactly (fy 21 gains, 9 none->wrong; as 10 / 11).
- Step 1 done: `data/vertex_residuals.csv`, `figures/fig_vertex_residual_vs_n.png`.
  Gains: early teeth tight on the line, late teeth walk out of the seed windows (seed P0 off median 2.9% vs 1.5% shared).
  none->wrong: low acceptance and large scatter already at n = 1-3.
- Reference scorecards (runs/t32_clean/sc/): local, local_screen2 fy/as/syn. Harness = scorecard per star (0 diffs).
- 15 T32 variants + local_seq_trim / local_trim screened (data/screen_summary.csv, screen_by_bin.csv).
  Leader local_seq_trim: fy 0.251 / wrong 0.021, as 0.142 / 0.026 (v7 0.254/0.038, 0.146/0.029).
- Official scorecards local_seq_trim, local_seq, local_trim fy/as/syn: dev_medium check_goals PASS vs clean v7;
  synthetic flags a 3-of-36-star 3-10d drop for local_seq_trim (local itself flags >30d, 1 of 8).
- Lost-star diagnosis: mean_height_snr on marginal long-period combs (data/leader_lost_stars.csv, fig_leader_lost_gates.png).
- Next: profiling (runs/t32_clean/prof/), example figures, speed figure, README.
- Profiling, figures, README, notebook (executed), docs updated. Report done 2026-11-19.
