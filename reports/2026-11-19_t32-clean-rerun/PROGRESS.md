# T32 clean rerun — progress log

- 2026-11-19: Stage-2 harness `tools/stage2_harness.py` (cache Stage 1, refit per variant). Reproduces
  `_v7` (local_screen2) per star on dev_medium_clean, both modes: 0/874 outcome or decline-gate diffs, max |dP|/P 1.6e-16.
- `local` via harness: fy success 0.264 / wrong 0.038 (v7 0.254 / 0.038); assigned 0.152 / 0.038 (v7 0.146 / 0.029).
  Transitions match T31's changed_stars filtered to clean stars exactly (fy 21 gains, 9 none->wrong; as 10 / 11).
- Step 1 done: `data/vertex_residuals.csv`, `figures/fig_vertex_residual_vs_n.png`.
  Gains: early teeth tight on the line, late teeth walk out of the seed windows (seed P0 off median 2.9% vs 1.5% shared).
  none->wrong: low acceptance and large scatter already at n = 1-3.
