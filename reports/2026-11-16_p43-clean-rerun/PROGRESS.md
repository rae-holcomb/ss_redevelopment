# P43 clean rerun — progress
- 2026-11-16: started. Reference scorecards of branch default `local_tol` (fy, assigned, synth200) running.
- Reference runs done (fy/assigned PASS vs clean _v7; synth FAIL on >30d bin, 1 of 8 stars). Committed.
- Diagnosis (harness on the 58+27 changed stars, reproduces scorecard exactly): main loss mechanism is
  outlier rejection dropping the tallest early teeth (lag-0 tooth pinned at the comb position fits 2-5x worse
  than under lsq) -> mean_height_snr fails; frac_vertex_tol is the sole blocker for ~5 FY stars only.
- Variants added: local_tol_pos, local_tol_trim, local_tol_trim_pos, local_tol_cs(_trim), local_tol_cs0(_trim).
- Running full-dataset harness (all variants, both modes) -> runs/p43_clean/harness/all_{mode}.pkl
- Full-dataset harness screen done (data/screen_summary.csv). trim variants lift success above _v7; pos ~no effect;
  cs small. Official scorecards next: local_tol_trim, local_tol_cs, local_tol_cs_trim (fy, assigned, synth).
- Official scorecards done for local_tol_trim, local_tol_cs_trim, local_tol_cs (all PASS dev_medium vs _v7 and vs
  local_tol; synth FAIL only on the >30d star local_tol already loses). Profiling done. Report written (README,
  figures, executed notebook). Recommendation: local_tol_trim.
