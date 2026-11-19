# P43 progress log

- 2026-11-15 03:00 Started. Branch `claude/p43-decayed-tail-22a749` (cut from `claude/local-tol-default`, a4f348c).
- Step 1: reference scorecards for `local_tol` (dev_medium full-year, assigned, synthetic 200) launched from a frozen
  snapshot of HEAD (`runs/p43/src_<sha>`), via `run_scorecards.sh`.
- 03:05 local_tol dev_medium full-year: success 0.248, wrong 0.067 (= T32's local_tol). check_goals vs _v7 PASS.
- Diagnosis (`diag_lost2.py`, `diag_reject.py`, T32's 30 lost star-modes, single process):
  - In most losses the correct candidate already PASSES frac_vertex_tol; it fails mean_height_snr (or HL / fph).
  - Cause: the MAD residual-RMS window rejection in `_fit_single_candidate` drops the TALLEST teeth (n = 0, 1, 2)
    under the local solver (absolute residual RMS scales with peak height; once n = 0 goes, the MAD shrinks and
    the next tall tooth goes too). The remaining decayed teeth give mean height SNR < 1.
    E.g. 015438: local drops n=0,1,2 -> snr 0.39; local without rejection -> 2.41; lsq drops only n=1 -> 2.03.
  - Negative fitted heights are rare among the losses (comb-tied heights mostly positive), so (a) and (c) target
    a minor mode. The vertex tail pattern "vvv...." (trailing teeth with no free vertex) is common -> (b).
  - (b) as literally written ("after the decayed-tail trim") is already what local_tol does: _vertex_stats only
    sees surviving windows. The trim never fires on these teeth because their comb-tied heights stay positive.
    So (b) is implemented as a vertex-tail excusal (trailing run failing the tol test, keep >= n_required windows).
- Variants added: local_tol_pos (a), local_tol_tail (b), local_tol_norej (no MAD rejection; new, from diagnosis),
  combos local_tol_norej_tail / _norej_pos, and (c) robust height fit: local_tol_tail_rh, local_tol_norej_rh.
- Quick screen on the 30 lost + 25 gained T32 star-modes: recovered lost (of 30) pos 1, tail 8, norej 14,
  norej_tail 16, tail_rh 8, norej_rh 15; T32 gains kept 25/25 except norej (24) and norej_rh (23).
- drift_a_frac: under local_tol the solver never fits a drift term (drift_a is always 0), so the feature is a
  separate quadratic fit to the free vertices (`comb_fit.comb_drift_frac`), NaN for lsq / < 4 vertices.
- 03:13 Reference runs done (runs/p43/sc/local_tol_{dmfy,dmas,syn}):
  full-year 0.248 / wrong 0.067, assigned 0.145 / 0.048 -- identical to T32's local_tol; PASS vs _v7 both.
  synthetic 200: 0.745 vs _v7 0.755, FAIL on the regression guard: >30d bin 1.000 -> 0.875 (1 of 8 stars), and
  3-10d -0.056 (just inside the 0.06 limit). This is the local_tol default itself, not P43 -- flag for the _v8 refresh.
- Tests: tests/test_p43_decayed_tail.py, 18 pass.
- 03:17 Harness screen (all 8 variants, both modes, rebuilds the Stage-1 cache) running.
- 03:34 Harness (dev_medium, 8 variants; local_tol reproduces the scorecard exactly), success / wrong:
  | variant | full-year | assigned |
  |---|---|---|
  | local_tol | 0.248 / 0.067 | 0.145 / 0.048 |
  | local_tol_pos (a) | 0.249 / 0.070 | 0.145 / 0.050 |
  | local_tol_tail (b) | 0.257 / 0.072 | 0.153 / 0.057 |
  | local_tol_norej | 0.277 / 0.113 | 0.165 / 0.070 |
  | local_tol_norej_tail | 0.288 / 0.133 | 0.175 / 0.088 |
  | local_tol_norej_pos | 0.278 / 0.123 | 0.168 / 0.074 |
  | local_tol_tail_rh (c) | 0.256 / 0.074 | 0.153 / 0.059 |
  | local_tol_norej_rh (c) | 0.277 / 0.102 | 0.166 / 0.067 |
  norej's 52 new full-year wrongs (all from none) are mostly long-P combs on short-P stars with mean SNR just
  over 1: hypothesis -- the kept lag-0 tooth (height ~ ACF(0)) props up mean_height_snr of noise combs; the MAD
  loop had been dropping it (its residual RMS is the largest), which was silently acting as a noise filter.
- Added `snr_skip_lag0` (mean_height_snr gate over non-lag-0 teeth; no threshold change). Harness:
  | local_tol_snr0 | 0.242 / 0.066 | 0.144 / 0.047 |
  | local_tol_tail_snr0 | 0.250 / 0.068 | 0.150 / 0.049 |
  | local_tol_norej_snr0 | 0.250 / 0.062 | 0.143 / 0.047 |
  | local_tol_norej_tail_snr0 | 0.256 / 0.064 | 0.150 / 0.049 |
  Hypothesis confirmed (norej wrong 0.113 -> 0.062). Scorecards next for norej_tail_snr0, norej_snr0, tail.
- 03:43-03:57 Scorecards local_tol_norej_tail_snr0: full-year 0.256 / wrong 0.064, but by-bin 10-30d 0.363 (_v7)
  -> 0.287: FAILS the regression guard vs _v7 (-0.076 > 0.06); PASS vs local_tol (-0.053). Assigned and synthetic
  see check_goals files. Harness by-bin: snr0 costs 10-30d (local_tol 0.339 -> snr0 0.316, norej_snr0 0.287);
  the 13 full-year losses vs local_tol are 10/13 in 10-30d, failing mean_height_snr at 0.85-0.98 with all 8 teeth
  kept (decayed late teeth now dilute the mean, and the lag-0 tooth no longer props it up).
- Decision: stop adding variants (each new idea was coming from the dev_medium judging set). Killed the queued
  norej_snr0 scorecard after 30 s (dominated in the harness; same 10-30d guard failure); scoring tail_snr0 and tail.
- 04:13-04:26 Scorecards local_tol_tail_snr0 and local_tol_tail (both modes + synthetic); all PASS vs local_tol;
  vs _v7 both pass dev_medium, synthetic FAILs on the same >30d star as local_tol (synth-s0-00099).
- 04:28-04:32 Profiles (2 rounds x 4 variants). 04:33 analysis, figures, notebook executed (0 errors).
- Full pytest under nice: 544 passed, 1 failed = the known test_default_scorecard_rows_bit_identical_to_baseline.
- README written; docs: PROJECT_CONTEXT changelog, TASKS_PROPOSED (P43 status, P44, P45), FEATURE_TABLE_SCHEMA.
