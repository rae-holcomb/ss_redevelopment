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
