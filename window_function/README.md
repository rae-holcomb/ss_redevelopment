# TESS phase-coverage figures

Recreation of **Fig. 4 of Rodel et al. (2024)**, MNRAS 529, 3739
([doi:10.1093/mnras/stae474](https://doi.org/10.1093/mnras/stae474)) —
the fraction of orbital phase covered by TESS as a function of a
simulated planet's orbital period — updated to current TESS pointings.

## Target

**TIC 167814656**, southern continuous-viewing-zone star, observed in
**45 Sectors**: 1–13, 27–39, 61–69, 87–90, 93–98. Baseline 2721 d
(7.45 yr), 1168 d on sky. Picked from `target_df_shortform.csv` as one of
the highest-`num_s` entries; the QLP FFI products reach Sectors 97–98,
two sectors beyond that CSV.

Window function comes from **FFI light curves** (QLP, with TESS-SPOC
filling Sectors 3 and 8 which QLP does not cover), NaN-flux and
default-quality cadences dropped. FFI rather than 2-min SPOC because the
SMARTS light curves this project is built on are themselves FFI cadence.
As the paper notes, a different pipeline's quality flags would shift some
gap edges slightly.

## Files

| file | what |
|---|---|
| `fetch_window.py` | Downloads the light curves, caches good-cadence times to `window_cadences.npz`. Re-run to update as new Sectors land. |
| `phase_coverage.py` | Coverage maths: cadences → contiguous intervals → exact union of phase arcs. |
| `make_figures.py` | Builds the three figures and `coverage_summary.txt`. |
| `fig0_window_map.png` | Where the 45 Sectors actually fall in time. |
| `fig1_rodel_fig4_recreation.png` | The paper's two-panel figure: idealised 27.4 d Sectors vs. a real 13-consecutive-Sector window function. |
| `fig2_current_pointings.png` | Extension to N = 1–45, contrasting back-to-back Sectors with the real clumped pointing pattern. |
| `coverage_summary.txt` | Covered fraction at reference periods, all three scenarios. |

## Method

Coverage is computed **exactly**, not by binning. Each contiguous run of
good cadences becomes an interval (padded by half a cadence at each end),
each interval is mapped to an arc on the [0, 1) phase circle, and the arcs
are unioned by a sorted running-maximum sweep. The fine oscillations in
the real-window curves are therefore genuine resonances between the trial
period and the repeating orbit/downlink gap pattern, not sampling noise.
Verified against a 2×10⁵-point brute-force phase grid to 6 decimal places.

## What changes versus the paper

1. **Modern Sectors lose less to gaps.** Mean per-Sector duty cycle rose
   from 0.919 in Cycle 1 to 0.967 in Cycles 5–8, so the real-window panel
   of Fig. 1 plateaus near 0.95 where the paper's sits near 0.80–0.85.
   The paper's headline point — gaps matter, do not assume a 27 d box —
   still holds, just less severely than in Cycle 1.
2. **Sectors are no longer 27 d.** Cycle 8 (S97, S98) uses ~55–57 d
   pointings; the mean Sector span over S87–98 is 32.6 d. The "27 d
   Sector" idealisation is now wrong in the optimistic direction too.
3. **Clumping dominates at long period.** For a star with many Sectors,
   the far larger effect is that those Sectors arrive in per-cycle clumps
   separated by year-long gaps. At fixed on-sky time and N = 45, coverage
   at P = 730 d is 0.98 back-to-back but **0.74** as actually observed;
   at N = 34 it is 0.95 vs **0.56**. The right panel of Fig. 2 shows the
   resulting non-monotonic structure — a deep trough near P ≈ 700 d where
   every clump lands on the same phase, and recoveries near 1100 and
   1900 d where they do not. Coverage stops being a smooth function of
   period, so a survey-completeness calculation cannot interpolate it.
