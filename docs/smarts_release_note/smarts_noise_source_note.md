---
title: "Periodic signals in the noise light curves of SMARTS-TESS DR1"
subtitle: "A data-release note with per-file and per-source flags"
author: "R. Holcomb [author list to be confirmed]"
date: "DRAFT, 2026-10-22"
---

# Summary

Each SMARTS-TESS DR1 light curve is a simulated spotted star injected into a real
TESS full-frame-image (FFI) light curve of a TIC `EXTENDED` source. These "noise light
curves" were chosen to be intrinsically constant, so that they would contribute only
instrumental systematics and noise. We recovered all 1,719 noise light curves from the
dataset and searched each for periodic signals. Our findings:

- **206 of the 1,719 noise light curves (12%) carry a persistent periodic signal.**
  They underlie **121,502 of the 1,000,000 SMARTS files (12.2%)**. Periods range from
  0.15 to 32 d (median 3.4 d), with amplitudes from about 100 ppm to 5.5%.
- **Most of these are contamination by neighbouring stars.**
  - 90 sources have a Gaia DR3 variable within 63″, including eclipsing binaries,
    Cepheids, RR Lyrae and δ Scuti stars.
  - 63 more sit in crowded fields.
  - Only 8 look like spacecraft-wide systematics.
- **The signals can be confirmed independently.** In TESS FFI light curves from three
  other reductions (QLP, TASOC, GSFC-ELEANOR-LITE), the known-variable contaminants are
  recovered at the same period 63% of the time, against a chance rate of 15%.
- **The contaminating signal is often recovered instead of the injected rotation
  period.** In a test sample of 1,000 SMARTS files, 51 of the 88 wrong periods
  recovered by a rotation-period search equal the period of the file's noise light
  curve.

We provide flags for every SMARTS file and every noise light curve, so that users can
remove or separately score the affected files.

# 1. How SMARTS light curves are built

SMARTS-TESS DR1 (Claytor et al. 2024) contains 1,000,000 simulated light curves of
spotted, differentially rotating stars. Each is sampled on a common 30-minute grid
covering one year of the TESS southern continuous viewing zone (CVZ). To give the
simulations realistic TESS noise, each one was combined with a real TESS FFI light
curve of a source classified as `EXTENDED` in the TESS Input Catalog (TIC). The
reasoning was that galaxies are intrinsically constant, so they carry only
systematics and noise. One noise light curve is drawn at random per simulation. There
are 1,719 distinct noise light curves in total, each reused in 210–1,082 files.
The noise source of each file is listed in `galaxy_files.csv`, keyed by the
simulation number (`SIM_ID`).

**The injection is multiplicative:** file flux = noise flux × simulated stellar flux.
We verified this using the noise light curves that contain corrupted orbits, where
the noise flux departs from 1 by factors of 3 to 100. There the residual
file − noise grows with the noise flux (by 6.2× where |noise − 1| > 3), while
file / noise − 1 stays flat (1.17×). A consequence is that a contaminant's
*relative* amplitude is the same in every file that uses that noise light curve.

# 2. How we tested the noise light curves

**Recovering each noise light curve.** For each noise source we took up to 80 of its
member files with the smallest expected injected spot amplitude. We used
`ACTIVITY` × |sin(`INCL` + mean spot-band latitude)| as a proxy for that amplitude.
We split them into two disjoint halves, A and B, and took the per-cadence median of
each half and of all of them. Because the injection is multiplicative and the stellar
signals are independent across files, the median recovers the noise light curve up to
a constant. Held-out member files confirm this: the median residual between member and
stack is 68 ppm for the stacked members, and its point-to-point scatter is 0.25 ppm.

**Periodicity and persistence.** We computed a Lomb–Scargle (LS) periodogram from 0.1
to 60 d for stacks A, B and A+B. We also ran a general-purpose rotation-period search
(ACF and periodogram based) on the A+B stack, treating it exactly like a SMARTS star.
A source is **periodic** if:

- the period search returns a period, *or*
- the A+B LS top peak has power ≥ 0.1 *and* the top peaks of halves A and B agree
  within 2%.

A signal present in both disjoint halves lives in the noise light curve, not in
leftover injected stellar signal. The reference period `P_ref` is the period-search
result where there is one, else the LS top peak. Fig. 1 shows examples.

**Common-mode test.** We took the top 8 principal components across all 1,719 stacks
as spacecraft-wide common modes, projected them out of each stack, and measured the
fraction of LS power at `P_ref` that survives (`cm_retained`). A signal driven by
shared systematics loses most of its power; a source-specific one keeps it.

**Neighbour lookups.** For each source we retrieved its TIC v8.2 entry and all Gaia
DR3 sources with G < 19 within 63″ (3 TESS pixels) via VizieR. We recorded:

- a dominant star-like source within 3″ of the target;
- neighbours (3–63″) brighter than G = Tmag + 1;
- neighbours flagged `VARIABLE`, with their Gaia DR3 variability classes.

**Classification.** Each periodic source receives the first class that applies, in
this order:

| class | rule | sources | files |
|---|---|---|---|
| `quiet` | not periodic | 1,513 | 878,498 |
| `systematic_common_mode` | `cm_retained` < 0.5 | 8 | 6,374 |
| `misclassified_star` | Gaia source within 3″ with star probability PSS > 0.9 and G < Tmag + 1.5 | 1 | 299 |
| `contamination_known_variable` | a Gaia DR3 variable within 63″ | 90 | 53,417 |
| `contamination_candidate_crowded` | within 4° of the LMC, or a Gaia neighbour brighter than G = Tmag + 1 | 63 | 35,369 |
| `source_specific_unexplained` | none of the above | 44 | 26,043 |

**Independent FFI light curves.** For all 206 periodic sources we downloaded the
QLP, TASOC or GSFC-ELEANOR-LITE light curves (MAST HLSPs) of the same TIC ID for three
sectors from sectors 1–5. No SPOC products exist for these targets. Each sector was
detrended with a running median of 4 × `P_ref` (between 0.5 and 5 d) and searched with
LS from 0.1 d up to the detrending window. A source counts as **confirmed** if one of
the three highest peaks of the stitched periodogram is within 3% of `P_ref`,
`P_ref`/2 or 2 × `P_ref`. Only `P_ref` < 5 d is testable: in single 27-day sectors,
spurious 13–15 d peaks "match" even for quiet sources, so 82 periodic sources with
`P_ref` ≥ 5 d were not tested. To measure the chance-match rate, we ran the same test
on 40 randomly chosen quiet sources, each assigned a period drawn from the periodic
sources (26 of them testable).

# 3. What we found

![**Figure 1.** Four noise light curves recovered from SMARTS files, colour-coded by class. Left: the full year (A+B stack). Right: a zoom with the two disjoint halves A (grey) and B (colour) overlaid; they agree to within the line width. From top: a 0.248-d, 5.5%-amplitude contaminant next to a Gaia eclipsing binary; a 0.955-d signal in a crowded field; a coherent 9.9-d, 3,900 ppm sinusoid with no identified cause; a quiet source.](figures/fig1_example_noise_sources.png)

**Contamination dominates.** All five of the largest signals (1.5–5.5% amplitude,
P = 0.2–1.6 d) sit next to Gaia eclipsing binaries or Cepheids. Several contaminants
change amplitude from sector to sector. That is expected when a neighbour's light
leaks into an aperture that is redefined each sector.

**Families of apertures share one variable.** Neighbouring `EXTENDED` sources often
have overlapping apertures, so a single variable star can contaminate several noise
light curves. For example, nine of the fourteen sources with TIC IDs between
10002480866 and 10002480885 are periodic: seven at P ≈ 0.83 d and two at P ≈ 1.4 d. More generally, 94 pairs of *distinct* noise light curves are correlated
at r = 0.5–0.99, in groups of 2–12 near-consecutive TIC IDs.

**Proximity to the LMC.** The southern CVZ overlaps the outskirts of the Large
Magellanic Cloud.

- 242 of the 1,719 noise sources lie within 4° of the LMC centre
  (RA, Dec = 80.89°, −69.76°). Their fields hold a median of about 90 Gaia sources
  (G < 19) within 63″, against 6 elsewhere.
- Of these, 62 (26%) are periodic, against 144 of 1,477 (10%) elsewhere.
- Together they underlie 137,980 files.

| class | sources | files | median `P_ref` (d) | median LS amplitude (ppm) | near LMC |
|---|---|---|---|---|---|
| `contamination_known_variable` | 90 | 53,417 | 2.6 | 1,015 | 53 |
| `contamination_candidate_crowded` | 63 | 35,369 | 4.0 | 420 | 6 |
| `source_specific_unexplained` | 44 | 26,043 | 4.4 | 543 | 0 |
| `systematic_common_mode` | 8 | 6,374 | 5.6 | 355 | 2 |
| `misclassified_star` | 1 | 299 | 0.37 | 315 | 1 |
| **all periodic** | **206** | **121,502** | 3.4 | | 62 |
| `quiet` | 1,513 | 878,498 | — | 226 | 180 |

![**Figure 2.** Left: LS amplitude against period for all 1,719 noise light curves. Periodic sources (colours) separate from the quiet red-noise population (grey) mainly at P < 5 d; the high-amplitude, short-period end is dominated by known-variable contamination. Right: sky positions; the dashed ellipse marks 4° around the LMC centre.](figures/fig2_population.png)

**Independent confirmation.** Table and Fig. 3 compare the confirmation rate of each
class with the chance-match rate measured on quiet sources. The p-values come from a
one-sided Fisher exact test against the controls.

| class (testable, `P_ref` < 5 d) | confirmed | rate [95% CI] | p vs. controls |
|---|---|---|---|
| `contamination_known_variable` | 38/60 | 0.63 [0.51, 0.74] | 4 × 10⁻⁵ |
| `contamination_candidate_crowded` | 13/34 | 0.38 [0.24, 0.55] | 0.05 |
| `source_specific_unexplained` | 4/25 | 0.16 [0.06, 0.35] | 0.63 |
| `systematic_common_mode` | 3/4 | 0.75 [0.30, 0.95] | 0.03 |
| `misclassified_star` | 1/1 | — | 0.19 |
| quiet controls (shuffled periods) | 4/26 | 0.15 [0.06, 0.34] | — |

- **Contamination by known variables is confirmed as a population.** Crowded-field
  candidates are confirmed at a lower but elevated rate.
- **The `source_specific_unexplained` class is confirmed only at the chance rate.**
  These signals are real in the SMARTS light curves: they persist in both disjoint
  halves, are specific to one source, and change the recovered period (Section 4). But
  independent reductions of the same targets show no excess at those periods.
  - The signals may therefore be specific to the reduction used to build the SMARTS
    noise light curves, through the aperture or background treatment.
  - Alternatively, they may be too faint to see in the other reductions.
  - We cannot tell these apart with the data in hand.
- **The amplitudes differ widely between reductions at the same period.** For
  example, one source shows 12,244 ppm in SMARTS but 336 ppm in QLP. This is what a
  constant target with a reduction-dependent fraction of a neighbour's light predicts.
  For the same reason, which neighbour leaks in can depend on the aperture. In the
  0.83-d family, QLP shows the ≈ 1.4-d signal of other family members instead.
- **`systematic_common_mode` cannot be confirmed this way.** All TESS reductions share
  spacecraft systematics, so a confirmation does not separate a few-day systematic
  from an astrophysical signal. Three of its eight members have P ≈ 3.2–3.3 d, close
  to a peak of one common mode at 3.4 d.

![**Figure 3.** Fraction of testable periodic sources (`P_ref` < 5 d) whose period is recovered in independent FFI light curves (QLP / TASOC / GSFC-ELEANOR-LITE), with Wilson 95% intervals. The grey band is the chance-match rate measured on quiet sources tested at shuffled periods.](figures/fig3_ffi_check.png)

# 4. Impact on period recovery

The rotation-period search sees the product of the noise light curve and the star. A
coherent signal in the noise light curve is therefore a second, unlabelled period in
every file that uses it. It is often stronger than the injected spots, especially for
low-activity or low-inclination stars.

We measured the effect on a test sample of 1,000 SMARTS files, stratified in period
from 0.1 to 180 d, using their
full-year light curves and a general-purpose rotation-period search:

- **51 of the 88 wrong periods (58%) equal, within 10%, the period the same search
  finds in the file's noise light curve alone.** If noise sources are shuffled among
  the files, the expected number of such coincidences is 0.4, and at most 3 in 200
  shuffles.
- **Files on periodic noise sources return a wrong period 43% of the time (55/128),
  against 3.8% (33/872) on quiet sources.**
- Correct recoveries (including periods in the differential-rotation range) differ
  much less: 22.7% on periodic sources against 25.1% on quiet ones. The noise period mostly turns what would have been
  "no detection" into a confident wrong answer.

The method is not failing on these files. It correctly measures a signal that the
labels do not contain. Any period-recovery method, or any model trained on SMARTS
labels, will see the same thing.

![**Figure 4.** Left: for test files whose noise light curve has a detectable period, the period recovered from the SMARTS file against the period found in the noise light curve alone. Most wrong answers (red) lie on the 1:1 line. Right: fraction of files with a wrong recovered period, by noise-source class.](figures/fig4_period_recovery.png)

# 5. How to use the flags

We suggest the following practice.

1. **Join the flags to every training and evaluation table** by `filename` (or
   `file_number` or `sim_id`). About 12% of files have, in effect, two true periods:
   the injected rotation period and `P_ref`.
2. **Training supervised models.** Exclude, or relabel, files on
   `contamination_known_variable`, `contamination_candidate_crowded` and
   `source_specific_unexplained` sources (197 sources, 114,829 files). At minimum,
   never score a candidate at `P_ref` (or `P_ref`/2, 2 × `P_ref`) as a negative.
   These signals are present in the light curves, and a good method *should* find
   them. `systematic_common_mode` sources (8) resemble systematics in real TESS data
   and can reasonably be kept.
3. **Reporting performance.** Report metrics on the `quiet` subset (`periodic` =
   False) alongside the full set. Treat a recovered period that matches the noise
   source's `P_ref` (within your tolerance, and at its P/2 and 2P aliases) as its own
   outcome, distinct from a genuine method error.
4. **Choosing your own threshold.** The `periodic` flag is a fairly inclusive
   threshold. The per-source signal-strength columns (`ls_P1_power`, `ls_P1_amp_ppm`,
   `ab_persistent`, `search_detected`) let you apply a stricter or looser cut without
   repeating the analysis.
5. **Sky position.** If your science case is sensitive to crowding, consider using
   `near_lmc` and the Gaia neighbour counts in the per-source table as additional
   selection variables.

# 6. Limitations

- **Long periods are only partly tested.** For `P_ref` ≥ 5 d (82 of 206 periodic
  sources), the independent-FFI test is not possible with single sectors. Their
  classification rests on the SMARTS light curves alone: A/B persistence, common-mode
  test and neighbour lookups.
  - At 5–30 d, real signals sit on a red-noise continuum. The quiet population
    itself has LS peaks with typical amplitudes of 100–500 ppm there (Fig. 2).
  - Our `periodic` flag requires either a period-search detection or LS power ≥ 0.1.
    A different threshold would move some sources between `quiet` and the periodic
    classes.
- **Thresholds are choices.** These include the LS power cut, the 2% A/B agreement,
  `cm_retained` < 0.5, the 63″ neighbour radius, the 4° LMC radius, the magnitude
  offsets, and the 3% FFI match tolerance. The classes are diagnostic, not
  definitive.
  - A Gaia variable within 63″ is weaker evidence near the LMC, where 67% of *quiet*
    sources also have one (against 8.6% elsewhere).
  - `contamination_known_variable` does not establish that the specific listed
    variable is the source of the signal.
- **Weak signals are only partly separated.** The stacks are built from the members
  with the smallest expected injected amplitude. A residual stellar contribution could
  remain at a level below the A/B persistence test. The member-minus-stack residuals
  (68 ppm median) bound this.
- **The near-LMC sources.** We note that 242 of the 1,719 noise sources lie within 4°
  of the LMC centre, in fields much more crowded than the rest of the sample, and that
  they are periodic about 2.6 times as often. Whether these targets fit the intended
  "constant galaxy" selection is a question for the dataset authors. We flag them
  (`near_lmc`) without making a recommendation.
- **The unexplained class.** Section 3 explains why we cannot say whether these
  signals are astrophysical or arise in the reduction. If they arise in the
  reduction, a similar effect at lower amplitude may be present in other noise light
  curves, below our detection threshold.

# 7. Data products and column reference

Two comma-separated tables accompany this note.

- **`smarts_noise_sources.csv`** has one row per noise light curve (1,719 rows).
- **`smarts_noise_source_flags.csv.gz`** has one row per SMARTS file (1,000,000 rows;
  about 31 MB compressed). Each row carries its noise source's columns, marked
  "both" below.

Booleans are `True`/`False`; missing values are empty.

**File identifiers (per-file table only)**

| column | description |
|---|---|
| `file_number` | 0-based file number, equal to the row index of the DR1 catalog (rows are sorted by period) |
| `filename` | `smarts-tess-v1.0-NNNNNN.fits`, with NNNNNN = `file_number` |
| `sim_id` | the catalog's `Simulation Number` (the original simulation index; the key of `galaxy_files.csv`) |

**Noise-source identity and classification (both tables)**

| column | description |
|---|---|
| `noise_tic` | TIC ID of the noise source |
| `noise_lc_file` | its noise light-curve filename as given in `galaxy_files.csv` (per-source table) |
| `classification` | one of the six classes in Section 2 |
| `periodic` | True if `search_detected`, or if `ls_periodic` |
| `P_ref_d` | reference period of the noise light curve (d): `search_P_d` if detected, else `ls_P1_d`. Empty for quiet sources |
| `near_lmc` | True if within 4° of the LMC centre (RA, Dec = 80.89°, −69.76°) |
| `ffi_check` | independent-FFI result for this source: `confirmed`, `not_confirmed`, `untestable` (`P_ref` ≥ 5 d), or `not_tested` (quiet sources) |

**Signal-strength metrics (both tables)**, measured on the A+B stack of the noise
light curve:

| column | description |
|---|---|
| `ls_P1_d` | period of the highest LS peak, 0.1–60 d |
| `ls_P1_power` | normalized LS power of that peak. Population median 0.015; 90th percentile 0.067 |
| `ls_P1_amp_ppm` | sinusoid semi-amplitude at `ls_P1_d` (ppm) |
| `ab_persistent` | True if the LS top peaks of the disjoint halves A and B agree within 2% |
| `ls_periodic` | `ab_persistent` and `ls_P1_power` ≥ 0.1 |
| `search_detected` | True if the rotation-period search returned a period for the stack |
| `search_P_d` | that period (d) |
| `cm_retained` | fraction of LS power at `P_ref` that survives removing 8 common modes; < 0.5 → `systematic_common_mode` |

**Per-source table only**

| column | description |
|---|---|
| `n_files` | number of SMARTS files that use this noise light curve |
| `ra_deg`, `dec_deg`, `Tmag` | TIC v8.2 position and TESS magnitude |
| `d_lmc_deg` | angular distance from the LMC centre (deg) |
| `n_gaia_nb` | Gaia DR3 sources with G < 19 at 3–63″ |
| `n_gaia_nb_bright` | of those, sources with G < Tmag + 1 |
| `n_gaia_nb_var` | of those, sources with Gaia `VarFlag` = VARIABLE |
| `gaia_nb_var_classes` | Gaia DR3 variability classes of those variables (`;`-separated; from `vari_classifier_result`) |
| `frac_corrupt` | fraction of the source's files with corrupted orbits (|flux − 1| > 0.1; these affect every file using the source) |
| `ffi_lc_authors` | HLSP(s) used for the FFI check (QLP, TASOC, GSFC-ELEANOR-LITE) |
| `ffi_sectors` | TESS sectors checked (`;`-separated) |
| `ffi_match` | for confirmed sources, which multiple matched: `P`, `P/2` or `2P` |
| `ffi_n_sectors_matching`, `ffi_n_sectors` | sectors whose own periodogram matched, out of sectors checked (testable sources) |

**Joining.** To join to the DR1 files, use `filename` or `file_number`. To join to
`galaxy_files.csv`, use `sim_id`. To join the two tables, use `noise_tic`.

# References

- Claytor, Z. R., et al. 2024, ApJ, 962, 47. *Recovery of TESS Stellar Rotation
  Periods Using Deep Learning.* (SMARTS)
- Holcomb, R. J., et al. 2022, ApJ, 936, 138. *SpinSpotter: An Automated Algorithm for
  Identifying Stellar Rotation Periods with Autocorrelation Analysis.*
- Gaia Collaboration 2023, A&A, 674, A1 (Gaia DR3); Eyer, L., et al. 2023, A&A, 674,
  A13 (DR3 variability).
- Stassun, K. G., et al. 2019, AJ, 158, 138 (TIC v8).
- Huang, C. X., et al. 2020, RNAAS, 4, 204 (QLP); Handberg, R., et al. 2021, AJ, 162,
  170 (TASOC); Powell, B. P., et al. 2022, RNAAS, 6, 111 (GSFC-ELEANOR-LITE).
