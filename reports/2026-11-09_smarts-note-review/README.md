# Referee review: "Periodic signals in the noise light curves of SMARTS-TESS DR1"

**Date:** 2026-11-09 · **Reviewer:** independent review session (did not author the note) · **Branch:** `claude/smarts-note-review-2d28d8` · **Status:** review only. No changes were made to the note, tables or code.

**Reviewed:** the live note at `.claude/worktrees/smarts-contamination-note-969cc1/docs/smarts_release_note/smarts_noise_source_note.md`, including Rae's uncommitted edits as of 2026-11-09 16:21. Line numbers below refer to that file. Also reviewed: its five figures, `smarts_noise_sources.csv`, the 1M-row `smarts_noise_source_flags.csv.gz` on the drive, `inspect_flags.ipynb`, `tools/audit/smarts_release_products.py`, the T20 scripts, and the phase-1 and phase-2 reports. The Data_Vault drive was mounted, so everything could be checked. All check scripts are in `checks/` (`c01`–`c10`); each runs in `ss_openblas`, except `c05b`, which runs in `codeastro`.

## Verdict: needs major changes before external release (mostly to the text)

The data products are **correct as computed**. Every number in the note that can be recomputed from the tables reproduces exactly. The per-file table agrees with Zach's own `galaxy_files.csv` for all 1,000,000 files, and the notebook runs.

The problems are in what the text claims, plus one gap in the classification.

- **(1) The `quiet` class is not clean.** At least 34 sources labelled `quiet` (19,554 files) carry short-period signals that are coherent across the whole year, at a local S/N above 20 in *both* halves. They miss the `periodic` cut because normalized LS power measures the fraction of total variance, not significance.
- **(2) Attribution claims are stronger than the evidence.** About 51 of the 90 `contamination_known_variable` sources would have a Gaia variable within 63″ by chance alone. The headline "most of these are contamination by neighbouring stars" rests on proximity, and the summary presents the FFI confirmation without its main limitations.
- **(3) The note depends on things an external reader cannot access or reproduce.** `galaxy_files.csv` is a private file from Zach, and the "general-purpose rotation-period search" that decides 189 of the 206 detections is undocumented.

Two things would make the note stronger than it is now, and both are cheap.

- **The Gaia neighbours' own periods match `P_ref` in 24 sources, against 5.4 expected by chance.** This is direct causal evidence.
- **21 periodic sources fall in 7 spatial families that share one period, against 2.7 expected by chance.** That points to a sky origin.

## Must fix (wrong or misleading)

**M1. Line 22 and lines 195–198: the `quiet` class contains coherent periodic signals, so "206 (12%)" is a lower bound and the "quiet subset" is not clean.**

- **Evidence** (`checks/c07*.py`): I split each A+B stack into the two halves of the year and measured the LS power at the source's period, relative to the median power within ±25% in frequency. Every quiet source whose top LS peak is below 1.5 d (33 of 33) has S/N > 10 in both halves. 34 quiet sources with P < 5 d have S/N > 20 in both halves (`checks/quiet_coherent.csv`).
  - They have periods of 0.10–2.2 d and amplitudes of 109–678 ppm (median 278 ppm), and lie under 19,554 files.
  - Their LS power is only 0.006–0.09, and the pipeline did not detect them.
  - They include three sources at 0.829 d, the period of the known 0.83-d family. 12 have a Gaia variable neighbour and 12 are near the LMC.
  - In Fig. 2 they are the grey points at P < 1 d.
- **Cause:** `ls_P1_power` is the fraction of variance explained, which red noise depresses. A 300-ppm sinusoid that stays coherent for a year can sit at power 0.01.
- **Fix:**
  - Either add a significance-based criterion (e.g. the time-split S/N above, or a false-alarm probability against a smoothed red-noise continuum) and reclassify,
  - or at minimum say plainly in the Summary, Section 4 and Appendix C that `quiet` means "not detected by our two criteria", that dozens of quiet sources show low-amplitude coherent short-period signals, and that 12% is a lower bound.
  - Recommendation 3 (line 195, "report metrics on the quiet subset") should carry this caveat.
  - Also rewrite line 199 ("a fairly inclusive threshold"): it is inclusive at long periods and exclusive at short ones.

**M2. Lines 25–28 and 87: "Most of these are contamination by neighbouring stars" overstates an association as causation.**

- **Evidence** (`checks/c04_stats.py`):
  - Given the quiet sources' rates (67% near the LMC, 8.6% elsewhere), **51 of the 90** `contamination_known_variable` sources would have a Gaia variable within 63″ even if the variable had nothing to do with the signal. The association is real (Mantel–Haenszel odds ratio 4.1; p = 2×10⁻⁴ near the LMC and 4×10⁻⁹ elsewhere), but the class label is wrong for many individual sources.
  - The crowded rule fires for 41% of off-LMC quiet sources (brighter neighbour within 63″) and 50% near the LMC.
  - The most common neighbour class in `contamination_known_variable` is **LPV (43 sources)**. For 18 of the 90, LPVs are the only neighbour class, and 2 have no class at all. LPVs (periods of tens to hundreds of days) cannot explain a few-day `P_ref`; the median `P_ref` of those 20 is 19 d. The summary's list "eclipsing binaries, Cepheids, RR Lyrae and δ Scuti" (line 27) omits the commonest class.
- **Positive evidence the note does not use:**
  - (a) **Period match** (`checks/c05b`, `c05c`). Gaia DR3 per-class period tables list a period for a neighbour of 61 of the 90 sources. In **24** of them, a neighbour's period matches `P_ref`, P/2 or 2P within 3%, against **5.4 expected** with `P_ref` shuffled (95th percentile 9, maximum 13 in 500 permutations). The matches come from 12 EBs, 10 Cepheids, 4 RR Lyrae, 4 short-timescale variables and 1 rotator. 18 of the 24 are also FFI-confirmed.
  - (b) **Spatial families** (`checks/c09_families.py`). 21 periodic sources form 7 groups within 5′ sharing one period, against 2.7 expected by chance (maximum 8). The largest group is the 9-member 0.83-d family.
- **Fix:**
  - Reword the summary to "associated with". For example: *"90 sources have a Gaia DR3 variable within 63″; for 24 of them a neighbour's catalogued period matches the signal (5 expected by chance), which identifies the contaminant. Near the LMC most sources have a variable neighbour regardless, so the class is a proximity label, not an identification."*
  - Add a column, e.g. `gaia_nb_period_match`, and report the chance rates in Appendix C next to line 367.
  - Make line 87's "Many contaminated sources are located near known variable stars" quantitative.

**M3. Lines 30–32: the summary's FFI bullet overstates the confirmation.**

- "The signals can be confirmed independently" holds only for `P_ref` < 5 d (124 of 206). Of those, 59 are confirmed, against **16.5 expected by chance** from the per-source null (S2), an excess of about 42.
- The test reaches population significance for two classes: known-variable (38/60) and crowded (13/34).
- For `source_specific_unexplained` the result is null (4/25 against 4.2 expected), and 82 sources were not testable.
- **Fix:** *"For sources with P < 5 d (124 of 206), independent FFI light curves (QLP, TASOC, GSFC-ELEANOR-LITE) recover the same period in 38 of 60 known-variable cases and 13 of 34 crowded-field cases, against about 15% expected by chance. The 44 unexplained sources are not recovered above chance, and periods ≥ 5 d could not be tested."*

**M4. Lines 51, 133 and 178–179: `galaxy_files.csv` is not public.**

- `docs/SMARTS_REFERENCE.md` §7 records it as a file Zach sent privately on 2026-10-21, stored on the drive only. An external reader cannot obtain it, so line 51 and the join instructions point to nothing.
- **Fix:** release the per-file noise-source mapping (it is already in our per-file table as `sim_id` → `noise_tic`, so ours suffices), or get Zach's agreement to publish `galaxy_files.csv` with the note. Then say where it lives. (Question Q1.)

**M5. Lines 158, 245 and 248–250: the detection criterion that decides most classifications is a black box.**

- 189 of the 206 periodic sources come from `search_detected`, and 102 of them are periodic *only* because of it. Yet the note describes it only as "a general-purpose rotation-period search (ACF and periodogram based)".
- Readers cannot judge its false-positive rate or reproduce `search_detected`. Appendix B reuses the same search, so its 51/88 result is method-specific too.
- **Fix:** add a paragraph giving the method:
  - the candidate generators;
  - the comb fit;
  - the acceptance / decline rule;
  - the period range;
  - and, if the code will be public, a citation or link.
- Also state that it was developed on the same SMARTS data, and that dev_medium (Appendix B's test sample) is that pipeline's own development set. Uncertain: whether dev_medium was used for tuning; Rae to confirm (Q4).

**M6. Lines 21, 75–76 and 145: the note tells users to remove segments it does not let them locate.**

- Only a per-source boolean is released. No segment start/end times are given, and the rule as written omits the 2-d merge gap from `smarts_release_products.py:55`. The flag count barely depends on that gap (167/170/170/172 sources for gaps of 0.5/1/2/5 d), but a user re-implementing it needs it.
- **Masking the flagged segments is not sufficient:** within flagged sources, outliers up to |g−1| = 3.7 remain outside the segments. 23 unflagged sources have at least 1 d of cadences with |g−1| > 0.1, and 2 have |g−1| > 1 (`checks/c03_corrupted.py`).
- **Fix:**
  - Release a small segment table (`noise_tic`, `t_start_d`, `t_end_d`), and state the merge rule.
  - Add: *"Also clip remaining cadences with |flux − 1| > 0.1; flagged and unflagged sources both have isolated outliers."*

## Should fix (weakens the science or clarity)

**S1. Lines 22 and 252–253: "persistent" means two different things.**

- The A/B test splits member *files*, which share the same noise light curve. It shows that a feature lives in the noise light curve, not that it is periodic. **93% of quiet sources pass it** (1,615/1,719 overall).
- For the 64 periodic sources whose `P_ref` differs from the LS top peak, it tests a different period altogether.
- The summary's "persistent periodic signal" will be read as "persistent in time".
- **Fix:** call it "present in both independent halves of the member files", state the 93% pass rate, and drop "persistent" from line 22. A time-split test (M1) would supply real persistence evidence. For periodic sources, the top peak recurs in both halves of the year in 56–59% of sources, against 6–12% for quiet ones.

**S2. Lines 296–303 and 306: the null and the p-values.**

- **The control null is fair.** Re-running the exact test on each source's *own* FFI light curves at 40 random trial periods (`checks/c06_ffi_null.py`, which reproduces `confirmed` exactly) gives chance rates of 13–16% in every class and for every HLSP author. 26 controls is a small sample, but it is unbiased. This is worth one sentence in the note.
- **The 26-control null understates the crowded class.** Against the per-source null, the crowded class is **13 confirmed against 4.9 expected (p = 4×10⁻⁴)**, and **9 against 2.0 (p = 5×10⁻⁵)** with the strict top-1 criterion. The note's Fisher p = 0.047 would not survive a Bonferroni correction for 5 classes (0.23), but the per-source result does.
- **The test is not liberal.** With only the top peak and only P (no aliases), known-variable sources still give 32 confirmed against 1.9 expected, and unexplained sources give 3 against 1.4.
- **Fix:** report the per-source expected counts (or the strict version) alongside the table, and state that p-values are one-sided and uncorrected.
- **Fix:** note that the controls contain no near-LMC sources (0/26, against 32% of tested periodic sources), even though this turned out not to matter.

**S3. Lines 305–314: the comparison with the unexplained class is confounded by amplitude, and the alternatives are incomplete.**

- **Amplitude drives confirmation:**

  | SMARTS amplitude (ppm) | confirmed |
  |---|---|
  | > 1,500 | 97% |
  | 600–1,500 | 43% |
  | 300–600 | 29% |
  | < 300 | 26% |

  The unexplained class has a median of 357 ppm against 902 ppm for known-variable sources. Below 1,500 ppm, known-variable sources are confirmed in 14/35 and unexplained in 2/23 (p = 0.015). So the difference is real but smaller than the headline suggests, and low S/N is a strong alternative, not a footnote.
- **Missing alternatives:**
  - (i) contamination by a variable not flagged in Gaia DR3, or lying beyond 63″, whose share of light depends on the aperture. That is aperture dependence, not a reduction artefact.
  - (ii) the detection itself being spurious. For P ≥ 5 d, the FFI test cannot help.
- **Fairness:** lines 311–312 and 380–383 lean towards "specific to the reduction used to build the SMARTS noise light curves". That points at Zach's pipeline on evidence equally consistent with low S/N. Line 382's speculation ("a similar effect … may be present in other noise light curves") has no supporting measurement.
- **Suggested wording:** *"Independent reductions do not recover these signals above the chance rate. Most are low-amplitude (median 360 ppm), where the test's recovery rate is about 25–30% even for other classes; they may also come from a neighbour not flagged by Gaia whose light enters the SMARTS aperture but not the others. We cannot distinguish these possibilities."* Drop the line-382 sentence, or ask Zach first (Q2).

**S4. Lines 256–259, 318–321 and 26–29: the common-mode test, and the "only 8 systematics" claim.**

- Only 4% of periodic sources fall below `cm_retained` 0.5, against 31% of quiet sources (at their LS peak). That supports the note's point.
- However, `cm_retained` reaches 55 (up to 1.32 appears in the first rows), so it is not a "fraction". The column reference (line 160) should say "ratio, can exceed 1".
- Eight global PCs cannot capture camera- or CCD-local systematics or scattered light. So "only 8 look like spacecraft-wide systematics" bounds the *global* systematics only.
- One `systematic_common_mode` source (3.27 d) shares its period with a `contamination_known_variable` neighbour within 5′ (`c09`). A neighbour-shared signal is an alternative to a systematic for the 3.2–3.3-d group, and the note should say so at line 320.

**S5. Lines 49–50: the train/validation split of the noise pool is missing, and ML users need it.**

- SMARTS uses two disjoint noise pools by `SIM_ID`: < 800,000 uses about 1,001 TICs and ≥ 800,000 uses about 719, sharing one TIC. "One noise light curve drawn at random per simulation" hides this.
- Add one sentence. The near-disjointness matters for leakage, and it is why Appendix B's sample (997 of 1,000 files in the ≥ 800k pool) samples only one pool.
- Also describe the SMARTS reduction briefly (TESScut + lightkurve regression correctors, Tmag ≤ 15; Claytor et al. 2022 §2). Readers need it to weigh line 311.

**S6. Lines 332–348: Appendix B is generalized beyond one pipeline and one sample.**

- "Any period-recovery method, or any model trained on SMARTS labels, will see the same thing" (lines 347–348) is not tested.
- A method that declines more often, or uses the injected amplitude, would behave differently. Wrong rates also depend on this pipeline's decline policy: it declines 70% of quiet-source files.
- **Fix:** *"Any method that reports the dominant periodic signal is exposed to the same effect; the size of the effect depends on the method."*
- Say that the 1,000 files come from one noise pool. Also say that 13 of the 51 coincidences are on `source_specific_unexplained` sources, so line 33's "the contaminating signal" should be "the noise-source signal".

**S7. Line 374: the residuals do not "bound" a residual stellar contribution.**

- The 68-ppm figure is the scatter of members about their own median, which is in-sample (line 241 says so). A component common to the stacked members would leave no trace in it.
- The argument that does work is that injected signals have independent periods and phases. A coherent residual at one period across both disjoint halves is therefore implausible.
- Say that, or quote the expected median bias (68 ppm × 1.253/√80 = 9.5 ppm for random phases, computed in Python).

**S8. Line 52: the multiplicative model is asserted without evidence or source.**

- The evidence reproduces (`checks/c10_injection.py`): where |g−1| > 3, f−g grows 6.2× while f/g−1 stays at 1.17× (14 TICs, 168 files).
- Claytor et al. say "convolved". Give the one-line evidence, because Section 2's whole argument depends on it.

**S9. Internal jargon and unexplained terms for an external reader.**

- "A+B stack" is used in captions (lines 81, 93) before Appendix A defines it.
- `P_ref` is defined only by forward reference.
- "test sample", "stratified" (line 333), "member files" and "held-out files" are internal shorthand.
- The table "Table and Fig. 4" (line 292) has no number.
- Figures are unnumbered in the Markdown. Pandoc numbers them in the LaTeX build, but the HTML fallback in `render.sh` does not.
- "Section 4" for products versus "Appendix A" for methods is fine, but a one-line roadmap after the Summary would help.
- The notebook's comment "the coherent 9.9-d 'unexplained' source in Fig. 1" (cell 8) is now Fig. 2.
- The notebook hard-codes `/Volumes/Data_Vault`; give external users a download path once the table has a public home.

## Optional (polish)

- Line 27: "within 63″" → "at 3–63″", as in the code and line 169. The target's own position (< 3″) is excluded.
- Line 106: "significantly higher contamination of variable stars … due to crowding". This is untested causally, and "significantly" isn't backed by a stated test. Suggested: "a much higher fraction of periodic sources (26% vs 10%), as expected in crowded fields."
- Line 145 / Section 2: state that the flag is computed on the A+B stack, and the 2-d merge rule.
- Line 9: for a MAST-facing note, "Drafted with the assistance of Claude (Anthropic); all content reviewed by the authors" is the conventional form. This is Rae's choice.
- References: add Paegert et al. 2021 (arXiv:2108.04778) for TIC **v8.2** (Stassun 2019 is v8). Add Feinstein et al. 2019 (PASP 131, 094502) for eleanor, which underlies GSFC-ELEANOR-LITE. Cite Lomb–Scargle (Lomb 1976; Scargle 1982) or the astropy implementation.
- Fig. 4 (left) shows 53 wrong points while the text says 51. Both are right, since the panel includes off-diagonal wrong answers, but the caption should say what the red count is.
- Line 283 says "three sectors"; all 124 tested sources do have exactly 3. Note the 5σ clipping step for completeness.

## Questions only Rae or Zach can answer

1. **Q1 (Zach):** may `galaxy_files.csv` (or our per-file mapping, which is equivalent) be published with the note? If not, how should users find a file's noise source?
2. **Q2 (Zach):** before publishing anything implying a reduction artefact (lines 311, 382), does Zach know of any aperture or background step that could create source-specific 1–5 d signals at a few hundred ppm? Would he rather the note stay neutral?
3. **Q3 (Zach):** were the near-LMC `EXTENDED` targets and the Tmag ≤ 15 cut intended? Is there a record of how the ~2,000 galaxies were chosen, of which 1,719 appear?
4. **Q4 (Rae):** was dev_medium used to tune the rotation-period search? If so, Appendix B's numbers are in-sample for that pipeline.
5. **Q5 (Rae):** should the 34 coherent "quiet" sources (M1) be reclassified before release (this changes both tables), or only caveated?
6. **Q6 (Rae):** is the corrupted-segment origin known to Zach? For example, sectors where the target fell near a CCD edge or bleed column. Segment start times are spread through the year, not concentrated in particular sectors.

## Verified and correct

- **Section 1:** 1,719 sources, 210–1,082 files each; the multiplicative evidence reproduces.
- **Section 2:** 170 sources and 107,712 files flagged; median segment 27.8 d; 150 sources with a single segment and 20 with two or three; maximum deviation 357; deviation above 1 in 132 sources; negative values in 125.
- **Section 3 table:** every count, file total, median `P_ref`, median amplitude and LMC count reproduces. So do: 206 sources and 121,502 files; periods 0.15–32 d with median 3.4 d; amplitudes 97 ppm to 5.5%; 242 near the LMC, 62 of them periodic (26%) against 144/1,477 (10%), ratio 2.63; 137,980 files; Gaia neighbours 89.5 near the LMC against 6 elsewhere; the five largest signals; 67% against 8.6%.
- **Section 4:** every documented column exists with the stated meaning. `periodic` = search OR ls_periodic exactly. `P_ref` follows the stated rule. LS power median 0.015 and 90th percentile 0.067. `ffi_check` agrees with `P_ref` < 5 d. 197 sources and 114,829 files in the exclusion classes.
- **Per-file table:** 1,000,000 unique `file_number` (0–999,999), `sim_id` and filenames. All 14 shared columns are identical to the per-source table. `noise_tic` matches Zach's `simulation_properties_sorted.csv` → `galaxy_files.csv` for 100% of files. Joining `galaxy_files.csv` on `file_number` instead would agree for only 0.08% of files, so the note's warning to use `sim_id` matters.
- **Appendix A:** the stacking description matches `galaxy_stack.py` (K = min(80, n//3), parity split, 4 top + 8 random held out). Residuals are 68 / 0.25 ppm for stacked files and 1,724 / 5.5 ppm for held-out files, against a typical 963 ppm. The classification order and rules match `galaxy_phase2_analysis.py`. The FFI method matches `ffi_lc_compare.py`.
- **FFI table:** every count, Wilson interval and one-sided Fisher p reproduces; there are 82 untestable sources. The 12,244 against 336 ppm example is TIC 10000584619. The 3.2–3.3-d trio is confirmed.
- **Appendix B:** 51/88, 55/128 = 43%, 33/872 = 3.8%, and 22.7% against 25.1% all reproduce. The "no detection → wrong" reading holds: declines fall from 70% to 29%.
- **Notebook:** `inspect_flags.ipynb` executes cleanly on a copy, and its consistency checks pass.
- **References:** Claytor et al. 2022 ApJ 927, 219; Handberg et al. 2021 AJ 162, 170; Powell et al. 2022 RNAAS 6, 111; Huang et al. 2020 RNAAS 4, 204; Gaia DR3 / Eyer et al. 2023; and the DOI all match their sources. They support the statements made (for the HLSPs, the HLSP pages say so).

## Not checked, or uncertain

- I did not re-run the rotation-period search, so its false-positive rate on red noise is unmeasured. That is the main remaining gap behind M5. The time-split test (M1) bounds it only indirectly: 102 pipeline-only detections pass the period-binned S/N test less often (14%) than LS detections (56%).
- At P ≥ 10 d my time-split S/N statistic has too few independent frequencies to mean much. The long-period periodic sources' top peak recurs in both halves 46–71% of the time, against about 6% for quiet ones, so I make no claim that they are spurious.
- The WAVELET-image claim (line 77) and the "files do not carry a quality flag" claim (line 71) were not re-inspected.
- Claytor et al. 2022's noise-selection text was checked only via `docs/SMARTS_REFERENCE.md` and the abstract, not by re-reading §2.
- The Gaia period match used DR3 per-class period tables for neighbours flagged `VARIABLE` at 3–63″, with a 3% tolerance at P, P/2 and 2P. The null permutes `P_ref` among the 63 sources that have neighbour periods.
