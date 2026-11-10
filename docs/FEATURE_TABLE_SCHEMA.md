# Ranker feature tables: schema (T10, 2026-11-10)

Produced by `tools/build_feature_tables.py`. Two tables per run (`--outdir`):
`candidates.csv.gz` (one row per candidate, the ranker's training rows) and
`stars.csv.gz` (one row per star: truth, the hand arbitration's final answer, counts).
Feature *definitions and rationale* live in `claude_files/FEATURE_DOCUMENTATION.txt`
(sections A–H, cited below). This file lists every column and says which are model
inputs. In code: `build_feature_tables.feature_columns(df)` returns the model inputs,
everything not in `NON_FEATURE_COLUMNS`.

## Which candidates are rows

Every candidate the comb fit saw, after its dedup step:

| kind | `fit_error` | `beyond_reach` | features evaluated at | fit columns |
|---|---|---|---|---|
| fitted, passed all gates | False | False | `fitted_P` | present |
| fitted, failed ≥ 1 gate | False | False | `fitted_P` | present |
| fit raised (too few windows, etc.) | True | False | `period` (P0) | NaN |
| P0 > searchable range R/2 (T16c), never fit | False | True | `period` (P0) | NaN |

Stars with no candidates at all (`decline_kind` `no_guesses` / `none_sane`) have no
candidate rows but are in `stars.csv.gz` with `n_rows` = 0.

Method-spectrum features (Section D) are evaluated against **all seven** Stage 1
methods' spectra, even though only the three pipeline methods (`pairwise_histogram`,
`lombscargle`, `acf_fft`) feed candidates into the fit.

## candidates.csv.gz

### Identifiers, grouping, provenance of the label (never model inputs)

| column | meaning |
|---|---|
| `star_id` | light curve id (`smarts-tess-v1.0-RRRRRR`); the ranking group |
| `noise_tic` | TIC of the TESS noise light curve (T20 galaxy map); group splits by this too |
| `noise_classification`, `noise_periodic`, `noise_exclude_from_training` | T20 flags (`noise_source_flags.csv.gz`), joined on catalog row |
| `noise_coherent_quiet` | noise TIC is on the 2026-11-09 review's list of 34 "quiet" sources with a coherent short-period signal (`reports/2026-11-10_t10-ranker-feature-tables/data/quiet_coherent.csv`); for T11's strict-quiet test |
| `is_rotator` | True for SMARTS rotators. T27's non-rotator negatives append with False |
| `label_source` | `smarts_injected` here; T27 rows get their own value |
| `cand_index` | row order within the star (fitted candidates first, best-first as the fit sorted them) |
| `t0`, `contributing_methods`, `failed_gates`, `fit_error_message` | bookkeeping strings / phases |
| `selected` | True for the candidate the hand arbitration reported (`fit_rotation_period`); at most one per star |

### Truth and labels (never model inputs)

| column | meaning |
|---|---|
| `true_period` | P_eq (FITS `PERIOD`) |
| `period_lo`, `period_hi` | active-band success range, `evaluation.active_band_period_range` |
| `period_bin` | `evaluation.period_bin(true_period)` |
| `P_eval` | the period the label is computed at: `fitted_P` if fitted, else `period` (P0) |
| `outcome` | `evaluation.classify_period(P_eval, true_period, REL_TOL, (period_lo, period_hi))` |
| `relevance` | **graded label: 2 = `correct` (within REL_TOL of P_eq), 1 = `diffrot` (inside the band), 0 = anything else** |
| `alias_class` | `half` / `double` / empty: the alias class, a column only (aliases have relevance 0) |

### Features (model inputs)

Candidate-generation provenance (FEATURE_DOCUMENTATION Section B): `period`,
`log_period`, `period_frac_of_baseline`, `method` (string; encode in T11), `rank`,
`strength`, `coverage_at_generation`, `n_duplicate_guesses`, `n_contributing_methods`.

Joint fit and gates (Section C; NaN for unfit rows): `n_cycles_spanned`, `fitted_P`,
`P_err_frac`, `t0_err_frac`, `redchi`, `log_redchi`, `n_peaks_used`,
`n_peaks_dropped`, `n_peaks_used_frac`, `n_teeth_available`, `n_peaks_required`,
`few_peaks`, `P_over_lc_baseline`, `half_tooth_ratio`, `frac_positive_heights`,
`frac_non_increasing_height`, `mean_height_snr`, `min_height_snr`, `max_height_snr`,
`std_height_snr`, `height_snr_slope`, `mean_curvature`, `std_curvature`,
`passes_min_peaks`, `passes_redchi`, `passes_height_snr`, `passed_gates`,
`mean/min/std_peak_trough_depth`, `mean/min/std_peak_trough_depth_snr`,
`n_acf_peaks_found`, `coverage_at_fit`.

Gate results (new in T10; from the scorecard's per-candidate records, T16b), one block
per gate in `comb_fit.GATE_NAMES` (`n_peaks`, `frac_positive_heights`,
`mean_height_snr`, `frac_vertex_in_window`, `half_lag_consistency`,
`few_peaks_height`): `gate_<g>_value`, `gate_<g>_pass`, `gate_<g>_threshold`,
`gate_<g>_enabled`; plus `passed_base_gates` (the four original comb-shape gates),
`n_failed_gates`, `acf_pair_frac_at_P` (ACF valid-pair fraction at lag P_fit),
`fit_error`, `beyond_reach`. Thresholds are constants except `n_peaks` (per-candidate
peak budget); T11 can drop the constant ones.

ACF-level context: `acf_std`, `baseline_days` (ACF lag range), `lc_baseline_days`,
`n_acf_points`.

Per-candidate S_ph and alias features (Sections F, G): `s_ph`, `acf_height_at_P`,
`acf_half_lag_contrast`, `acf_odd_even_contrast`, `fold_half_corr`,
`fold_half_corr_at_2P`. See the report for the diffrot check on these.

Method-spectrum features (Section D): `<m>_nearest_peak_{rank,frac_power,rel_distance,width,fap|snr}`
for `m` in `ls`, `ls_short`, `fft`, `fft_short`, `fft_highpass`, `wavelet`;
`fft_highpass_n_windows_tested`, `fft_highpass_best_window`;
`hist_nearest_bin_{rank,frac_count,rel_distance}`; `n_methods_recognize_as_peak`,
`combined_method_strength`.

Star-level light-curve features (Section A, same for every row of a star): `r_var`,
`flicker_std`, `flux_std`, `flux_skewness`, `flux_kurtosis`, `duty_cycle`,
`von_neumann_eta`, `fliper_band_1/2/3`.

Coverage context (Section H): `n_sectors_observed`, `n_contiguous_segments`,
`longest_contiguous_days`, `largest_gap_days` (star-level); `P_over_longest_segment`,
`n_cycles_observed_total`, `tess_timescale_rel_dist`, `P_over_reach` (candidate P / R).

Cross-candidate agreement (Section E; fitted rows only): `n_agreeing_candidates`,
`n_agreeing_methods`, `best_competing_redchi`, `redchi_ratio_to_best_competing`,
`redchi_ratio_to_global_best`, `is_harmonic_of_global_best`,
`harmonic_factor_of_global_best`, `is_global_best_redchi`.

## stars.csv.gz

| group | columns |
|---|---|
| ids / context | `star_id`, `is_rotator`, `label_source`, manifest columns when present (`row`, `sim_id`, `pbin`, `eval_pbin`, `amp_tier`, `shear_class`, `stratum`, `coverage_class`, `sampling_weight`, `coverage_weight`, `combined_weight`, `train_order`, `nested_1k`), the five `noise_*` columns |
| truth | `true_period`, `period_bin`, `period_lo`, `period_hi`, `band_width`, `band_width_class` |
| coverage (scorecard v3/v4 descriptors) | `sectors_requested`, `n_sectors`, `sectors`, `total_baseline_days`, `longest_contiguous_days`, `largest_gap_days`, `duty_cycle`, `acf_reach`, `P_over_reach` (**P_eq / R, truth-derived**, unlike the candidate column of the same name), `searchable`, `flux_amp` |
| Stage 1 | `stage1_<method>_outcome` (best outcome in that method's list), `stage1_<method>_error` if it raised |
| hand arbitration | `final_period`, `final_outcome`, `pipeline_success`, `decline_kind`, `decline_gates`, `n_pooled` |
| counts | `n_fit_candidates`, `n_fit_error`, `n_beyond_reach`, `n_rows`, `max_relevance`, `n_relevant` (rows with relevance ≥ 1), `n_relevant_fitted` |
| run | `error`, `traceback`, `seconds` |

`final_*` columns reproduce `benchmarks/scorecard.py`'s `per_star.csv` for the same
star and code (checked in the report and in `tests/test_build_feature_tables.py`).

## Appending non-rotator negatives (T27)

Use the same builder on T27's noise-only light curves, with `true_period` NaN. Rows get
`is_rotator` = False, `label_source` = e.g. `smarts_noise_only`, `relevance` = 0,
`outcome` = `wrong` (or `none`). `period_lo`/`period_hi` NaN. Every feature column has
the same meaning, so `pd.concat([rotators, negatives])` is a valid table. A star with
`is_rotator` False and `max_relevance` 0 is a correct "decline" target for T11.
(`build_one_star` takes `is_rotator` and `label_source` from the task `context` when
present, default True / `smarts_injected`.)
