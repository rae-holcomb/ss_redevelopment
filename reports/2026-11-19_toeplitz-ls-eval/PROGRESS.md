# PROGRESS — Toeplitz-LS evaluation (P50)

- [x] Step 1 fit check (no compute). Toeplitz-LS (Rutkowski & Poleski, arXiv 2611.20300v2), MIT, pure C via ctypes,
  numpy-only Python dependency, no internal threads, no BLAS. nterms=1 + normalization="standard" = floating-mean
  generalized LS R^2 (same as astropy standard, fit_mean=True, center_data=True). `tls.power(Nf, df, f0, ...)` takes any
  regular grid, so astropy's autopower grid can be reproduced exactly. Nothing rules it out.
  Key context: SMARTS light curves are regridded to 30-min cadence by `load_smarts_fits` (not 2-min), so N is ~1.3k
  finite points per sector, Nf = baseline*12 c/d*spp ~ 300 per sector, ~4k full-year. LS is ~6-11 ms of a
  0.35-0.6 s star in T31's profile (≈2%).
- [x] Step 2 build: commit c0aa3ec0d9347956ebce3db59ed8ea3d4dfc4edc (2026-11-15), `make generic` and `make native`
  with Apple clang 13.1.6 in the scratchpad; loaded via PYTHONPATH, nothing installed in ss_openblas.
  Smoke test (synthetic N=1200, 323 freqs): tls vs astropy exact (cython) 3e-14; tlsf 3e-6; astropy `fast`
  (our current path) vs exact 2e-4.
- [x] Step 3 accuracy (5 dev_small_clean stars x assigned/full-year x LS/LS_short grids = 20 periodograms;
  `accuracy.py` -> data/accuracy.csv, candidates_top10.csv): tls vs exact <= 4e-14; tlsf <= 7e-7; astropy fast
  (today) vs exact 4e-7 .. 1e-4. Top-10 lists: tls == exact in 20/20; astropy fast == exact in 19/20 (201481
  full-year LS_short differs). NOTE: this run started while the other session's stage2_harness was running
  (my gate wasn't in place yet); it's a ~1 min light job and accuracy is timing-independent. Later runs gated.
- [x] Step 3 speed (timing.py, rerun clean after run 1 was flagged; run 1 kept as data/timing_run1.csv) and end-to-end (profile_pipeline, 5 stars x 3 repeats, both modes; end_to_end.py)
- [ ] Report, P50 entry, push
