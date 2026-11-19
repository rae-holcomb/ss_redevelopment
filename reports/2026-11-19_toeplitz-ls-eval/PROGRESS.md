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
- [ ] Step 3 accuracy on ~5 dev_small_clean stars (assigned + full-year)
- [ ] Step 3 speed vs N, vs Nf, end-to-end share
- [ ] Report, P50 entry, push
