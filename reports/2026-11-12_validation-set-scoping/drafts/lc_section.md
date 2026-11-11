### 6. Light-curve products

Scope note: rows marked ✔ were checked against the paper or the MAST HLSP page this session; the rest are from
the papers as I know them and should be confirmed on one file before a loader is written.

| Product | Photometry / crowding | What it does to long periods | Mag limit, sectors | Cluster stars | Loader needs |
|---|---|---|---|---|---|
| **TESS-SPOC FFI** (Caldwell+2020) ✔ | SPOC aperture photometry; targets need crowding metric ≥ 0.8 ✔ | `PDCSAP_FLUX`: cotrending basis vectors fitted per sector. It attenuates or distorts signals longer than ~10–13 d (the reason Claytor+2024 built custom light curves). `SAP_FLUX` is in the same file, untouched. | ≤160k stars per sector; Tmag ≤ 13.5, H ≤ 10, d < 100 pc, all 2-min targets ✔. Started at S14, since back-filled; per-star coverage is patchy (example SCVZ star: 12 SPOC sectors vs 42 QLP). | Poor in dense clusters (crowding cut); OK for nearby open clusters | Standard lightkurve-style FITS (`TIME`, `SAP_FLUX`, `PDCSAP_FLUX`, `QUALITY`); per-sector files to stitch. Closest to an existing format. |
| **QLP** (Huang+2020; Kunimoto+2021/22) ✔ | Aperture photometry on TICA-calibrated FFIs; no deblending | `KSPSAP_FLUX` is spline-detrended for transit search and removes rotation beyond ~1 d. `SAP_FLUX` (raw) is kept. | All stars with Tmag ≤ 13.5 in every sector ✔; the most complete sector coverage | Same crowding as SPOC, without the cut | Per-sector FITS; read `SAP_FLUX`, never `KSPSAP_FLUX` |
| **eleanor / GSFC-eleanor-lite** (Feinstein+2019; Powell+2022) | Aperture + PCA (SPOC CBVs) or analytic-Gaussian PSF | `CORR_FLUX`/PCA regress out long trends much like PDC; `RAW_FLUX` is available | Tmag ≤ 16 in eleanor-lite (patchy by sector); the eleanor code itself is no longer actively maintained | Gaussian PSF is crude in crowded fields | Different column names; needs its own adapter |
| **TGLC** (Han & Brandt 2023) ✔ | **ePSF forward model with Gaia DR3 priors; removes flux from neighbours** ✔ | **`cal_psf_flux`/`cal_aper_flux` are normalised and detrended with a wotan biweight, 1-day window** ✔. That erases rotation beyond ~1 d. Use the raw `psf_flux`/`aperture_flux` columns. | Tmag ≤ 16, all available sectors ✔; negative fluxes for faint stars near variables in S1–26 ✔ | **Best deblending for clusters and crowded fields** | Raw flux plus our own mild systematics correction; per-sector FITS |
| **CDIPS** (Bouma+2019) ✔ | **Image-subtraction photometry** | Three flux versions: raw (IRM), PCA-detrended and TFA-detrended ✔. TFA/PCA remove long trends; IRM keeps them. | G_RP < 16 **cluster and young-star targets only** (~1M, built from Cantat-Gaudin, Kounkel, etc.), 30-min/10-min, sectors 1–55ish ✔ (5 releases) | **Designed for clusters**, but has no field stars, so it can't serve as a general-purpose source | Own FITS layout (multiple apertures, IRM/PCA/TFA); adapter needed |
| (unpopular CPM; Hattori+2022) | Causal pixel model on TESScut cutouts | Can be tuned to preserve long signals; used for TESS rotation | Any star, built on demand from cutouts | Pixel-level, OK in moderate crowding | Built locally from TESScut cutouts |

**Which flux matters for us.** SMARTS noise came from "galaxy" TESScut cutouts reduced with lightkurve regression
correctors, closer to SAP than to PDCSAP (`SMARTS_REFERENCE.md` §5). A fair real-star test therefore uses light curves
**in the same state as the training noise.** Products that detrend on ~1 d windows (QLP `KSPSAP`, TGLC `cal_*`, CDIPS
TFA) would turn every star beyond a few days into a "non-rotator". Those columns must not be used.

**Recommendation.**
- **Primary: TESS-SPOC FFI, read as `SAP_FLUX` (keep `PDCSAP_FLUX` for comparison).** It is the most standard format
  (one small adapter beside `load_smarts_fits`) and covers most Tmag ≤ 13.5 stars from S14 on. Running both fluxes gives
  the PDC long-period-attenuation measurement for free.
- **Fallback: TGLC raw `aperture_flux` / `psf_flux`.** Use it for cluster members and anything with TIC
  `contratio` > 0.1, and for stars that SPOC does not cover, down to Tmag 16. Its Gaia-based deblending is the right tool
  for Pleiades, Praesepe and NGC 2516 fields.
- **Not primary:**
  - **QLP:** use only its `SAP_FLUX`, and only when SPOC is missing; same aperture issues, more sectors.
  - **CDIPS:** clusters only, not field negatives; but its IRM flux is a useful independent cross-check on cluster
    rotators.
  - **eleanor:** unmaintained.
- **Gold standard, later:** if both products prove too lossy at P > 13 d, the long-period answer is TESScut cutouts
  reduced exactly as SMARTS noise was (lightkurve `RegressionCorrector`) or with unpopular/CPM. That is larger
  (≈ 0.5–1 MB per sector per star for a 10×10 cutout at 30-min cadence, more at 200 s) and is Rae's local job.
