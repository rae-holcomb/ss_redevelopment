# WG6 reference: the RGES PIT variable-star light-curve set

**There are two WG6 products on the drive:**

- the **raw** archival set, `wg6_roman_data_raw/` (§1–7);
- the **resampled** Roman-cadence set, `wg6_roman_data_resampled/` (§8), received 2026-11-11.

For Roman work, use the resampled set (improved noise).

Local reference so future sessions don't have to re-derive it. Compiled 2026-11-12 (T28 scoping,
`reports/2026-11-12_wg6-dataset-scoping/`) from three sources:

1. **The files themselves** on the drive, `/Volumes/Data_Vault/wg6_roman_data_raw/` (`$WG6_DATA_ROOT`): every primary header
   (inventory), plus a sample of tables per set. **This is the ground truth, and it wins wherever 2 and 3 disagree.**
2. **The generating code** `github.com/rges-pit/variable_catalog_xmatch`, commit `5982d92` (2025-10-01), read in a
   scratch clone (not vendored).
3. **`README.txt` / `README.docx`** shipped with the download. These are copied from other people's notes, and partly
   out of date.

Companion: `docs/ROMAN_REFERENCE.md` (GBTDS survey design; draft).

---

## 1. The headline: these are archival light curves, not Roman simulations

The README describes a pipeline that goes GP fit → resample onto Roman GBTDS time sampling → transform filters to
Roman bands → apply the Roman noise model. **Neither the code nor the files reflect that pipeline.**

- The repo only **cross-matches catalogs and compiles archival photometry**: OGLE-IV I/V from the OGLE OCVS web
  server; UKIRT microlensing-survey H/K from a local disk copy (2015–2017); and TESS 20-s PDCSAP for the flare stars.
- No GP, resampling, filter-transformation or Roman-noise code exists in the repo. A grep for
  `gaussian|celerite|tinygp|noise|cadence|resampl|F146` finds nothing relevant.
- Every file on the drive has the same `*_multiband_lc.fits` layout, written by `utils.output_multiband_lc`:
  - extensions are `LC_I`, `LC_V`, `LC_H<k>`, `LC_K<k>` (OGLE/UKIRT), or `0`, `1`, ... (one per TESS sector);
  - columns are `HJD, mag, mag_error` (TESS: `BTJD, mag, mag_err, PDCSAP_FLUX, PDCSAP_FLUX_ERR`);
  - times are real calendar epochs (2001–2024), not Roman seasons.

*(§1–7 describe the raw set. WG6 has since shipped the Roman-resampled product: see §8.)*

So, for our purposes, the raw set is **a labelled library of real bulge variable stars (plus nearby TESS M dwarfs)
in ground-based I band**. It is not Roman-like photometry. Making it Roman-like is our job (or WG6's next release):
cut it to ~70 d seasons, resample to 12.1–15 min, and scale the noise to F146 at bulge magnitudes. See §6.

## 2. Sets on the drive (as of 2026-11-11)

Extracted with `tools/extract_wg6.py` (resumable, atomic) to `$WG6_DATA_ROOT/extracted/<tarball stem>/`.
The original `RRLYR_lightcurves/` at the root is an identical earlier extraction of `RRLYR_lightcurves.tgz`.
Use `extracted/` only.

Counts are files = stars (one file per star). Source catalogue and class definitions come from the OGLE Collection
of Variable Stars (OCVS) for the bulge, `https://www.astrouw.edu.pl/ogle/ogle4/OCVS/blg/<cls>/`.

| Class | Folder(s) under `extracted/` | Stars | Source / truth catalog | Primary band, median points (all bands) | Truth periods joined | Catalog period range (5–50–95%) |
|---|---|---|---|---|---|---|
| **ROT** spotted rotators | `Rotators_lightcurves` | 3,172 | OCVS `rot` (Iwanek+2024, 18,441 in the bulge) | OGLE I, 12,379 | 3,172 (100%) | 1.0 / 25.7 / 74.6 d |
| **FLARE** M dwarfs | `flare_lightcurves-003` | 390 | Günther+2020 (297), Medina+2020 (93) | TESS 20 s PDCSAP, 211,181 (all sectors) | 273 (70%) rotation P | 0.32 / 1.4 / 124 d |
| **ECL** eclipsing binaries | `ECL_lightcurves1`, `ECL_lightcurves2-002`, `ECL_lightcurves3` | 47,668 (17,105 + 29,370 + 1,193) | OCVS `ecl` (37,270 non-contact, 10,395 contact, 3 CV) | OGLE I, 7,401 | 47,660 (99.98%) | 0.31 / 0.58 / 8.2 d (P_orb) |
| **ELL** ellipsoidal | `ELL_lightcurves` | 4,227 | OCVS `ecl/ell.dat` | OGLE I, 7,351 | 4,227 (100%) | 0.36 / 1.39 / 227 d (P_orb) |
| **HB** heartbeat stars | `HB_lightcurves` | 94 | OCVS `hb` (Wrona+2022) | OGLE I, 12,704 | 94 (100%) | 128 / 246 / 437 d (P_orb) |
| **RRLYR** | `RRLYR_lightcurves` | 6,809 | OCVS `rrlyr` (4,964 RRab, 1,828 RRc, 17 RRd/aRRd) | OGLE I, 6,719 | 6,809 (100%) | 0.27 / 0.51 / 0.68 d |
| **DSCT** δ Scuti | `DSCT_lightcurves` | 1,160 | OCVS `dsct` | OGLE I, 11,314 | 1,149 (99%) | 0.049 / 0.075 / 0.15 d |
| **CEP** classical Cepheids | `CEP_lightcurves` | 25 | OCVS `cep` (F, 1O, multimode) | OGLE I, 1,898 | 25 (100%) | 0.29 / 1.1 / 4.4 d |
| **T2CEP** type II Cepheids | `T2CEP_lightcurves` | 224 | OCVS `t2cep` | OGLE I, 8,375 | 224 (100%) | 1.2 / 8.5 / 36.7 d |
| **LPV** (all Miras) | `LPV_lightcurves` | 4,373 | OCVS `lpv/Miras.dat` | OGLE I, 1,906 | 4,373 (100%) | 171 / 319 / 490 d |
| **VVV-LPV** | `VVV_lightcurves/VVV_lc` | 401 | VVV (Nikzat+2022); numeric VVV IDs | UKIRT H/K only, **50** | 0 (not joined) | — |
| **BLAP** | `BLAP_lightcurves` | 9 | OGLE BLAPs (not in OCVS `blg/`) | OGLE I, 14,024 | 0 | LS ~0.02–0.05 d |
| **DN** dwarf novae | `DwarfNovae_lightcurves` | 198 | OGLE DN (Mróz+2015) | OGLE I, 5,877 | 0 (outbursts, no period) | — |
| **TR** OGLE transits | `Transits_lightcurves` | 74 | OGLE-TR candidates | OGLE I, 14,251 | 0 | — |
| **Total** | | **68,824** | | | | |

Download layout: `ECL_lightcurves2-002.tgz` and `flare_lightcurves-003.tgz` sit at the root. The other tarballs are
inside the Google Drive zips (`-001`: ECL1, VVV, T2CEP, LPV; `-004`: README.docx, HB, Rotators, ELL; `-005`: CEP, DN,
ECL3; `-006`: BLAP, Transits, DSCT). There are no zips `-002`/`-003`, which is consistent with those two `.tgz` names.
Extracted size is 22 GB on the drive.

**Truth tables on the drive**, from `tools/fetch_ogle_truth.py`, downloaded 2026-11-11 17:17 UTC with Rae's OK. There
are 35 files, 105.8 MB in all, under `extracted/_truth/<cls>/`, all from `https://www.astrouw.edu.pl/ogle/ogle4/OCVS/blg/`.
The full URL/size list is in `extracted/_truth/manifest.csv`, with a copy in
`reports/2026-11-12_wg6-dataset-scoping/data/ogle_truth_manifest.csv`. Main tables:

- `rot/rot.dat` (1.07 MB)
- `ecl/ecl.dat` (30.6 MB) and `ecl/ell.dat` (1.83 MB)
- `ecl/ident.dat` (45.1 MB; subtypes)
- `rrlyr/RRab.dat`, `RRc.dat`, `RRd.dat`, `aRRd.dat` (7.2 MB in all)
- `dsct/dsct.dat` (2.0 MB)
- `lpv/Miras.dat` (2.1 MB)
- `t2cep/t2cep.dat` (139 kB)
- `cep/cep*.dat` (21 kB)
- `hb/hb.dat` (48 kB)

The join is in `tools/audit/wg6_truth.py`, and the result in `reports/2026-11-12_wg6-dataset-scoping/data/truth_joined.csv.gz`.

Not in this download, although the README mentions them: Be stars ("being executed"), CVs (only dwarf novae are
present), and a non-periodic kernel set.

## 3. Header keys (primary HDU)

| Key | Meaning | Notes |
|---|---|---|
| `NAME` | OGLE ID (`OGLE-BLG-RRLYR-03103`), VVV numeric source ID, or TIC ID (flares) | Join key to truth catalogs |
| `RA`, `DEC` | deg, ICRS | |
| `VARTYPE` | WG6 class string | **ELL files say `eclipsing_binary`**; Cepheids say `cephied` / `cephied_type2` (typos) |
| `NDATA_<band>` | rows in each extension | `NDATA_I`, `NDATA_V`, `NDATA_H0`, `NDATA_K1`, ...; TESS `NDATA_0`, ... |
| `UKSRC<k>` | UKIRT source table for extension `H<k>`/`K<k>` (year, field, CCD) | |
| `SECTOR<k>`, `CCD<k>` | TESS sector / CCD of extension `<k>` (flares only) | |
| `VSOURCE`..`TSOURCE` | constant boilerplate (`OGLE`, `OGLE`, `UKIRT`, `UKIRT`, `TESS`) on every file | **Not provenance: `TSOURCE='TESS'` appears on OGLE stars too** |

**No truth parameter is stored in any file**: no period, amplitude, epoch, subtype or mean magnitude. Truth has to
be joined by `NAME` from the source catalogs (§4).

## 4. Truth parameters: where they live

| Set | Truth source | In the WG6 repo? |
|---|---|---|
| OGLE classes (ECL, ELL, RRLYR, ROT, DSCT, CEP, T2CEP, LPV, HB, BLAP, DN, TR) | OCVS `<cls>/ident.dat` (ID, subtype, coordinates) + **`<cls>/cat.dat` / `<cls>.dat` (period, mean I, amplitude, epoch)** | `ident.dat` only (subtype, no period) for 9 classes; **no `cat.dat`** |
| Flare stars | Günther+2020 (AJ 159, 60) per-flare table: amplitude, FWHM, energy, **rotation period `Per`** (886 of 1,228 stars); Medina+2020 (ApJ 905, 107): **`Prot`** (72 of 125), flare rate | yes (`data/*_mrt.txt`) |
| VVV LPVs | Nikzat+2022 (A&A 660, A35) tables B1/B2 (period) | yes (`data/VVV_*.fits`) |

Period conventions that matter (OCVS, standard; **verify once the cat.dat files are joined**):

- **ECL: OCVS gives the orbital period P_orb.** For near-equal minima, the strongest LS/ACF peak is P_orb/2.
- **ELL: OCVS gives the orbital period.** The light curve has two maxima per orbit, so the photometric period is P_orb/2.
- **ROT: the photometric (spot) period**, directly comparable to a rotation period.
- **RRLYR / CEP / T2CEP / DSCT / HB / BLAP: the pulsation (or HB orbital) period.**

## 5. Discrepancies: README vs code vs files (trust the files, then the code)

1. **README: "lightcurve resampling (GP) – complete" and "the Roman filter noise model is applied".** Neither the code nor
   the files reflect this. The files are raw archival photometry (§1).
2. **README: "magnitude-shifted copies ... to show more realistic uncertainties."** None are on the drive. One file per star, at
   the star's real magnitude.
3. **README lists CEP, T2CEP, DSCT, ECL, ELL, HB, LPV, RRLYR, flares, CVs/DN.** The download also has **Rotators (OGLE
   spotted variables), Transits (OGLE-TR), BLAPs and a VVV LPV set**, which aren't mentioned. There are no CV (non-DN)
   or Be files.
4. **Time stamps: CEP and T2CEP `LC_I`/`LC_V` HJD values are ~4.9e6.** OCVS already ships full HJD for those classes,
   and `utils.parse_ogle_lightcurve` adds +2450000 again. Subtract 2450000 when the value is > 3e6 (done in
   `tools/audit/wg6_profile.fix_time`).
5. **`fetch_ogle_photometry` tests `Type == 'cephied'`, but the catalog types are `cepheid` / `cepheid_type2`.** As
   committed, Cepheids would get no OGLE data. The files do have it, and their `VARTYPE` says `cephied`, so the
   data were made with a different (uncommitted) version of the code.
6. **ELL files carry `VARTYPE='eclipsing_binary'`.** Use the folder or `NAME` prefix (`OGLE-BLG-ELL-`) for the class, never `VARTYPE`.
7. **`VSOURCE`..`TSOURCE` are hard-coded constants.** They don't record which surveys contributed.
8. **The code doesn't run as committed:** `variable_catalog.py` imports a missing `event_cat_build`;
   `repack_tess_lc.py` calls `output_multiband_lc` with 5 arguments (the signature takes 4); the README names a
   `compile_lightcurves.py` that doesn't exist (the real script is `extract_lightcurves_from_index.py`). So the
   files were made by code that differs from the public repo.
9. **The "RGES field" is a 4-deg-radius circle** (~50 deg², centred at l = 359.68, b = −1.65). The actual GBTDS
   footprint is ~1.7 deg², so most stars lie outside it. That's fine for a training library.
10. **TESS flare stars are not in the bulge**: they are an all-sky sample of nearby M dwarfs (Günther: TESS sectors 1–2;
    Medina: within 15 pc). The light curves are 20-s cadence PDCSAP (`FAST-LC`), only quality == 0 cadences, and the
    flux is converted to magnitude with ZP 20.44.

## 6. Quirks that would confuse a rotation pipeline

Measured on a 524-star sample (30 per class, 100 ROT; `tools/audit/wg6_profile.py`). Report figures are in
`reports/2026-11-12_wg6-dataset-scoping/`.

1. **Ground-based sampling, not Roman-like.** OGLE I covers 2010–2024 (DN from 2001), with ~240–270 d seasons,
   daytime gaps, and a median spacing of ~20–60 min within a night (CEP/T2CEP/LPV fields: hours to a day).
   - The densest 70 d window of a well-sampled star has ~1,250 points. A 12.1-min GBTDS season has 8,331.
   - On a regular 15-min grid only ~15–18% of cadences are filled, and `compute_acf`'s `min_valid_frac=0.3` cut then
     truncates the ACF after ~17 lags. At 60 min the ACF survives, but the comb fit still found nothing on 3 of 3 ROT
     test stars.
   - **The data must be resampled (Roman-like) before it can go through the pipeline.**
2. **Daily aliases dominate raw periodograms.** The 1-d window peak (0.997 d) is the top LS peak for 19% of ROT stars,
   many HB stars and the DN/TR sets. Our quick check skipped peaks at 0.95–1.05 d, 0.49–0.51 d and >300 d. Roman (space)
   data won't have these, but any analysis of the archival data will.
3. **EB and ELL periods are orbital.** Lomb–Scargle finds P_orb/2: 79/90 ECL and 28/30 ELL in the check. Score EBs
   against both P_orb and P_orb/2.
4. **ROT periods are photometric spot periods and reliable.** 88/89 measurable ROT stars match OCVS within 10% and 1 is at P/2.
   The other 11 are unmeasurable in raw LS because the daily alias dominates. ROT is **giant-dominated**: I = 15.0 / 16.8 / 18.3
   (5/50/95%) at bulge distance, I amplitude 0.06 / 0.14 / 0.31 mag. That is much larger than a Sun-like spot signal.
   37% have P > 35 d, beyond one season's R/2. OCVS remarks flag 52 of the WG6 ROT stars as flaring and 4 as RS CVn (EB IDs).
5. **HB light curves show harmonics** of P_orb (P/2, P/3, P/5 in the check) plus the daily alias.
6. **T2CEP (1–37 d) sits squarely in the rotation range**: strictly periodic and saw-tooth, so it is a good contaminant.
   CEP (0.3–4 d) and RRLYR (0.27–0.68 d) contaminate the sub-day/few-day bins; DSCT is mostly below 0.1 d.
7. **Noise.** The OGLE I point-to-point scatter goes from ~3 mmag at I ≈ 13 to ~100 mmag at I ≈ 19. SMARTS-TESS 30-min
   p2p is 0.23–2.0 mmag (5–95%). OGLE errors (`mag_error`) are smaller than the measured p2p for faint stars.
   Magnitudes are quantised to 1 mmag.
8. **Flare stars are TESS, not bulge.** They have 20-s PDCSAP, 1–several sectors (one extension each), ~25 d per
   sector, and a TESS-mag 6–6.5 mmag p2p at T ≈ 11. Literature periods come from different methods (Günther: TESS 1–2
   sectors; Medina: MEarth/TESS, up to 180 d). 14/22 match our quick LS, and the misses are mostly P > 1 sector.
9. **Every file has V, H and K extensions of a few to a few hundred points.** UKIRT H/K come from 2015–2017 only. Ignore
   them for periods; they could feed colour-amplitude features.

## 7. How to load one (until a `claude_files/wg6.py` exists)

```python
from astropy.io import fits
import numpy as np
with fits.open(path) as h:
    d = h["LC_I"].data                      # OGLE I band (or h[1] for a TESS-sector file)
    t = np.asarray(d["HJD"], float)
    t = np.where(t > 3e6, t - 2450000, t)   # quirk 5.4
    flux = 10 ** (-0.4 * (d["mag"] - np.median(d["mag"])))   # relative flux, median 1
```

---

## 8. The resampled (Roman-cadence, Roman-filter) set: `wg6_roman_data_resampled/`

Received 2026-11-11 (~30 GB of zips, **42 GB uncompressed**). **Not extracted:** our tools read each FITS member in place
from the zips (`tools/audit/wg6r_profile.py`; root `$WG6_RESAMPLED_ROOT`). Profiled in
`reports/2026-11-12_wg6-dataset-scoping/` ("Resampled set" section), from a sample of 138 objects: 12 per OGLE class and
30 FL objects at every magnitude copy.

### 8.1 What it is (READMEs and files agree, with the exceptions listed below)

- The light curves are **GP models of the archival OGLE/UKIRT light curves** (§1–7), evaluated on the Roman time grid,
  translated to Roman filters, and given noise. The one exception is **FL**: TESS M-dwarf templates (rotation + flares).
- **Filters and cadence:**
  - **F146**: 50,700 points at **12.1 min**;
  - **F087**: 1,704 points every **6 h**;
  - **F213**: 1,704 points every **6 h**, offset +3 h from F087.
- **Seasons:** 6 seasons of exactly **71.00 d**: Aug 14–Oct 23 and Feb 10–Apr 21, in consecutive years 2027–2030
  (JD 2461631.5–2462613.5).
  - Each season is contiguous (no gap > 1 h).
  - Seasons are separated by 109–115 d.
  - No off-season points.
- **File layout:** each FITS has a PRIMARY header with only `NAME` and `VARTYPE`, plus `F087`, `F146` and `F213`
  binary tables.
  - Columns `mag, mag_error` (simple), or `JD, mag, mag_error` (improved `_final`).
  - Simple-noise times are in `simple_noise/drive-download-*/roman_times_{shortcadence,longcadence,longcadence2}.npy`
    (F146, F087, F213).
- **No truth parameters in any file.** Join by OGLE ID to the OCVS tables (§4) with `wg6_truth.py --resampled`
  (writes `resampled/truth.csv.gz`). For FL, join the TIC in the file name to Günther/Medina.
- **Truth matches:** all OGLE-class stars join to OCVS (100%).
  - The 1,000 resampled **ECL stars are not in the raw WG6 ECL set**: none overlap, although all are in OCVS.
  - The other classes are 97–100% in the raw set.
  - CV (dwarf novae) is in the raw set, but has no period table.
- **The underlying signal is strictly periodic.** For ECL and T2CEP, the season-to-season folded-profile correlation is
  1.000 and the amplitude is constant. FL rotation amplitudes are also constant across seasons (e.g. 0.0600–0.0603 mag),
  so **there is no spot evolution**.

### 8.2 Contents (files / objects)

| Class | simple | simple_shifted (+8 mag) | improved (`_final`) | FL non-final (improved folder) |
|---|---|---|---|---|
| CEP | 24 | 24 | — | |
| DSCT | 1,000 | 1,000 | 1,000 | |
| ECL | 1,000 | 1,000 | 1,000 | |
| ELL | 1,000 | 1,000 | 1,000 | |
| HB | 93 | 93 | 93 | |
| LPV (Miras) | 991 | 976 | 991 | |
| RRLYR | 991 | 990 | 991 | |
| T2CEP | 100 | 100 | 100 | |
| CV (dwarf novae) | — | — | 1,100 files = 110 stars × 10 (`_ind<k>_<j>`) | |
| FL (M-dwarf rotation + flares) | — | — | 12,000 files = 2,000 objects × WFmag 16–21 | 10,000 = 2,000 × WFmag 16/18/20/22/24 |
| **ROT (OGLE spotted rotators)** | **absent** | **absent** | **absent** | |

**FL objects:** `FL_<idx>_<TIC or "flat">_<k>_WFmag_<m>`. There are 2,000 objects on **65 TESS templates plus "flat"**
(79 objects). 54 templates (1,600 objects) have literature periods: mostly 0.2–5 d, plus one at 13.3 d and one at 89 d.
Each template is reused 19–41 times with different `k`, which appears to be a flare draw. **Split by template, never by object.**

**No ROT:** the WG6 repo's unmerged branch `add_rotators` (commit `4130e4e`) adds the OGLE ROT/BLAP/transit catalogs to the
*archival* extraction (the raw `Rotators_lightcurves.tgz`), but no resampled ROT set was shipped. BLAP and TR are absent
too, and so is the non-periodic kernel set the README mentions.

### 8.3 The two noise variants (measured; READMEs are vague)

Every variant carries **the same underlying light curve**: correlation with simple is 0.998–1.000 for all OGLE classes
except LPV. They differ only in the magnitude offset and the noise draw.

| | simple | simple_shifted | improved (`_final`) |
|---|---|---|---|
| Magnitude | original OGLE/UKIRT-based brightness (F146 ~10–18) | simple **+8.000 mag** exactly (F146 18–26) | simple + **0, ~4.45 or ~6.68 mag** (step shift; stars already fainter than ~15.4 are not shifted) → F146 ~14–20; FL at WFmag 16–21 |
| Error model σ(mag) | **one deterministic curve**: 0.002–0.006 dex scatter across stars; floor **1.0 mmag** (bright), 2.2 mmag at 18, 5.9 at 20, 18 at 22, 69 at 24, ~300 at 26 | same curve as simple | **star-dependent**: 0.05 dex scatter (up to ×1.5) at fixed magnitude, as expected from the stated blending/crowding term. Median **below** the simple curve at 17–20 mag (1.27 vs 1.65 mmag at 17; 4.3 vs 5.9 at 20). Same 1.0 mmag floor |
| Per-point error | tracks the instantaneous magnitude (within-star corr(mag, log σ) = 0.997) | 0.986 | 0.956 |
| Noise draw | white Gaussian at the quoted σ: autocorrelation lags 1–5 ≈ 0, binned RMS/(σ/√N) = 1.00 at 4, 16, 64 points | same | same |
| Times | in `.npy` side files | in `.npy` side files | `JD` column in each file |
| NaN cadences | none seen | none seen | FL only: 0.04–0.3% of points |

- **What "improved" adds:** per-star crowding/blending scatter in σ. I don't see added correlated noise or diluted
  amplitudes: the 5–95% amplitude matches simple to within ~1%. So "blending" acts on the error budget only, as far as these
  files show.
- **What neither variant has:** red noise, systematics, flux dilution, saturation, or season-to-season zero-point offsets.
  That makes both optimistic for real Roman data.
- **LPV is different:** the LPV signal itself differs between variants (correlation 0.96–0.999; the difference series is
  fully correlated, ac1 = 1.0), so the LPV GP was re-drawn per variant.
- **Unresolved:** the exact improved magnitude-shift rule (why 4.45 vs 6.68); what the CV `_ind<k>_<j>` copies are;
  what FL `k` is; why "improved" σ is *lower* than simple at 17–20 mag; and how the FL non-final files (WFmag up to 24,
  σ on the simple curve) relate to the "simple noise" model.

### 8.4 Period conventions (unchanged from the raw set)

LS over 6 seasons on F146 (improved), sample of 12 per class:
- ECL 11/12 and ELL 12/12 at **P_orb/2**;
- HB at P_orb/2 or P_orb/3;
- RRLYR, DSCT and T2CEP all 12/12 match;
- CEP (simple) 12/12;
- **FL: 23/25 objects match the template's literature rotation period at every magnitude copy** (138/150 files).

LPV can't be checked with a 300 d LS cap.

### 8.5 Loading (until `claude_files/wg6.py` exists)

```python
import io, zipfile, numpy as np
from astropy.io import fits
with zipfile.ZipFile(zip_path) as zf, fits.open(io.BytesIO(zf.read(member))) as h:
    d = h["F146"].data
    t = d["JD"] if "JD" in d.columns.names else np.load(".../roman_times_shortcadence.npy")
    flux = 10 ** (-0.4 * (d["mag"] - np.nanmedian(d["mag"])))   # NaN-safe (FL has NaN cadences)
# one season: split t at gaps > 1 d (6 × 71 d)
```

