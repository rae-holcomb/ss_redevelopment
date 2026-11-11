### 1. Literature check: are white dwarfs and red-clump stars good "non-rotators"?

**What we need from a negative.** We need a star with *no detectable periodic signal anywhere in 0.1–180 d in a TESS FFI
light curve*. That is a property of the light curve, not of the star's spin. "Slow rotator" is good enough only when the
expected spot signal is below TESS noise **and** the star has no other periodic signal (pulsation, binarity, granulation
peaks, a blended neighbour) in that range. A negative set is mostly a test of how often the pipeline invents a period
from systematics and red noise. For that, "astrophysically quiet in 0.1–180 d" matters more than "not rotating".

**Red clump (Claytor & Tayar 2025 used them as noise templates, but in *Kepler*).**
- *Most don't show rotation, but not all.* Ceillier+2017 found rotation in 2.08% of 17,377 oscillating Kepler red giants
  (361 stars), yet in **~15% of 575 low-mass (M < 1.1 M☉) red-clump stars**. They read the excess as mergers and
  engulfment on the RGB. Gaulme+2020 found ~8% of ~4,500 bright Kepler giants with rotational modulation and argue that
  low-mass giants gain angular momentum on the way to the clump (engulfment or merger). Tayar+2015 report rapid rotators
  (APOGEE v sin i) among low-mass giants. **So a red-clump sample selected only on the CMD has a several-% to ~15%
  rotator contamination,** with periods of tens of days, inside our search range.
- *Oscillations and granulation are in range at the short end.* A clump star has ν_max ≈ 30–40 μHz (P ≈ 7–9 h ≈ 0.3 d),
  with granulation at a similar or slightly longer timescale (Mathur+2011: granulation timescale scales as 1/ν_max).
  Both sit right at our 0.1–1 d bins, as quasi-periodic red-noise humps rather than coherent peaks. Claytor & Tayar 2025
  report that their CNN detected **608 asteroseismic frequencies** from red giants as "periods", and that the clump
  light curves were noisier than the rest of the CMD. **The days-to-tens-of-days issue the brief raised belongs to the
  high-luminosity RGB/AGB** (ν_max ≲ 1 μHz → P ≳ 10 d, semi-regular variables). The M_G cut excludes most of those,
  but a CMD-only selection (−0.5 < M_G < 1.5) does admit some upper-RGB stars and AGB bump stars.
- *Blending.* Giants are bright and distant, so at a fixed Tmag they sit in more crowded fields than dwarfs. Claytor &
  Tayar 2025 found that even PSF-deblended Kepler background light curves were still blended for up to 63% of
  periodic pairs. TESS's 21″ pixels make this worse.
- *Usable form:* the clean version of this idea is **asteroseismically classified He-burning (clump) giants in the
  Kepler field with no rotation detected in four years of Kepler data** (Yu+2018 phase = HeB, minus every Ceillier+2017
  rotator or polluted star). Kepler is ~10x more precise than TESS FFIs, so a star with no Kepler rotation signal has
  none TESS can see either. We should still expect the pipeline to "detect" the ~0.3 d oscillation hump; that is a known
  astrophysical signal, so score it as its own outcome (`seismic`), not as a rotation false positive.

**White dwarfs.**
- *Variability classes that hit our range:*
  - Spotted or magnetic WDs rotate at minutes to hours (Oliveira da Rosa+2024: 318 periods among 9,285 TESS WDs;
    median ~3.9 h for single WDs).
  - ZZ Ceti / DBV / GW Vir pulsators sit at 100–1500 s, below our 0.1 d floor except as aliases in 30-min FFIs.
  - Reflection and ellipsoidal effects in WD + M-dwarf and double-WD binaries run from hours to a few days.
  - Steen+2024 confirmed 105 Gaia-variable WDs (binaries, spotted, pulsators; median amplitude 1.4%).
  - Gaia DR3's classifier labels 910 sources "WD" variables.
- *Most of those signals sit below 0.1 d*, so a correctly working pipeline should report "no period in 0.1–180 d" for
  nearly all WDs. That makes them a clean test of systematics-only false positives, which is what we want.
- *But they are faint.* Of the Gentile Fusillo+2021 high-confidence WDs (P_WD > 0.75), only **157 have G < 14 and 566
  have G < 15**; 2,034 have G < 16. At G ≈ 15–16 a TESS FFI star is photon- and background-dominated. The sample tests
  "noise only" well, but the noise is in the faint regime, unlike the Tmag 9–13 stars we most care about.
- *Crowding:* WDs are faint neighbours of brighter stars more often than not, so a TESS "detection" in a WD aperture is
  usually a neighbour's signal. That is a useful blend test, but it is not a clean negative unless TGLC-style PSF
  photometry is used.

**Verdict.**
- **Red clump: usable with care.** Use asteroseismically classified Kepler-field clump stars with no Kepler rotation
  (≈2,300 at Kp ≤ 12; 4,450 at Kp ≤ 13). Do not use a CMD-only clump selection: it carries an unknown several-% to 15%
  rotator rate. Score a 0.2–0.5 d hump as `seismic`, not as a rotation FP.
- **White dwarfs: usable as a small, faint, systematics-only control** (~20–40 stars at G ≲ 14.5). They don't represent
  the brightness of the main sample, and crowding has to be checked star by star.
- **"Slow rotator" is good enough only where the spot signal is below TESS noise.** That is true for clump giants
  (P ~ 100+ d, tiny amplitudes; rapid rotators excluded by Kepler). It is not true for an arbitrary slow-rotating
  dwarf: TESS can see a 20 d K dwarf at 1% amplitude.
- **Complements the literature supports:**
  1. **Kepler dwarfs with no detected period** (McQuillan+2014 non-periodic, after removing Santos+2019/2021
     detections): ~58,600 stars, 4000–6200 K. Same logic as the clump: no rotation signal in four years of Kepler, so any
     confident TESS period is a false positive or a blend. Caveat: activity cycles and evolving spots mean "not detected
     in Q3–Q14" is not "never variable", but it is the strongest label available.
  2. **Inactive old FGK dwarfs with spectroscopic rotation** (log R'HK ≲ −4.9 and P_rot from activity time series, e.g.
     Mount Wilson / HARPS samples). They are few and bright (Tmag < 8, saturation risk), so they are a small
     "bright-star" negative block for later.
  3. Hot A/F stars are **not** quiet: δ Sct / γ Dor pulsate at 0.3–3 d. Gaia DR3 flags 748,058 DSCT|GDOR|SXPHE
     candidates and 54,476 `vari_ms_oscillator` stars.
