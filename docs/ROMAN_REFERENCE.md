# Roman GBTDS reference (DRAFT, 2026-11-12)

**Status: draft.** Compiled in the T28 scoping session (`reports/2026-11-12_wg6-dataset-scoping/`). Each line is
tagged with how far it is confirmed:

- **[code]**: in the WG6 generating code (`rges-pit/variable_catalog_xmatch`, commit `5982d92`, 2025-10-01).
- **[web]**: from NASA/STScI public pages, read 2026-11-11. Not checked against the final ROTAC/APT survey definition.
- **[unconfirmed]**: recalled or older design. Check it before relying on it.

## 0. What the WG6 code assumes about Roman: almost nothing

The WG6 repo has **no Roman cadence, season, filter or noise model.** Its only Roman-specific object is
`rges_survey_definition.RGESSurvey`:

- one circular footprint of radius 4.0 deg [code];
- centred at RA 267.8359, Dec −30.0608 (l = 359.68, b = −1.65) [code];
- that circle is about 50 deg², roughly 30× the real GBTDS footprint (~1.7 deg², below). It is a generous selection
  region for picking known variables, not the survey footprint.

The README's "GP resampling to Roman's Bulge Survey time sampling", "filter transformations" and "Roman filter noise
model" steps **are not in the repo, and the files on the drive have not been through them** (they are archival
OGLE/UKIRT/TESS photometry). See `docs/WG6_REFERENCE.md` §1. So every number below comes from outside WG6.

## 1. Survey design

| Quantity | Value | Source |
|---|---|---|
| Fields | 6: five ~0.28 deg² bulge fields plus one Galactic Centre field, ~1.7 deg² total | [web] STScI/NASA GBTDS pages; arXiv:2505.10574 |
| Seasons | 6 (3 early, 3 late in the mission) | [web] NASA GBTDS technical page |
| Season length | ~72 d visibility window, twice a year; plan on ~70 d usable | [web] STScI ("~72 days twice per year") |
| Total time | ~15 months of observing time allocated | [web] NASA |
| Main cadence | **12.1 min in F146** (wide, ~0.93–2.0 µm) | [web] NASA; older Penny+2019 design was 15 min in W146 [unconfirmed for final] |
| Exposure | 67 s per F146 visit | [web] arXiv:2505.10574 |
| Colour filters | F087 and F213, each **every 12 h or faster** | [web] NASA. Older design: alternate blue/red every 6 h [unconfirmed] |
| Off-season / between-season sampling | **Not confirmed.** ROTAC discussed low-cadence epochs in intervening seasons (for astrometry and long-timescale events); rate unknown | [unconfirmed] |
| Gap between early and late seasons | Several years (the 3 early seasons sit near mission start, the 3 late ones near the end) | [web], exact dates [unconfirmed] |

## 2. What this means for a rotation pipeline

Computed in Python in the scoping notebook:

- **Points per season:** 12.1-min cadence over 70 d gives ~8,300 F146 points per season; 15 min gives ~6,700. A
  TESS year at 30 min has 17,520.
- **Nyquist period:** 2 × 12.1 min = 0.0168 d. The short-period limit is set by the methods (`P_min`), not by sampling.
- **Per-season P_max:** one contiguous ~70 d season gives R/2 = 35 d. R/3 is ~23 d (the "low-20s d" in
  `docs/TASKS.md` corresponds to R/3, the 3-peak ACF criterion). Seasons several months or years apart
  can't be stitched for rotation (spots evolve), so treat each season as one batch (T16h).
- **Colour bands:** F087/F213 at ≤12 h cadence give ~140 points per season. That is useful for colour-dependent
  amplitude (a pulsator-vs-spot discriminant), but too sparse for periods below ~1 d.
- **Faintness and crowding:** bulge targets are mostly F146 ~ 15–24 mag, with heavy blending. A Roman ranker will need
  blending/crowding features (T28 note). No WG6 file carries a blending estimate.

## 3. Open items to confirm (before the Roman scorecard)

1. The final cadence (12.1 vs 15 min) and the per-field cadence: does the Galactic Centre field have the same cadence?
2. The exact season start/stop dates and the length of each season (~60 d minimum requirement vs ~72 d window).
3. Whether there is off-season sampling, and at what rate.
4. The F146 noise model, i.e. precision vs magnitude for the 67 s exposure. The WG6 README mentions one, but it isn't in the
   repo or the data. Candidates: the Roman exposure-time calculator (`pandeia`/ETC tables), or the Wilson+2023 pixel-level
   simulations (arXiv:2305.16204).

Sources: [NASA GBTDS technical page](https://science.nasa.gov/mission/roman-space-telescope/galactic-bulge-time-domain-survey-technical/),
[STScI GBTDS docs](https://roman-docs.stsci.edu/roman-community-defined-surveys/galactic-bulge-time-domain-survey),
[arXiv:2505.10574](https://arxiv.org/html/2505.10574v2), [arXiv:2305.16204](https://arxiv.org/pdf/2305.16204).
