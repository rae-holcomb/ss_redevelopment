# SMARTS reference: how the training light curves were generated

Local reference so future sessions don't need to re-derive this. Compiled
2026-10-13 from three sources, cross-checked against each other:

1. **Claytor et al. 2022, ApJ 927, 219** (arXiv:2104.14566), Sec. II–III and Table 1:
   the SMARTS-TESS (DR1) paper.
2. **butterpy source code** (github.com/zclaytor/butterpy), as of commit
   `35c119f` (2021-04-26, the era when DR1 was generated): `generate.py`,
   `butterpy/spots.py`, `butterpy/regions.py`.
3. **The actual FITS headers of 10,000 SMARTS-TESS files** already processed in
   `results_products/batch1/summary.csv`. This is the empirical ground truth, and it
   wins wherever 1 and 2 disagree.

Other links: MAST HLSP page https://archive.stsci.edu/hlsp/smarts (DOI
10.17909/davg-m919). DR2 (Kepler/KBONUS, 1M sims, red-clump noise) is described in
Claytor & Tayar 2025 (arXiv:2506.03248) and is not used here.

---

## 1. Simulation input parameters (SMARTS-TESS DR1)

| Quantity | FITS key | Range | Distribution | Notes |
|---|---|---|---|---|
| Equatorial period P_eq | `PERIOD` | 0.1–180 d | **uniform** | Label used everywhere. The repo's `generate.py` says 0.1–365, but headers confirm 0.1–180. |
| Activity level A (x solar) | `ACTIVITY` | 0.1–10 | log-uniform | Headers: median 0.98, mean 2.1. `generate.py` says uniform, which is wrong for the released data. |
| Activity cycle length | `CYCLE` | 1–40 yr | log-uniform | |
| Cycle overlap | `OVERLAP` | 0.1 yr – cycle length | log-uniform | |
| Inclination i | `INCL` | 0–pi/2 | uniform in sin^2 i | **Header is in radians.** |
| Min. *average* spot latitude | `MINLAT` | 0–40 deg | uniform | |
| Max. *average* spot latitude | `MAXLAT` | MINLAT+5 to 80 deg | uniform | |
| Spot lifetime tau_spot | `TSPOT` | 1–10 (units of P_eq) | log-uniform | Decay e-folding = tau_spot x P_eq; emergence = max(2 d, tau_spot P_eq / 5). |
| Differential-rotation shear | `DIFFROT` | +0.1–+1 (50%), 0 (25%), −1 to −0.1 (25%) | log-uniform in \|shear\| | Headers: 50.4% / 24.6% / 25.1%. **Sign convention: see Sec. 2.** |
| Butterfly pattern | `BFLY` | True (80%) / False (20%) | | From `generate.py`; headers: 80.4% True. |

Empirical header ranges (n=10,000) match the table: PERIOD 0.101–179.98 d,
ACTIVITY 0.100–9.997, INCL 0.013–1.567 rad, MINLAT 0.004–39.99, MAXLAT
5.3–80.0, DIFFROT −0.999 to 0.999, TSPOT 1.000–9.995.

Implied period coverage (uniform P_eq): P < 1 d is 0.5% of the corpus, P < 3 d
is 1.6%, P < 10 d is 5.5%, and P < 30 d is 16.6%. Most of the corpus is beyond what a
1-yr baseline ACF method can measure, so **always stratify subsets by period.**

## 2. Differential rotation: exact convention

butterpy code (`spots.py`, 2021, unchanged in meaning through the 2023 refactor):

    omega(lat) = omega_eq - shear * omega_eq * sin^2(lat)
    =>  P(lat) = P_eq / (1 - shear * sin^2(lat))

- **shear > 0 is solar-like** (poles slower): spot periods are *longer* than P_eq.
- **shear < 0 is anti-solar** (poles faster): spot periods are *shorter* than P_eq.
- The paper's printed Eq. 1, Omega = (2pi/P_eq)(1 + alpha sin^2 lat), has the
  **opposite sign** to the code. Its own Table 1 note ("anti-solar ... by allowing
  the shear parameter to be negative") matches the code. **Trust the code.**

Size of the effect, P(lat)/P_eq:

| shear | 20 deg | 40 deg | 60 deg | 80 deg |
|---|---|---|---|---|
| +1.0 | 1.13 | 1.70 | 4.00 | 33.2 |
| +0.5 | 1.06 | 1.26 | 1.60 | 1.94 |
| +0.1 | 1.01 | 1.04 | 1.08 | 1.11 |
| −0.1 | 0.99 | 0.96 | 0.93 | 0.91 |
| −0.5 | 0.94 | 0.83 | 0.73 | 0.67 |
| −1.0 | 0.90 | 0.71 | 0.57 | 0.51 |

## 3. Spot latitudes

`MINLAT`/`MAXLAT` are bounds on the **average** emergence latitude, not hard
limits (`regions.py`):

- Spots are placed on a 16-bin latitude grid spanning
  [max(MINLAT − 7, 1), MAXLAT + 7] deg, on both hemispheres.
- **Butterfly (80%)**: the mean emergence latitude decays exponentially through the
  cycle, lat_avg = MAXLAT * exp(−phase * ln(MAXLAT/MINLAT)), so from MAXLAT toward
  MINLAT. The scatter is lat_rms = MAXLAT/5 − phase (MAXLAT − MINLAT)/7. Emergence
  probability per bin is proportional to exp(−((lat − lat_avg)/lat_rms)^2).
- **Random (20%)**: lat_avg = (MINLAT + MAXLAT)/2 and lat_rms = MAXLAT − MINLAT,
  constant.
- Emergence rate scales with the activity level, times a sin^2(pi * phase) cycle
  envelope.

## 4. Light curve flux model

Per spot k (Claytor+2022, Eqs. 2–8):
delta_F_k(t) = f_k(t) * max(cos beta_k(t), 0), with
cos beta = cos(phi_k(t)) cos(lat_k) sin(i) + sin(lat_k) cos(i), and
f_k^max = 3e-4 * A * (delta_beta_k / <delta_beta_k>)^2 (bipole separation),
using a Gaussian emergence/decay envelope in time.

## 5. Noise and observing pattern

- Noise light curves come from about 2,000 galaxies in the southern TESS CVZ (Tmag ≤ 15,
  excluding known AGN). Galaxies were chosen because they should be intrinsically
  constant, so they carry only systematics.
- They were built with a custom FFI pipeline (TESScut + lightkurve regression
  correctors), closer to SAP than PDCSAP.
- Each sector was median-normalized, then stitched over 13 sectors (~1 yr). Gaps were
  linearly interpolated to the 30-min cadence, and missing start/end cadences
  were filled with the mean.
- The galaxies were split 1,000 train / 1,000 validation+test, with one random galaxy per simulation.
  The paper says the simulation was "convolved" with the galaxy light curve; the
  exact operation isn't stated.
- **Consequence for this repo:** `preprocessing.load_smarts_fits` re-detects the
  interpolated gaps, because SMARTS ships them filled.
- The same galaxy background appears under many simulations (1M sims over about 1,000
  training galaxies, so roughly 1,000 reuses each). Keep this in mind for duplicate and
  leakage checks (task T1).

## 6. Implications already acted on

- **Scoring**: a measurement matching the period of the active latitude band
  counts as success (`evaluation.DIFFROT`), computed from `PERIOD`, `DIFFROT`,
  `MINLAT`, `MAXLAT`. See `claude_files/evaluation.py`.
- **Synthetic data**: `claude_files/synthetic.py` draws shear, spot-latitude band
  and inclination from the distributions above, keeping log-uniform periods for
  development focus.
