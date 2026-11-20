"""Figures for the P49 Roman noise-survey report. Pure numbers from the cited papers + WG6 curves in
tools/make_roman_rotators.py. No drive, no downloads. Called by notebook.ipynb; also `python figlib.py`.

Every literature number used here is in LIT below with its source; the CCD-equation curve is *our* evaluation of the
parameters those papers list, not a curve copied from them (none of them tabulates sigma vs magnitude in text).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_roman_rotators as mr  # noqa: E402

D, F = HERE / "data", HERE / "figures"
MMAG_PER_PPT = 2.5 / np.log(10) / 1000 * 1000  # 1 ppt = 1.0857 mmag

# --- Literature parameters (see README table for citations) ---------------------------------------------------------
# Wilson+2023 (arXiv:2305.16204) Table 1: ZP 27.648 (1 e-/s), sky 4.25 + thermal 0.98 e-/pix/s, CDS read noise 16 e-.
# GBTDS exposure today: 66-67 s (STScI GBTDS page / ROTAC). Aperture 3x3 = 9 px (Penny+2019, Tamburo+2023).
# Aperture flux fraction 0.83 is an ASSUMPTION (reported only in a model summary of Wilson+2023 III.4; unverified).
CCD = dict(zp=27.648, bkg=4.25 + 0.98, rn=16.0, npix=9, texp=66.0, apfrac=0.83)
FLOORS_MMAG = {"Penny+2019 / Tamburo+2023 floor (1 mmag)": 1.0,
               "Wilson+2023 conservative floor (1.3 ppt)": 1.3 * MMAG_PER_PPT}
# Anchor points quoted in text (one exposure each): (F146, sigma_mmag, label)
ANCHORS = [(21.2, 2.5 / np.log(10) / 100 * 1000, "STScI: S/N~100 at F146=21.2 (66 s)"),
           (21.15, 10.0, "Penny+2019: 0.01 mag at W149=21.15 (46.8 s)"),
           (21.0, 2.5 / np.log(10) / 100 * 1000, "Tamburo+2023: 1% at F146~21 (46.8 s)")]
SAT = {"saturation (Penny+2019 W149 ~14.8; Weiss+2025 ~15)": (14.8, 15.5),
       "semi-saturation (Claytor & Tayar 2026: ~16-18)": (16.0, 18.0)}
WG6_SIMPLE = (np.array([14, 17, 18, 20, 22, 24]), np.array([1.0, 1.65, 2.2, 5.9, 18.0, 69.0]))  # WG6_REFERENCE 8.3
ROT_AMP_BAND = (0.4, 2.0)  # old G/K dwarf Rvar(F146) priors in mmag, P34 report


def ccd_sigma_mmag(mag, floor_mmag=0.0, **kw):
    p = {**CCD, **kw}
    s = p["apfrac"] * 10 ** (-0.4 * (np.asarray(mag, float) - p["zp"])) * p["texp"]
    var = s + p["npix"] * (p["bkg"] * p["texp"] + p["rn"] ** 2)
    sig = 2.5 / np.log(10) * np.sqrt(var) / s * 1000
    return np.sqrt(sig ** 2 + floor_mmag ** 2)


def save_tables():
    D.mkdir(exist_ok=True)
    m = np.arange(14.0, 24.01, 0.25)
    df = pd.DataFrame({"f146": m, "wg6_improved_mmag": mr.sigma_f146_mmag(m),
                       "wg6_simple_mmag": 10 ** np.interp(m, WG6_SIMPLE[0], np.log10(WG6_SIMPLE[1])),
                       "ccd_nofloor_mmag": ccd_sigma_mmag(m),
                       "ccd_penny_floor_mmag": ccd_sigma_mmag(m, 1.0),
                       "ccd_wilson_floor_mmag": ccd_sigma_mmag(m, 1.3 * MMAG_PER_PPT)})
    df.to_csv(D / "sigma_vs_f146.csv", index=False, float_format="%.4g")
    return df


def fig_sigma():
    df = save_tables()
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    for name, (lo, hi) in SAT.items():
        a.axvspan(lo, hi, color="0.85" if "semi" in name else "0.65", alpha=0.6, lw=0, label=name)
    a.axhspan(*ROT_AMP_BAND, color="#fdd49e", alpha=0.6, lw=0, label="old G/K dwarf Rvar(F146), P34 prior")
    a.plot(df.f146, df.wg6_improved_mmag, "k-", lw=2.2, label="WG6 improved (romrot_v0 uses this)")
    a.plot(df.f146, df.wg6_simple_mmag, "k--", lw=1.4, label="WG6 simple")
    a.plot(df.f146, df.ccd_nofloor_mmag, color="#2c7bb6", lw=1, ls=":", label="CCD eq., Wilson+23 Table 1 params, 66 s, no floor")
    a.plot(df.f146, df.ccd_penny_floor_mmag, color="#2c7bb6", lw=1.8, label="  + 1 mmag floor (Penny+19, Tamburo+23)")
    a.plot(df.f146, df.ccd_wilson_floor_mmag, color="#d7191c", lw=1.4, label="  + 1.3 ppt floor (Wilson+23 conservative)")
    for (m, s, lab), mk in zip(ANCHORS, ["o", "s", "^"]):
        a.plot(m, s, mk, color="#5e3c99", ms=7, ls="none", label=lab)
    a.set_yscale("log"); a.set_ylim(0.3, 120)
    a.set_ylabel("per-point sigma, 1 F146 exposure (mmag)")
    a.legend(fontsize=7, loc="upper left"); a.grid(alpha=0.3, which="both")
    a.set_title("Roman F146 per-exposure white-noise models vs WG6 improved sigma")
    b.axhline(1, color="k", lw=0.8)
    b.plot(df.f146, df.ccd_penny_floor_mmag / df.wg6_improved_mmag, color="#2c7bb6", lw=1.8, label="CCD + 1 mmag / WG6 improved")
    b.plot(df.f146, df.ccd_wilson_floor_mmag / df.wg6_improved_mmag, color="#d7191c", lw=1.4, label="CCD + 1.3 ppt / WG6 improved")
    b.plot(df.f146, df.wg6_simple_mmag / df.wg6_improved_mmag, "k--", lw=1.2, label="WG6 simple / improved")
    b.set_ylabel("ratio"); b.set_xlabel("F146 (AB mag)"); b.set_ylim(0.4, 2.5)
    b.legend(fontsize=7); b.grid(alpha=0.3)
    fig.tight_layout(); F.mkdir(exist_ok=True)
    fig.savefig(F / "sigma_vs_f146.png", dpi=130); plt.close(fig)
    return df


# --- Timescale chart ------------------------------------------------------------------------------------------------
# (label, t_lo_days, t_hi_days, kind, modelled_in) ; kind: white / correlated / step / gap / static
MIN, HR = 1 / 1440, 1 / 24
TIMESCALES = [
    ("Photon + sky + read noise (per 66 s exposure)", 66 / 86400, 12.1 * MIN, "white", "WG6, Penny, Wilson, Tamburo, rimtimsim"),
    ("1/f read noise (ms-s; row banding within a read)", 1e-3 / 86400, 66 / 86400, "white", "none (white across exposures)"),
    ("Burn-in (~20 s exponential, ~0.1%)", 5 / 86400, 60 / 86400, "correlated", "none"),
    ("Persistence after saturation (1/t, 20 s - 30 min)", 20 / 86400, 30 * MIN, "correlated", "romanisim (Sanchez+23)"),
    ("Dither: 36 mosaics/season; 12-tile cycle ~3 h", 12.1 * MIN, 36 * 12.1 * MIN, "correlated", "none (pixel-phase/flat errors)"),
    ("F087/F213 interleave (every 3 h, each 6 h)", 3 * HR, 6 * HR, "gap", "WG6 (F146 grid unaffected)"),
    ("Instrumental red-noise limit discussed (<~12 h)", 0.5 * HR, 12 * HR, "correlated", "Wilson+23 (as depth floor only)"),
    ("Granulation / oscillation (giants)", 1 * HR, 2.0, "correlated", "Weiss+25, gadfly (signal, not noise)"),
    ("Snapshot passes (start/mid/end; ~35 d apart)", 30.0, 40.0, "gap", "none"),
    ("Slow sky / thermal / zero-point drift in a season", 5.0, 70.5, "correlated", "none (Penny: time-dependent sky only)"),
    ("Season (~70.5 d high cadence)", 70.5, 72.0, "gap", "WG6 grid (71 d)"),
    ("Spring/fall 180 deg roll: different pixels, PSF, blend mix", 180.0, 190.0, "step", "none"),
    ("Inter-season gap (~110 d) and early/late gap (years)", 109.0, 1500.0, "step", "WG6 grid (109-115 d only)"),
    ("Blend geometry change from proper motion", 365.0, 3000.0, "correlated", "none (pipeline re-runs positions)"),
]
KCOL = {"white": "#bababa", "correlated": "#d7191c", "step": "#fdae61", "gap": "#2c7bb6", "static": "#999999"}


def fig_timescales():
    pd.DataFrame(TIMESCALES, columns=["effect", "t_lo_d", "t_hi_d", "kind", "modelled_in"]).to_csv(
        D / "timescales.csv", index=False, float_format="%.4g")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for i, (lab, lo, hi, kind, mod) in enumerate(TIMESCALES[::-1]):
        ax.barh(i, max(np.log10(hi) - np.log10(lo), 0.08), left=np.log10(lo), color=KCOL[kind], height=0.6)
        ax.text(np.log10(hi) + 0.08, i, f"{lab}  [{mod}]", va="center", fontsize=7)
    # x axis is log10(days); manual ticks
    ticks = {"1 ms": 1e-3 / 86400, "1 s": 1 / 86400, "1 min": MIN, "12 min": 12.1 * MIN, "1 h": HR, "1 d": 1.0,
             "10 d": 10.0, "50 d": 50.0, "1 yr": 365.0, "5 yr": 1826.0}
    ax.set_xticks(np.log10(list(ticks.values()))); ax.set_xticklabels(list(ticks.keys()), fontsize=8)
    ax.axvspan(0, np.log10(50), color="#ffffbf", alpha=0.8, lw=0, zorder=-1, label="rotation 1-50 d")
    ax.axvline(np.log10(35.5), color="k", ls=":", lw=1, label="R/2 of one 71-d season")
    ax.set_xlim(np.log10(1e-3 / 86400), np.log10(1826) + 4.2)
    ax.set_yticks([])
    for k, c in KCOL.items():
        if k != "static":
            ax.barh(-5, 0, color=c, label={"white": "white across 12-min points", "correlated": "time-correlated",
                                           "step": "step / offset", "gap": "gap / interruption"}[k])
    ax.set_ylim(-0.7, len(TIMESCALES) - 0.3)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("GBTDS noise/systematics timescales vs rotation (yellow: 1-50 d; dotted: R/2 = 35.5 d)\n"
                 "[brackets]: which model or simulator includes it", fontsize=9)
    fig.tight_layout(); fig.savefig(F / "timescales.png", dpi=130); plt.close(fig)


# --- F146 / Kepler amplitude ratio (D5) -----------------------------------------------------------------------------
H, C, K = 6.626e-34, 2.998e8, 1.381e-23
BANDS = {"Kepler (top-hat 0.43-0.89 um)": (0.43e-6, 0.89e-6), "F146 (top-hat 0.93-2.00 um)": (0.93e-6, 2.00e-6)}


def band_flux(T, band):
    lam = np.linspace(*band, 400)
    B = 2 * H * C ** 2 / lam ** 5 / np.expm1(H * C / (lam * K * T))
    return np.trapz(B * lam, lam)  # photon-weighted (lam factor)


def spot_ratio(teff, dT):
    """(1 - F_spot/F_phot) in F146 divided by the same in Kepler; same filling factor cancels (small-coverage limit)."""
    kp, f1 = BANDS.values()
    c = lambda b: 1 - band_flux(teff - dT, b) / band_flux(teff, b)  # noqa: E731
    return c(f1) / c(kp)


def fig_amp_ratio():
    T = np.linspace(3500, 6500, 61)
    weiss = 0.493 + 0.058 * (T / 5000) + 0.018 * (T / 5000) ** 2  # Weiss+2025 eq. 5 (T5000 = Teff/5000, our reading)
    rows = {"teff": T, "weiss2025_osc": weiss, "bb_dT_small": [spot_ratio(t, 1.0) for t in T]}
    for f in (0.15, 0.3):
        rows[f"bb_dT_{int(f*100)}pct"] = [spot_ratio(t, f * t) for t in T]
    df = pd.DataFrame(rows); df.to_csv(D / "f146_kepler_amp_ratio.csv", index=False, float_format="%.4g")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(T, weiss, "k-", lw=2, label="Weiss+2025 eq. 5 (oscillations; Gadfly)")
    ax.plot(T, df.bb_dT_small, color="#2c7bb6", lw=1.2, ls=":", label="blackbody, small dT (our calc.)")
    ax.plot(T, df.bb_dT_15pct, color="#2c7bb6", lw=1.6, label="blackbody spot, dT = 0.15 Teff (our calc.)")
    ax.plot(T, df.bb_dT_30pct, color="#d7191c", lw=1.6, label="blackbody spot, dT = 0.30 Teff (our calc.)")
    ax.axhline(0.7, color="#fdae61", lw=2.5, label="romrot_v0 factor 0.7 (Claytor & Tayar 53%/76%)")
    ax.set_xlabel("Teff (K)"); ax.set_ylabel("amplitude ratio F146 / Kepler band")
    ax.set_ylim(0.3, 0.95); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title("D5: F146/Kepler variability amplitude ratio (top-hat bands, no faculae, no limb darkening)", fontsize=8)
    fig.tight_layout(); fig.savefig(F / "f146_amp_ratio.png", dpi=130); plt.close(fig)
    return df


def main():
    df = fig_sigma(); fig_timescales(); amp = fig_amp_ratio()
    sel = df[df.f146.isin([14, 16, 17, 18, 19, 20, 21, 22])]
    print(sel.round(3).to_string(index=False))
    print(amp.iloc[::10].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
