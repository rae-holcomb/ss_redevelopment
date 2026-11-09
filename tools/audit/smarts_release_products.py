"""Public data products for the SMARTS-TESS DR1 noise-source contamination note.

Builds, from the T20 audit outputs, the tables and figures that accompany
``docs/smarts_release_note/`` (a data-release note for SMARTS users, to be
published alongside the dataset on MAST):

- ``smarts_noise_sources.csv`` (1,719 rows, one per noise light curve; small,
  committed to the repo next to the note);
- ``smarts_noise_source_flags.csv.gz`` (1,000,000 rows, one per SMARTS file;
  written atomically to the drive under ``extracted/release_note/``);
- four figures (PNG) for the note.

Internal columns (the SIM_ID 800k pool split, catalog row bookkeeping, our own
training-exclusion policy) are dropped; every public column is documented in
the note's "Column reference" section. ``P_ref`` and the derived per-source
quantities are copied unchanged from T20 (``galaxy_phase2_analysis.py``); the
script only renames, selects and joins.

Inputs:
    <data root>/extracted/galaxy_ids/noise_source_flags.csv.gz   (per file, T20)
    reports/2026-10-21_t20-noise-sources-phase2/data/sources.csv  (per source, T20)
    reports/2026-10-21_t20-noise-sources-phase2/data/ffi_compare_all.csv
    reports/2026-10-21_t20-noise-sources-phase2/data/example_stacks.npz, example_sources.csv
    reports/2026-10-21_t20-noise-sources-phase2/data/devmedium_vs_source.csv

Usage::

    PY=/opt/anaconda3/envs/ss_openblas/bin/python
    $PY tools/audit/smarts_release_products.py sources     # per-source table (needs the drive)
    $PY tools/audit/smarts_release_products.py files       # per-file table -> drive (~1 min)
    $PY tools/audit/smarts_release_products.py figures     # note figures (no drive needed)
    $PY tools/audit/smarts_release_products.py figure-corrupted   # corrupted-segment example (needs the drive)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import smarts_data as sd  # noqa: E402

T20 = REPO / "reports" / "2026-10-21_t20-noise-sources-phase2" / "data"
NOTE_DIR = REPO / "docs" / "smarts_release_note"
SOURCES_OUT = NOTE_DIR / "smarts_noise_sources.csv"
FILES_OUT_NAME = "smarts_noise_source_flags.csv.gz"
LMC_RA, LMC_DEC, LMC_NEAR_DEG = 80.89, -69.76, 4.0   # as in galaxy_phase2_analysis.py
# Corrupted segment: |flux - 1| > CORRUPT_DEV cadences, joined across gaps of up to CORRUPT_MERGE_D,
# lasting at least CORRUPT_MIN_D with at least CORRUPT_MIN_FRAC of its cadences beyond CORRUPT_DEV.
CORRUPT_DEV, CORRUPT_MERGE_D, CORRUPT_MIN_D, CORRUPT_MIN_FRAC = 0.1, 2.0, 10.0, 0.2
CADENCES_PER_DAY = 48

CLASS_ORDER = ["contamination_known_variable", "contamination_candidate_crowded",
               "source_specific_unexplained", "systematic_common_mode", "misclassified_star", "quiet"]
CLASS_COLORS = {"contamination_known_variable": "#d62728", "contamination_candidate_crowded": "#ff7f0e",
                "source_specific_unexplained": "#1f77b4", "systematic_common_mode": "#9467bd",
                "misclassified_star": "#2ca02c", "quiet": "#b0b0b0"}

# T20 column -> public column (per source). Order here is the CSV's column order.
SOURCE_COLUMNS = {
    "tic": "noise_tic",
    "classification": "classification",
    "periodic": "periodic",
    "P_ref": "P_ref_d",
    "P1_all": "ls_P1_d",
    "pow1_all": "ls_P1_power",
    "amp1_ppm_all": "ls_P1_amp_ppm",
    "persist": "ab_persistent",
    "ls_periodic": "ls_periodic",
    "pipe_success": "search_detected",
    "pipe_P": "search_P_d",
    "cm_retained": "cm_retained",
    "near_lmc": "near_lmc",
}
SOURCE_EXTRA = {   # per-source table only
    "n_files": "n_files",
    "ra": "ra_deg",
    "dec": "dec_deg",
    "Tmag": "Tmag",
    "d_lmc_deg": "d_lmc_deg",
    "n_nb": "n_gaia_nb",
    "n_nb_brighter": "n_gaia_nb_bright",
    "n_nb_var": "n_gaia_nb_var",
    "nb_var_classes": "gaia_nb_var_classes",
}
FFI_COLUMNS = ["ffi_check", "ffi_lc_authors", "ffi_sectors", "ffi_match", "ffi_n_sectors_matching",
               "ffi_n_sectors"]


def ffi_results() -> pd.DataFrame:
    """Per-TIC independent-FFI check result, for the periodic sources only.

    Quiet control sources were tested at a *shuffled* period to measure the
    chance-match rate, so their result says nothing about their own signal and
    is left out (they get ``ffi_check = not_tested``).
    """
    f = pd.read_csv(T20 / "ffi_compare_all.csv")
    f = f[~f.control.astype(bool)].copy()
    f["ffi_check"] = np.where(~f.testable.astype(bool), "untestable",
                              np.where(f.confirmed.astype(bool), "confirmed", "not_confirmed"))
    f["ffi_match"] = f["match"].where(f.confirmed.astype(bool))
    tested = f.testable.astype(bool)
    f["ffi_n_sectors_matching"] = f.n_sectors_matching.where(tested)
    f["ffi_n_sectors"] = f.n_sectors.where(tested)
    f = f.rename(columns={"authors": "ffi_lc_authors", "sectors": "ffi_sectors", "tic": "noise_tic"})
    return f.set_index("noise_tic")[FFI_COLUMNS]


def corrupted_segments(g: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) cadence of each corrupted segment in noise light curve ``g``."""
    dev = np.abs(g - 1) > CORRUPT_DEV
    i = np.flatnonzero(dev)
    if not len(i):
        return []
    br = np.flatnonzero(np.diff(i) > CORRUPT_MERGE_D * CADENCES_PER_DAY)
    runs = zip(np.r_[i[0], i[br + 1]], np.r_[i[br], i[-1]])
    return [(int(a), int(b)) for a, b in runs
            if (b - a + 1) / CADENCES_PER_DAY >= CORRUPT_MIN_D and dev[a:b + 1].mean() >= CORRUPT_MIN_FRAC]


def corrupted_flags(root: Path) -> pd.Series:
    """Per noise TIC: does its stacked noise light curve contain a corrupted segment?"""
    sdir = sd.extract_dir(root, "galaxy_stack")
    S = np.load(sdir / "stacks.npy", mmap_mode="r")[:, 2]
    tics = pd.read_csv(sdir / "stack_index.csv").tic
    return pd.Series([bool(corrupted_segments(np.asarray(g, float))) for g in S], index=tics.values,
                     name="has_corrupted_segment")


def public_sources(root: Path | None = None) -> pd.DataFrame:
    """The 1,719-row public per-source table (all columns documented in the note).

    With ``root`` (the drive), includes ``has_corrupted_segment``; figures don't need it.
    """
    src = pd.read_csv(T20 / "sources.csv")
    assert len(src) == 1719 and src.tic.is_unique
    out = src[list(SOURCE_COLUMNS) + list(SOURCE_EXTRA)].rename(columns={**SOURCE_COLUMNS, **SOURCE_EXTRA})
    for c in ("periodic", "ab_persistent", "ls_periodic", "search_detected", "near_lmc"):
        out[c] = out[c].fillna(False).astype(bool)
    # P_ref of a quiet source is just its red-noise LS peak; blank it so nobody treats it as a period.
    out.loc[~out.periodic, "P_ref_d"] = np.nan
    out = out.join(ffi_results(), on="noise_tic")
    out["ffi_check"] = out.ffi_check.fillna("not_tested")
    for c in ("ffi_n_sectors_matching", "ffi_n_sectors", "n_gaia_nb", "n_gaia_nb_bright", "n_gaia_nb_var"):
        out[c] = out[c].astype("Int64")
    if root is not None:
        out.insert(out.columns.get_loc("near_lmc") + 1, "has_corrupted_segment",
                   out.noise_tic.map(corrupted_flags(root)).astype(bool))
    out.insert(1, "noise_lc_file", "TIC" + out.noise_tic.astype(str) + "_full.lc.fits")
    return out


def public_files(root: Path, src: pd.DataFrame) -> pd.DataFrame:
    """The 1,000,000-row public per-file table: file identifiers + the noise source's flags."""
    flags = pd.read_csv(sd.extract_dir(root, "galaxy_ids") / "noise_source_flags.csv.gz",
                        usecols=["row", "sim_id", "tic", "classification", "P_ref", "periodic", "near_lmc"])
    assert len(flags) == 1_000_000 and flags.row.is_unique
    # Consistency with the per-source table (both written by galaxy_phase2_analysis.py).
    chk = flags.drop_duplicates("tic").merge(src, left_on="tic", right_on="noise_tic")
    assert len(chk) == flags.tic.nunique() == 1719
    assert (chk.classification_x == chk.classification_y).all()
    assert (chk.periodic_x.astype(bool) == chk.periodic_y).all()
    pm = chk.periodic_y
    assert np.allclose(chk.P_ref[pm], chk.P_ref_d[pm])
    per_file = src[["noise_tic"] + [c for c in src.columns if c in SOURCE_COLUMNS.values()
                                   and c != "noise_tic"] + ["has_corrupted_segment", "ffi_check"]]
    out = pd.DataFrame({"file_number": flags.row,
                        "filename": [f"smarts-tess-v1.0-{r:06d}.fits" for r in flags.row],
                        "sim_id": flags.sim_id, "noise_tic": flags.tic})
    out = out.merge(per_file, on="noise_tic", how="left", validate="many_to_one")
    return out.sort_values("file_number").reset_index(drop=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a binomial fraction k/n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig_examples(outdir: Path):
    """Fig. 1: four noise light curves (full year + zoom with halves A and B overlaid)."""
    import matplotlib.pyplot as plt
    z = np.load(T20 / "example_stacks.npz")
    ex = pd.read_csv(T20 / "example_sources.csv").set_index("galaxy")
    picks = [(10002408894, 192, 6), (10000439468, 72, 6), (10000297576, 270, 30), (10000584382, 300, 6)]
    tic2gal = dict(zip(ex.tic, ex.index))
    t = np.arange(z["stacks"].shape[-1]) / 48.0
    fig, axes = plt.subplots(len(picks), 2, figsize=(10, 7.5), gridspec_kw=dict(width_ratios=[2.2, 1]))
    for (tic, t0, dt), (a0, a1) in zip(picks, axes):
        i = int(np.where(z["galaxy"] == tic2gal[tic])[0][0])
        row = ex.loc[tic2gal[tic]]
        cls = row.classification
        A, B, AB = z["stacks"][i]
        a0.plot(t, AB, ",", color=CLASS_COLORS[cls], rasterized=True)
        lo, hi = np.nanpercentile(AB, [0.5, 99.5])
        pad = 0.15 * (hi - lo)
        a0.set_ylim(lo - pad, hi + pad)
        per = f"P_ref = {row.P_ref:.3g} d" if row.periodic else "no significant period"
        a0.set_title(f"TIC {tic}: {cls} ({per}; {row.n_files} files)", fontsize=9, loc="left")
        m = (t >= t0) & (t < t0 + dt)
        a1.plot(t[m], A[m], lw=1.6, color="0.3", label="half A")
        a1.plot(t[m], B[m], lw=0.8, color=CLASS_COLORS[cls] if cls != "quiet" else "C1", label="half B")
        a1.set_ylim(*np.nanpercentile(AB[m], [0.5, 99.5]) + np.array([-1, 1]) * 0.1 * np.ptp(
            np.nanpercentile(AB[m], [0.5, 99.5])))
        a1.tick_params(labelsize=8)
        a0.tick_params(labelsize=8)
    axes[0, 1].legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("time (d)")
    axes[-1, 1].set_xlabel("time (d), zoom")
    fig.supylabel("normalized flux of the noise light curve", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "fig1_example_noise_sources.png", dpi=150)
    plt.close(fig)


def fig_population(outdir: Path, src: pd.DataFrame):
    """Fig. 2: LS amplitude vs period by class, and sky positions with the LMC region."""
    import matplotlib.pyplot as plt
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.3))
    for cls in reversed(CLASS_ORDER):
        s = src[src.classification == cls]
        P = s.P_ref_d.where(s.periodic, s.ls_P1_d)
        lab = f"{cls} ({len(s)})"
        kw = dict(s=6, alpha=0.5) if cls == "quiet" else dict(s=14, alpha=0.9)
        a0.scatter(P, s.ls_P1_amp_ppm, color=CLASS_COLORS[cls], label=lab, **kw)
        a1.scatter(s.ra_deg, s.dec_deg, color=CLASS_COLORS[cls], **kw)
    a0.set(xscale="log", yscale="log", xlabel="period (d): P_ref, or LS peak for quiet sources",
           ylabel="LS semi-amplitude of the top peak (ppm)")
    h, lab = a0.get_legend_handles_labels()
    a0.legend(h[::-1], lab[::-1], fontsize=7, loc="upper right")
    th = np.linspace(0, 2 * np.pi, 200)
    a1.plot(LMC_RA + LMC_NEAR_DEG * np.cos(th) / np.cos(np.radians(LMC_DEC)),
            LMC_DEC + LMC_NEAR_DEG * np.sin(th), "k--", lw=1)
    a1.invert_xaxis()
    a1.set(xlabel="RA (deg)", ylabel="Dec (deg)", title="dashed: 4° around the LMC centre")
    a1.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_population.png", dpi=150)
    plt.close(fig)


def fig_ffi(outdir: Path):
    """Fig. 3: fraction of testable sources confirmed in independent FFI light curves, vs the null."""
    import matplotlib.pyplot as plt
    f = pd.read_csv(T20 / "ffi_compare_all.csv")
    f = f[f.testable.astype(bool)]
    ctrl = f[f.control.astype(bool)]
    k0, n0 = int(ctrl.confirmed.sum()), len(ctrl)
    rows = [(c, int(g.confirmed.sum()), len(g)) for c, g in f[~f.control.astype(bool)].groupby("classification")]
    rows = sorted(rows, key=lambda r: CLASS_ORDER.index(r[0]), reverse=True)
    rows.insert(0, (f"quiet controls (shuffled period)", k0, n0))
    fig, ax = plt.subplots(figsize=(7, 3.3))
    lo0, hi0 = wilson(k0, n0)
    ax.axvspan(lo0, hi0, color="0.88", label=f"chance-match null {k0}/{n0}, 95% CI")
    for y, (c, k, n) in enumerate(rows):
        lo, hi = wilson(k, n)
        col = CLASS_COLORS.get(c, "0.4")
        ax.errorbar(k / n, y, xerr=[[k / n - lo], [hi - k / n]], fmt="o", color=col, capsize=3)
    ax.set_yticks(range(len(rows)), [f"{c} ({k}/{n})" for c, k, n in rows], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction confirmed in independent FFI light curves (P_ref < 5 d)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "fig3_ffi_check.png", dpi=150)
    plt.close(fig)


def fig_impact(outdir: Path, src: pd.DataFrame):
    """Fig. 4: on 1,000 SMARTS files, the recovered period vs the noise source's own period."""
    import matplotlib.pyplot as plt
    d = pd.read_csv(T20 / "devmedium_vs_source.csv")
    raw = pd.read_csv(T20 / "sources.csv", usecols=["galaxy", "classification", "pipe_P", "pipe_success"])
    d = d.merge(raw, on="galaxy", how="left")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw=dict(width_ratios=[1, 1.1]))
    m = d.pipe_success.fillna(False).astype(bool) & d.final_period.notna()
    style = {"correct": ("#2ca02c", "correct"), "diffrot": ("#bcbd22", "correct (diff. rotation)"),
             "wrong": ("#d62728", "wrong"), "alias": ("#ff7f0e", "P/2 or 2P alias"),
             }
    outcome = d.final_outcome.replace({"half": "alias", "double": "alias"})
    for o, (col, lab) in style.items():
        s = d[m & (outcome == o)]
        a0.scatter(s.pipe_P, s.final_period, s=12, color=col, label=f"{lab} ({len(s)})")
    a0.plot([0.1, 60], [0.1, 60], "k:", lw=0.8)
    a0.set(xscale="log", yscale="log", xlabel="period found in the noise light curve alone (d)",
           ylabel="period recovered from the SMARTS file (d)")
    a0.set_title(f"{int(m.sum())} files with a recovered period whose noise source has one too", fontsize=9)
    a0.legend(fontsize=7)
    rows = []
    for c in reversed(CLASS_ORDER):
        g = d[d.classification == c]
        if len(g):
            rows.append((c, int((g.final_outcome == "wrong").sum()), len(g)))
    a1.barh(range(len(rows)), [k / n for _, k, n in rows], color=[CLASS_COLORS[c] for c, _, _ in rows])
    for y, (_, k, n) in enumerate(rows):
        a1.text(k / n + 0.01, y, f"{k}/{n}", va="center", fontsize=8)
    a1.set_yticks(range(len(rows)), [c for c, _, _ in rows], fontsize=8)
    a1.set_xlabel("fraction of files whose recovered period is wrong")
    a1.set_xlim(0, 0.65)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_period_recovery.png", dpi=150)
    plt.close(fig)


def fig_corrupted(outdir: Path, root: Path, tic: int = 10000441469):
    """Example noise light curve with corrupted segments (needs the T20 stacks on the drive)."""
    import matplotlib.pyplot as plt
    sdir = sd.extract_dir(root, "galaxy_stack")
    i = int(np.where(pd.read_csv(sdir / "stack_index.csv").tic == tic)[0][0])
    g = np.asarray(np.load(sdir / "stacks.npy", mmap_mode="r")[i, 2], float)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(np.arange(g.size) / 48.0, g, lw=0.4, color="#d62728", rasterized=True)
    ax.set_yscale("symlog", linthresh=2)
    ax.axhline(1, color="k", lw=0.6, ls=":")
    ax.set(xlabel="time (d)", ylabel="normalized noise flux (symlog)")
    ax.set_title(f"TIC {tic}", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(outdir / "fig_corrupted_segments.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    """CLI: ``sources`` | ``files`` | ``figures``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["sources", "files", "figures", "figure-corrupted"])
    ap.add_argument("--data-root", default=None, help="SMARTS data root (default $SMARTS_DATA_ROOT or the drive)")
    args = ap.parse_args(argv)
    root = None
    if args.what in ("sources", "files", "figure-corrupted"):
        root = sd.require_drive(sd.data_root(args.data_root))
    if args.what == "figure-corrupted":
        fig_corrupted(NOTE_DIR / "figures", root)
        return
    src = public_sources(root)
    if args.what == "sources":
        NOTE_DIR.mkdir(parents=True, exist_ok=True)
        with sd.atomic_path(SOURCES_OUT) as tmp:
            src.to_csv(tmp, index=False, float_format="%.6g")
        print(f"wrote {SOURCES_OUT} ({len(src)} rows)")
        print(src.groupby("classification").agg(n=("noise_tic", "size"), files=("n_files", "sum")))
        print(src.ffi_check.value_counts())
    elif args.what == "files":
        out = public_files(root, src)
        final = sd.extract_dir(root, "release_note") / FILES_OUT_NAME
        with sd.atomic_path(final) as tmp:
            out.to_csv(tmp, index=False, float_format="%.6g", compression="gzip")
        print(f"wrote {final} ({len(out)} rows, {final.stat().st_size / 1e6:.1f} MB)")
        print(out.groupby("classification").size())
    else:
        figdir = NOTE_DIR / "figures"
        figdir.mkdir(parents=True, exist_ok=True)
        fig_examples(figdir)
        fig_population(figdir, src)
        fig_ffi(figdir)
        fig_impact(figdir, src)
        print(f"wrote figures to {figdir}")


if __name__ == "__main__":
    main()
