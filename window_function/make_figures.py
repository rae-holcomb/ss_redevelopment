"""Build the phase-coverage figures.

Figure 1 is a direct recreation of Rodel et al. (2024) Fig. 4: an
idealised contiguous window function against a real TESS light curve's
window function, for up to 13 consecutive Sectors.

Figure 2 extends it to current TESS pointings. The same 45 real Sector
windows are used twice: once laid back to back (the inter-cycle gaps
removed) and once at the times they were actually observed. Total on-sky
time is identical panel to panel at fixed N, so the difference between
them isolates the cost of the clumped, multi-cycle pointing pattern that
long-baseline TESS targets actually have.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase_coverage import (
    SECTOR_LENGTH,
    consecutive_runs,
    idealised_intervals,
    load_window,
    phase_coverage,
    stacked_intervals,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "window_cadences.npz")

# Sectors labelled in the extension figure; a subset keeps 45 curves readable.
LABEL_N = [1, 2, 3, 5, 8, 13, 21, 34, 45]
# The star's most recent unbroken run of 13 Sectors, used for Figure 1.
RUN_13 = list(range(27, 40))


def set_style():
    """Apply a serif, paper-like plotting style.

    Returns
    -------
    None
    """
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 13,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "figure.dpi": 130,
    })


def sequential_intervals(windows, sectors):
    """Lay each Sector's real window immediately after the previous one.

    Preserves every real within-Sector gap (orbit downlinks, momentum
    dumps, scattered-light masking) and every real Sector length, but
    removes the long waits between observing cycles. Combined with
    `stacked_intervals` on the same Sector list this isolates the effect
    of the pointing pattern at fixed total on-sky time.

    Parameters
    ----------
    windows : dict
        Sector number -> (n, 2) interval array.
    sectors : sequence of int
        Sectors to lay out, in the order given.

    Returns
    -------
    ndarray, shape (n, 2)
        Intervals in the shifted, back-to-back time frame.
    """
    out, cursor = [], 0.0
    for s in sectors:
        iv = windows[s]
        out.append(iv - iv[0, 0] + cursor)
        cursor = out[-1][-1, 1]
    return np.vstack(out)


def edge_labels(ax, curves, periods, x_frac=0.985, fontsize=9.5):
    """Write each curve's label at the right edge, coloured to match it.

    Labels are nudged apart vertically so that closely-spaced curves stay
    individually readable, in the style of the original figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes holding the curves.
    curves : list of (str, ndarray, color)
        Label, coverage array, and colour for each curve.
    periods : ndarray
        The period grid the coverage arrays were evaluated on.
    x_frac : float
        Fractional x position of the label anchor within the axes.
    fontsize : float
        Label font size.

    Returns
    -------
    None
    """
    idx = int(x_frac * (len(periods) - 1))
    entries = sorted(((c[idx], lab, col) for lab, c, col in curves), reverse=True)
    min_sep = 0.031
    placed = []
    for y, lab, col in entries:
        if placed and y > placed[-1] - min_sep:
            y = placed[-1] - min_sep
        placed.append(y)
        ax.text(periods[idx], y, lab, color=col, fontsize=fontsize,
                ha="right", va="center")


def figure1(windows, out_path):
    """Recreate Rodel+2024 Fig. 4 with a current TESS light curve.

    Left panel is the paper's idealised scenario, contiguous gap-free
    27.4 d Sectors. Right panel is the real FFI window function of the
    target's most recent unbroken 13-Sector run.

    Parameters
    ----------
    windows : dict
        Sector number -> interval array.
    out_path : str
        Where to write the PNG.

    Returns
    -------
    None
    """
    periods = np.linspace(1.0, 700.0, 5000)
    colors = plt.cm.tab10(np.arange(10) % 10)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), sharey=True)
    for ax, mode in zip(axes, ("ideal", "real")):
        curves = []
        for n in range(1, 14):
            col = colors[(n - 1) % 10]
            if mode == "ideal":
                iv = idealised_intervals(n)
            else:
                iv = stacked_intervals(windows, RUN_13[:n])
            cov = phase_coverage(iv, periods)
            ax.plot(periods, cov, color=col, lw=1.2)
            curves.append((f"{n} Sector" + ("" if n == 1 else "s"), cov, col))
        edge_labels(ax, curves, periods)
        ax.set_xlim(0, 700)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Period [days]")
        ax.minorticks_on()
    axes[0].set_ylabel("Fraction of phase covered")
    axes[0].set_title("Idealised 27.4 d Sectors, contiguous", fontsize=12)
    axes[1].set_title(f"Real TESS window function, Sectors {RUN_13[0]}"
                      f"–{RUN_13[-1]}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def figure2(windows, sectors, out_path):
    """Extend the comparison to all 45 Sectors of current TESS coverage.

    Left panel lays those Sectors back to back; right panel uses the real
    observing times, in which the Sectors fall into per-cycle clumps
    separated by year-long gaps. Total on-sky time is identical between
    panels at fixed N, so the difference is purely the pointing pattern.

    Parameters
    ----------
    windows : dict
        Sector number -> interval array.
    sectors : list of int
        The target's full observed Sector list, ascending.
    out_path : str
        Where to write the PNG.

    Returns
    -------
    None
    """
    periods = np.linspace(1.0, 3000.0, 7000)
    n_max = len(sectors)
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=1, vmax=n_max)

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6), sharey=True)
    for ax, mode in zip(axes, ("back-to-back", "as observed")):
        for n in range(1, n_max + 1):
            col = cmap(norm(n))
            subset = sectors[:n]
            iv = (sequential_intervals(windows, subset) if mode == "back-to-back"
                  else stacked_intervals(windows, subset))
            cov = phase_coverage(iv, periods)
            emph = n in LABEL_N
            ax.plot(periods, cov, color=col, lw=1.6 if emph else 0.6,
                    alpha=1.0 if emph else 0.4, zorder=3 if emph else 2)
        ax.set_xlim(0, 3000)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Period [days]")
        ax.minorticks_on()
    axes[0].set_ylabel("Fraction of phase covered")
    axes[0].set_title("Same Sectors laid back to back", fontsize=12)
    axes[1].set_title("Sectors at their actual observing times", fontsize=12)

    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.915, 0.12, 0.016, 0.78])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Number of Sectors")
    cb.set_ticks([1, 5, 13, 21, 34, 45])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def figure_window_map(windows, sectors, tic, out_path):
    """Plot where the observed Sectors actually fall on the time axis.

    Context for the coverage figures: shows the per-cycle clumping and the
    long gaps that drive the difference between the two panels of
    Figure 2.

    Parameters
    ----------
    windows : dict
        Sector number -> interval array.
    sectors : list of int
        Observed Sectors, ascending.
    tic : str
        Target name for the title.
    out_path : str
        Where to write the PNG.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(13.6, 2.6))
    runs = consecutive_runs(sectors)
    for run in runs:
        for s in run:
            for lo, hi in windows[s]:
                ax.axvspan(lo, hi, ymin=0.25, ymax=0.75, color="#2b6cb0", lw=0)
        span = stacked_intervals(windows, run)
        mid = 0.5 * (span[0, 0] + span[-1, 1])
        lab = f"S{run[0]}" if len(run) == 1 else f"S{run[0]}–{run[-1]}"
        ax.text(mid, 0.86, lab, ha="center", va="center", fontsize=10)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time [BTJD]")
    ax.set_title(f"{tic}: observed TESS Sectors ({len(sectors)} total, "
                 f"{stacked_intervals(windows, sectors)[-1, 1] - stacked_intervals(windows, sectors)[0, 0]:.0f} d baseline)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def write_summary(windows, sectors, tic, out_path):
    """Tabulate covered phase fraction at a handful of reference periods.

    Writes the three scenarios (idealised contiguous Sectors, the real
    Sector windows laid back to back, and the real windows at their actual
    times) side by side, plus the per-cycle duty-cycle trend.

    Parameters
    ----------
    windows : dict
        Sector number -> interval array.
    sectors : list of int
        Observed Sectors, ascending.
    tic : str
        Target name.
    out_path : str
        Where to write the text table.

    Returns
    -------
    None
    """
    probe = np.array([10.0, SECTOR_LENGTH, 100.0, 200.0, 365.25, 730.5,
                      1000.0, 1500.0, 2000.0])
    lines = [f"{tic}: fraction of orbital phase covered",
             "",
             "N    scenario       " + "".join(f"{p:>8.0f}" for p in probe) + "   [days]"]
    for n in LABEL_N:
        subset = sectors[:n]
        scenarios = [("idealised", idealised_intervals(n)),
                     ("back-to-back", sequential_intervals(windows, subset)),
                     ("as observed", stacked_intervals(windows, subset))]
        for name, iv in scenarios:
            cov = phase_coverage(iv, probe)
            lines.append(f"{n:<4d} {name:<14s}" + "".join(f"{v:8.3f}" for v in cov))
        lines.append("")

    lines.append("mean per-Sector duty cycle and Sector length by era:")
    for label, era in [("Cycle 1 (S1-13)", range(1, 14)),
                       ("Cycle 3 (S27-39)", range(27, 40)),
                       ("Cycles 5-6 (S61-69)", range(61, 70)),
                       ("Cycles 7-8 (S87-98)", range(87, 99))]:
        have = [s for s in era if s in windows]
        if not have:
            continue
        duty = [(windows[s][:, 1] - windows[s][:, 0]).sum()
                / (windows[s][-1, 1] - windows[s][0, 0]) for s in have]
        span = [windows[s][-1, 1] - windows[s][0, 0] for s in have]
        lines.append(f"  {label:<22s} duty={np.mean(duty):.3f}  "
                     f"mean span={np.mean(span):5.1f} d")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()

    set_style()
    windows, tic = load_window(args.cache)
    sectors = sorted(windows)

    figure_window_map(windows, sectors, tic,
                      os.path.join(args.outdir, "fig0_window_map.png"))
    figure1(windows, os.path.join(args.outdir, "fig1_rodel_fig4_recreation.png"))
    figure2(windows, sectors, os.path.join(args.outdir, "fig2_current_pointings.png"))
    write_summary(windows, sectors, tic,
                  os.path.join(args.outdir, "coverage_summary.txt"))


if __name__ == "__main__":
    main()
