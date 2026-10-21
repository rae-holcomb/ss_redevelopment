"""Collect the small data tables behind the T20 phase-1 report (benchmark timings, validation, spot-check LCs).

Reads the galaxy_pass outputs and one galaxy_groups output directory (drive
needed), writes CSV/NPZ files into ``--outdir`` (the report's ``data/``).
The spot-check streams one bundle once to pull a few member light curves.

Usage::

    $PY tools/audit/galaxy_phase1_data.py --groups DIR --outdir reports/<folder>/data
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import galaxy_pass as gp  # noqa: E402
import smarts_data as sd  # noqa: E402

T1_GROUPS = HERE.parents[1] / "reports/2026-10-15_t1-smarts-catalog-audit/data/fingerprint_groups.csv"


def bench_table(pass_dir: Path) -> pd.DataFrame:
    """One row per finished bundle from its NNN_meta.json."""
    recs = [json.loads(p.read_text()) for p in sorted(pass_dir.glob("*_meta.json"))]
    df = pd.DataFrame(recs)
    df[["load1", "load5", "load15"]] = pd.DataFrame(df.pop("host_load").tolist())
    return df


def t1_validation(assign: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare our galaxy IDs with T1's noise groups (pairs at r > 0.8) on the rows both cover."""
    t1 = pd.read_csv(T1_GROUPS)
    m = t1.merge(assign[["row", "galaxy_id", "corr_best"]], on="row", how="inner")
    i, j = np.triu_indices(len(m), 1)
    same_t1 = m.noise_group.values[i] == m.noise_group.values[j]
    same_us = m.galaxy_id.values[i] == m.galaxy_id.values[j]
    res = dict(n_rows_overlap=len(m), n_t1_rows=len(t1),
               t1_same_pairs=int(same_t1.sum()), our_same_pairs=int(same_us.sum()),
               both=int((same_t1 & same_us).sum()), t1_only=int((same_t1 & ~same_us).sum()),
               ours_only=int((~same_t1 & same_us).sum()))
    pairs = pd.DataFrame(dict(row_a=m.row.values[i], row_b=m.row.values[j],
                              t1_same=same_t1, ours_same=same_us))
    return pairs[pairs.t1_same != pairs.ours_same], res


def per_bundle_sizes(assign: pd.DataFrame, bundles) -> pd.DataFrame:
    """Files per galaxy within each bundle, split by SIM_ID pool."""
    a = assign[assign.bundle.isin(bundles)].assign(pool=lambda d: np.where(d.sim_id >= 800_000, "ge800k", "lt800k"))
    return a.groupby(["bundle", "pool", "galaxy_id"]).size().rename("n_files").reset_index()


def spotcheck(root: Path, assign: pd.DataFrame, sib: pd.DataFrame, bundle: int, out: Path) -> pd.DataFrame:
    """Pull raw flux of a few galaxies' members (and one sibling pair) from one bundle."""
    a = assign[assign.bundle == bundle]
    cnt = a.groupby("galaxy_id").size()
    rng = np.random.default_rng(0)
    gals = list(rng.choice(cnt[cnt >= 4].index.values, 3, replace=False))
    in_b = set(cnt.index)
    s = sib[sib.galaxy_a.isin(in_b) & sib.galaxy_b.isin(in_b) & (sib["corr"] > 0.8)]
    if len(s):
        gals += [int(s.galaxy_a.iloc[0]), int(s.galaxy_b.iloc[0])]
    pick = a[a.galaxy_id.isin(gals)].groupby("galaxy_id").head(4)
    want = set(pick.row)
    flux = {}
    with gp.open_bundle_stream(root / sd.BUNDLE_DIR / sd.bundle_name(bundle)) as tf:
        for m in tf:
            if m.isfile() and int(m.name[-11:-5]) in want:
                t, f = gp.parse_lightcurve(tf.extractfile(m).read())
                flux[int(m.name[-11:-5])] = f
    rows = pick.row.values
    np.savez_compressed(out, time=t, rows=rows, flux=np.stack([flux[r] for r in rows]))
    return pick


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--groups", required=True, help="galaxy_groups output directory")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--spot-bundle", type=int, default=90)
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    pass_dir = Path(root, sd.EXTRACT_DIR, gp.PASS_DIR)
    g, out = Path(args.groups), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    bench_table(pass_dir).to_csv(out / "bench_meta.csv", index=False)
    for f in ("galaxy_table.csv", "galaxy_siblings.csv", "groups_summary.json"):
        shutil.copy(g / f, out / f)
    assign = pd.read_csv(g / "galaxy_assign.csv.gz")
    stats = gp.load_pass(pass_dir, sorted(assign.bundle.unique().tolist()))
    # per-file table for the three benchmark bundles only (keeps data/ small)
    assign[assign.bundle.isin([1, 90, 180])].merge(stats[["row", "slow_amp_ppm"]], on="row") \
        [["row", "bundle", "galaxy_id", "galaxy_best", "corr_best", "corr_second", "sim_id", "period", "max_abs_dev",
          "p2p_ppm", "slow_rms_ppm", "slow_amp_ppm"]].to_csv(out / "galaxy_assign.csv.gz", index=False)
    mism, res = t1_validation(assign)
    mism.to_csv(out / "t1_validation_mismatches.csv", index=False)
    (out / "t1_validation.json").write_text(json.dumps(res, indent=1))
    print("T1 validation:", res)
    per_bundle_sizes(assign, [1, 90, 180]).to_csv(out / "per_bundle_group_sizes.csv", index=False)
    sib = pd.read_csv(g / "galaxy_siblings.csv")
    pick = spotcheck(root, assign, sib, args.spot_bundle, out / "spotcheck_flux.npz")
    pick.to_csv(out / "spotcheck_members.csv", index=False)
    print("spot-check galaxies:", sorted(pick.galaxy_id.unique()))


if __name__ == "__main__":
    main()
