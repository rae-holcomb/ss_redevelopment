"""T20 phase 2: recover each noise source's own light curve by amplitude-aware stacking of its SMARTS members.

Every SMARTS file = one butterpy spot signal injected into one of 1,719 TESS
noise light curves (TIC IDs from Zach Claytor; ``galaxy_tic_map.py``). The
injected signals have unrelated periods and phases, so a per-cadence median
over many members of one TIC leaves the shared noise light curve. Following
Rae's amplitude caution (T20 decisions, 2026-10-21):

- **select**: per TIC, rank members by an expected injected amplitude
  ``E = ACTIVITY * |sin(INCL + mean band latitude)|`` (activity sets spot
  area; the sine is the visible-hemisphere factor of Claytor+2022 Eq. 4 for
  a spot at the band's mean latitude). The ``K = min(--k, n // 3)`` lowest-E
  members build the estimate, split by rank parity into two disjoint halves
  A and B (for the persistence test). ``--n-check-top`` highest-E and
  ``--n-check-random`` random other members are held out as checks: their
  residual (member - estimate) should look like their own injected signal.
- **extract**: one resumable streaming pass over the bundles that saves the
  raw flux of the selected files only (per bundle, float32, atomic).
- **stack**: per TIC, the median over A, over B and over A+B (plus the
  robust spread across members), and per check/stack member the residual
  statistics against the A+B estimate.

Outputs under ``<root>/extracted/galaxy_stack/``: ``selection.csv``,
``flux/NNN_{flux,rows}.npy``, ``stacks.npy`` (n_tic, 3, 17520 float32; A, B, all),
``stack_spread.npy``, ``stack_index.csv``, ``residuals.csv``.

Usage::

    $PY tools/audit/galaxy_stack.py select
    $PY tools/audit/galaxy_stack.py extract --all --resume --n-workers 3     # ~25 min, drive
    $PY tools/audit/galaxy_stack.py stack
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "claude_files"))
import determinism  # noqa: E402

if __name__ == "__main__":  # entry point only; pool workers inherit the env vars
    determinism.pin_blas_threads()

import argparse  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import galaxy_pass as gp  # noqa: E402
import smarts_data as sd  # noqa: E402

STACK_DIR = "galaxy_stack"
ROLES = ("A", "B", "check")


def expected_amplitude(cat) -> np.ndarray:
    """E = ACTIVITY * |sin(INCL + mean band latitude)| (ranking only; see module docstring)."""
    lat = np.radians(0.5 * (cat["MINLAT"] + cat["MAXLAT"]))
    return cat["ACTIVITY"] * np.abs(np.sin(cat["INCL"] + lat))


def select(root: Path, k: int, n_top: int, n_rand: int, seed: int) -> pd.DataFrame:
    """Per-TIC stacking (A/B) and check members. Returns rows with galaxy, tic, role, E, rank."""
    m = pd.read_csv(Path(root, sd.EXTRACT_DIR, "galaxy_ids", "galaxy_map.csv.gz"))
    cat = np.load(Path(root, sd.EXTRACT_DIR, "catalog", "smarts_catalog.npz"))
    m["E"] = expected_amplitude({c: cat[c] for c in ("ACTIVITY", "INCL", "MINLAT", "MAXLAT")})[m.row]
    rng = np.random.default_rng(seed)
    out = []
    for g, d in m.groupby("galaxy", sort=True):
        d = d.sort_values("E", kind="stable")
        kk = min(k, len(d) // 3)
        st = d.iloc[:kk].assign(role=np.where(np.arange(kk) % 2 == 0, "A", "B"))
        rest = d.iloc[kk:]
        top = rest.iloc[-n_top:]
        others = rest.iloc[:-n_top] if n_top else rest
        rnd = others.iloc[rng.choice(len(others), min(n_rand, len(others)), replace=False)]
        out += [st, top.assign(role="check"), rnd.assign(role="check")]
    sel = pd.concat(out, ignore_index=True)
    sel["E_rank_frac"] = sel.groupby("galaxy").E.rank(pct=True)
    return sel[["row", "sim_id", "tic", "galaxy", "pool", "bundle", "role", "E", "E_rank_frac"]]


def extract_bundle(n: int, root: str, rows: np.ndarray, outdir: str) -> dict:
    """Stream bundle ``n`` and save the raw flux of ``rows`` (module-level, picklable)."""
    root, outdir = Path(root), Path(outdir)
    sd.require_drive(root)
    want = set(int(r) for r in rows)
    t0 = time.time()
    got_rows, flux = [], []
    with gp.open_bundle_stream(root / sd.BUNDLE_DIR / sd.bundle_name(n)) as tf:
        for m in tf:
            if not m.isfile():
                continue
            match = gp._ROW_RE.search(m.name)
            if match and int(match.group(1)) in want:
                flux.append(gp.parse_lightcurve(tf.extractfile(m).read())[1])
                got_rows.append(int(match.group(1)))
    order = np.argsort(got_rows)
    with sd.atomic_path(outdir / f"{n:03d}_flux.npy") as tmp:
        np.save(tmp, np.stack(flux)[order] if flux else np.zeros((0, gp.N_CAD), np.float32))
    with sd.atomic_path(outdir / f"{n:03d}_rows.npy") as tmp:   # written last = done marker
        np.save(tmp, np.asarray(got_rows, np.int64)[order])
    return dict(bundle=n, n_want=len(want), n_got=len(got_rows), wall_s=time.time() - t0)


def load_selected(fdir: Path, rows_wanted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gather the extracted flux of ``rows_wanted`` (any order) via memory-mapped per-bundle files.

    Returns (rows found, float32 flux (n, 17520)) in the order of ``rows_wanted``
    (rows not extracted are dropped).
    """
    want = pd.Index(np.asarray(rows_wanted))
    rows_out, flux_out = [], []
    for rp in sorted(fdir.glob("*_rows.npy")):
        r = np.load(rp)
        hit = np.flatnonzero(np.isin(r, want))
        if len(hit):
            F = np.load(rp.with_name(rp.name.replace("_rows", "_flux")), mmap_mode="r")
            rows_out.append(r[hit])
            flux_out.append(np.asarray(F[hit]))
    rows = np.concatenate(rows_out)
    flux = np.concatenate(flux_out)
    order = pd.Index(rows).get_indexer(want[want.isin(rows)])
    return rows[order], flux[order]


def robust_scatter(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """1.4826 * median absolute deviation along ``axis`` (NaN-aware)."""
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - med), axis=axis)


def stack_block(sel: pd.DataFrame, rows: np.ndarray, F: np.ndarray) -> tuple[dict, list]:
    """Stacks and residual stats for the galaxies in ``sel`` (rows/F: their extracted flux)."""
    idx = pd.Index(rows)
    stacks, spread, recs = {}, {}, []
    for g, d in sel.groupby("galaxy"):
        i = {r: idx.get_indexer(d.row[d.role == r]) for r in ROLES}
        i = {r: v[v >= 0] for r, v in i.items()}
        ab = np.concatenate([i["A"], i["B"]])
        S = np.stack([np.median(F[i["A"]], 0), np.median(F[i["B"]], 0), np.median(F[ab], 0)])
        stacks[g] = S.astype(np.float32)
        spread[g] = robust_scatter(F[ab]).astype(np.float32)
        est = S[2]
        for role in ROLES:
            for j in i[role]:
                res = F[j] - est
                ok = np.isfinite(res) & (np.abs(est - 1) < gp.OUTLIER_THRESH)
                recs.append(dict(row=int(rows[j]), galaxy=g, role=role,
                                 res_rms_ppm=float(robust_scatter(res[ok]) * 1e6),
                                 res_p2p_ppm=float(robust_scatter(np.diff(res[ok])) / np.sqrt(2) * 1e6),
                                 ratio_rms_ppm=float(robust_scatter(F[j][ok] / est[ok]) * 1e6)))
    return dict(stacks=stacks, spread=spread), recs


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["select", "extract", "stack"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--k", type=int, default=80, help="max stacking members per TIC (A+B)")
    ap.add_argument("--n-check-top", type=int, default=4)
    ap.add_argument("--n-check-random", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261021)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--bundles", type=int, nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n-workers", type=int, default=3)
    ap.add_argument("--block", type=int, default=250, help="TICs per gather block in 'stack'")
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    out = sd.extract_dir(root, STACK_DIR)
    fdir = sd.extract_dir(root, STACK_DIR, "flux")

    if args.step == "select":
        sel = select(root, args.k, args.n_check_top, args.n_check_random, args.seed)
        with sd.atomic_path(out / "selection.csv") as tmp:
            sel.to_csv(tmp, index=False)
        print(sel.groupby("role").size().to_dict(), f"{sel.galaxy.nunique()} TICs")
        return

    sel = pd.read_csv(out / "selection.csv")
    if args.step == "extract":
        bundles = list(range(1, sd.N_BUNDLES + 1)) if args.all or not args.bundles else args.bundles
        if args.resume:
            bundles = [n for n in bundles if not (fdir / f"{n:03d}_rows.npy").exists()]
        by_b = {n: d.row.values for n, d in sel.groupby("bundle")}
        print(f"{len(bundles)} bundles to extract -> {fdir}", flush=True)
        t0, failed = time.time(), []
        with ProcessPoolExecutor(args.n_workers, initializer=determinism.init_worker) as ex:
            futs = {ex.submit(extract_bundle, n, str(root), by_b.get(n, np.zeros(0, int)), str(fdir)): n
                    for n in bundles}
            for i, fu in enumerate(as_completed(futs), 1):
                n = futs[fu]
                try:
                    r = fu.result()
                    flag = "" if r["n_got"] == r["n_want"] else "  MISSING ROWS"
                    print(f"[{i}/{len(bundles)}] bundle {n:03d}: {r['n_got']}/{r['n_want']} files, "
                          f"{r['wall_s']:.1f} s{flag}  [{time.time() - t0:.0f} s]", flush=True)
                except (OSError, EOFError, SystemExit) as e:
                    failed.append(n)
                    print(f"bundle {n:03d} FAILED ({type(e).__name__}: {e}); rerun with --resume", flush=True)
        print(f"done in {time.time() - t0:.0f} s; failed: {failed or 'none'}")
        sys.exit(1 if failed else 0)

    # stack
    gals = np.sort(sel.galaxy.unique())
    all_stacks = np.full((len(gals), 3, gp.N_CAD), np.nan, np.float32)
    all_spread = np.full((len(gals), gp.N_CAD), np.nan, np.float32)
    recs, t0 = [], time.time()
    for b0 in range(0, len(gals), args.block):
        blk = gals[b0:b0 + args.block]
        s = sel[sel.galaxy.isin(blk)]
        rows, F = load_selected(fdir, s.row.values)
        res, r = stack_block(s, rows, F.astype(np.float64))
        for g in blk:
            if g in res["stacks"]:
                all_stacks[np.searchsorted(gals, g)] = res["stacks"][g]
                all_spread[np.searchsorted(gals, g)] = res["spread"][g]
        recs += r
        print(f"  stacked TICs {b0}-{b0 + len(blk) - 1}: {len(rows)} files  [{time.time() - t0:.0f} s]",
              flush=True)
    idx = sel.groupby("galaxy").agg(tic=("tic", "first"), pool=("pool", "first"),
                                    n_A=("role", lambda x: int((x == "A").sum())),
                                    n_B=("role", lambda x: int((x == "B").sum())),
                                    E_max_stack=("E", lambda x: float(x.max()))).reset_index()
    with sd.atomic_path(out / "stacks.npy") as tmp:
        np.save(tmp, all_stacks)
    with sd.atomic_path(out / "stack_spread.npy") as tmp:
        np.save(tmp, all_spread)
    with sd.atomic_path(out / "stack_index.csv") as tmp:
        idx.to_csv(tmp, index=False)
    with sd.atomic_path(out / "residuals.csv") as tmp:
        pd.DataFrame(recs).merge(sel[["row", "E", "E_rank_frac"]], on="row").to_csv(tmp, index=False)
    print(f"stacks: {all_stacks.shape} -> {out}")


if __name__ == "__main__":
    main()
