"""T20 phase 1: one streaming pass per SMARTS bundle -> noise fingerprints, corrupt-noise flags, stats.

Every SMARTS-TESS file is one butterpy spot signal injected into one of ~2,000
real TESS "quiet galaxy" light curves, and no header says which galaxy. Files
that share a galaxy share the identical noise realization, so this pass
records, for every file in a bundle:

- a **noise fingerprint**: the first difference of the raw flux (which
  suppresses the slow spot signal; at P = 0.1 d the spot signal is still
  weak against the galaxy's 30-min noise, see the T20 phase-1 report),
  with linearly interpolated gap runs, constant fills and non-finite values
  zeroed (they are shared by every galaxy with the same gap template and
  would fake a match), robustly standardized (median / MAD), clipped at
  +/-8 sigma and stored as int8 (x ``FP_SCALE``). This is T1's
  ``noise_fingerprint.fingerprint`` at full resolution (17,519 values),
  vectorized; ``tools/audit/galaxy_groups.py`` clusters these into galaxy IDs;
- the **corrupt-noise flags** of ``flux_outliers.py``: max |flux - 1|, the
  number of cadences with |flux - 1| > 0.1 and the orbits hit (27 equal
  slices of the 365-d grid);
- **per-file stats**: point-to-point scatter, a robust amplitude of the slow
  component (12-h bin means), fill/gap counts and a hash of the fill mask.

Outputs, per bundle, on the drive (``<root>/extracted/galaxy_pass/``),
each written atomically:

- ``NNN_fp.npy``: int8 (n_files, 17519) fingerprints, rows in the same order as
- ``NNN_stats.csv``: one row per file (written last: its presence marks the bundle done);
- ``NNN_meta.json``: timings (wall, time blocked on the tar stream, compute).

``--resume`` skips bundles whose ``NNN_stats.csv`` and ``NNN_fp.npy`` exist.
A bundle is all-or-nothing: an unplug or sleep-induced I/O error mid-bundle
leaves no partial output, and the rerun redoes only that bundle.

Usage::

    PY=/opt/anaconda3/envs/ss_openblas/bin/python
    $PY tools/audit/galaxy_pass.py --bundles 1 90 180 --n-workers 3
    $PY tools/audit/galaxy_pass.py --all --resume --n-workers 3   # full corpus (Rae, overnight)
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
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tarfile  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import smarts_data as sd  # noqa: E402

# Fixed SMARTS-TESS DR1 file layout (T1: every file is 155,520 B; checked per file below).
FILE_SIZE = 155_520
LC_HDR_OFFSET = 2880          # LIGHTCURVE BINTABLE header block
LC_DATA_OFFSET = 5760         # 17,520 rows x (time, flux) big-endian float32
N_CAD = 17_520
N_FP = N_CAD - 1
FP_CLIP = 8.0                 # sigma
FP_SCALE = 15.0               # int8 = round(clip(z) * FP_SCALE); |z| <= 8 -> |q| <= 120
OUTLIER_THRESH = 0.1          # |flux - 1| (flux_outliers.py default)
N_ORBITS = 27
BIN = 24                      # 12-h bins for the slow-component amplitude (17,520 = 730 x 24)
PASS_DIR = "galaxy_pass"
_ROW_RE = re.compile(r"smarts-tess-v1\.0-(\d{6})\.fits$")


def parse_lightcurve(buf: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, flux) float32 arrays from the raw bytes of one SMARTS FITS file.

    Fast path reads the fixed binary-table offset directly; anything that
    doesn't look exactly like the standard layout falls back to astropy.
    """
    if (len(buf) == FILE_SIZE and buf[LC_HDR_OFFSET:LC_HDR_OFFSET + 8] == b"XTENSION"
            and b"EXTNAME = 'LIGHTCURVE'" in buf[LC_HDR_OFFSET:LC_DATA_OFFSET]
            and b"NAXIS2  =                17520" in buf[LC_HDR_OFFSET:LC_DATA_OFFSET]):
        x = np.frombuffer(buf, ">f4", count=2 * N_CAD, offset=LC_DATA_OFFSET).reshape(-1, 2)
        return x[:, 0].astype(np.float32), x[:, 1].astype(np.float32)
    from astropy.io import fits
    with fits.open(io.BytesIO(buf)) as h:
        d = h["LIGHTCURVE"].data
        return np.asarray(d["time"], np.float32), np.asarray(d["flux"], np.float32)


def collinear(f: np.ndarray) -> np.ndarray:
    """Boolean array (len n-2): True at k when points k, k+1, k+2 are exactly collinear (linear fill).

    "Exactly" means a second difference within a few float32 ulps of the flux level,
    as in ``lc_spotcheck.interp_runs``.
    """
    tol = 8 * np.finfo(np.float32).eps * np.nanmedian(np.abs(f))
    return np.abs(np.diff(f, 2)) <= tol


def interp_diff_mask(f: np.ndarray, flat: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask over ``np.diff(f)``: True for diffs touching an exactly-linear (interpolated) run.

    Vectorized equivalent of T1's rule (``noise_fingerprint.fingerprint``): for
    every run ``(s, e)`` from ``lc_spotcheck.interp_runs(f, min_run=3)``, mask
    diffs ``max(s - 1, 0) .. e``. A collinear triple at k (see :func:`collinear`)
    therefore masks diffs k-1 .. k+2. ~100x faster than the Python loop.
    """
    flat = collinear(f) if flat is None else flat
    m = np.zeros(len(f) - 1, bool)
    m[:-1] |= flat          # diff k
    m[1:] |= flat           # diff k+1
    m[2:] |= flat[:-1]      # diff k+2
    m[:-2] |= flat[1:]      # diff k-1
    return m


def orbit_index(time: np.ndarray) -> np.ndarray:
    """Orbit number 1..27 of each cadence (27 equal time slices, as in flux_outliers.py)."""
    t = time.astype(float)
    return np.minimum((t / (t.max() + 1e-9) * N_ORBITS).astype(int) + 1, N_ORBITS)


def _run_len(x: np.ndarray) -> int:
    """Length of the leading run of values exactly equal to ``x[0]``."""
    neq = np.flatnonzero(x != x[0])
    return int(neq[0]) if len(neq) else len(x)


def file_record(flux32: np.ndarray, orbit: np.ndarray) -> tuple[dict, np.ndarray]:
    """Per-file stats dict and int8 fingerprint (length 17,519) from one raw flux array.

    ``orbit`` is the per-cadence orbit index from :func:`orbit_index`.
    """
    f = flux32.astype(np.float64)
    finite = np.isfinite(f)
    dev = np.abs(f - 1)
    bad_pt = finite & (dev > OUTLIER_THRESH)
    orbits_hit = np.unique(orbit[bad_pt])

    d = np.diff(f)
    flat = collinear(np.where(finite, f, 0.0))
    interp = interp_diff_mask(f, flat)
    const = d == 0
    nonfin = ~np.isfinite(d)
    masked = interp | const | nonfin
    good = d[~masked]
    if len(good) >= 100:
        med = float(np.median(good))
        scale = float(1.4826 * np.median(np.abs(good - med)))
    else:
        med, scale = 0.0, 0.0
    if scale > 0:
        z = np.clip((np.where(masked, med, d) - med) / scale, -FP_CLIP, FP_CLIP)
        fp = np.rint(z * FP_SCALE).astype(np.int8)
        fp[masked] = 0
    else:
        fp = np.zeros(N_FP, np.int8)

    # Slow component: 12-h bin means over cadences that are finite, not fill, not in a bad orbit.
    pt_fill = np.zeros(N_CAD, bool)
    pt_fill[1:-1] |= flat                          # interior points of linear runs
    pt_fill[1:] |= const                           # constant fills (both ends of each
    pt_fill[:-1] |= const                          # repeated pair)
    ok = finite & ~pt_fill & ~np.isin(orbit, orbits_hit)
    fb = np.where(ok, f, 0.0).reshape(-1, BIN)
    nb = ok.reshape(-1, BIN).sum(1)
    use = nb >= BIN // 2
    bm = fb.sum(1)[use] / nb[use]
    if len(bm) >= 20:
        p5, p95 = np.percentile(bm, [5, 95])
        slow_amp = float(p95 - p5)
        slow_rms = float(1.4826 * np.median(np.abs(bm - np.median(bm))))
    else:
        slow_amp = slow_rms = np.nan

    rec = dict(
        max_abs_dev=float(np.nanmax(dev)) if finite.any() else np.nan,
        n_dev_gt_thresh=int(bad_pt.sum()),
        orbits_hit=" ".join(map(str, orbits_hit)),
        flux_min=float(np.nanmin(f)) if finite.any() else np.nan,
        flux_max=float(np.nanmax(f)) if finite.any() else np.nan,
        n_nonfinite=int((~finite).sum()),
        n_interp_diff=int(interp.sum()),
        n_const_diff=int(const.sum()),
        lead_const=_run_len(f),
        trail_const=_run_len(f[::-1]),
        fp_n_used=int((~masked).sum()),
        p2p_ppm=scale / np.sqrt(2) * 1e6,
        slow_amp_ppm=slow_amp * 1e6,
        slow_rms_ppm=slow_rms * 1e6,
        n_slow_bins=int(use.sum()),
        fill_hash=hashlib.blake2b(np.packbits(masked).tobytes(), digest_size=8).hexdigest(),
    )
    return rec, fp


def out_paths(outdir: Path, n: int) -> dict:
    """Output file paths of bundle ``n``."""
    return {k: Path(outdir, f"{n:03d}_{k}.{ext}")
            for k, ext in (("fp", "npy"), ("stats", "csv"), ("meta", "json"))}


def bundle_done(outdir: Path, n: int) -> bool:
    """True if bundle ``n`` has complete outputs (stats CSV is written last)."""
    p = out_paths(outdir, n)
    return p["stats"].exists() and p["fp"].exists()


@contextlib.contextmanager
def open_bundle_stream(path: Path):
    """Yield a streaming ``tarfile`` over a .tar.gz, decompressed by an external ``gzip -dc``.

    Profiled 2026-10-21 on bundle 100 (533 MB): ``tarfile.open(mode="r|gz")``
    costs 7.9 s wall / 7.4 s Python CPU, while piping through ``gzip -dc`` costs
    4.0 s wall / 1.5 s Python CPU (decompression runs in its own process, in
    parallel with the per-file compute). Falls back to ``r|gz`` if gzip is missing.
    A non-zero gzip exit (e.g. the drive vanished) raises OSError.
    """
    gz = shutil.which("gzip")
    if gz is None:
        with tarfile.open(path, "r|gz") as tf:
            yield tf
        return
    proc = subprocess.Popen([gz, "-dc", str(path)], stdout=subprocess.PIPE, bufsize=1 << 20)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            yield tf
        proc.stdout.close()
        if proc.wait() != 0:
            raise OSError(f"gzip -dc {path} exited with {proc.returncode}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def process_bundle(n: int, root: str, outdir: str) -> dict:
    """Stream bundle ``n`` once and write its fingerprints, stats and timing meta (module-level, picklable).

    Returns the meta dict (timings and counts).
    """
    root, outdir = Path(root), Path(outdir)
    sd.require_drive(root)
    t_wall0, t_cpu0 = time.time(), time.process_time()
    t_io = 0.0
    recs, fps = [], []
    t_ref, orbit = None, None
    with open_bundle_stream(root / sd.BUNDLE_DIR / sd.bundle_name(n)) as tf:
        while True:
            a = time.time()
            m = tf.next()
            if m is None:
                t_io += time.time() - a
                break
            if not m.isfile():
                t_io += time.time() - a
                continue
            buf = tf.extractfile(m).read()
            t_io += time.time() - a
            match = _ROW_RE.search(m.name)
            time32, flux32 = parse_lightcurve(buf)
            if t_ref is None:
                t_ref, orbit = time32, orbit_index(time32)
            rec, fp = file_record(flux32, orbit)
            recs.append(dict(row=int(match.group(1)) if match else -1, bundle=n,
                             member=Path(m.name).name, size=m.size,
                             time_ok=bool(np.array_equal(time32, t_ref)), **rec))
            fps.append(fp)
    t_read_compute = time.time() - t_wall0
    p = out_paths(outdir, n)
    fp_arr = np.stack(fps) if fps else np.zeros((0, N_FP), np.int8)
    with sd.atomic_path(p["fp"]) as tmp:
        np.save(tmp, fp_arr)
    meta = dict(bundle=n, n_files=len(recs), wall_s=time.time() - t_wall0,
                tar_stream_s=t_io, compute_s=t_read_compute - t_io,
                write_s=time.time() - t_wall0 - t_read_compute,
                cpu_s=time.process_time() - t_cpu0,
                fp_bytes=int(fp_arr.nbytes), host_load=list(_loadavg()),
                fp_clip=FP_CLIP, fp_scale=FP_SCALE, outlier_thresh=OUTLIER_THRESH)
    with sd.atomic_path(p["meta"]) as tmp:
        tmp.write_text(json.dumps(meta, indent=1))
    with sd.atomic_path(p["stats"]) as tmp:
        pd.DataFrame(recs).to_csv(tmp, index=False)
    return meta


def _loadavg() -> tuple:
    """1/5/15-min load averages (to spot interference from other jobs)."""
    import os
    try:
        return os.getloadavg()
    except OSError:
        return (np.nan,) * 3


def load_pass(outdir: Path, bundles=None, fingerprints: bool = False):
    """Concatenate per-bundle stats (and optionally fingerprints, memory-mapped) for finished bundles.

    Returns ``stats`` DataFrame, or ``(stats, [fp memmaps in the same order])``.
    """
    outdir = Path(outdir)
    bundles = bundles or [n for n in range(1, sd.N_BUNDLES + 1) if bundle_done(outdir, n)]
    stats = pd.concat([pd.read_csv(out_paths(outdir, n)["stats"]) for n in bundles],
                      ignore_index=True)
    if not fingerprints:
        return stats
    return stats, [np.load(out_paths(outdir, n)["fp"], mmap_mode="r") for n in bundles]


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--outdir", default=None, help="default: <root>/extracted/galaxy_pass")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bundles", type=int, nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip bundles with complete outputs")
    ap.add_argument("--n-workers", type=int, default=3)
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    outdir = Path(args.outdir) if args.outdir else sd.extract_dir(root, PASS_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    bundles = list(range(1, sd.N_BUNDLES + 1)) if args.all else args.bundles
    if args.resume:
        bundles = [n for n in bundles if not bundle_done(outdir, n)]
    print(f"{len(bundles)} bundles to do -> {outdir}  ({args.n_workers} workers)", flush=True)
    t0, failed = time.time(), []
    if args.n_workers == 1:
        for n in bundles:
            m = process_bundle(n, str(root), str(outdir))
            print(f"bundle {n:03d}: {m['n_files']} files, {m['wall_s']:.1f} s "
                  f"[{time.time() - t0:.0f} s elapsed]", flush=True)
    else:
        with ProcessPoolExecutor(args.n_workers, initializer=determinism.init_worker) as ex:
            futs = {ex.submit(process_bundle, n, str(root), str(outdir)): n for n in bundles}
            for i, fu in enumerate(as_completed(futs), 1):
                n = futs[fu]
                try:
                    m = fu.result()
                    print(f"[{i}/{len(bundles)}] bundle {n:03d}: {m['n_files']} files, "
                          f"{m['wall_s']:.1f} s (stream {m['tar_stream_s']:.1f}, "
                          f"compute {m['compute_s']:.1f})  [{time.time() - t0:.0f} s elapsed]",
                          flush=True)
                except (OSError, EOFError, tarfile.TarError, SystemExit) as e:
                    failed.append(n)
                    print(f"bundle {n:03d} FAILED ({type(e).__name__}: {e}); rerun with --resume",
                          flush=True)
    print(f"done in {time.time() - t0:.0f} s; failed: {failed or 'none'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
