"""Unpack the RGES PIT WG6 variable-star light-curve archives onto the external drive.

Layout produced (all under ``$WG6_DATA_ROOT``, default
``/Volumes/Data_Vault/wg6_roman_data``)::

    extracted/_tgz/<NAME>.tgz        inner tarballs pulled out of the Google Drive zips
    extracted/<NAME>/...             contents of each tarball (one folder per tarball)
    extracted/<NAME>/.done           marker written last; its presence means "complete"

Design (CLAUDE.md drive rules):

- Checks the drive is mounted first and stops with "plug in the Data_Vault drive" if not.
- Resumable: a tarball whose ``.done`` marker exists is skipped; a zip member already
  present in ``_tgz/`` with the right size is skipped.
- Atomic: every zip member is written to ``<name>.part`` then renamed; every tarball is
  unpacked into ``extracted/.<NAME>.tmp/`` and the folder is renamed into place only
  when complete. An unplug mid-run leaves at most a ``.part`` file or a ``.tmp`` folder,
  which the next run deletes and redoes.
- Light on the laptop: one archive at a time, no worker pools.
- Safe on untrusted archives: ``tarfile`` extraction uses ``filter="data"`` (no absolute
  paths, no ``..``, no device files, no links leaving the destination).

Usage::

    python tools/extract_wg6.py                 # everything
    python tools/extract_wg6.py --only RRLYR     # one tarball (name stem, case-sensitive)
    python tools/extract_wg6.py --list           # show what is done / pending
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

DEFAULT_ROOT = "/Volumes/Data_Vault/wg6_roman_data"


def resolve_root(cli_root: str | None) -> Path:
    """Return the WG6 data root (CLI > ``WG6_DATA_ROOT`` env > default) after a mount check.

    Exits with a clear message if the drive is not mounted.
    """
    root = Path(cli_root or os.environ.get("WG6_DATA_ROOT", DEFAULT_ROOT))
    if not root.is_dir():
        sys.exit(f"WG6 data root {root} not found: plug in the Data_Vault drive "
                 "(or set WG6_DATA_ROOT / --root if it mounted elsewhere, e.g. 'Data_Vault 1').")
    return root


def tarball_stem(name: str) -> str:
    """Folder name for a tarball: ``ECL_lightcurves2-002.tgz`` -> ``ECL_lightcurves2-002``."""
    for ext in (".tar.gz", ".tgz"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def pull_zip_members(root: Path) -> list[Path]:
    """Copy each inner ``.tgz`` out of the Google Drive zips into ``extracted/_tgz`` atomically.

    Returns the list of inner tarball paths (existing or newly written). Other members
    (e.g. ``README.docx``) are copied too, to ``extracted/_docs``.
    """
    tgz_dir = root / "extracted" / "_tgz"
    doc_dir = root / "extracted" / "_docs"
    tgz_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for zpath in sorted(root.glob("drive-download-*.zip")):
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name  # flatten; never trust member paths
                if not name or info.is_dir():
                    continue
                dest = (tgz_dir if name.endswith((".tgz", ".tar.gz")) else doc_dir) / name
                if dest.exists() and dest.stat().st_size == info.file_size:
                    print(f"  skip (present) {name}")
                else:
                    part = dest.with_name(dest.name + ".part")
                    print(f"  zip -> {dest.relative_to(root)} ({info.file_size / 1e6:.0f} MB)")
                    with zf.open(info) as src, open(part, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=16 << 20)
                    os.replace(part, dest)
                if dest.parent == tgz_dir:
                    out.append(dest)
    return out


def extract_tarball(tgz: Path, extracted: Path) -> None:
    """Unpack one tarball into ``extracted/<stem>/`` via a temp folder, then mark it done."""
    stem = tarball_stem(tgz.name)
    final = extracted / stem
    if (final / ".done").exists():
        print(f"  skip (done) {stem}")
        return
    tmp = extracted / f".{stem}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)  # leftover from an interrupted run
    if final.exists():
        shutil.rmtree(final)  # incomplete (no .done) -> redo
    tmp.mkdir(parents=True)
    print(f"  untar {tgz.name} ({tgz.stat().st_size / 1e9:.2f} GB) ...", flush=True)
    n = 0
    with tarfile.open(tgz, "r:*") as tf:
        for member in tf:
            tf.extract(member, tmp, filter="data")
            n += 1
    (tmp / ".done").write_text(f"{n} members from {tgz.name}\n")
    os.replace(tmp, final)
    print(f"  done {stem}: {n} members")


def main(argv=None) -> None:
    """CLI entry point: pull tarballs out of zips, then unpack each tarball in turn."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="WG6 data root (default: $WG6_DATA_ROOT or %s)" % DEFAULT_ROOT)
    ap.add_argument("--only", default=None, help="only this tarball stem prefix (e.g. RRLYR, Rotators)")
    ap.add_argument("--list", action="store_true", help="list tarballs and their status, then exit")
    args = ap.parse_args(argv)

    root = resolve_root(args.root)
    extracted = root / "extracted"
    extracted.mkdir(exist_ok=True)

    tarballs = sorted(set(root.glob("*.tgz")) | set(root.glob("*.tar.gz")))
    if not args.list:
        print("Step 1: inner tarballs from zips")
        tarballs += pull_zip_members(root)
    else:
        tarballs += sorted((extracted / "_tgz").glob("*.tgz"))
    tarballs = sorted(set(tarballs), key=lambda p: p.name)
    if args.only:
        tarballs = [t for t in tarballs if t.name.startswith(args.only)]

    if args.list:
        for t in tarballs:
            state = "done" if (extracted / tarball_stem(t.name) / ".done").exists() else "pending"
            print(f"{state:8s} {t.name}")
        return
    print("Step 2: unpack tarballs (one at a time)")
    for t in tarballs:
        extract_tarball(t, extracted)


if __name__ == "__main__":
    main()
