"""Shared helpers for reading the SMARTS-TESS DR1 corpus off the external drive.

The corpus lives on an often-unplugged external drive (see CLAUDE.md,
"SMARTS data location"). Every script in ``tools/audit/`` goes through these
helpers so that:

- the data root comes from ``--data-root`` / ``$SMARTS_DATA_ROOT`` and is
  checked *before* any work starts (a clear "plug in the drive" message,
  never a failure deep inside a loop);
- extracted products go to the drive (``<root>/extracted/``), never in bulk
  to the laptop;
- every output file is written atomically (temp file in the same directory,
  then ``os.replace``), so an unplug mid-write leaves no corrupt partial file.

Layout under the data root (default ``/Volumes/Data_Vault/smarts_data``)::

    tess_retry/hlsp_smarts_tess_ffi_NNN_tess_v1.0_sim.tar.gz   (180 period bundles)
    tess_retry/hlsp_smarts_tess_ffi_all_tess_v1.0_cat.tar.gz   (catalog)
    extracted/                                                 (our outputs)
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from pathlib import Path

DEFAULT_DATA_ROOT = "/Volumes/Data_Vault/smarts_data"
BUNDLE_DIR = "tess_retry"
EXTRACT_DIR = "extracted"
CATALOG_TARBALL = "hlsp_smarts_tess_ffi_all_tess_v1.0_cat.tar.gz"
CATALOG_FITS = "hlsp_smarts_tess_ffi_all_tess_v1.0_sim.fits"
N_BUNDLES = 180


def bundle_name(n: int) -> str:
    """Tarball filename of period bundle ``n`` (1..180; holds periods n-1 to n days)."""
    return f"hlsp_smarts_tess_ffi_{n:03d}_tess_v1.0_sim.tar.gz"


def data_root(cli_value: str | None = None) -> Path:
    """Resolve the SMARTS data root: CLI value, then $SMARTS_DATA_ROOT, then the default."""
    return Path(cli_value or os.environ.get("SMARTS_DATA_ROOT") or DEFAULT_DATA_ROOT)


def require_drive(root: Path) -> Path:
    """Exit with a clear message unless the data root (and its bundle dir) is reachable.

    Call this first in every entry point that touches the drive.
    """
    root = Path(root)
    if not (root / BUNDLE_DIR).is_dir():
        parent = root.parent if root.parent != root else root
        mounted = parent.exists()
        msg = (f"SMARTS data root not found: {root / BUNDLE_DIR}\n"
               + ("" if mounted else f"({parent} is not mounted.)\n")
               + "Plug in the Data_Vault drive, or point --data-root / "
                 "$SMARTS_DATA_ROOT at it (macOS may mount it as 'Data_Vault 1').")
        sys.exit(msg)
    return root


def extract_dir(root: Path, *parts: str) -> Path:
    """Return (and create) ``<root>/extracted/<parts...>`` on the drive."""
    d = Path(root, EXTRACT_DIR, *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextlib.contextmanager
def atomic_path(final: Path):
    """Yield a temp path next to ``final``; rename it onto ``final`` only on success.

    The temp file keeps ``final``'s suffix so writers that dispatch on the
    extension (numpy, pandas, matplotlib) behave normally.
    """
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{final.stem}.", suffix=f".part{final.suffix}",
                               dir=final.parent)
    os.close(fd)
    os.chmod(tmp, 0o644)  # mkstemp creates 0600; outputs should be normally readable
    try:
        yield Path(tmp)
        os.replace(tmp, final)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
        raise
