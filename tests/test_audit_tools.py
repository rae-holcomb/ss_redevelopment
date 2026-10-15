"""Tests for tools/audit helpers that don't need the SMARTS drive."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))
import smarts_data as sd  # noqa: E402
from lc_spotcheck import interp_runs  # noqa: E402


def test_interp_runs_finds_linear_fill():
    """A linearly filled gap inside noise is found with the right bounds."""
    rng = np.random.default_rng(0)
    f = 1 + 1e-3 * rng.standard_normal(500)
    f[100:140] = np.linspace(f[99], f[140], 42)[1:-1]
    runs = interp_runs(f.astype(np.float32).astype(float))
    assert any(s <= 100 and e >= 139 for s, e in runs)
    assert all(e - s < 60 for s, e in runs)


def test_atomic_path_leaves_nothing_on_failure(tmp_path):
    """atomic_path only creates the final file on success, and cleans up its temp file."""
    final = tmp_path / "out.csv"
    with pytest.raises(RuntimeError):
        with sd.atomic_path(final) as tmp:
            tmp.write_text("partial")
            raise RuntimeError
    assert not final.exists() and list(tmp_path.iterdir()) == []
    with sd.atomic_path(final) as tmp:
        tmp.write_text("ok")
    assert final.read_text() == "ok"


def test_require_drive_exits_cleanly(tmp_path):
    """A missing data root stops with a 'plug in the drive' message, not a traceback."""
    with pytest.raises(SystemExit, match="Plug in the Data_Vault drive"):
        sd.require_drive(tmp_path / "nope")
