"""datasets/train_v0.csv (T10 ranker training sample): pool rule, nesting, balance. No drive needed."""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "datasets" / "train_v0.csv"


@pytest.fixture(scope="module")
def man():
    """The committed training manifest."""
    return pd.read_csv(MAN)


def test_training_pool_only(man):
    """Every row is from the SIM_ID < 800k training pool and a quiet noise source."""
    assert (man.sim_id < 800_000).all()
    assert man.noise_classification.eq("quiet").all()


def test_disjoint_from_eval_sets(man):
    """No training row (or noise TIC, where known) is in dev_small, dev_medium or the holdout."""
    for name in ("dev_small", "dev_medium", "holdout"):
        other = pd.read_csv(ROOT / "datasets" / f"{name}.csv")
        assert not set(man.row) & set(other.row), name


def test_nested_1k_first(man):
    """The first 1,000 rows by train_order are the nested_1k subset, balanced like the full set."""
    m = man.sort_values("train_order")
    assert len(man) == 3000 and m.nested_1k.iloc[:1000].all() and not m.nested_1k.iloc[1000:].any()
    share_1k = m[m.nested_1k].pbin.value_counts(normalize=True)
    share_all = m.pbin.value_counts(normalize=True)
    assert (share_1k - share_all).abs().max() < 0.01


def test_relative_paths(man):
    """Paths are relative to SMARTS_DATA_ROOT and point into extracted/train_v0/."""
    assert man.path.str.startswith("extracted/train_v0/").all()
    assert not man.path.str.startswith("/").any()
    assert man.row.is_unique
