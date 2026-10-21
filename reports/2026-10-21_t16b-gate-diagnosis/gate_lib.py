"""
gate_lib.py -- shared loading + offline gate-replay helpers for the T16b report.

Reads the instrumented scorecard outputs (per_star.csv, per_candidate.csv)
and the pb_scan noise candidate dump, and replays alternative gate rules on
the recorded per-candidate gate values (candidate fits don't depend on the
gating rule, so any rule can be evaluated without re-fitting).

Replay approximation: the selected candidate is the lowest-redchi candidate
that passes the rule. The real pipeline additionally applies the
factor-of-2 alias guard among passing candidates (needs the ACF, not
recorded); `replay_agreement` measures how often the replay of the current
rule reproduces the real outcome.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"

GATES = ("n_peaks", "frac_positive_heights", "mean_height_snr", "frac_vertex_in_window",
         "half_lag_consistency", "few_peaks_height")
GATE_SHORT = {"n_peaks": "n_peaks", "frac_positive_heights": "pos_heights", "mean_height_snr": "height_snr",
              "frac_vertex_in_window": "vertex", "half_lag_consistency": "half_lag",
              "few_peaks_height": "few_height"}
SUCCESS = ("correct", "diffrot")
ALIAS = ("half", "double")

# dataset label -> run directory
SYNTH = {"full": "t16b_synth200_full", "consecutive-2": "t16b_synth200_consecutive-2",
         "gapped-2": "t16b_synth200_gapped-2", "single": "t16b_synth200_single"}
DEV_MEDIUM = "t16b_dev_medium_assigned"
NOISE = "t16b_noise"
# held-out validation (rules were chosen on the sets above)
VALID = {"s1-single": "t16b_synth200s1_single", "s1-gapped-2": "t16b_synth200s1_gapped-2"}
NOISE_VALID = "t16b_noise_off1500"
BIN_ORDER = ["<1d", "1-3d", "3-10d", "10-30d", "30-60d", "60-120d", ">120d"]


def load_run(name: str, runs: Path = RUNS) -> tuple:
    """(per_star, per_candidate) DataFrames of one scorecard run directory."""
    d = runs / name
    ps = pd.read_csv(d / "per_star.csv", float_precision="round_trip")
    pc = pd.read_csv(d / "per_candidate.csv", float_precision="round_trip")
    return ps, pc


DATA = Path(__file__).resolve().parent / "data"


def load_data(data: Path = DATA) -> tuple:
    """(stars, candidates) compact tables written by make_data.py, with
    candidate class labels (cls: success / alias / wrong / noise)."""
    s = pd.read_csv(data / "stars.csv.gz", float_precision="round_trip", low_memory=False)
    c = pd.read_csv(data / "candidates.csv.gz", float_precision="round_trip", low_memory=False)
    c["redchi"] = c.redchi.fillna(np.inf)  # non-finite redchi sorts last, as in the pipeline
    c["cls"] = cand_class(c.outcome)
    for g in GATES:
        c[f"{g}_pass"] = c[f"{g}_pass"].astype(bool)
        c[f"{g}_enabled"] = c[f"{g}_enabled"].astype(bool)
    return s, c


def load_synth(runs: Path = RUNS) -> tuple:
    """All synthetic coverage runs concatenated, with a `dataset` column."""
    stars, cands = [], []
    for label, name in SYNTH.items():
        ps, pc = load_run(name, runs)
        ps["dataset"] = label
        pc["dataset"] = label
        stars.append(ps)
        cands.append(pc)
    return pd.concat(stars, ignore_index=True), pd.concat(cands, ignore_index=True)


def load_noise(runs: Path = RUNS) -> pd.DataFrame:
    """pb_scan noise candidates, with dataset = coverage-equivalent label and star_id."""
    df = pd.read_csv(runs / NOISE / "noise_candidates.csv", float_precision="round_trip")
    df["dataset"] = df.n_sectors.map({1: "single", 2: "consecutive-2", 13: "full"})
    df["star_id"] = "noise-" + df.n_sectors.astype(str) + "-" + df.seed.astype(str)
    return df


def fitted(pc: pd.DataFrame) -> pd.DataFrame:
    """Only candidates that were actually fitted (have gate values). A
    non-finite redchi (stored as NaN) becomes +inf, as fit_rotation_period
    sorts it (last)."""
    out = pc[pc.P_fit.notna()].copy()
    out["redchi"] = out.redchi.fillna(np.inf)
    return out


def cand_class(outcome: pd.Series) -> pd.Series:
    """success / alias / wrong / noise label of a candidate's fitted period."""
    return np.select([outcome.isin(SUCCESS), outcome.isin(ALIAS), outcome.eq("noise")],
                     ["success", "alias", "noise"], "wrong")


def best_per_star(pc: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Lowest-redchi candidate per (dataset, star) among rows where mask is True."""
    sub = pc[mask]
    if sub.empty:
        return sub
    idx = sub.groupby(["dataset", "star_id"]).redchi.idxmin()
    return sub.loc[idx.values]


# ------------------------------------------------------------------ gate rules

def gate_pass(pc: pd.DataFrame, overrides: dict | None = None) -> pd.DataFrame:
    """Per-gate pass booleans for each candidate under the current thresholds,
    with optional per-gate overrides {gate: fn(pc) -> bool Series}."""
    out = pd.DataFrame(index=pc.index)
    for g in GATES:
        p = pc[f"{g}_pass"].astype(bool)
        en = pc[f"{g}_enabled"].astype(bool)
        out[g] = p | ~en
    for g, fn in (overrides or {}).items():
        en = pc[f"{g}_enabled"].astype(bool)
        out[g] = fn(pc).astype(bool) | ~en
    return out


def replay(pc: pd.DataFrame, overrides: dict | None = None, fallback_min: float | None = None,
           apply_mask: pd.Series | None = None, stars: pd.DataFrame | None = None) -> pd.DataFrame:
    """Replay a gate rule; returns one row per (dataset, star_id) with the
    replayed outcome ('none' when nothing passes) and selected P_fit.

    overrides : {gate: fn(pc) -> bool Series} replacing that gate's pass test.
    fallback_min : emulate fit_rotation_period's half_lag_fallback_min tier.
    apply_mask : if given, overrides only apply where True (e.g. short
        coverage); elsewhere the current gates are used.
    stars : every (dataset, star_id) to report (default: stars with at
        least one fitted candidate); stars without candidates are 'none'.
    """
    cur = gate_pass(pc)
    new = gate_pass(pc, overrides)
    if apply_mask is not None:
        m = np.repeat(apply_mask.to_numpy()[:, None], new.shape[1], axis=1)
        new = new.where(m, cur)
    ok = new.all(axis=1)
    if fallback_min is not None:
        base = new[["n_peaks", "frac_positive_heights", "mean_height_snr", "frac_vertex_in_window",
                    "few_peaks_height"]].all(axis=1)
        fb = base & (pc.half_lag_contrast.fillna(-np.inf) > fallback_min)
        any_ok = ok.groupby([pc.dataset, pc.star_id]).transform("any")
        ok = ok | (~any_ok & fb)
    sel = best_per_star(pc, ok)
    stars = pc[["dataset", "star_id"]].drop_duplicates() if stars is None else stars[["dataset", "star_id"]]
    res = stars.merge(sel[["dataset", "star_id", "outcome", "P_fit"]], how="left", on=["dataset", "star_id"])
    res["outcome"] = res.outcome.fillna("none")
    return res


def outcome_counts(res: pd.DataFrame, stars: pd.DataFrame | None = None, by=("dataset",)) -> pd.DataFrame:
    """success / alias / wrong / noise / none counts of a replay, grouped by
    `by` (columns of res, or of `stars` merged in on (dataset, star_id))."""
    r = res if stars is None else res.merge(stars, on=["dataset", "star_id"], how="left")
    r = r.assign(cls=np.where(r.outcome.eq("none"), "none", cand_class(r.outcome)))
    out = r.groupby(list(by)).cls.value_counts().unstack(fill_value=0)
    for c in ("success", "alias", "wrong", "noise", "none"):
        if c not in out:
            out[c] = 0
    return out[["success", "alias", "wrong", "noise", "none"]]


def replay_agreement(ps: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: fraction of stars where the current-rule replay
    reproduces the real final outcome class (the gap is the alias guard)."""
    r = replay(pc, stars=ps)
    m = ps[["dataset", "star_id", "final_outcome"]].merge(r, on=["dataset", "star_id"], how="left")
    real = np.where(m.final_outcome.eq("none"), "none", cand_class(m.final_outcome))
    rep = np.where(m.outcome.eq("none"), "none", cand_class(m.outcome))
    m["agree"] = real == rep
    return m.groupby("dataset").agree.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "agree"})


def tradeoff(stars: pd.DataFrame, cands: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Replay every rule; one row per (rule, dataset) with outcome counts
    (success / alias / wrong / noise / none) and deltas vs rule 'current'.

    rules : {name: (overrides, fallback_min, short_only)} (see rules.py);
        short_only applies the overrides only to stars with n_sectors <= 2.
    """
    nsec = cands[["dataset", "star_id"]].merge(stars[["dataset", "star_id", "n_sectors"]],
                                               on=["dataset", "star_id"], how="left").n_sectors
    short = pd.Series((nsec <= 2).to_numpy(), index=cands.index)
    rows = []
    for name, (ov, fb, short_only) in rules.items():
        r = replay(cands, ov, fb, apply_mask=short if short_only else None, stars=stars)
        oc = outcome_counts(r).assign(rule=name)
        rows.append(oc.reset_index())
    out = pd.concat(rows, ignore_index=True)
    base = out[out.rule == "current"].set_index("dataset")
    for c in ("success", "alias", "wrong", "noise", "none"):
        out[f"d_{c}"] = out[c] - out.dataset.map(base[c])
    return out
