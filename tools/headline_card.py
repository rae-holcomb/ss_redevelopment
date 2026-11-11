"""Headline card: a small, fixed-definition table every method/ranker report can quote (P28).

Computes from existing per-star outputs only (no pipeline runs):

* **hand arbitration**: a scorecard ``per_star.csv`` (``final_period`` = the pipeline's answer,
  NaN when it declined);
* **ranker**: a ``*_top1.csv`` from ``tools/train_ranker.py`` / ``train_ranker_negatives.py``
  (``top_period``, ``top_score``; the ranker answers when ``top_score >= threshold``).

Every period is re-scored here with ``evaluation.classify_period`` (``REL_TOL``), so the card never
depends on how an upstream table labelled its outcomes.

Rows (fixed definitions; see ``ROW_DEFINITIONS``)
-------------------------------------------------
R0  raw success: unweighted mean over the dataset's rotators (continuity with older reports).
R1  success at the real TESS coverage mix, P <= 30 d: per-star ``coverage_weight`` from the
    manifest (assigned coverage only; n/a for full-year, whose coverage is fixed).
R2  success re-weighted to a named period population (default ``mcq14_like``), times the coverage
    weight when the mode is assigned. All periods 0.1-180 d.
R3  correct-only under the R2 weighting (success = correct + diffrot; this drops diffrot).
R4  precision of reported periods at stated non-rotator prevalences, under the R2 weighting:
    ``prec(f) = f*S / (f*A + (1-f)*FP)``, with S = success rate, A = answer rate per rotator, and
    FP = answer rate per non-rotator (from galaxy-stack negatives, coverage-weighted when assigned).
R5  Claytor+2022-comparable: fraction of the *most credible half* of all rotators whose reported
    period is within 10% / 20% of the injected equatorial period (no diffrot credit). Credibility =
    ranker score; hand arbitration has no score, so its answered stars come first and declined
    stars fill the half as misses. Unweighted, like Claytor's test-set numbers.

Each number carries a bootstrap standard error (stars resampled with replacement, seed 0).

Example
-------
    python tools/headline_card.py --mode assigned --manifest datasets/dev_medium.csv \\
        --hand "hand (T5)=benchmarks/baselines/dev_medium_assigned_t5_tables/per_star.csv" \\
        --ranker "ranker v0=/path/to/runs/t27_ranker/v0_assigned_top1.csv" \\
        --negatives /path/to/runs/t27_ranker/v0_assigned_top1.csv --out runs/headline_assigned

writes ``headline.json`` and ``headline.md`` in ``--out``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
from evaluation import (CORRECT, DIFFROT, PERIOD_BIN_EDGES, PERIOD_BIN_LABELS,  # noqa: E402
                        REL_TOL, classify_period, period_bin)

P_MIN, P_MAX = 0.1, 180.0          # SMARTS period range (days)
P_SHORT_MAX = 30.0                 # R1 restricts to P <= 30 d (beyond reach for most real coverage)
DEFAULT_PREVALENCES = (0.1, 0.3, 0.5)
DEFAULT_POPULATION = "mcq14_like"
RANKER_V0_THRESHOLD = 1.3571535348892212   # runs/t27_ranker/results.json, models.v0.thresholds.v0_original
N_BOOT = 200

# Period populations, each a callable giving the CDF in P over [P_MIN, P_MAX] (unnormalized OK).
# mcq14_like: log10 P ~ N(1.1, 0.35) (median ~12.6 d), a rough stand-in for the McQuillan+2014 Kepler
#   rotation-period distribution, written FROM MEMORY. It is an open question for Rae (which
#   population; whether to histogram the real catalogue instead).
POPULATIONS = {
    "mcq14_like": lambda p: norm.cdf((np.log10(p) - 1.1) / 0.35),
    "loguniform": lambda p: np.log(p),
    "smarts_uniform": lambda p: np.asarray(p, float),
}
POPULATION_NOTES = {
    "mcq14_like": "log10 P ~ N(1.1, 0.35), truncated to 0.1-180 d; McQuillan+2014-like, from memory",
    "loguniform": "log-uniform in P over 0.1-180 d",
    "smarts_uniform": "uniform in P over 0.1-180 d (the SMARTS corpus)",
    "none": "the dataset's own period mix (no re-weighting)",
}

ROW_DEFINITIONS = {
    "R0_success_raw": "success (correct + diffrot), unweighted over the dataset's rotators",
    "R1_success_tess_cov_P30": "success at the real TESS coverage mix (coverage_weight), P <= 30 d; assigned only",
    "R2_success_pop": "success re-weighted to the named period population (x coverage_weight if assigned)",
    "R3_correct_pop": "correct-only (no diffrot credit), same weighting as R2",
    "R4_precision": "precision of reported periods at non-rotator prevalence f, same weighting as R2",
    "R5_claytor_half": "within 10% / 20% of the injected P_eq, most credible half of all rotators, unweighted",
}


# ----------------------------------------------------------------------------- loading
def _standard_frame(df: pd.DataFrame, answer: pd.Series, best: pd.Series, score) -> pd.DataFrame:
    """Return the common per-star frame used by every row (one row per rotator)."""
    out = pd.DataFrame({
        "star_id": df["star_id"].to_numpy(),
        "true_period": df["true_period"].to_numpy(float),
        "period_lo": df["period_lo"].to_numpy(float),
        "period_hi": df["period_hi"].to_numpy(float),
        "answer_period": np.asarray(answer, float),
        "best_period": np.asarray(best, float),
        "score": np.asarray(score, float) if score is not None else np.full(len(df), np.nan),
    })
    if "coverage_class" in df:
        out["coverage_class"] = df["coverage_class"].to_numpy()
    elif "coverage" in df:
        out["coverage_class"] = df["coverage"].to_numpy()
    out["period_bin"] = [period_bin(p) for p in out.true_period]
    return out


def load_hand(path) -> pd.DataFrame:
    """Load a scorecard per_star.csv as a standard frame (answer = final_period; NaN = declined)."""
    df = pd.read_csv(path)
    return _standard_frame(df, df["final_period"], df["final_period"], None)


def load_ranker(path, threshold: float = RANKER_V0_THRESHOLD) -> pd.DataFrame:
    """Load a ranker top-1 CSV (rotator rows only) as a standard frame.

    The ranker answers ``top_period`` when ``top_score >= threshold`` and declines otherwise;
    ``best_period`` keeps the top-1 period regardless, for the R5 most-credible-half row.
    """
    df = pd.read_csv(path)
    if "is_rotator" in df:
        df = df[df["is_rotator"].astype(bool)].reset_index(drop=True)
    answer = df["top_period"].where(df["top_score"] >= threshold)
    return _standard_frame(df, answer, df["top_period"], df["top_score"])


def load_negatives(path, threshold: float = RANKER_V0_THRESHOLD) -> pd.DataFrame:
    """Load non-rotator rows of a ranker top-1 CSV: who answered (hand / ranker) and the coverage class."""
    df = pd.read_csv(path)
    df = df[~df["is_rotator"].astype(bool)].reset_index(drop=True)
    return pd.DataFrame({
        "coverage_class": df["coverage"].to_numpy(),
        "hand_answered": df["final_period"].notna().to_numpy(),
        "ranker_answered": (df["top_score"] >= threshold).to_numpy(),
    })


# ----------------------------------------------------------------------------- scoring
def add_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add boolean outcome columns, scoring every period with evaluation.classify_period.

    success / correct use REL_TOL and the active-band range (as the scorecard does); within10 /
    within20 compare ``best_period`` with the injected P_eq only (Claytor-style, no diffrot credit).
    """
    f = frame.copy()
    outc = [classify_period(m, p, REL_TOL, (lo, hi))
            for m, p, lo, hi in zip(f.answer_period, f.true_period, f.period_lo, f.period_hi)]
    f["outcome"] = outc
    f["answered"] = f.answer_period.notna()
    f["success"] = f.outcome.isin([CORRECT, DIFFROT])
    f["correct"] = f.outcome == CORRECT
    f["within10"] = [classify_period(m, p, 0.10) == CORRECT for m, p in zip(f.best_period, f.true_period)]
    f["within20"] = [classify_period(m, p, 0.20) == CORRECT for m, p in zip(f.best_period, f.true_period)]
    return f


def population_bin_shares(population: str) -> dict:
    """Return {period-bin label: share} for a named population over [P_MIN, P_MAX]."""
    if population not in POPULATIONS:
        raise ValueError(f"unknown population {population!r}; choose from {sorted(POPULATIONS) + ['none']}")
    cdf = POPULATIONS[population]
    edges = np.clip(np.asarray(PERIOD_BIN_EDGES, float), P_MIN, P_MAX)
    mass = np.diff(cdf(edges))
    mass = mass / mass.sum()
    return dict(zip(PERIOD_BIN_LABELS, mass.tolist()))


def coverage_weights(frame: pd.DataFrame, manifest: pd.DataFrame) -> np.ndarray:
    """Per-star coverage_weight joined from the manifest on star_id (1.0 where missing)."""
    m = manifest.drop_duplicates("star_id").set_index("star_id")["coverage_weight"]
    return frame["star_id"].map(m).fillna(1.0).to_numpy(float)


def real_coverage_shares(manifest: pd.DataFrame) -> pd.Series:
    """Real TESS star-season share of each coverage class implied by the manifest's coverage_weight.

    coverage_weight = real share / set share, so real share(class) = sum of coverage_weight / N.
    """
    s = manifest.groupby("coverage_class")["coverage_weight"].sum()
    return s / s.sum()


def population_weights(frame: pd.DataFrame, population: str, base=None) -> np.ndarray:
    """Per-star weights whose period-bin marginal matches ``population`` (``base`` reweighted per bin).

    ``base`` (e.g. coverage weights) sets the relative weight of stars inside a bin; each bin is then
    scaled so its total weight equals the population's bin share. Bins absent from the frame drop out
    and the remaining shares are renormalized. ``population='none'`` returns ``base`` unchanged.
    """
    w = np.ones(len(frame)) if base is None else np.asarray(base, float).copy()
    if population == "none":
        return w / w.sum()
    shares = population_bin_shares(population)
    bins = frame["period_bin"].to_numpy()
    present = [b for b in PERIOD_BIN_LABELS if np.any(bins == b)]
    tot = sum(shares[b] for b in present)
    out = np.zeros(len(frame))
    for b in present:
        sel = bins == b
        out[sel] = w[sel] / w[sel].sum() * shares[b] / tot
    return out


def precision_at(prevalence: float, success: float, answered: float, fp_rate: float) -> float:
    """Precision of reported periods when a fraction ``prevalence`` of observed stars are rotators."""
    den = prevalence * answered + (1 - prevalence) * fp_rate
    return float(prevalence * success / den) if den > 0 else float("nan")


def credible_half(frame: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Boolean mask of the most credible half (ceil(N/2)) of rotators.

    With a score: the highest-scored half. Without (hand arbitration): answered stars first, then
    declined stars, which count as misses because their ``best_period`` is NaN. If more than half
    answered, a seeded random subset of the answered stars is taken.
    """
    n = len(frame)
    k = int(math.ceil(n / 2))
    rng = np.random.default_rng(seed)
    if frame["score"].notna().any():
        order = np.argsort(-frame["score"].fillna(-np.inf).to_numpy(), kind="stable")
    else:
        ans = np.flatnonzero(frame["answered"].to_numpy())
        dec = np.flatnonzero(~frame["answered"].to_numpy())
        order = np.r_[rng.permutation(ans), dec]
    mask = np.zeros(n, bool)
    mask[order[:k]] = True
    return mask


# ----------------------------------------------------------------------------- the card
def _point(f: pd.DataFrame, w_cov, w_pop, half, fp_rate, prevalences, mode) -> dict:
    """All card numbers for one (possibly resampled) frame."""
    out = {"R0_success_raw": float(f.success.mean())}
    if mode == "assigned":
        short = f.true_period.to_numpy() <= P_SHORT_MAX
        ws = w_cov[short]
        out["R1_success_tess_cov_P30"] = float(np.sum(ws * f.success.to_numpy()[short]) / ws.sum())
    else:
        out["R1_success_tess_cov_P30"] = None
    S = float(np.sum(w_pop * f.success) / w_pop.sum())
    A = float(np.sum(w_pop * f.answered) / w_pop.sum())
    out["R2_success_pop"] = S
    out["R3_correct_pop"] = float(np.sum(w_pop * f.correct) / w_pop.sum())
    out["answer_rate_pop"] = A
    out["R4_precision"] = ({f"{p:g}": precision_at(p, S, A, fp_rate) for p in prevalences}
                           if fp_rate is not None else None)
    out["R5_within10_half"] = float(f.within10[half].mean())
    out["R5_within20_half"] = float(f.within20[half].mean())
    return out


def compute_card(frame: pd.DataFrame, mode: str, manifest: pd.DataFrame | None = None,
                 population: str = DEFAULT_POPULATION, fp_rate: float | None = None,
                 prevalences=DEFAULT_PREVALENCES, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Compute the headline card for one system (one standard frame with outcomes added).

    Parameters
    ----------
    frame : standard frame from load_hand / load_ranker, passed through add_outcomes.
    mode : "assigned" or "fullyear". Coverage weighting (R1, and inside R2-R4) applies to assigned only.
    manifest : dataset manifest with star_id and coverage_weight (required for assigned).
    population : key of POPULATIONS, or "none".
    fp_rate : answer rate on non-rotators (R4); None leaves R4 empty.
    n_boot : bootstrap resamples for standard errors (0 disables).

    Returns
    -------
    dict with "values", "se" (same keys), "n", and the settings used.
    """
    if mode not in ("assigned", "fullyear"):
        raise ValueError("mode must be 'assigned' or 'fullyear'")
    if "success" not in frame:
        frame = add_outcomes(frame)
    frame = frame.reset_index(drop=True)
    if mode == "assigned":
        if manifest is None:
            raise ValueError("assigned mode needs a manifest with coverage_weight")
        w_cov = coverage_weights(frame, manifest)
    else:
        w_cov = np.ones(len(frame))
    w_pop = population_weights(frame, population, base=w_cov)
    half = credible_half(frame, seed)
    values = _point(frame, w_cov, w_pop, half, fp_rate, prevalences, mode)

    se = {}
    if n_boot:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(frame), len(frame))
            fb = frame.iloc[idx].reset_index(drop=True)
            wcb = w_cov[idx]
            draws.append(_point(fb, wcb, population_weights(fb, population, base=wcb),
                                credible_half(fb, seed), fp_rate, prevalences, mode))
        for k, v in values.items():
            if isinstance(v, dict):
                se[k] = {p: float(np.std([d[k][p] for d in draws], ddof=1)) for p in v}
            elif v is not None:
                se[k] = float(np.std([d[k] for d in draws], ddof=1))
            else:
                se[k] = None
    return {"values": values, "se": se, "n": int(len(frame)), "n_answered": int(frame.answered.sum()),
            "population": population, "mode": mode, "fp_rate": fp_rate,
            "prevalences": list(prevalences)}


def negatives_fp_rate(neg: pd.DataFrame, column: str, mode: str, manifest: pd.DataFrame | None) -> float:
    """Answer rate on non-rotators; coverage classes weighted to the real TESS mix in assigned mode."""
    if mode == "assigned" and manifest is not None:
        real = real_coverage_shares(manifest)
        per = neg.groupby("coverage_class")[column].mean()
        common = per.index.intersection(real.index)
        return float((per[common] * real[common]).sum() / real[common].sum())
    return float(neg[column].mean())


def card_markdown(cards: dict, title: str = "") -> str:
    """Render {system label: card} as a markdown table (value ± bootstrap SE)."""
    labels = list(cards)
    first = cards[labels[0]]

    def cell(c, key, sub=None):
        v = c["values"].get(key)
        s = c["se"].get(key) if c["se"] else None
        if sub is not None:
            v = None if v is None else v[sub]
            s = None if s is None else s[sub]
        if v is None:
            return "n/a"
        return f"{v:.3f} ± {s:.3f}" if s is not None else f"{v:.3f}"

    pop = first["population"]
    rows = [("R0 success, raw (unweighted)", "R0_success_raw", None),
            ("R1 success, TESS coverage mix, P ≤ 30 d", "R1_success_tess_cov_P30", None),
            (f"R2 success, population `{pop}`", "R2_success_pop", None),
            (f"R3 correct-only, population `{pop}`", "R3_correct_pop", None)]
    for p in first["prevalences"]:
        rows.append((f"R4 precision at rotator prevalence {p:g}", "R4_precision", f"{p:g}"))
    rows += [("R5 within 10%, most credible half", "R5_within10_half", None),
             ("R5 within 20%, most credible half", "R5_within20_half", None)]
    lines = []
    if title:
        lines.append(f"**{title}**\n")
    lines.append("| Row | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for name, key, sub in rows:
        lines.append(f"| {name} | " + " | ".join(cell(cards[l], key, sub) for l in labels) + " |")
    lines.append("| n rotators / answered | " + " | ".join(
        f"{cards[l]['n']} / {cards[l]['n_answered']}" for l in labels) + " |")
    lines.append("| non-rotator FP rate (R4 input) | " + " | ".join(
        "n/a" if cards[l]["fp_rate"] is None else f"{cards[l]['fp_rate']:.4f}" for l in labels) + " |")
    return "\n".join(lines) + "\n"


def _parse_pairs(items) -> list:
    """Parse repeated LABEL=PATH options."""
    out = []
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected LABEL=PATH, got {it!r}")
        label, path = it.split("=", 1)
        out.append((label.strip(), path.strip()))
    return out


def main(argv=None):
    """CLI: compute the card for each --hand / --ranker system and write headline.json + headline.md."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["assigned", "fullyear"])
    ap.add_argument("--manifest", default=str(ROOT / "datasets" / "dev_medium.csv"),
                    help="manifest with star_id + coverage_weight (used in assigned mode)")
    ap.add_argument("--hand", action="append", help="LABEL=per_star.csv (repeatable)")
    ap.add_argument("--ranker", action="append", help="LABEL=top1.csv (repeatable)")
    ap.add_argument("--ranker-threshold", type=float, default=RANKER_V0_THRESHOLD)
    ap.add_argument("--negatives", help="top1.csv with non-rotator rows (is_rotator False) for R4 FP rates")
    ap.add_argument("--population", default=DEFAULT_POPULATION, choices=sorted(POPULATIONS) + ["none"])
    ap.add_argument("--prevalences", default=",".join(f"{p:g}" for p in DEFAULT_PREVALENCES))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)

    manifest = pd.read_csv(args.manifest) if args.manifest and os.path.exists(args.manifest) else None
    prevalences = tuple(float(x) for x in args.prevalences.split(","))
    neg = load_negatives(args.negatives, args.ranker_threshold) if args.negatives else None
    systems = [(lab, "hand", load_hand(p), p) for lab, p in _parse_pairs(args.hand)]
    systems += [(lab, "ranker", load_ranker(p, args.ranker_threshold), p) for lab, p in _parse_pairs(args.ranker)]
    if not systems:
        raise SystemExit("give at least one --hand or --ranker")
    cards = {}
    for label, kind, frame, path in systems:
        fp = None if neg is None else negatives_fp_rate(
            neg, "hand_answered" if kind == "hand" else "ranker_answered", args.mode, manifest)
        card = compute_card(add_outcomes(frame), args.mode, manifest, args.population, fp,
                            prevalences, args.n_boot)
        card.update(kind=kind, source=str(path), fp_source=args.negatives)
        cards[label] = card
    os.makedirs(args.out, exist_ok=True)
    meta = {"definitions": ROW_DEFINITIONS, "population_note": POPULATION_NOTES[args.population],
            "rel_tol": REL_TOL, "ranker_threshold": args.ranker_threshold, "mode": args.mode}
    tmp = os.path.join(args.out, "headline.json.tmp")
    with open(tmp, "w") as fh:
        json.dump({"meta": meta, "cards": cards}, fh, indent=1)
    os.replace(tmp, os.path.join(args.out, "headline.json"))
    md = card_markdown(cards, f"Headline card, {args.mode}, population `{args.population}`")
    with open(os.path.join(args.out, "headline.md"), "w") as fh:
        fh.write(md)
    print(md)
    return cards


if __name__ == "__main__":
    main()
