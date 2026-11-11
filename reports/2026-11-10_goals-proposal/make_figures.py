"""Build the proposed-goal table (data/goal_table.csv) and margin figure from goals_proposed.json + v5 baselines."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "benchmarks"))
from check_goals import expand, get_path, OPS  # noqa: E402
BASES = {"datasets/dev_medium.csv,data_root=data/smarts_dev,sectors=ignored": "dev_medium_fullyear_v5",
         "datasets/dev_medium.csv,data_root=data/smarts_dev,sectors=manifest": "dev_medium_assigned_v5"}
SHORT = {"sectors=ignored": "dev_medium full-year", "sectors=manifest": "dev_medium assigned"}

def build() -> pd.DataFrame:
    """Return one row per (goal, expanded metric path) with current value and pass/fail."""
    goals = json.loads((ROOT / "benchmarks/goals_proposed.json").read_text())["goals"]
    rows = []
    for g in goals:
        m = json.loads((ROOT / f"benchmarks/baselines/{BASES[g['dataset']]}.json").read_text())
        for p in expand(m, g["metric"]):
            v = get_path(m, p)
            rows.append(dict(id=g["id"], dataset=g["dataset"].split(",")[-1], metric=p, op=g["op"], target=g["target"],
                             value=v, passes=bool(OPS[g["op"]](v, g["target"]))))
    return pd.DataFrame(rows)

def plot(df: pd.DataFrame, out: Path) -> None:
    """Dot-and-line chart: value vs threshold per goal, grouped by dataset, pass green / fail red."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw=dict(height_ratios=[df[df.dataset == d].shape[0] for d in ["sectors=ignored", "sectors=manifest"]]))
    for ax, d in zip(axes, ["sectors=ignored", "sectors=manifest"]):
        s = df[df.dataset == d].reset_index(drop=True)
        for i, r in s.iterrows():
            c = "#2a9d4a" if r.passes else "#d6342c"
            ax.plot([r.value, r.target], [i, i], color="0.6", lw=1.5, zorder=1)
            ax.scatter(r.value, i, color=c, s=60, zorder=3)
            ax.scatter(r.target, i, marker="|", color="k", s=200, zorder=2)
            ax.text(max(r.value, r.target) + 0.01, i, f"{r.value:.3f} {r.op} {r.target:g}", va="center", fontsize=8)
        ax.set_yticks(range(len(s))); ax.set_yticklabels([f"{r.id}  {r.metric.replace('final.', '')}" for _, r in s.iterrows()], fontsize=8)
        ax.invert_yaxis(); ax.set_xlim(0, 1.15); ax.set_title(SHORT[d] + "  (dot = current v5 baseline, bar = threshold)", fontsize=10)
    axes[1].set_xlabel("metric value (rate)")
    fig.tight_layout(); fig.savefig(out, dpi=130)

if __name__ == "__main__":
    here = Path(__file__).parent
    df = build(); df.to_csv(here / "data/goal_table.csv", index=False); plot(df, here / "figures/goal_margins.png")
    print(df.to_string())
