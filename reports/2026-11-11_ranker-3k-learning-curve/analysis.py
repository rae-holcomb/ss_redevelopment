"""Ranker v0 learning curve with the 3,000-star point (coordinator, 2026-11-11).

Reads runs/t11_ranker/results/n{250,500,1000,3000}_base_ndcg.json (tools/train_ranker.py output) and
writes a summary table plus figures. The 250/500/1000 points were trained on the train_1k table and
3000 on train_3k (a superset); tuned hyperparameters are shared.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SIZES = [250, 500, 1000, 3000]
BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]


def load(n: int) -> dict:
    """Load one learning-curve result JSON."""
    return json.load(open(ROOT / f"runs/t11_ranker/results/n{n}_base_ndcg.json"))


def rows() -> pd.DataFrame:
    """Flatten OOF and dev results into one long table (size, split, rule, subset, bin, metrics)."""
    out = []
    for n in SIZES:
        d = load(n)
        blocks = {"oof": d["oof"], **{f"dev_{k}": v for k, v in d["dev"].items()}}
        for split, blk in blocks.items():
            for key, lst in blk.items():
                rule, subset = key.split("|")
                for r in lst:
                    out.append(dict(size=n, split=split, rule=rule, subset=subset, threshold=d["thresholds"].get(rule), **r))
    return pd.DataFrame(out)


def main() -> None:
    """Write the learning-curve table and figures."""
    df = rows()
    df.to_csv(OUT / "data/learning_curve.csv", index=False)
    a = df[(df.subset == "all") & (df.bin == "all")]
    show = a[a.rule.isin(["no_decline", "match_hand_precision"])].pivot_table(
        index=["split", "rule"], columns="size", values="success")
    print(show.round(3).to_string())
    print(a[a.rule == "match_hand_precision"].pivot_table(index="split", columns="size", values=["wrong", "declined"]).round(3).to_string())
    pb = df[(df.subset == "all") & (df.rule == "match_hand_precision") & df.split.str.startswith("dev") & df["size"].isin([1000, 3000])]
    print(pb.pivot_table(index=["split", "bin"], columns="size", values="success").reindex(BINS + ["all"], level=1).round(3).to_string())
    print({n: load(n)["thresholds"]["match_hand_precision"] for n in SIZES})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, split in zip(axes, ["oof", "dev_fullyear", "dev_assigned"]):
        for rule, ls in [("no_decline", "-"), ("match_hand_precision", "--")]:
            s = a[(a.split == split) & (a.rule == rule)].sort_values("size")
            ax.plot(s["size"], s["success"], ls, marker="o", label=f"success, {rule}")
            if rule == "match_hand_precision":
                ax.plot(s["size"], s["wrong"] + s["alias"], ":", marker="x", color="crimson", label="wrong+alias, matched precision")
        ax.set_xscale("log"); ax.set_xticks(SIZES, [str(n) for n in SIZES])
        ax.set_title({"oof": "training OOF", "dev_fullyear": "dev_medium full-year", "dev_assigned": "dev_medium assigned"}[split])
        ax.set_xlabel("training stars"); ax.set_ylim(0, 0.6); ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction of stars"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "figures/fig_learning_curve.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split in zip(axes, ["dev_fullyear", "dev_assigned"]):
        for i, n in enumerate([1000, 3000]):
            s = pb[(pb.split == split) & (pb["size"] == n)].set_index("bin").reindex(BINS)
            ax.bar([j + (i - 0.5) * 0.38 for j in range(len(BINS))], s["success"], 0.38, label=f"{n} stars")
        ax.set_xticks(range(len(BINS)), BINS); ax.set_title(f"{split}: success at matched precision"); ax.set_xlabel("true-period bin")
    axes[0].set_ylabel("success"); axes[0].legend()
    fig.tight_layout(); fig.savefig(OUT / "figures/fig_per_bin_1k_vs_3k.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
