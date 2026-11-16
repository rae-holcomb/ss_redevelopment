"""Headline table (markdown) from the official scorecard metrics in data/metrics/ and the clean _v7 baselines."""
import json
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = {"dmfy": "dev_medium_clean_fullyear_v7", "dmas": "dev_medium_clean_assigned_v7", "syn": "synth200_s0_v7"}
rows = []
for tag, base in BASE.items():
    items = [("_v7 (local_screen2)", ROOT / f"benchmarks/baselines/{base}.json")]
    items += [(p.stem.rsplit("_", 1)[0], p) for p in sorted((HERE / "data/metrics").glob(f"*_{tag}.json"))]
    for name, p in items:
        m = json.loads(p.read_text())
        o = m["final"]["overall"]
        # overall entries are fractions of n
        rows.append(dict(dataset=tag, variant=name, n=o["n"], success=o["success"], wrong=o["wrong"],
                         alias=o["alias"], none=o["none"],
                         precision=o["success"] / (1 - o["none"]) if o["none"] < 1 else float("nan"),
                         median_s=m["runtime"].get("median_seconds")))
df = pd.DataFrame(rows)
df.to_csv(HERE / "data/summary_table.csv", index=False)
cols = list(df.columns)
print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
for _, r in df.iterrows():
    print("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
