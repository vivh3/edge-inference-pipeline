#!/usr/bin/env python3
"""Plot the headline figure: result age over time, one line per admission policy.

matplotlib is an optional dependency (`pip install -e '.[plot]'`).  The CSVs
in results/ are the primary artifact; this only draws them, so a missing
plotting library never blocks a run.

    python3 tools/plot_results.py results/simulated --out results/simulated/result_age.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


def load(path: str):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return [], []
    t0 = float(rows[0]["capture_ts"])
    return (
        [float(r["capture_ts"]) - t0 for r in rows],
        [float(r["result_age"]) for r in rows],
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="directory containing <policy>/results.csv")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; install with: pip install -e '.[plot]'", file=sys.stderr)
        print("the CSVs in", args.root, "hold the data regardless.", file=sys.stderr)
        return 1

    series = []
    for name in sorted(os.listdir(args.root)):
        csv_path = os.path.join(args.root, name, "results.csv")
        if os.path.isfile(csv_path):
            x, y = load(csv_path)
            if x:
                series.append((name, x, y))
    if not series:
        print(f"no results.csv found under {args.root}", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, x, y in series:
        ax.plot(x, y, marker=".", linewidth=1.2, label=name)
    ax.set_xlabel("time since first capture (s)")
    ax.set_ylabel("result age: capture -> publish (s)")
    ax.set_title("Result age under overload, by admission policy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = args.out or os.path.join(args.root, "result_age.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
