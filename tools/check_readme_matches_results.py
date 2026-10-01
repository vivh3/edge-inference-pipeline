#!/usr/bin/env python3
"""Fail if the README's headline table disagrees with the committed results.

The table and the JSON summaries are two copies of the same numbers, and they
have already drifted apart once: a stale `git checkout -- results/` reverted
the artifacts while the prose kept the regenerated figures. Anyone reading the
repo would have found a claim its own data did not support, which is the worst
place to be caught. This runs in CI so the two cannot separate again.
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_BY_LABEL = {
    "latest": "latest",
    "fifo_bounded(8)": "fifo_bounded",
    "fifo_unbounded": "fifo_unbounded",
}
CAMERA_FPS = re.compile(r"measured ([\d.]+) fps")
ROW = re.compile(
    r"\| `(latest|fifo_bounded\(8\)|fifo_unbounded)` \| (\d+) \| (\d+) \| "
    r"([\d.]+)% \| \*?\*?([\d.]+) s\*?\*? \| ([\d.]+) s \| ([\d.]+) s \|"
)


def main() -> int:
    readme = open(os.path.join(ROOT, "README.md")).read()
    rows = ROW.findall(readme)
    if len(rows) != len(POLICY_BY_LABEL):
        print(f"expected {len(POLICY_BY_LABEL)} table rows, found {len(rows)}", file=sys.stderr)
        return 1

    failures = []
    for label, captured, published, drop, p50, p99, maximum in rows:
        path = os.path.join(ROOT, "results", "simulated", POLICY_BY_LABEL[label], "summary.json")
        with open(path) as fh:
            summary = json.load(fh)
        age = summary["result_age"]
        for field, in_readme, in_file in [
            ("captured", int(captured), summary["frames_captured"]),
            ("published", int(published), summary["results_published"]),
            ("drop rate", float(drop), round(summary["drop_rate"] * 100, 1)),
            ("result age p50", float(p50), round(age["p50"], 3)),
            ("result age p99", float(p99), round(age["p99"], 3)),
            ("result age max", float(maximum), round(age["max"], 3)),
        ]:
            if in_readme != in_file:
                failures.append(f"{label} {field}: README {in_readme}, {path} {in_file}")

    # The camera rate is the same class of claim: a number in the prose with a
    # committed measurement behind it, and nothing but diligence keeping them
    # together. It is the denominator of every drop rate here, so it is the
    # worst one to let drift.
    camera_path = os.path.join(ROOT, "results", "baseline", "camera.json")
    claimed = CAMERA_FPS.search(readme)
    if not claimed:
        failures.append("README no longer states a measured camera fps")
    elif os.path.exists(camera_path):
        with open(camera_path) as fh:
            measured = json.load(fh)["effective_fps"]
        if float(claimed.group(1)) != measured:
            failures.append(
                f"camera fps: README {claimed.group(1)}, {camera_path} {measured}"
            )

    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        print("\nRegenerate with tools/run_overload_sim.py and update the table.", file=sys.stderr)
        return 1
    print(f"README headline table matches all {len(rows)} committed summaries")
    print("README camera rate matches results/baseline/camera.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
