#!/usr/bin/env python3
"""Plot wall time, CPU time and peak memory from benches/reads.py results.

    uv run --no-project --with matplotlib python benches/plot_reads.py
    uv run --no-project --with matplotlib python benches/plot_reads.py \
        --results target/compare/reads-results-na12878.json --output docs/benchmark-human.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# kbolt carries the colour; the tools it is compared with stay neutral.
TOOLS = [
    ("kbolt", "kbolt", "#1b9e77"),
    ("kmc", "KMC 3.2.4", "#9a9a9a"),
    ("fastk", "FastK 1.2", "#9a9a9a"),
]


# What each dataset of benches/reads.py is called in the title.
DATASETS = {"celegans": "C. elegans reads", "na12878": "human reads (NA12878)"}


def median(values):
    return sorted(values)[len(values) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "target/compare/reads-results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/benchmark.png")
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    # Labels are tool/tTHREADS, then /mGIB for a memory limit.
    threads = max(int(label.split("/")[1][1:]) for label in report["count"])
    memory = next((label.split("/")[2][1:] for label in report["count"]
                   if label.count("/") == 2), None)
    suffix = f"/t{threads}" + (f"/m{memory}" if memory else "")
    cases = {tool: report["count"][f"{tool}{suffix}"] for tool, _, _ in TOOLS}
    rounds = len(cases["kbolt"]["runs"])
    panels = [
        ("Wall time", "s", "{:.1f} s",
         {tool: case["summary"]["median_seconds"] for tool, case in cases.items()}),
        ("CPU time", "s", "{:.0f} s",
         {tool: median([run["user_seconds"] + run["system_seconds"] for run in case["runs"]])
          for tool, case in cases.items()}),
        ("Peak memory", "GiB", "{:.1f} GiB",
         {tool: case["summary"]["max_peak_rss_bytes"] / 2**30 for tool, case in cases.items()}),
    ]

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 3.5), sharey=True)
    rows = range(len(TOOLS))
    for ax, (title, unit, form, values) in zip(axes, panels):
        longest = max(values.values())
        for row, (tool, _, color) in zip(rows, TOOLS):
            ax.barh(row, values[tool], height=0.58, color=color)
            ax.text(values[tool] + longest * 0.02, row, form.format(values[tool]), va="center",
                    fontsize=12, fontweight="bold" if tool == "kbolt" else "normal", color="#222222")
        ax.set_title(title, loc="left")
        ax.set_xlabel(unit, color="#555555")
        ax.set_xlim(0, longest * 1.24)
        ax.set_yticks(list(rows), [label for _, label, _ in TOOLS])
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors="#555555")
        ax.grid(True, axis="x", color="#e3e3e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    name = DATASETS.get(report["dataset"].split("-")[0], report["dataset"])
    limit = f" within {memory} GiB of memory" if memory else ""
    fig.suptitle(f"Counting 31-mers in {report['bases'] / 1e9:.2f} Gbp of {name}{limit} "
                 "(Lower is Better)", fontsize=16, y=0.98)
    note = (f" Limits: kbolt --memory {memory}G, KMC -m{memory}, FastK -M{memory}."
            if memory else "")
    fig.text(0.5, 0.01, f"Median of {rounds} runs at {threads} threads on an Apple M5 "
             f"(4 performance + 6 efficiency cores, 24 GB of memory).{note}",
             ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor="white")
    print(args.output)


if __name__ == "__main__":
    main()
