#!/usr/bin/env python3
"""Plot counting throughput against threads from benches/compare.py results.

    uv run --no-project --with matplotlib python benches/plot_compare.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOOLS = [
    ("kbolt", "kbolt", "#1b9e77", "o"),
    ("fastk", "FastK 1.2", "#d95f02", "s"),
    ("jellyfish", "Jellyfish 2.3.1", "#7570b3", "^"),
]
PANELS = [
    ("arab", "Reference genome (Arabidopsis, 120 Mbp)"),
    ("sampled", "Sequencing reads (134 Mbp FASTQ)"),
    ("sampled-gz", "Same reads, gzip-compressed"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "target/compare/results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/benchmark.png")
    args = parser.parse_args()
    count = json.loads(args.results.read_text())["count"]

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 5.6))
    for ax, (key, title) in zip(axes, PANELS):
        top = 0
        for tool, label, color, marker in TOOLS:
            runs = sorted((int(name.split("/t")[1]), case["summary"]["median_million_bases_per_second"])
                          for name, case in count[key].items() if name.startswith(f"{tool}/t"))
            threads, throughput = zip(*runs)
            top = max(top, *throughput)
            ax.plot(threads, throughput, color=color, marker=marker, linewidth=2.2,
                    markersize=6, label=label)
        ax.set_title(title)
        ax.set_xlabel("Threads")
        ax.set_xticks(threads)
        ax.set_xlim(0.5, threads[-1] + 0.5)
        ax.set_ylim(0, top * 1.12)
        ax.grid(True, color="#e3e3e3", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Throughput (million bases / s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=len(TOOLS),
               frameon=True, framealpha=1, edgecolor="#cccccc")
    fig.suptitle("k-mer Counting Throughput, k = 31 (Higher is Better)", fontsize=16, y=0.985)
    fig.text(0.5, 0.01, "Median of 3 runs on an Apple M5 (4 performance + 6 efficiency cores). "
             "All three tools report identical counts.", ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor="white")
    print(args.output)


if __name__ == "__main__":
    main()
