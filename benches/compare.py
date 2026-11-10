#!/usr/bin/env python3
"""Compare kbolt with Jellyfish and FastK on identical exact-counting work.

All tools count canonical 31-mers, keep k-mers seen once, and report exact
counts; their dumped tables were checked against kbolt before timing (full
sorted diffs for E. coli and the sampled reads, distinct/total counts and the
complete count histogram for Arabidopsis).

Two stages are timed separately:
  count  input -> each tool's native table, written to target/compare/out
  dump   native table -> tab-separated text on /dev/null (single-threaded)
kbolt can also write text directly while counting ("count-tsv").

Jellyfish needs its hash size up front; it gets twice the true distinct count
(rounded up to a power of two internally), which favours it. It cannot read
gzip, so it decompresses through its own `-g` generator (gzcat), whose CPU time
it accumulates. FastK requires .fa/.fq names, so inputs are symlinks under
target/compare/inputs, created on each run.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

from dataset import ROOT, fasta_metadata, measure, positive, summarize

WORK = ROOT / "target/compare"
INPUTS = WORK / "inputs"
OUT = WORK / "out"
TMP = WORK / "tmp"
SOURCES = {
    "ecoli.fa": ROOT / "data/ecoli/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna",
    "arab.fa": ROOT / "data/arabidopsis/ncbi_dataset/data/GCF_000001735.4/GCF_000001735.4_TAIR10.1_genomic.fna",
    "sampled.fq": ROOT / "target/read-profile/inputs/sampled.fq",
    "sampled.fq.gz": ROOT / "target/read-profile/inputs/sampled.fq.gz",
}
# name: (input file, Jellyfish -s = about twice the distinct 31-mers)
DATASETS = {
    "ecoli": ("ecoli.fa", "16M"),
    "arab": ("arab.fa", "256M"),
    "sampled": ("sampled.fq", "8M"),
    "sampled-gz": ("sampled.fq.gz", "8M"),
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def bases(name):
    path = INPUTS / DATASETS[name][0]
    if path.suffix == ".fa":
        return fasta_metadata(path.resolve())["bases"]
    # generate_inputs.py writes fixed 307-byte records with 150 bases.
    return SOURCES["sampled.fq"].stat().st_size // 307 * 150


def count_command(tool, name, threads, kbolt):
    file, size = DATASETS[name]
    source = str(INPUTS / file)
    if tool == "kbolt":
        return [kbolt, "count", "-k", "31", "-t", str(threads), "--format", "binary",
                source, "-o", str(OUT / f"{name}.kbolt")]
    if tool == "kbolt-tsv":
        return [kbolt, "count", "-k", "31", "-t", str(threads), source, "-o", "/dev/null"]
    if tool == "jellyfish":
        command = ["jellyfish", "count", "-m", "31", "-s", size, "-t", str(threads), "-C",
                   "-o", str(OUT / f"{name}.jf")]
        if file.endswith(".gz"):
            generator = WORK / f"{name}.generator"
            generator.write_text(f"gzcat {INPUTS / file}\n")
            return command + ["-g", str(generator)]
        return command + [source]
    if tool == "fastk":
        return ["FastK", "-k31", "-t1", f"-T{threads}", f"-P{TMP}", f"-N{OUT / (name + '_fk')}",
                source]
    raise ValueError(tool)


def dump_command(tool, name, kbolt):
    if tool == "kbolt":
        return [kbolt, "dump", str(OUT / f"{name}.kbolt")]
    if tool == "jellyfish":
        return ["jellyfish", "dump", "-c", "-t", str(OUT / f"{name}.jf")]
    if tool == "fastk":
        return ["Tabex", "-A", str(OUT / f"{name}_fk.ktab")]
    raise ValueError(tool)


def setup():
    for directory in (INPUTS, OUT, TMP):
        directory.mkdir(parents=True, exist_ok=True)
    for link, target in SOURCES.items():
        path = INPUTS / link
        if path.is_symlink() or path.exists():
            path.unlink()
        path.symlink_to(target)


def run_cases(cases, rounds, warmup):
    """Run each (label, command) case `warmup` times untimed, then `rounds`
    times in rotated order. Returns {label: [measurement, ...]}."""
    for _ in range(warmup):
        for _, command in cases:
            measure(command)
    results = {label: [] for label, _ in cases}
    for round_index in range(rounds):
        shift = round_index % len(cases)
        for label, command in cases[shift:] + cases[:shift]:
            results[label].append(measure(command))
            print(f"  {label}: {results[label][-1]['wall_seconds']:.3f}s", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kbolt", type=Path, default=ROOT / "target/release/kbolt")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--threads", nargs="+", type=positive, default=[1, 4, 8])
    parser.add_argument("--tools", nargs="+", default=["kbolt", "kbolt-tsv", "jellyfish", "fastk"])
    parser.add_argument("--runs", type=positive, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=WORK / "results.json")
    args = parser.parse_args()
    setup()
    kbolt = str(args.kbolt.resolve())
    versions = {
        "kbolt": {"path": kbolt, "sha256": digest(kbolt),
                  "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
        "jellyfish": subprocess.check_output(["jellyfish", "--version"], text=True).strip(),
        "fastk": subprocess.check_output(["brew", "list", "--versions", "fastk"], text=True).strip(),
    }
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "versions": versions,
        "method": "warm cache; fresh processes; rotated rounds; native tables written to "
                  "target/compare/out; text to /dev/null; Jellyfish -s = 2x distinct",
        "count": {},
        "dump": {},
    }
    for name in args.datasets:
        size = bases(name)
        print(f"== {name} ({size:,} bases)", flush=True)
        cases = [(f"{tool}/t{threads}", count_command(tool, name, threads, kbolt))
                 for threads in args.threads for tool in args.tools]
        results = run_cases(cases, args.runs, args.warmup)
        report["count"][name] = {
            label: {"command": command, "runs": results[label], "summary": summarize(results[label], size)}
            for label, command in cases
        }
        dumps = [(tool, dump_command(tool, name, kbolt))
                 for tool in args.tools if tool != "kbolt-tsv"]
        results = run_cases(dumps, args.runs, args.warmup)
        report["dump"][name] = {
            tool: {"command": command, "runs": results[tool], "summary": summarize(results[tool], size)}
            for tool, command in dumps
        }
        args.output.write_text(json.dumps(report, indent=2))
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"Saved {args.output}")
    for stage in ("count", "dump"):
        print(f"\n{stage}: median wall seconds (peak RSS GiB)")
        for name, cases in report[stage].items():
            row = "  ".join(f"{label}={case['summary']['median_seconds']:.3f}"
                            f"({case['summary']['max_peak_rss_bytes'] / 2**30:.2f})"
                            for label, case in cases.items())
            print(f"  {name}: {row}")


if __name__ == "__main__":
    main()
