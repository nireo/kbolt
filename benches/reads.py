#!/usr/bin/env python3
"""Compare kbolt with KMC and FastK on real sequencing reads.

The input is C. elegans run SRR065390 from the European Nucleotide Archive:
two gzip-compressed FASTQ files, 67.6 million reads of 100 bases (6.76 Gbp),
with 487,817,339 distinct canonical 31-mers, 77% of them seen once. Unlike
the inputs of compare.py, its count table is far larger than the CPU caches.

    python3 benches/reads.py --fetch    # download 5 GB to data/celegans, once
    python3 benches/reads.py

--dataset na12878 counts human reads instead: NA12878 run SRR8848156, two
gzip-compressed FASTQ files of 95.9 million reads of 150 bases each (28.76
Gbp, 16 GB compressed), with 8,185,470,424 distinct canonical 31-mers. They
do not fit in the memory of a laptop, so give every tool a limit:

    python3 benches/reads.py --dataset na12878 --fetch
    python3 benches/reads.py --dataset na12878 --memory 4 --tools kbolt kmc fastk --warmup 0

Every tool counts canonical 31-mers, keeps k-mers seen once, and writes its
native table to target/compare/out. kbolt's and KMC's tables hold identical
counts (compared as complete count histograms). KMC and FastK flags are as in
compare.py; kbolt's -t excludes its two input readers, so compare CPU time too.

--reads N counts only the first N reads of the first file, written once to
target/compare/inputs as plain FASTQ, for quicker iteration.

--memory G [G ...] gives every tool the same memory limit in GiB (kbolt
--memory, KMC -m, FastK -M), which kbolt then counts on disk; 0 means each
tool's default, where kbolt counts in memory. Compare peak RSS: the limits
are targets, and each tool interprets its own differently. --idle P waits for
the CPU to be P% idle before every timed run, for machines with other jobs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

from compare import INPUTS, OUT, TMP, WORK, digest, run_cases
from dataset import ROOT, nonnegative, positive, summarize, wait_idle

# Per dataset: directory, ENA directory, and file name: (bytes, MD5) as
# published by ENA, read length, reads in all files.
DATASETS = {
    "celegans": {
        "data": ROOT / "data/celegans",
        "ena": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR065/SRR065390/",
        "files": {
            "SRR065390_1.fastq.gz": (2476071286, "08a64d5483edcc42ef652933ba8c6b47"),
            "SRR065390_2.fastq.gz": (2509181096, "6a5cf958b5ab4d30910bb5d0494916d4"),
        },
        "read_bases": 100,
        "reads": 67617092,
    },
    "na12878": {
        "data": ROOT / "data/na12878",
        "ena": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR884/006/SRR8848156/",
        "files": {
            "SRR8848156_1.fastq.gz": (7921603800, "b21ab6d5330c2b054a19437ad6996c12"),
            "SRR8848156_2.fastq.gz": (9149021851, "7e3294454da6b27c43bcf6359e7b989d"),
        },
        "read_bases": 150,
        "reads": 191729882,
    },
}
DATA = ENA = FILES = READ_BASES = READS = None
TOOLS = ["kbolt", "kmc", "fastk"]
PART = 32 << 20


def fetch_file(name, size, md5):
    """Download in parallel ranges (ENA is slow per connection) and check MD5."""
    target = DATA / name
    if target.exists() and target.stat().st_size == size:
        print(f"{name}: already present", flush=True)
        return
    partial = target.with_name(name + ".part")
    descriptor = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(descriptor, size)

    def part(start):
        end = min(start + PART, size) - 1
        for attempt in range(8):
            try:
                request = urllib.request.Request(ENA + name, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise OSError(f"HTTP {response.status}")
                    offset = start
                    while block := response.read(1 << 20):
                        os.pwrite(descriptor, block, offset)
                        offset += len(block)
                if offset == end + 1:
                    return
            except OSError:
                time.sleep(2 + attempt)
        raise OSError(f"{name}: bytes {start}-{end} failed")

    with ThreadPoolExecutor(12) as pool:
        list(pool.map(part, range(0, size, PART)))
    os.close(descriptor)
    checksum = hashlib.md5()
    with partial.open("rb") as source:
        for block in iter(lambda: source.read(1 << 22), b""):
            checksum.update(block)
    if checksum.hexdigest() != md5:
        sys.exit(f"{name}: MD5 mismatch ({checksum.hexdigest()}); delete {partial} and retry")
    partial.rename(target)
    print(f"{name}: downloaded, MD5 verified", flush=True)


def subset(reads):
    """The first `reads` reads of the first file as plain FASTQ."""
    path = INPUTS / f"{DATA.name}-{reads}.fq"
    if not path.exists():
        with gzip.open(DATA / next(iter(FILES)), "rb") as source, path.open("wb") as out:
            for _ in range(4 * reads):
                out.write(source.readline())
    return path


def count_command(tool, inputs, name, threads, memory, kbolt, kmc):
    """`memory` is a limit in GiB, or 0 for the tool's default."""
    files = [str(path) for path in inputs]
    if tool == "kbolt":
        limit = ["--memory", f"{memory}G", "--temp-dir", str(TMP)] if memory else []
        return [kbolt, "count", "-k", "31", "-t", str(threads), "--format", "binary", "--timings",
                *limit, *files, "-o", str(OUT / f"{name}.kbolt")]
    if tool == "kmc":
        listing = WORK / f"{name}.lst"
        listing.write_text("".join(f"{file}\n" for file in files))
        limit = [f"-m{memory}"] if memory else []
        return [kmc, "-k31", "-ci1", "-cs4294967295", "-fq", f"-t{threads}", "-hp", *limit,
                f"@{listing}", str(OUT / f"{name}_kmc"), str(TMP)]
    if tool == "fastk":
        limit = [f"-M{memory}"] if memory else []
        return ["FastK", "-k31", "-t1", f"-T{threads}", *limit, f"-P{TMP}",
                f"-N{OUT / (name + '_fk')}", *files]
    raise ValueError(tool)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="celegans", choices=list(DATASETS))
    parser.add_argument("--fetch", action="store_true", help="download the reads and exit")
    parser.add_argument("--reads", type=positive, help="count only the first N reads of the first file")
    parser.add_argument("--kbolt", type=Path, default=ROOT / "target/release/kbolt")
    parser.add_argument("--kmc", type=Path, default=WORK / "tools/kmc/bin/kmc")
    parser.add_argument("--tools", nargs="+", default=["kbolt", "kmc"], choices=TOOLS)
    parser.add_argument("--threads", nargs="+", type=positive, default=[8])
    parser.add_argument("--memory", nargs="+", type=nonnegative, default=[0], metavar="G",
                        help="memory limits in GiB to give every tool; 0 is its default")
    parser.add_argument("--idle", type=positive, metavar="P",
                        help="wait until the CPU is P%% idle before each timed run")
    parser.add_argument("--runs", type=positive, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=WORK / "reads-results.json")
    args = parser.parse_args()
    global DATA, ENA, FILES, READ_BASES, READS
    dataset = DATASETS[args.dataset]
    DATA, ENA, FILES = dataset["data"], dataset["ena"], dataset["files"]
    READ_BASES, READS = dataset["read_bases"], dataset["reads"]
    if args.fetch:
        DATA.mkdir(parents=True, exist_ok=True)
        for name, (size, md5) in FILES.items():
            fetch_file(name, size, md5)
        return
    missing = [name for name in FILES if not (DATA / name).exists()]
    if missing:
        sys.exit(f"missing {', '.join(missing)} under {DATA}; run with --fetch first")
    for directory in (INPUTS, OUT, TMP):
        directory.mkdir(parents=True, exist_ok=True)
    if args.reads:
        name, inputs, bases = f"{args.dataset}-{args.reads}", [subset(args.reads)], args.reads * READ_BASES
    else:
        name, inputs, bases = args.dataset, [DATA / file for file in FILES], READS * READ_BASES
    kbolt, kmc = str(args.kbolt.resolve()), str(args.kmc.resolve())
    cases = [(f"{tool}/t{threads}" + (f"/m{memory}" if memory else ""),
              count_command(tool, inputs, name, threads, memory, kbolt, kmc))
             for threads in args.threads for memory in args.memory for tool in args.tools]
    print(f"== {name} ({bases:,} bases)", flush=True)
    results = run_cases(cases, args.runs, args.warmup,
                        before=(lambda: wait_idle(args.idle)) if args.idle else None)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "kbolt": {"path": kbolt, "sha256": digest(kbolt),
                  "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
        "method": "warm cache; fresh processes; rotated rounds; native tables written to "
                  "target/compare/out; KMC -ci1 -cs4294967295 with disk bins; memory limits as "
                  "in each command",
        "dataset": name,
        "bases": bases,
        "count": {label: {"command": command, "runs": results[label],
                          "summary": summarize(results[label], bases)}
                  for label, command in cases},
    }
    args.output.write_text(json.dumps(report, indent=2))
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"Saved {args.output}\n\nmedian wall seconds, median CPU seconds, peak RSS GiB")
    for label, case in report["count"].items():
        cpu = sorted(run["user_seconds"] + run["system_seconds"] for run in case["runs"])
        print(f"  {label}: {case['summary']['median_seconds']:.2f} s wall, "
              f"{cpu[len(cpu) // 2]:.1f} s CPU, "
              f"{case['summary']['max_peak_rss_bytes'] / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
