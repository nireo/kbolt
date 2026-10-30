#!/usr/bin/env python3
"""Repeatable CLI benchmark for a local FASTA file (macOS and Linux)."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    ROOT / "data/ecoli/ncbi_dataset/data/GCF_000005845.2"
    / "GCF_000005845.2_ASM584v2_genomic.fna"
)


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def capture(*command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def fasta_metadata(path):
    digest = hashlib.sha256()
    records = bases = 0
    with path.open("rb") as source:
        for line in source:
            digest.update(line)
            if line.startswith(b">"):
                records += 1
            else:
                sequence = line.rstrip(b"\r\n")
                if sequence and not records:
                    raise ValueError("input must be an uncompressed FASTA file")
                bases += len(sequence)
    if not records:
        raise ValueError("input must contain at least one FASTA record")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "records": records,
        "bases": bases,
    }


def measure(command):
    # wait4 gives this child's own peak RSS, rather than a cumulative maximum
    # across previous children. stderr goes to a file to avoid pipe deadlocks.
    with tempfile.TemporaryFile() as errors:
        started = time.perf_counter()
        with subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=errors
        ) as child:
            while True:
                try:
                    _, status, usage = os.wait4(child.pid, 0)
                    break
                except InterruptedError:
                    continue
            elapsed = time.perf_counter() - started
            child.returncode = os.waitstatus_to_exitcode(status)
            if child.returncode:
                errors.seek(0)
                message = errors.read().decode(errors="replace").strip()
                raise RuntimeError(
                    f"kbolt exited with {child.returncode}: {message}"
                )
    # Darwin reports bytes; Linux reports KiB.
    rss_bytes = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    return {
        "wall_seconds": elapsed,
        "user_seconds": usage.ru_utime,
        "system_seconds": usage.ru_stime,
        "peak_rss_bytes": rss_bytes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="uncompressed FASTA (default: downloaded E. coli)")
    parser.add_argument("-k", type=positive, default=31)
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--capacity", type=nonnegative, default=0,
                        help="initial distinct-k-mer capacity hint (default: 0)")
    parser.add_argument("--format", choices=["tsv", "binary"], default="tsv",
                        help="output encoding to benchmark (default: tsv)")
    parser.add_argument("--runs", type=positive, default=10)
    parser.add_argument("--warmup", type=positive, default=2)
    args = parser.parse_args()
    if sys.platform not in ("darwin", "linux"):
        parser.error("peak memory measurement requires macOS or Linux")
    path = args.input.expanduser().resolve()
    if not path.is_file():
        parser.error(f"FASTA not found: {path}")
    metadata = fasta_metadata(path)

    print("Building release binary (outside measured runs)...", flush=True)
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "--bin", "kbolt",
         "--target-dir", str(ROOT / "target")],
        cwd=ROOT, check=True,
    )
    binary = ROOT / "target/release/kbolt"
    command = [str(binary), "count", "-k", str(args.k)]
    if args.forward:
        command.append("--forward")
    command.extend(["--capacity", str(args.capacity)])
    command.extend(["--format", args.format])
    command.append(str(path))

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": capture("git", "rev-parse", "HEAD"),
        "git_status": capture("git", "status", "--porcelain"),
        "rustc": capture("rustc", "-Vv"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "input": metadata,
        "command": command,
        "stdout": "discarded to /dev/null; output serialization and writes included",
        "cache": "warm; input scanned and warmup runs completed before timing",
        "warmup_runs": args.warmup,
    }
    print(f"Input: {path.name}; {metadata['bases']:,} bases", flush=True)
    for index in range(args.warmup):
        measure(command)
        print(f"Warmup {index + 1}/{args.warmup} complete", flush=True)

    runs = []
    for index in range(args.runs):
        result = measure(command)
        runs.append(result)
        print(
            f"Run {index + 1:2}/{args.runs}: {result['wall_seconds']:.4f} s, "
            f"{result['peak_rss_bytes'] / 1024**2:.1f} MiB peak RSS",
            flush=True,
        )
    times = [run["wall_seconds"] for run in runs]
    median = statistics.median(times)
    report["runs"] = runs
    report["summary"] = {
        "median_seconds": median,
        "mean_seconds": statistics.mean(times),
        "stdev_seconds": statistics.stdev(times) if len(times) > 1 else 0,
        "min_seconds": min(times),
        "max_seconds": max(times),
        "median_million_bases_per_second": metadata["bases"] / median / 1e6,
        "max_peak_rss_bytes": max(run["peak_rss_bytes"] for run in runs),
    }
    directory = ROOT / "target/dataset-benchmarks"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = directory / f"{stamp}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Median: {median:.4f} s; {metadata['bases'] / median / 1e6:.2f} Mbases/s")
    print(f"Results: {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f"benchmark: {error}")
