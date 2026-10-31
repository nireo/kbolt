#!/usr/bin/env python3
"""Repeatable CLI benchmark for a local FASTA file (macOS and Linux)."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
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
            errors.seek(0)
            diagnostic = errors.read().decode(errors="replace")
    # Darwin reports bytes; Linux reports KiB.
    rss_bytes = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    result = {
        "wall_seconds": elapsed,
        "user_seconds": usage.ru_utime,
        "system_seconds": usage.ru_stime,
        "peak_rss_bytes": rss_bytes,
    }
    match = re.search(r"timings: counting=([0-9.]+)s merging=([0-9.]+)s output=([0-9.]+)s", diagnostic)
    if match:
        for name, value in zip(("counting_seconds", "merging_seconds", "output_seconds"), match.groups()):
            result[name] = float(value)
    return result


def summarize(runs, bases):
    times = [run["wall_seconds"] for run in runs]
    median = statistics.median(times)
    result = {
        "median_seconds": median,
        "mean_seconds": statistics.mean(times),
        "stdev_seconds": statistics.stdev(times) if len(times) > 1 else 0,
        "min_seconds": min(times),
        "max_seconds": max(times),
        "median_million_bases_per_second": bases / median / 1e6,
        "max_peak_rss_bytes": max(run["peak_rss_bytes"] for run in runs),
    }
    for phase in ("counting", "merging", "output"):
        key = f"{phase}_seconds"
        if all(key in run for run in runs):
            result[f"median_{key}"] = statistics.median(run[key] for run in runs)
    return result


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
    parser.add_argument("--threads", type=positive, nargs="+", default=[1],
                        help="worker counts to sweep; 1 uses the serial path (default: 1)")
    parser.add_argument("--compare-binary", type=Path,
                        help="also measure a saved baseline executable")
    parser.add_argument("--binary", type=Path,
                        help="benchmark an existing binary instead of building")
    parser.add_argument("--timings", action="store_true",
                        help="collect counting/merge/output wall times; requires a current binary")
    parser.add_argument("--runs", type=positive, default=10)
    parser.add_argument("--warmup", type=positive, default=2)
    args = parser.parse_args()
    if sys.platform not in ("darwin", "linux"):
        parser.error("peak memory measurement requires macOS or Linux")
    path = args.input.expanduser().resolve()
    if not path.is_file():
        parser.error(f"FASTA not found: {path}")
    metadata = fasta_metadata(path)

    if args.binary:
        binary = args.binary.expanduser().resolve()
        if not binary.is_file():
            parser.error(f"binary not found: {binary}")
    else:
        print("Building release binary (outside measured runs)...", flush=True)
        subprocess.run(
            ["cargo", "build", "--release", "--locked", "--bin", "kbolt",
             "--target-dir", str(ROOT / "target")],
            cwd=ROOT, check=True,
        )
        binary = ROOT / "target/release/kbolt"
    executables = [(binary, "current")]
    if args.compare_binary:
        previous = args.compare_binary.expanduser().resolve()
        if not previous.is_file():
            parser.error(f"comparison binary not found: {previous}")
        executables.append((previous, "baseline"))
    cases = []
    for executable, implementation in executables:
        binary_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
        for threads in dict.fromkeys(args.threads):
            command = [str(executable), "count", "-k", str(args.k)]
            if args.forward:
                command.append("--forward")
            command.extend(["--capacity", str(args.capacity), "--format", args.format])
            if threads != 1:
                command.extend(["--threads", str(threads)])
            if args.timings:
                command.append("--timings")
            command.append(str(path))
            cases.append({"implementation": implementation, "threads": threads,
                          "command": command, "binary_sha256": binary_hash, "runs": []})
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = capture("sysctl", "-n", "machdep.cpu.brand_string")
    elif Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": capture("git", "rev-parse", "HEAD"),
        "git_status": capture("git", "status", "--porcelain"),
        "rustc": capture("rustc", "-Vv"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "input": metadata,
        "binary": str(binary),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "cases": cases,
        "stdout": "discarded to /dev/null; output serialization and writes included",
        "cache": "warm; input scanned and warmup runs completed before timing",
        "warmup_runs": args.warmup,
    }
    print(f"Input: {path.name}; {metadata['bases']:,} bases", flush=True)
    for case in cases:
        for _ in range(args.warmup):
            measure(case["command"])
        print(f"Warmups complete: {case['implementation']}/{case['threads']}", flush=True)
    # Rotate the order across rounds to reduce bias from gradual machine drift.
    for index in range(args.runs):
        offset = index % len(cases)
        for case in cases[offset:] + cases[:offset]:
            result = measure(case["command"])
            case["runs"].append(result)
            print(
                f"{case['implementation']}/{case['threads']}, run {index + 1:2}/{args.runs}: "
                f"{result['wall_seconds']:.4f} s, "
                f"{result['peak_rss_bytes'] / 1024**2:.1f} MiB peak RSS",
                flush=True,
            )
    serial = None
    for case in cases:
        case["summary"] = summarize(case["runs"], metadata["bases"])
        if case["threads"] == 1 and case["implementation"] == "current":
            serial = case["summary"]["median_seconds"]
    for case in cases:
        summary = case["summary"]
        if serial is not None:
            summary["speedup_vs_serial"] = serial / summary["median_seconds"]
        print(f"{case['implementation']}/{case['threads']}: median {summary['median_seconds']:.4f} s; "
              f"{summary['median_million_bases_per_second']:.2f} Mbases/s")
    directory = ROOT / "target/dataset-benchmarks"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = directory / f"{stamp}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results: {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f"benchmark: {error}")
