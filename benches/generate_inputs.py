#!/usr/bin/env python3
"""Generate the synthetic FASTQ inputs used by compare.py and pipeline.py.

Writes three files to target/read-profile/inputs (or --output):

  sampled.fq      894,784 reads of 150 bases sampled from a fixed 256 kbp random
                  genome; every tenth read has one substitution (about 512x
                  coverage with sequencing errors)
  sampled.fq.gz   sampled.fq as a single level-1 gzip stream
  homopolymer.fq  the same number of reads, every base A

Headers and qualities are fixed, so every record is 307 bytes. Output is
deterministic: the same command always writes identical files.
"""

import argparse
import gzip
import hashlib
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
MASK = (1 << 64) - 1
READ = 150
RECORDS = 128 * 1024 * 1024 // READ


def random_genome(length):
    """SplitMix64 bases with a fixed seed."""
    state, genome = 42, bytearray()
    for _ in range(length):
        state = (state + 0x9E3779B97F4A7C15) & MASK
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
        value ^= value >> 31
        genome.append(b"ACGT"[value & 3])
    return genome


def write_reads(path, reads):
    with path.open("wb") as out:
        for read in reads:
            out.write(b"@r\n" + read + b"\n+\n" + b"I" * READ + b"\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(path, RECORDS * READ, "bases", digest, flush=True)


def sampled_reads(genome):
    """Reads at positions from a fixed 64-bit LCG."""
    state = 42
    for index in range(RECORDS):
        state = (state * 6364136223846793005 + 1) & MASK
        start = (state >> 16) % (len(genome) - READ + 1)
        read = genome[start : start + READ]
        if index % 10 == 0:
            read[75] = ord("C") if read[75] == ord("A") else ord("A")
        yield read


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "target/read-profile/inputs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sampled = args.output / "sampled.fq"
    write_reads(sampled, sampled_reads(random_genome(256 * 1024)))
    # A zero timestamp and no stored name keep the gzip bytes reproducible.
    with sampled.open("rb") as source, (args.output / "sampled.fq.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1) as out:
            shutil.copyfileobj(source, out)
    write_reads(args.output / "homopolymer.fq", (b"A" * READ for _ in range(RECORDS)))


if __name__ == "__main__":
    main()
