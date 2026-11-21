# kbolt

kbolt counts k-mers in DNA. Point it at a genome or a batch of sequencing reads
and it lists every substring of length k along with how many times it shows up.
The counts are exact, not estimates. It reads FASTA and FASTQ, gzipped or not,
and you can use it from the command line or as a Rust library.

It's built to be fast on an ordinary machine. Everything stays in memory, so it
works best when the distinct k-mers fit in RAM. Here's how it compares with KMC
and FastK, two widely used exact counters, on a real read set (C. elegans
SRR065390: two gzipped FASTQ files, 6.76 Gbp, 488 million distinct 31-mers):

![Wall time, CPU time and peak memory of kbolt, KMC and FastK counting 31-mers in 6.76 Gbp of C. elegans reads](docs/benchmark.png)

Counting canonical 31-mers at 8 threads (median of 3 runs, Apple M5 with 24 GB
of memory):

| | Wall time | CPU time | Peak memory |
|---|---:|---:|---:|
| kbolt | **13.1 s** | 103 s | 6.5 GiB |
| KMC | 33.4 s | 206 s | 11.2 GiB |
| FastK | 45.0 s | 196 s | 11.4 GiB |

kbolt is 2.5× faster than KMC and 3.4× faster than FastK, on about half the
CPU time and under 60% of the memory. kbolt and KMC report identical counts.

Read sets whose k-mers do not fit in memory can be counted on disk with
`--memory`, as KMC and FastK do. The same reads with every tool limited to
2 GiB (kbolt `--memory 2G`, KMC `-m2`, FastK `-M2`), median of 3 runs:

| | Wall time | CPU time | Peak memory |
|---|---:|---:|---:|
| kbolt | **16.6 s** | 115 s | 1.4 GiB |
| KMC | 35.3 s | 200 s | 1.9 GiB |
| FastK | 44.5 s | 195 s | 4.2 GiB |

The result is identical to counting in memory.

On smaller inputs the margin is similar or larger, here with Jellyfish as
well. Wall time (median of 3 runs):

| Workload | Threads | kbolt | FastK | KMC | Jellyfish |
|---|---:|---:|---:|---:|---:|
| Arabidopsis genome, 120 Mbp | 1 | **2.73 s** | 6.52 s | 4.12 s | 29.4 s |
|  | 8 | **0.76 s** | 2.02 s | 1.60 s | 6.61 s |
| Synthetic reads, 134 Mbp FASTQ | 1 | **0.56 s** | 2.08 s | 4.87 s | 6.29 s |
|  | 8 | **0.14 s** | 0.61 s | 1.63 s | 2.82 s |
| Same reads, gzip | 1 | **0.56 s** | 2.38 s | 4.88 s | 6.64 s |
|  | 8 | **0.23 s** | 0.81 s | 1.94 s | 2.42 s |

All four tools report identical counts on these inputs.

## Usage

```sh
cargo install --path .

kbolt count -k 31 reads.fastq.gz > counts.tsv
kbolt count -k 31 -t 8 sample_R1.fastq.gz sample_R2.fastq.gz -o counts.tsv
kbolt count -k 31 -t 8 --format binary reads.fastq.gz -o counts.kbolt
kbolt count -k 31 -t 8 --memory 4G reads.fastq.gz -o counts.tsv
kbolt dump counts.kbolt > counts.tsv
```

- Output is `KMER<TAB>COUNT` per line, in no particular order.
- Reverse complements are combined by default; `--forward` counts each strand
  separately.
- Format and compression are detected from the content. With no input files,
  kbolt reads stdin.
- `-t N` sets the number of counting threads. `--progress` and `--timings`
  report on stderr.
- By default every distinct k-mer is held in RAM. `--memory SIZE` (k ≤ 32)
  counts within about that much memory instead: the input is first split
  into bins, kept in memory up to half the limit and otherwise in a temporary
  file under `--temp-dir` (a little under a byte per input base).

See `kbolt count --help` for all options.

## Library

```rust
use kbolt::{Counter, StrandMode};

fn main() -> Result<(), kbolt::Error> {
    let mut counter = Counter::new(3, StrandMode::Canonical)?;
    counter.add_record(b"ACGT")?;
    counter.add_record(b"acg")?;

    for (kmer, count) in counter.counts() {
        println!("{} {count}", String::from_utf8_lossy(&kmer)); // ACG 3
    }
    Ok(())
}
```

For many short reads, `add_records` is faster than repeated `add_record`
calls. `ParallelCounter` counts on several threads and `finish()` returns a
normal `Counter`. `DiskCounter` takes input the same way but counts on disk
within a memory limit and writes its result straight to a file.

## How it works

For k ≤ 32, each k-mer is packed into 64 bits and passed through an invertible
hash. The top hash bits choose a partition, and the rest are stored in a single
64-bit slot alongside a small count. Rare large counts move to a side table, so
every count stays exact. Slots sit in 64-byte buckets of eight, so a lookup
touches one cache line.

With several threads there are thousands of small tables. Updates are queued
per group of tables and applied in bulk, one table at a time, so each table is
pulled into the CPU cache once and updated there instead of missing the cache
on every k-mer. Tables grow in small steps and are returned to the OS when
outgrown, which keeps memory close to what the k-mers need. Longer k-mers use
ordinary hash maps.

With `--memory`, counting takes two passes. The first cuts each read into
super-k-mers, runs of neighbouring k-mers that share a minimizer, and appends
them to one of 512 bins; every copy of a k-mer lands in the same bin. Bins
stay in memory while the limit allows and go to a temporary file after that.
The second pass counts one bin at a time with the same tables and writes its
k-mers out before the next, so memory follows the bins being counted, not the
input. Reads overlap, so most super-k-mers of a read set occur many times;
equal ones are merged first and their k-mers counted once, with the number of
copies.

## Development

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo bench
```

On x86-64, the table lookup uses the widest vector instructions the build
enables: SSE2 by default, AVX2 or AVX-512 with
`RUSTFLAGS="-C target-cpu=native" cargo build --release` (or
`-C target-cpu=x86-64-v3`). A binary built that way only runs on processors
with those instructions.

`python3 benches/reads.py --tools kbolt kmc fastk` reruns the real-read
comparison after `python3 benches/reads.py --fetch` has downloaded the reads
(5 GB); FastK comes from Homebrew and KMC from its GitHub releases. Add
`--memory 0 2` to also run every tool with a 2 GiB limit. Redraw the
chart with `uv run --no-project --with matplotlib python benches/plot_reads.py`.
`python3 benches/compare.py` reruns the small-input comparison (inputs from
`benches/generate_inputs.py`), and `benches/plot_compare.py` plots its
throughput against threads.
