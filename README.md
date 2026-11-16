# kbolt

kbolt counts k-mers in DNA. Point it at a genome or a batch of sequencing reads
and it lists every substring of length k along with how many times it shows up.
The counts are exact, not estimates. It reads FASTA and FASTQ, gzipped or not,
and you can use it from the command line or as a Rust library.

It's built to be fast on an ordinary machine. Everything stays in memory, so it
works best when the distinct k-mers fit in RAM. Here's how it compares with
FastK, KMC and Jellyfish, three widely used exact counters, on the same inputs:

![k-mer counting throughput of kbolt, FastK, KMC and Jellyfish at 1, 4 and 8 threads](docs/benchmark.png)

Wall time to count canonical 31-mers (median of 3 runs, Apple M5):

| Workload | Threads | kbolt | FastK | KMC | Jellyfish |
|---|---:|---:|---:|---:|---:|
| Arabidopsis genome, 120 Mbp | 1 | **2.47 s** | 6.57 s | 4.11 s | 29.7 s |
|  | 8 | **0.73 s** | 1.86 s | 1.65 s | 6.81 s |
| Reads, 134 Mbp FASTQ | 1 | **0.53 s** | 2.12 s | 4.90 s | 6.85 s |
|  | 8 | **0.13 s** | 0.61 s | 1.56 s | 2.65 s |
| Same reads, gzip | 1 | **0.56 s** | 2.38 s | 4.89 s | 6.98 s |
|  | 8 | **0.22 s** | 0.80 s | 1.95 s | 2.26 s |

kbolt is 2.6–4.7× faster than FastK, 1.7–12× faster than KMC and 9–20× faster
than Jellyfish here, and all four report identical counts.

These inputs are small. KMC is built for read sets larger than memory, which
it splits into bins on disk, and it pays fixed costs here that matter less at
that scale; kbolt cannot count those inputs at all.

## Usage

```sh
cargo install --path .

kbolt count -k 31 reads.fastq.gz > counts.tsv
kbolt count -k 31 -t 8 sample_R1.fastq.gz sample_R2.fastq.gz -o counts.tsv
kbolt count -k 31 -t 8 --format binary reads.fastq.gz -o counts.kbolt
kbolt dump counts.kbolt > counts.tsv
```

- Output is `KMER<TAB>COUNT` per line, in no particular order.
- Reverse complements are combined by default; `--forward` counts each strand
  separately.
- Format and compression are detected from the content. With no input files,
  kbolt reads stdin.
- `-t N` sets the number of counting threads. `--progress` and `--timings`
  report on stderr.
- Every distinct k-mer is held in RAM; there is no disk spill.

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
normal `Counter`.

## How it works

For k ≤ 32, each k-mer is packed into 64 bits and passed through an invertible
hash. The top hash bits choose a partition, and the rest are stored in a single
64-bit slot alongside a small count. Rare large counts move to a side table, so
every count stays exact. Slots sit in 64-byte buckets of eight, so a lookup
touches one cache line, and updates are buffered so buckets can be prefetched
ahead of time. Longer k-mers use ordinary hash maps.

## Development

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo bench
```

`python3 benches/compare.py` reruns the comparison (FastK and Jellyfish from
Homebrew, KMC from its GitHub releases, inputs from
`benches/generate_inputs.py`). Redraw the chart with
`uv run --no-project --with matplotlib python benches/plot_compare.py`.
