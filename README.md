# kbolt

kbolt is a Rust library and CLI that counts DNA k-mers exactly in memory. It
packs k-mers of up to 32 bases into 64-bit integers, uses 128-bit packed keys
through k=64, and uses byte strings for longer k-mers.

## CLI

Install from this checkout:

```sh
cargo install --path .
```

Count canonical k-mers from one or more FASTA/FASTQ files:

```sh
kbolt count -k 31 reads.fastq.gz --output counts.tsv --progress
kbolt count -k 31 --threads 4 --format binary reads.fastq.gz -o counts.kbolt --timings
kbolt count -k 31 -t 8 --readers 2 sample_R1.fastq.gz sample_R2.fastq.gz -o counts.tsv
kbolt count -k 21 sample_R1.fastq.gz sample_R2.fastq.gz > counts.tsv
cat reference.fasta | kbolt count -k 31 > counts.tsv
kbolt count -k 31 --forward reference.fasta > forward.tsv
kbolt count --help
```

- Input format and gzip compression are detected from content, not extensions.
  Files may use different formats/compression. Concatenated gzip members work.
- No input paths means stdin; use `-` to include stdin explicitly (at most once).
  `--output -` (the default) writes to stdout.
- Output is headerless `KMER<TAB>COUNT`, in unspecified order. Canonical mode
  combines reverse complements by default; `--forward` keeps them separate.
- Input is parsed in 64 KiB fragments, retaining up to `k-1` sequence bytes at
  counting-batch boundaries. Even enormous single-line FASTA records, headers, and FASTQ
  quality lines are not loaded in full. Windows cross wrapped lines and chunks,
  but never records or files. Paired-end reads are separate records, not joined.
- `--threads N` (`-t N`) selects persistent counting workers. The default of 1
  retains the direct serial path. Parallel workers extract keys from reusable
  batches that grow from 64 KiB to 1 MiB, and buffer updates by key partition.
  Each key lives in exactly one partition; workers acquire a partition once per batch and combine repeated
  updates locally. Finalization and output retain those maps without a global
  merge. `--capacity` is a total initial hint divided among partitions, not a
  memory limit. Batch dispatch uses a bounded Crossbeam queue.
- `--readers N` selects concurrent input readers when `--threads` exceeds 1.
  The default is 2, capped by the number of inputs. One input is parsed on the
  calling thread; multiple readers independently parse/decompress files into
  the shared bounded queue. Reader threads are additional to counting workers.
  Use `--readers 1` for sequential input. Concurrent readers accept regular files;
  stdin, devices, and FIFOs retain sequential input order. Gzip decoding uses `zlib-rs`;
  one ordinary gzip stream still has one decoder. Parallel readers do not split a
  compressed file into arbitrary byte ranges.
- `--timings` prints counting (including parsing/decompression and worker startup),
  finalization (`merging` for compatibility), output, and total wall times to
  stderr. The benchmark runner additionally measures process teardown and peak RSS.
- **The distinct-k-mer table still lives entirely in RAM.** Streaming bounds
  input buffering, not counting memory; there is no disk spill or memory limit.
- FASTA and FASTQ support wrapped sequence lines, LF/CRLF, and a final line
  without a newline. Headers must start with `>`/`@` in column one and include a
  name. Sequence bytes may be ASCII letters, `.` or `-`; case is ignored and
  anything other than A/C/G/T breaks a valid run. FASTA blank sequence lines
  are allowed; leading blank lines and FASTA comment lines are not supported.
- FASTQ requires a `+` separator and exactly one printable ASCII quality score
  (`!` through `~`) per sequence byte. Quality lines may wrap independently of
  sequence lines. Optional `+` text is ignored; qualities are validated, not
  used for filtering. Empty FASTQ records require an empty quality line.
- Malformed input produces a nonzero exit status and a diagnostic on stderr.
  Output is opened only after all inputs have been validated, so input errors
  do not truncate an existing output file. Output write failures can leave a
  partial file. Empty input is accepted and contributes no records.
- `--progress` reports records, sequence bases (including ambiguity), distinct
  k-mers, and elapsed time on stderr, approximately once a second while counting
  and once at completion. With parallel workers, periodic reports show submitted
  bases; the exact distinct count is available after finalization. Concurrent
  readers report per-file progress with a path prefix; the final summary combines
  every input. Counts alone go to stdout/the output file.

## Library usage

```rust
use kbolt::{Counter, StrandMode};

fn main() -> Result<(), kbolt::Error> {
    let mut counter = Counter::new(3, StrandMode::Canonical)?;

    // Add each read or sequence record separately.
    counter.add_record(b"ACGT")?;
    counter.add_record(b"acg")?;

    for (kmer, count) in counter.counts() {
        println!("{} {count}", String::from_utf8_lossy(&kmer));
    }

    Ok(())
}
```

Output:

```text
ACG 3
```

Canonical mode combines the counts of a k-mer and its reverse complement.
Both `ACG` and `CGT` contribute to `ACG` in this example. Use `StrandMode::Forward`
to count them separately.

For parallel library counting, submit each complete record to
`ParallelCounter::new(k, strand, workers)?`, then call `finish()?` to obtain a
normal `Counter` retaining its key partitions. `finish_with_stats()?` also returns
counting and finalization wall times. Errors from workers can surface during
finalization. Dropping the parallel
counter closes its bounded queue and joins all workers. `Counter::merge` consumes
another counter with the same k/strand mode, adding counts with overflow checks.
Batch dispatch uses `crossbeam-channel`.

For fragmented records, call `start_record()` and then `add_sequence(fragment)?`
for each fragment; windows cross fragments but not `start_record()` boundaries.
Use `parallel.input()` to create independent producers for scoped reader threads.
Each producer supports the same record/fragment methods; call `input.finish()?`
to submit its final partial batch and report submission errors. Producer borrows
prevent finishing the counter until all producers have been released. A dropped
producer also attempts to submit its partial batch. Finished partitioned counters
support subsequent `add_record`, `merge`, and both output formats.

## Development

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo bench --bench count
cargo bench --bench parallel
```

See [the benchmark guide](benches/README.md) for workloads and baseline comparisons.
