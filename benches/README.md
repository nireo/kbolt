# Counting benchmarks

Run the optimized suite or smoke-test its correctness checks:

```sh
cargo bench --bench count
cargo bench --bench count -- --test
```

Save a baseline before a change, then compare against it:

```sh
cargo bench --bench count -- --save-baseline reference
cargo bench --bench count -- --baseline reference
```

Filter a case or use longer measurements:

```sh
cargo bench --bench count -- random_long/canonical/31
cargo bench --bench count -- --sample-size 30 --measurement-time 10
```

## Workloads and scope

Each workload contains 1,000,000 input bases:

- fixed-seed random DNA in one long record;
- the same DNA split into 150-base reads;
- a homopolymer;
- a repeating 12-base motif;
- random DNA with every 100th base replaced by N.

Each runs at k=21, 31, 33, and 63 in forward and canonical modes: 40 cases total.
Untimed assertions check the total number of valid windows. Unit tests check
exact keys and counts.

Timing includes creating a fresh counter, allocation, counting, and destruction.
Input generation, correctness assertions, key decoding, and file I/O are excluded.
Packed keys use randomized FoldHash; larger byte-string keys retain the standard
hasher. Results are observed through `std::hint::black_box` without decoding.

Criterion reports elapsed time and input bases/second (`Melem/s` means million
bases/second), including ambiguous input bases. Defaults are 10 samples, 1 second
of warmup, and a target measurement time of 2 seconds per case.

Keep hardware, compiler, power mode, and background load consistent. Randomized
hash seeds introduce variation; do not treat differences within uncertainty as
improvements. Criterion artifacts and baselines live in `target/criterion/` and
are not committed. These benchmarks do not measure peak memory or end-to-end
file processing.

## Parallel counting and merge benchmarks

```sh
cargo bench --bench parallel
cargo bench --bench parallel -- --test
cargo bench --bench parallel -- 'parallel/sampled_reads_150/.*/31'
KBOLT_BENCH_BASES=16777216 cargo bench --bench parallel
```

The default input size is 4 MiB (adjustable with `KBOLT_BENCH_BASES`, at least
150 bases). Set `KBOLT_BENCH_WORKLOAD=sampled_reads_150` (or `random_long` or
`homopolymer`) to skip other workloads' untimed counting checks. Workloads are
one uniform random record, reads sampled from a fixed
256 KiB random genome, and a homopolymer. Reads are 150 bases; every tenth read
has one substitution. At the default size this approximates 16x coverage with
sequencing errors. Generation uses fixed seeds and is outside timing.

Each workload runs canonical k=31 and k=33, comparing serial counting with 1, 2,
4, and 8 persistent workers. `workers_N` names the Crossbeam worker pipeline.
The one-worker parallel case measures pipeline overhead; CLI `--threads 1`
instead uses the direct serial path. Timing includes thread startup, copying
input into reusable bounded batches, parallel extraction, partitioned updates,
finalization, and destruction. Finalization retains key partitions without map
merging. Total windows and distinct-key counts are checked before timing;
functional tests compare exact key/count maps across record/chunk boundaries and
all representations.

Benchmark names include input size to keep different-size baselines separate.
`merge/<workload>/<bases>_bases/31` isolates merging two counters. The counters
are constructed outside timing with `iter_batched`; merging and destruction are timed. A single
long record is split into two independent records for this test, so its boundary
windows are intentionally absent. This measures map merging, not chunk semantics.

For complete CLI wall time, phase timings, and peak RSS on local references,
see [the dataset runner](datasets.md). Keep benchmark processes isolated from
other benchmarks, builds, and tests. Small inputs expose startup costs; the
larger reference and configurable synthetic sizes exercise tables beyond cache.

See [the recorded parallel baseline](baseline.md) for measured speed, memory,
phase timings, and the saved raw results.

The [profiling investigation](read-profile.md) explains the former merge and
input-stage bottlenecks. The [partitioned implementation measurements](partitioned.md)
compare the resulting pipeline, gzip backend, and concurrent readers against the
saved previous executable, with replay instructions.

The [historical Crossbeam/DashMap comparison](crates.md) retains the dependency
experiment results. DashMap has been removed from the current implementation.
