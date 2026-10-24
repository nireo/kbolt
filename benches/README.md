# Counting benchmarks

Run the full suite (optimized builds are automatic):

```sh
cargo bench --bench count
```

Smoke-test every case without collecting statistics:

```sh
cargo bench --bench count -- --test
```

Save a reference, then compare subsequent changes against it:

```sh
cargo bench --bench count -- --save-baseline naive
cargo bench --bench count -- --baseline naive
```

Filter to a workload or case by passing its name:

```sh
cargo bench --bench count -- random_long/canonical/31
```

## Work measured

Every workload contains 1,000,000 input bases. Random DNA uses fixed-seed
SplitMix64, so inputs stay identical between runs. Cases cover:

- one long random record (many distinct keys);
- the same random sequence split into 150-base reads (last read is 100 bases);
- a homopolymer (one repeated key);
- a repeating 12-base motif (a small set of repeated keys);
- long random DNA with every 100th base replaced by N.

Each runs at k=21 and k=31 in forward and canonical modes: 20 cases total.
Before timing, each case checks that the sum of counts matches the number of
valid windows. Functional tests remain the oracle for exact key correctness.

Each timed iteration starts with a fresh counter. Timing includes map creation,
allocations, counting all records, and destruction. It excludes input generation,
validation assertions, parsing, sorting, and file I/O. The resulting counts are
passed through `std::hint::black_box`.

Criterion reports elapsed time and input bases/second (`elem/s`); `Melem/s` means
million bases/second, not distinct k-mers/second. Ambiguous bases are included in
the input-base total. Defaults are 1 second of warmup and 10 samples over a target
of 2 seconds per case; use longer runs for small differences:

```sh
cargo bench --bench count -- --sample-size 30 --measurement-time 10
```

Keep the machine, Rust version, power mode, input, and background load consistent.
Record `rustc -Vv` and machine details with any reported results. Do not interpret
small differences within the reported uncertainty as improvements. HashMap uses
randomized hashing, so some run-to-run variation is expected. Criterion artifacts
and saved baselines live under `target/criterion/` (not committed). These
benchmarks do not measure peak memory or end-to-end file processing.
