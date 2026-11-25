# kbolt (R bindings)

R bindings for [kbolt](https://github.com/nireo/kbolt), a fast exact DNA k-mer
counter written in Rust, built with [extendr](https://extendr.github.io).

## Installation

Requires Rust (`rustc` >= 1.85 and `cargo`):

```sh
R CMD INSTALL bindings/r
```

## Usage

```r
library(kbolt)

# Count directly from FASTA/FASTQ files (plain or gzipped) in parallel
counts <- kbolt_count_files(c("R1.fastq.gz", "R2.fastq.gz"), k = 31, threads = 8)
print(counts)
length(counts)

# Vectorized lookups directly in Rust memory (no R string interning overhead)
counts[c("ACGTACGTACGTACGTACGTACGTACGTACG", "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")]
kbolt_get(counts, c("ACGTACGTACGTACGTACGTACGTACGTACG"))

# K-mer spectrum and most abundant k-mers as data.frames
spectrum <- kbolt_histogram(counts)
top20 <- kbolt_top_k(counts, 20)

# Export filtered counts as a data.frame
df <- as.data.frame(counts, min_count = 2)

# Count huge datasets within a memory budget on disk, then stream in chunks
distinct <- kbolt_count_files_to_disk(
  c("R1.fastq.gz", "R2.fastq.gz"),
  output = "counts.kbolt",
  k = 31,
  memory = "2G",
  threads = 8
)

db <- kbolt_open_database("counts.kbolt")
kbolt_database_header(db)
chunk <- kbolt_database_read(db, n = 100000)
kbolt_database_close(db)
```

## Notes

- All k-mer counts and totals are returned as R `numeric` (`double`), which
  represents integers exactly up to $2^{53}$ without overflowing R's 32-bit
  `integer` type.
- `kbolt_counter` keeps the hash table in Rust memory and frees it
  automatically when garbage-collected. For large read sets, avoid calling
  `as.data.frame(counts)` without `min_count`, since R interns every unique
  character string in its global `CHARSXP` cache.

## Testing

```sh
R CMD INSTALL bindings/r
Rscript bindings/r/tests/test_kbolt.R
```
