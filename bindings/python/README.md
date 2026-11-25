# kbolt (Python bindings)

Python bindings for [kbolt](https://github.com/nireo/kbolt), a fast exact DNA
k-mer counter written in Rust.

## Usage

```python
import kbolt

# Count directly from FASTA/FASTQ files (plain or gzipped) in parallel
counts = kbolt.count_files(["R1.fastq.gz", "R2.fastq.gz"], k=31, threads=8)

# Fast lookups directly in Rust memory (no Python dict needed)
print(len(counts))
print(counts["ACGTACGTACGTACGTACGTACGTACGTACG"])
print(counts.get_many(["ACGT...", "TGCA..."]))

# K-mer spectrum and most abundant k-mers
spectrum = counts.histogram()
top20 = counts.top_k(20)

# Single-pass bulk export into NumPy arrays (fixed-width ASCII bytes or packed uint64 for k <= 32)
keys, vals = counts.to_numpy(min_count=2)
packed_keys, packed_vals = counts.to_numpy(min_count=2, packed=True)

# Count huge datasets within a memory budget on disk, then stream from Python
distinct = kbolt.count_files_to_disk(
    ["R1.fastq.gz", "R2.fastq.gz"],
    "counts.kbolt",
    k=31,
    memory="2G",
    threads=8,
)
with kbolt.open_database("counts.kbolt") as db:
    for kmer, count in db:
        ...
```

## Notes

- `Counter` is **not** safe for concurrent mutation and reading from multiple
  Python threads; concurrent access raises `RuntimeError` via PyO3's borrow
  checker.
- `to_dict()`, `items()`, `keys()`, and `iter(counter)` materialize Python
  objects for every matching k-mer. For large datasets, prefer `to_numpy()`,
  `histogram()`, `top_k()`, `get_many()`, or `write_binary()` + `open_database()`.
- Wheels target Python's `abi3` (3.11+) limited API (standard GIL builds).

## Development

```sh
cd bindings/python
maturin develop --release
pytest
```
