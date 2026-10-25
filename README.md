# kbolt

kbolt is a Rust library that counts DNA k-mers in memory. It packs k-mers of up
to 32 bases into 64-bit integers and uses byte strings for longer k-mers.

## Usage

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

## Development

```sh
cargo test
cargo bench --bench count
```

See [the benchmark guide](benches/README.md) for workloads and baseline comparisons.
