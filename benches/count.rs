use std::hint::black_box;
use std::time::Duration;

use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use kbolt::{Counter, StrandMode};

const BASES: usize = 1_000_000;

// Fixed-seed SplitMix64: reproducible inputs without an extra random dependency.
fn random_dna(len: usize) -> Vec<u8> {
    let mut state = 42_u64;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
            value ^= value >> 31;
            b"ACGT"[(value & 3) as usize]
        })
        .collect()
}

fn count_records(records: &[Vec<u8>], k: usize, strand: StrandMode) -> Counter {
    let mut counter = Counter::new(k, strand).unwrap();
    for record in records {
        counter.add_record(record).unwrap();
    }
    counter
}

fn bench_count(c: &mut Criterion) {
    // All input generation and record splitting happen outside timed sections.
    let random = random_dna(BASES);
    let mut ambiguous = random.clone();
    for base in ambiguous.iter_mut().step_by(100) {
        *base = b'N';
    }
    let reads = random.chunks(150).map(<[u8]>::to_vec).collect();
    let motif = b"ACGTTGCAAGTC";
    let workloads = [
        ("random_long", vec![random]),
        ("random_reads_150", reads),
        ("homopolymer", vec![vec![b'A'; BASES]]),
        (
            "repeated_motif",
            vec![(0..BASES).map(|i| motif[i % motif.len()]).collect()],
        ),
        ("ambiguous_long", vec![ambiguous]),
    ];

    for (name, records) in workloads {
        let bases: usize = records.iter().map(Vec::len).sum();
        let mut group = c.benchmark_group(format!("count/{name}"));
        // One element is one input base, including ambiguous bases.
        group.throughput(Throughput::Elements(bases as u64));
        group.sampling_mode(SamplingMode::Flat);

        for k in [21, 31] {
            // Untimed sanity check: count exactly the valid windows, not merely keys.
            let expected: u64 = records
                .iter()
                .flat_map(|record| record.windows(k))
                .filter(|window| window.iter().all(|base| b"ACGT".contains(base)))
                .count() as u64;

            for (mode, strand) in [
                ("forward", StrandMode::Forward),
                ("canonical", StrandMode::Canonical),
            ] {
                let check = count_records(&records, k, strand);
                assert_eq!(
                    check.counts().map(|(_, count)| count).sum::<u64>(),
                    expected
                );
                drop(check);

                group.bench_with_input(BenchmarkId::new(mode, k), &k, |b, &k| {
                    b.iter(|| {
                        // Fresh map each iteration; construction, allocation, counting,
                        // and destruction are all included in the measurement.
                        let counter = count_records(black_box(&records), k, strand);
                        // Observe storage without decoding keys: output conversion
                        // was not part of the byte-key baseline either.
                        black_box(&counter);
                    });
                });
            }
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = bench_count
}
criterion_main!(benches);
