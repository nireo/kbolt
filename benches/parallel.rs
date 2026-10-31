use std::hint::black_box;
use std::time::Duration;

use criterion::{
    BatchSize, BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use kbolt::{Counter, ParallelCounter, StrandMode};

mod common;

fn serial(records: &[Vec<u8>], k: usize) -> Counter {
    let mut counter = Counter::new(k, StrandMode::Canonical).unwrap();
    for record in records {
        counter.add_record(record).unwrap();
    }
    counter
}

fn parallel(records: &[Vec<u8>], k: usize, threads: usize) -> Counter {
    let mut counter = ParallelCounter::new(k, StrandMode::Canonical, threads).unwrap();
    for record in records {
        counter.add_record(record).unwrap();
    }
    counter.finish().unwrap()
}

fn bench_parallel(c: &mut Criterion) {
    let bases = std::env::var("KBOLT_BENCH_BASES")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(4 * 1024 * 1024);
    assert!(bases >= 150, "KBOLT_BENCH_BASES must be at least 150");
    let selected = std::env::var("KBOLT_BENCH_WORKLOAD").ok();
    assert!(
        selected
            .as_deref()
            .is_none_or(|name| matches!(name, "random_long" | "sampled_reads_150" | "homopolymer")),
        "unknown KBOLT_BENCH_WORKLOAD"
    );
    let genome = common::random_dna(256 * 1024);
    let mut state = 42_u64;
    let reads: Vec<Vec<u8>> = (0..bases / 150)
        .map(|index| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let start = (state >> 16) as usize % (genome.len() - 149);
            let mut read = genome[start..start + 150].to_vec();
            // One substitution in every tenth read: duplication with sequencing errors.
            if index % 10 == 0 {
                read[75] = if read[75] == b'A' { b'C' } else { b'A' };
            }
            read
        })
        .collect();
    for (name, records) in [
        ("random_long", vec![common::random_dna(bases)]),
        ("sampled_reads_150", reads),
        ("homopolymer", vec![vec![b'A'; bases]]),
    ] {
        if selected.as_deref().is_some_and(|selected| selected != name) {
            continue;
        }
        let bases: usize = records.iter().map(Vec::len).sum();
        let mut group = c.benchmark_group(format!("parallel/{name}/{bases}_bases"));
        group.throughput(Throughput::Elements(bases as u64));
        group.sampling_mode(SamplingMode::Flat);
        for k in [31, 33] {
            let expected: u64 = records
                .iter()
                .map(|r| r.len().saturating_sub(k - 1) as u64)
                .sum();
            let reference = serial(&records, k);
            assert_eq!(
                reference.counts().map(|(_, count)| count).sum::<u64>(),
                expected
            );
            let distinct = reference.len();
            drop(reference);
            group.bench_with_input(BenchmarkId::new("serial", k), &k, |b, &k| {
                b.iter(|| black_box(serial(black_box(&records), k)));
            });
            for threads in [1, 2, 4, 8] {
                let check = parallel(&records, k, threads);
                assert_eq!(check.len(), distinct);
                assert_eq!(
                    check.counts().map(|(_, count)| count).sum::<u64>(),
                    expected
                );
                drop(check);
                group.bench_with_input(
                    BenchmarkId::new(format!("workers_{threads}"), k),
                    &k,
                    |b, &k| b.iter(|| black_box(parallel(black_box(&records), k, threads))),
                );
            }
        }
        group.finish();
        // Isolate the cost of combining two maps. Setup counts half the records
        // (or half of a long record) outside the timed region; output/drop is timed.
        let split: Vec<Vec<u8>> = if records.len() == 1 {
            records[0]
                .chunks(records[0].len().div_ceil(2))
                .map(<[u8]>::to_vec)
                .collect()
        } else {
            records
        };
        let middle = split.len() / 2;
        c.bench_function(&format!("merge/{name}/{bases}_bases/31"), |b| {
            b.iter_batched(
                || (serial(&split[..middle], 31), serial(&split[middle..], 31)),
                |(mut left, right)| {
                    left.merge(right).unwrap();
                    black_box(left);
                },
                BatchSize::PerIteration,
            );
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).warm_up_time(Duration::from_secs(1)).measurement_time(Duration::from_secs(2));
    targets = bench_parallel
}
criterion_main!(benches);
