//! Bounded batches and persistent worker-local counters. Synchronization happens
//! once per batch, never in the per-base or per-k-mer counting loops.

use crossbeam_channel as channel;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::{Counter, Error, StrandMode};

const BATCH_BASES: usize = 256 * 1024;

#[derive(Default)]
struct Batch {
    bases: Vec<u8>,
    ends: Vec<usize>,
}

impl Batch {
    fn count(&self, counter: &mut Counter) -> Result<(), Error> {
        let mut start = 0;
        for &end in &self.ends {
            counter.add_record(&self.bases[start..end])?;
            start = end;
        }
        Ok(())
    }

    fn clear(&mut self) {
        self.bases.clear();
        self.ends.clear();
    }
}

/// Wall-clock phases measured from construction to completion, excluding output.
#[derive(Debug, Clone, Copy)]
pub struct ParallelStats {
    /// Input submission and counting, including waiting for all workers.
    pub counting: Duration,
    /// Merging worker-local maps into the final counter.
    pub merging: Duration,
}

/// An exact streaming counter with persistent worker-local maps.
///
/// Records are copied into reusable, bounded batches. Long records are split
/// with k-1 bases of overlap; windows never cross separate `add_record` calls.
/// At most two batches per worker are queued. Input buffering is O(workers *
/// (256 KiB + k)); count tables are unbounded and can duplicate keys across workers.
/// Dropping the counter closes the queue and joins all workers.
pub struct ParallelCounter {
    sender: Option<channel::Sender<Batch>>,
    recycled: channel::Receiver<Batch>,
    workers: Vec<JoinHandle<Result<Counter, Error>>>,
    failure: Arc<Mutex<Option<Error>>>,
    batch: Batch,
    k: usize,
    started: Instant,
}

impl ParallelCounter {
    /// Start `threads` counting workers. The caller submits input separately.
    pub fn new(k: usize, strand: StrandMode, threads: usize) -> Result<Self, Error> {
        Self::with_capacity(k, strand, threads, 0)
    }

    /// Start workers, dividing a total distinct-key capacity hint among them.
    /// The hint is not a limit. Duplicate keys across workers may require more memory.
    pub fn with_capacity(
        k: usize,
        strand: StrandMode,
        threads: usize,
        capacity: usize,
    ) -> Result<Self, Error> {
        let started = Instant::now();
        if k == 0 {
            return Err(Error::InvalidK);
        }
        let queued = threads
            .checked_mul(2)
            .filter(|&n| n > 0)
            .ok_or(Error::InvalidThreads)?;
        let (sender, receiver) = channel::bounded::<Batch>(queued);
        let (recycle, recycled) = channel::unbounded();
        let mut result = Self {
            sender: Some(sender),
            recycled,
            workers: Vec::new(),
            failure: Arc::new(Mutex::new(None)),
            batch: Batch::default(),
            k,
            started,
        };
        result
            .workers
            .try_reserve(threads)
            .map_err(|_| Error::ReserveFailed)?;
        for index in 0..threads {
            let mut counter = Counter::with_capacity(
                k,
                strand,
                capacity / threads + usize::from(index < capacity % threads),
            )?;
            let receiver = receiver.clone();
            let recycle = recycle.clone();
            let failure = Arc::clone(&result.failure);
            let worker = thread::Builder::new()
                .spawn(move || {
                    loop {
                        let next = receiver.recv();
                        let Ok(mut batch) = next else { break };
                        if let Err(error) = batch.count(&mut counter) {
                            *failure.lock().unwrap() = Some(error);
                            return Err(error);
                        }
                        batch.clear();
                        if recycle.send(batch).is_err() {
                            break;
                        }
                    }
                    Ok(counter)
                })
                .map_err(|_| Error::WorkerFailed)?;
            result.workers.push(worker);
        }
        Ok(result)
    }

    /// Submit one complete record. Short records add nothing. Errors from workers
    /// may be reported later by `finish`; success here only means input was accepted.
    pub fn add_record(&mut self, sequence: &[u8]) -> Result<(), Error> {
        if sequence.len() < self.k {
            return Ok(());
        }
        // Assign each window start to exactly one piece. The right overlap is
        // context only and is never counted again in the following piece.
        let windows = sequence.len() - self.k + 1;
        for start in (0..windows).step_by(BATCH_BASES) {
            let end = start + (windows - start).min(BATCH_BASES) + self.k - 1;
            let piece = &sequence[start..end];
            if !self.batch.ends.is_empty()
                && self.batch.bases.len().saturating_add(piece.len()) > BATCH_BASES
            {
                self.submit()?;
            }
            self.batch.bases.extend_from_slice(piece);
            self.batch.ends.push(self.batch.bases.len());
            if self.batch.bases.len() >= BATCH_BASES {
                self.submit()?;
            }
        }
        Ok(())
    }

    fn submit(&mut self) -> Result<(), Error> {
        if self.batch.ends.is_empty() {
            return Ok(());
        }
        let replacement = self.recycled.try_recv().unwrap_or_default();
        let batch = std::mem::replace(&mut self.batch, replacement);
        self.sender.as_ref().unwrap().send(batch).map_err(|_| {
            self.failure
                .lock()
                .ok()
                .and_then(|error| *error)
                .unwrap_or(Error::WorkerFailed)
        })
    }

    /// Wait for all counting and merge the maps without decoding their keys.
    pub fn finish(self) -> Result<Counter, Error> {
        self.finish_with_stats().map(|(counter, _)| counter)
    }

    /// Finish counting and return wall-clock counting and merge durations.
    pub fn finish_with_stats(mut self) -> Result<(Counter, ParallelStats), Error> {
        self.submit()?;
        self.sender.take();
        let mut counters = Vec::with_capacity(self.workers.len());
        // Pop individually: on error, Drop still joins every remaining worker.
        while let Some(worker) = self.workers.pop() {
            counters.push(worker.join().map_err(|_| Error::WorkerFailed)??);
        }
        let counting = self.started.elapsed();
        let merging = Instant::now();
        // Pairwise reduction lets independent merges run in parallel, and frees
        // source maps at each level instead of retaining them for a serial fold.
        while counters.len() > 1 {
            counters.sort_unstable_by_key(Counter::len);
            if counters.len() == 2 {
                let mut target = counters.pop().unwrap();
                target.merge(counters.pop().unwrap())?;
                counters.push(target);
                break;
            }
            counters = thread::scope(|scope| {
                let mut merges = Vec::with_capacity(counters.len() / 2);
                while counters.len() >= 2 {
                    let mut target = counters.pop().unwrap();
                    let source = counters.pop().unwrap();
                    merges.push(
                        thread::Builder::new()
                            .spawn_scoped(scope, move || {
                                target.merge(source)?;
                                Ok(target)
                            })
                            .map_err(|_| Error::WorkerFailed)?,
                    );
                }
                for merge in merges {
                    counters.push(merge.join().map_err(|_| Error::WorkerFailed)??);
                }
                Ok::<_, Error>(counters)
            })?;
        }
        let counter = counters.pop().unwrap();
        Ok((
            counter,
            ParallelStats {
                counting,
                merging: merging.elapsed(),
            },
        ))
    }
}

impl Drop for ParallelCounter {
    fn drop(&mut self) {
        self.sender.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Counts;
    use std::collections::BTreeMap;

    fn table(counter: &Counter) -> BTreeMap<Vec<u8>, u64> {
        counter.counts().collect()
    }

    #[test]
    fn parallel_matches_serial_across_batches_records_and_representations() {
        let mut long = b"acGTTGCAGATTCGAC".repeat(BATCH_BASES / 16 + 16);
        for index in [BATCH_BASES - 1, BATCH_BASES, BATCH_BASES + 1] {
            long[index] = b'N';
        }
        let mut state = 42_u64;
        let random: Vec<u8> = (0..4096)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                b"ACGT"[(state & 3) as usize]
            })
            .collect();
        let records = [
            b"AC".as_slice(),
            b"GT",
            b"",
            &long,
            b"ACGTNNacgt",
            &long,
            &random,
        ];
        for k in [1, 3, 31, 32, 33, 63, 64, 65] {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let mut serial = Counter::new(k, strand).unwrap();
                for record in records {
                    serial.add_record(record).unwrap();
                }
                for threads in [1, 3, 4] {
                    let mut parallel =
                        ParallelCounter::with_capacity(k, strand, threads, 7).unwrap();
                    for record in records {
                        parallel.add_record(record).unwrap();
                    }
                    assert_eq!(
                        table(&parallel.finish().unwrap()),
                        table(&serial),
                        "k={k}, strand={strand:?}, threads={threads}"
                    );
                }
            }
        }
    }

    #[test]
    fn tiny_records_are_batched_without_crossing_record_boundaries() {
        let mut parallel = ParallelCounter::new(3, StrandMode::Canonical, 2).unwrap();
        for _ in 0..100_000 {
            parallel.add_record(b"ACG").unwrap();
            parallel.add_record(b"CGT").unwrap();
            parallel.add_record(b"AC").unwrap();
        }
        assert_eq!(
            table(&parallel.finish().unwrap()),
            BTreeMap::from([(b"ACG".to_vec(), 200_000)])
        );
    }

    #[test]
    fn overlap_larger_than_batch_and_empty_input_are_supported() {
        let k = BATCH_BASES + 1;
        let mut parallel = ParallelCounter::new(k, StrandMode::Forward, 2).unwrap();
        parallel.add_record(&vec![b'A'; k + 1]).unwrap();
        assert_eq!(
            table(&parallel.finish().unwrap()),
            BTreeMap::from([(vec![b'A'; k], 2)])
        );
        assert!(
            ParallelCounter::new(31, StrandMode::Canonical, 3)
                .unwrap()
                .finish()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn checked_merge_matches_serial_and_rejects_incompatible_counters() {
        for k in [1, 32, 33, 64, 65] {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let sequence = b"ACGTTGCAAGTC".repeat(20);
                let mut left = Counter::new(k, strand).unwrap();
                let mut right = Counter::new(k, strand).unwrap();
                let mut expected = Counter::new(k, strand).unwrap();
                for record in [sequence.as_slice(), &[b'A'; 100]] {
                    left.add_record(record).unwrap();
                    expected.add_record(record).unwrap();
                }
                right.add_record(&sequence).unwrap();
                expected.add_record(&sequence).unwrap();
                left.merge(right).unwrap();
                assert_eq!(table(&left), table(&expected));

                let mut overflowing = Counter::new(k, strand).unwrap();
                overflowing.add_record(&vec![b'A'; k]).unwrap();
                match &mut overflowing.counts {
                    Counts::Packed(map) => *map.values_mut().next().unwrap() = u64::MAX,
                    Counts::Packed128(map) => *map.values_mut().next().unwrap() = u64::MAX,
                    Counts::Bytes(map) => *map.values_mut().next().unwrap() = u64::MAX,
                }
                let before = table(&overflowing);
                let mut delta = Counter::new(k, strand).unwrap();
                delta.add_record(&vec![b'A'; k]).unwrap();
                assert_eq!(overflowing.merge(delta), Err(Error::CountOverflow));
                assert_eq!(table(&overflowing), before);
            }
        }
        let mut counter = Counter::new(3, StrandMode::Forward).unwrap();
        for (k, strand) in [(4, StrandMode::Forward), (3, StrandMode::Canonical)] {
            assert_eq!(
                counter.merge(Counter::new(k, strand).unwrap()),
                Err(Error::IncompatibleCounters)
            );
        }
    }

    #[test]
    fn validates_configuration_and_joins_workers_on_failure_or_drop() {
        assert!(matches!(
            ParallelCounter::new(0, StrandMode::Forward, 2),
            Err(Error::InvalidK)
        ));
        for threads in [0, usize::MAX] {
            assert!(matches!(
                ParallelCounter::new(3, StrandMode::Forward, threads),
                Err(Error::InvalidThreads)
            ));
        }
        let mut failed = ParallelCounter::new(3, StrandMode::Forward, 2).unwrap();
        failed
            .workers
            .push(thread::spawn(|| Err(Error::CountOverflow)));
        assert!(matches!(failed.finish(), Err(Error::CountOverflow)));

        let mut dropped = ParallelCounter::new(31, StrandMode::Canonical, 2).unwrap();
        dropped.add_record(&vec![b'A'; BATCH_BASES * 10]).unwrap();
        drop(dropped);

        let mut disconnected = ParallelCounter::new(31, StrandMode::Canonical, 2).unwrap();
        let (sender, receiver) = channel::bounded(1);
        drop(receiver);
        disconnected.sender = Some(sender);
        assert_eq!(
            disconnected.add_record(&vec![b'A'; BATCH_BASES + 31]),
            Err(Error::WorkerFailed)
        );
    }
}
