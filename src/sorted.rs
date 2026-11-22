//! Output in k-mer order.
//!
//! Counts are held by hash, so their k-mers come out in no order that means
//! anything to a reader. Putting them in k-mer order takes one more pass
//! over the distinct k-mers: each is dropped into a bucket chosen by its
//! leading bases, and then the buckets are sorted and written one after the
//! other. A bucket is small enough to sort in memory whatever the size of
//! the whole, so the work grows in proportion to the number of k-mers, and
//! with a memory limit the buckets wait in a temporary file, sixteen bytes
//! per distinct k-mer, of which only the ones being sorted are in memory.
//!
//! Without a memory limit the buckets are a second copy of the counts in
//! memory.

use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

use crate::database::{self, KeyOrder, SectionWriter};
use crate::disk::Spill;
use crate::table::{self, PackedCounts};
use crate::{
    Counter, Counts, decode_packed128_into, format_count, write_ordered, write_packed_tsv,
    write_packed64_tsv,
};

/// Bytes of a record in a bucket: the key and its count.
const RECORD: usize = 16;
/// Records collected before they are sorted and handed to their buckets
/// (2 MiB), few enough to sort in cache and enough to give a bucket more
/// than a handful at a time.
const BATCH: usize = 1 << 17;
/// Records a bucket should hold when everything is in memory: what sorts in
/// a core's cache.
const BUCKET_RECORDS: u64 = 1 << 17;
/// The most buckets, as bits of the key that choose one.
const MAX_BUCKET_BITS: u32 = 16;
/// Bytes of a bucket collected before they are written to the file.
const MIN_BLOCK: usize = 4 * 1024;
const MAX_BLOCK: usize = 256 * 1024;

fn poisoned<T>(_: T) -> io::Error {
    io::Error::other("a sorting thread failed")
}

/// A key and its count.
pub(crate) type Record = [u64; 2];

/// The bytes of `records`, as they are kept in the temporary file.
fn bytes(records: &[Record]) -> &[u8] {
    // SAFETY: a record is sixteen bytes without padding, all initialized.
    unsafe { std::slice::from_raw_parts(records.as_ptr().cast(), std::mem::size_of_val(records)) }
}

fn bytes_mut(records: &mut [Record]) -> &mut [u8] {
    // SAFETY: as above, and every byte pattern is a valid record.
    unsafe {
        std::slice::from_raw_parts_mut(records.as_mut_ptr().cast(), std::mem::size_of_val(records))
    }
}

/// One bucket's records, in no order.
#[derive(Default)]
struct Bucket {
    /// The latest records.
    open: Vec<Record>,
    /// Earlier ones in memory, or in the file: offset and number.
    held: Vec<Vec<Record>>,
    written: Vec<(u64, u32)>,
    records: u64,
}

/// Records on their way to a sorter, and the space to group them by bucket.
#[derive(Default)]
pub(crate) struct Batch {
    records: Vec<Record>,
    grouped: Vec<Record>,
    /// Where each bucket's records start in `grouped` (then, while they are
    /// moved there, where its next one goes).
    starts: Vec<usize>,
}

/// Collects `(key, count)` records of packed keys, and returns them a bucket
/// of leading key bits at a time, each sorted: together, in key order.
pub(crate) struct Sorter {
    /// A key's bucket is its bits from this one up.
    shift: u32,
    buckets: Vec<Mutex<Bucket>>,
    /// Records at which a bucket's open ones are moved on.
    block: usize,
    /// Where they go, unless they stay in memory.
    spill: Option<Spill>,
    /// What decides the order in which buckets are sorted at once.
    admission: Option<Admission>,
}

impl Sorter {
    /// A sorter that keeps `records` records of `key_bits` bits in memory.
    pub(crate) fn in_memory(key_bits: u32, records: u64) -> Self {
        Self::new(
            key_bits,
            records.div_ceil(BUCKET_RECORDS),
            MAX_BLOCK,
            None,
            None,
        )
    }

    /// A sorter that keeps its records in a temporary file under
    /// `directory`, and sorts within about `memory` bytes on `threads`
    /// threads. `records` is the most it will be given.
    pub(crate) fn on_disk(
        key_bits: u32,
        records: u64,
        memory: u64,
        threads: usize,
        directory: &Path,
    ) -> io::Result<Self> {
        // Half the memory holds the buckets being sorted; one should fit
        // an eighth of it per thread even if it is several times the
        // average. An eighth holds the records not yet written.
        let sorting = memory / 2;
        let bucket = (sorting / 8 / threads.max(1) as u64 / RECORD as u64).max(1);
        let most = ((memory / 8 / MIN_BLOCK as u64).max(1) + 1).next_power_of_two() / 2;
        let buckets = records.div_ceil(bucket).min(most);
        let block = usize::try_from(memory / 8 / buckets.max(1).next_power_of_two())
            .unwrap_or(MAX_BLOCK)
            .clamp(MIN_BLOCK, MAX_BLOCK);
        Ok(Self::new(
            key_bits,
            buckets,
            block,
            Some(Spill::create(directory)?),
            Some(Admission::new(sorting.max(1))),
        ))
    }

    /// A sorter with at least `buckets` buckets, if the key has the bits,
    /// that moves a bucket's records on every `block` bytes.
    fn new(
        key_bits: u32,
        buckets: u64,
        block: usize,
        spill: Option<Spill>,
        admission: Option<Admission>,
    ) -> Self {
        let bits = buckets
            .max(1)
            .next_power_of_two()
            .trailing_zeros()
            .min(MAX_BUCKET_BITS)
            .min(key_bits);
        Self {
            shift: key_bits - bits,
            buckets: (0..1_usize << bits).map(|_| Mutex::default()).collect(),
            block: block / RECORD,
            spill,
            admission,
        }
    }

    #[inline(always)]
    fn bucket(&self, key: u64) -> usize {
        // (A shift by all 64 bits, with a single bucket, is zero.)
        key.checked_shr(self.shift).unwrap_or(0) as usize
    }

    pub(crate) fn buckets(&self) -> usize {
        self.buckets.len()
    }

    /// Add a record to `batch`, handing the batch on when it is full.
    #[inline(always)]
    pub(crate) fn push(&self, batch: &mut Batch, key: u64, count: u64) -> io::Result<()> {
        batch.records.push([key, count]);
        if batch.records.len() >= BATCH {
            self.flush(batch)?;
        }
        Ok(())
    }

    /// Hand the records of `batch` to their buckets, leaving it empty.
    pub(crate) fn flush(&self, batch: &mut Batch) -> io::Result<()> {
        let Batch {
            records,
            grouped,
            starts,
        } = batch;
        // Group the records by bucket, a counting sort, so that each bucket
        // is locked once for all of its records.
        starts.clear();
        starts.resize(self.buckets.len() + 1, 0);
        for record in records.iter() {
            starts[self.bucket(record[0]) + 1] += 1;
        }
        let mut start = 0;
        for next in starts.iter_mut() {
            start += std::mem::replace(next, start);
        }
        grouped.clear();
        grouped.resize(records.len(), [0; 2]);
        for &record in records.iter() {
            // (Entry `bucket + 1` holds the bucket's start until its records
            // are placed, and its end, the next one's start, after.)
            let next = &mut starts[self.bucket(record[0]) + 1];
            grouped[*next] = record;
            *next += 1;
        }
        records.clear();
        for (bucket, bounds) in self.buckets.iter().zip(starts.windows(2)) {
            let run = &grouped[bounds[0]..bounds[1]];
            if run.is_empty() {
                continue;
            }
            let mut bucket = bucket.lock().map_err(poisoned)?;
            let bucket = &mut *bucket;
            bucket.records += run.len() as u64;
            bucket.open.extend_from_slice(run);
            if bucket.open.len() < self.block {
                continue;
            }
            match &self.spill {
                Some(spill) => {
                    // (A number of records fits 32 bits: split more.)
                    for part in bucket.open.chunks(1 << 26) {
                        let offset = spill.append(bytes(part))?;
                        bucket.written.push((offset, part.len() as u32));
                    }
                    bucket.open.clear();
                }
                None => {
                    let block = std::mem::take(&mut bucket.open);
                    bucket.held.push(block);
                }
            }
        }
        Ok(())
    }

    /// Empty `bucket` into `records`, sorted by key.
    fn take(&self, bucket: usize, records: &mut Vec<Record>) -> io::Result<()> {
        let bucket = std::mem::take(&mut *self.buckets[bucket].lock().map_err(poisoned)?);
        records.clear();
        records.reserve(bucket.records as usize);
        for block in &bucket.held {
            records.extend_from_slice(block);
        }
        for &(offset, length) in &bucket.written {
            let start = records.len();
            records.resize(start + length as usize, [0; 2]);
            let spill = self.spill.as_ref().expect("written blocks have a file");
            spill.read(bytes_mut(&mut records[start..]), offset)?;
        }
        records.extend_from_slice(&bucket.open);
        records.sort_unstable_by_key(|record| record[0]);
        Ok(())
    }

    /// Hand every entry of `counts` to the sorter, from the calling thread.
    pub(crate) fn feed(&self, counts: &PackedCounts, batch: &mut Batch) -> io::Result<()> {
        for (key, count) in counts.entries() {
            self.push(batch, key, count)?;
        }
        self.flush(batch)
    }

    /// Hand every entry of `counts` to the sorter, on `threads` threads.
    fn feed_concurrently(&self, counts: &PackedCounts, threads: usize) -> io::Result<()> {
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads.clamp(1, counts.tables().len()))
                .map(|_| {
                    scope.spawn(|| -> io::Result<()> {
                        let mut batch = Batch::default();
                        loop {
                            let partition = next.fetch_add(1, Ordering::Relaxed);
                            if partition >= counts.tables().len() {
                                return self.flush(&mut batch);
                            }
                            for (key, count) in counts.partition_entries(partition) {
                                self.push(&mut batch, key, count)?;
                            }
                        }
                    })
                })
                .collect();
            workers
                .into_iter()
                .try_for_each(|worker| worker.join().map_err(poisoned)?)
        })
    }

    /// Sort the buckets on up to `threads` threads and write each, in key
    /// order, as `encode` makes of its records.
    pub(crate) fn write(
        &self,
        threads: usize,
        writer: &mut dyn Write,
        encode: impl Fn(&[Record], &mut dyn Write) -> io::Result<()> + Sync,
    ) -> io::Result<()> {
        write_ordered(threads, self.buckets(), writer, |bucket, writer| {
            let result = (|| {
                let bytes = self.buckets[bucket].lock().map_err(poisoned)?.records * RECORD as u64;
                let _admitted = match &self.admission {
                    Some(admission) => Some(admission.enter(bucket, bytes)?),
                    None => None,
                };
                let mut records = Vec::new();
                self.take(bucket, &mut records)?;
                encode(&records, writer)
            })();
            // A thread that stops here never takes its later buckets; the
            // threads waiting for their turn after those must stop as well.
            if let (Err(_), Some(admission)) = (&result, &self.admission) {
                admission.fail();
            }
            result
        })
    }
}

/// Lets buckets into memory in their order, as many at once as fit a limit.
///
/// The order matters: output is written in bucket order, so the bucket
/// whose turn it is to be written must never wait for memory held by later
/// ones, which in turn wait to be written.
struct Admission {
    total: u64,
    /// The next bucket to let in, the bytes free for it, and whether
    /// sorting has failed, so that no further bucket is let in.
    state: Mutex<(usize, u64, bool)>,
    changed: Condvar,
}

struct Admitted<'a>(&'a Admission, u64);

impl Admission {
    fn new(total: u64) -> Self {
        Self {
            total,
            state: Mutex::new((0, total, false)),
            changed: Condvar::new(),
        }
    }

    /// Wait for `bucket`'s turn and for `bytes` to be free; a bucket larger
    /// than the whole limit goes alone.
    fn enter(&self, bucket: usize, bytes: u64) -> io::Result<Admitted<'_>> {
        let bytes = bytes.min(self.total);
        let mut state = self.state.lock().map_err(poisoned)?;
        while !state.2 && (state.0 != bucket || state.1 < bytes) {
            state = self.changed.wait(state).map_err(poisoned)?;
        }
        if state.2 {
            return Err(io::Error::other("sorting stopped after an error"));
        }
        (state.0, state.1) = (bucket + 1, state.1 - bytes);
        self.changed.notify_all();
        Ok(Admitted(self, bytes))
    }

    /// Stop letting buckets in, and release the threads waiting to enter.
    fn fail(&self) {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .2 = true;
        self.changed.notify_all();
    }
}

impl Drop for Admitted<'_> {
    fn drop(&mut self) {
        self.0
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .1 += self.1;
        self.0.changed.notify_all();
    }
}

/// Write the records of packed keys, sorted, as rows of text.
pub(crate) fn tsv_rows(k: usize) -> impl Fn(&[Record], &mut dyn Write) -> io::Result<()> + Sync {
    move |records, writer| {
        write_packed64_tsv(records.iter().map(|&[key, count]| (key, count)), k, writer)
    }
}

/// Write the records of packed keys, sorted, as sections of a database in
/// k-mer order.
pub(crate) fn binary_sections(records: &[Record], writer: &mut dyn Write) -> io::Result<()> {
    let mut sections = SectionWriter::new(writer);
    for &[key, count] in records {
        sections.ascending(key, count)?;
    }
    sections.finish()
}

impl Counter {
    /// Write headerless `KMER<TAB>COUNT` rows in ascending order of the
    /// k-mers (A < C < G < T).
    ///
    /// The counts are sorted for this in a second copy, sixteen bytes per
    /// distinct k-mer for k <= 32. Does not flush the writer; write failures
    /// may leave partial output.
    pub fn write_tsv_sorted(&self, mut writer: impl Write) -> io::Result<()> {
        self.write_sorted(&mut writer, false)
    }

    /// Write a database (see [`crate::database`]) in ascending order of the
    /// k-mers; otherwise as [`Counter::write_tsv_sorted`].
    pub fn write_binary_sorted(&self, mut writer: impl Write) -> io::Result<()> {
        let entries = self.len() as u64;
        writer.write_all(&database::header(
            self.k,
            self.strand,
            KeyOrder::Kmer,
            true,
            entries,
        ))?;
        self.write_sorted(&mut writer, true)
    }

    fn write_sorted(&self, writer: &mut dyn Write, binary: bool) -> io::Result<()> {
        let k = self.k;
        if let Counts::Packed(counts) = &self.counts {
            let sorter = Sorter::in_memory(table::key_bits(k), counts.len() as u64);
            sorter.feed_concurrently(counts, self.output_threads)?;
            return if binary {
                sorter.write(self.output_threads, writer, binary_sections)
            } else {
                sorter.write(self.output_threads, writer, tsv_rows(k))
            };
        }
        // Wider keys are few enough in practice to sort in one piece.
        let key_bytes = k.div_ceil(4);
        if matches!(self.counts, Counts::Packed128(_)) {
            let mut entries: Vec<((u64, u64), u64)> = self
                .tables()
                .flat_map(|table| match table {
                    Counts::Packed128(counts) => counts.iter().map(|(&key, &count)| (key, count)),
                    _ => unreachable!("every table holds keys of the same width"),
                })
                .collect();
            entries.sort_unstable();
            if !binary {
                return write_packed_tsv::<64, _>(
                    entries.into_iter(),
                    k,
                    writer,
                    decode_packed128_into,
                );
            }
            let mut sections = SectionWriter::new(writer);
            for ((high, low), count) in entries {
                let key = (u128::from(high) << 64) | u128::from(low);
                sections.raw(&key.to_le_bytes()[..key_bytes], count)?;
            }
            return sections.finish();
        }
        let mut entries: Vec<(&[u8], u64)> = self
            .tables()
            .flat_map(|table| match table {
                Counts::Bytes(counts) => counts.iter().map(|(key, &count)| (key.as_slice(), count)),
                _ => unreachable!("every table holds keys of the same width"),
            })
            .collect();
        entries.sort_unstable();
        if !binary {
            let mut suffix = [0; 22];
            for (key, count) in entries {
                writer.write_all(key)?;
                writer.write_all(format_count(count, &mut suffix))?;
            }
            return Ok(());
        }
        let mut sections = SectionWriter::new(writer);
        let mut packed = vec![0; key_bytes];
        for (key, count) in entries {
            database::pack_bases(key, &mut packed);
            sections.raw(&packed, count)?;
        }
        sections.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::DatabaseReader;
    use crate::{ParallelCounter, StrandMode};

    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn sorter_returns_every_record_in_key_order_from_memory_and_from_disk() {
        let mut random = Random(11);
        let directory = std::env::temp_dir();
        for (key_bits, records) in [(2, 4), (10, 900), (62, 200_000), (64, 300_000)] {
            let mask = u64::MAX >> (64 - key_bits);
            // Distinct keys, as a counter's are.
            let mut keys: Vec<u64> = (0..records).map(|_| random.next() & mask).collect();
            keys.sort_unstable();
            keys.dedup();
            let expected: Vec<(u64, u64)> = keys.iter().map(|&key| (key, key % 1000 + 1)).collect();
            let sorters = [
                Sorter::in_memory(key_bits, expected.len() as u64),
                // Many buckets and small blocks: nearly everything is written.
                Sorter::on_disk(key_bits, expected.len() as u64, 1 << 20, 3, &directory).unwrap(),
                // A limit so small that it allows a single bucket.
                Sorter::on_disk(key_bits, expected.len() as u64, 4096, 3, &directory).unwrap(),
            ];
            for (index, sorter) in sorters.iter().enumerate() {
                // Fed in arbitrary order from several threads.
                let mut shuffled = expected.clone();
                for at in (1..shuffled.len()).rev() {
                    shuffled.swap(at, (random.next() % (at as u64 + 1)) as usize);
                }
                std::thread::scope(|scope| {
                    for part in shuffled.chunks(shuffled.len().div_ceil(3)) {
                        scope.spawn(move || {
                            let mut batch = Batch::default();
                            for (index, &(key, count)) in part.iter().enumerate() {
                                sorter.push(&mut batch, key, count).unwrap();
                                // Batches of every size, not only full ones.
                                if index % 5000 == 4999 {
                                    sorter.flush(&mut batch).unwrap();
                                }
                            }
                            sorter.flush(&mut batch).unwrap();
                        });
                    }
                });
                let mut output = Vec::new();
                sorter
                    .write(3, &mut output, |records, writer| {
                        records.iter().try_for_each(|&[key, count]| {
                            writer.write_all(&key.to_le_bytes())?;
                            writer.write_all(&count.to_le_bytes())
                        })
                    })
                    .unwrap();
                let actual: Vec<(u64, u64)> = output
                    .chunks_exact(16)
                    .map(|record| {
                        (
                            u64::from_le_bytes(record[..8].try_into().unwrap()),
                            u64::from_le_bytes(record[8..].try_into().unwrap()),
                        )
                    })
                    .collect();
                assert!(actual == expected, "key_bits={key_bits} sorter={index}");
                if index == 1 && records > 1000 {
                    assert!(sorter.buckets() > 1);
                }
            }
        }
    }

    #[test]
    fn a_failing_encoder_stops_the_others_waiting_for_their_turn() {
        let sorter = Sorter::on_disk(40, 100_000, 1 << 20, 4, &std::env::temp_dir()).unwrap();
        assert!(sorter.buckets() >= 16);
        let mut batch = Batch::default();
        for key in 0..100_000_u64 {
            sorter.push(&mut batch, key << 23, 1).unwrap();
        }
        sorter.flush(&mut batch).unwrap();
        // The third bucket fails; the thread that owns it never asks for its
        // later ones, behind which the other threads queue.
        let result = sorter.write(4, &mut Vec::new(), |records, writer| {
            if records[0][0] >> sorter.shift == 2 {
                return Err(io::Error::other("injected"));
            }
            writer.write_all(bytes(records))
        });
        assert!(result.is_err());
    }

    #[test]
    fn sorted_outputs_ascend_and_hold_the_same_counts_for_every_key_width() {
        let mut random = Random(3);
        // An invalid base now and then, far enough apart for long k-mers.
        let sequence: Vec<u8> = (0..30_000)
            .map(|_| match random.next() % 2000 {
                0 => b'N',
                value => b"ACGTacgt"[(value % 8) as usize],
            })
            .collect();
        for k in [1, 5, 31, 32, 33, 64, 65, 80] {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let mut serial = Counter::new(k, strand).unwrap();
                serial.add_record(&sequence).unwrap();
                let mut parallel = ParallelCounter::new(k, strand, 3).unwrap();
                parallel.add_record(&sequence).unwrap();
                let parallel = parallel.finish().unwrap();
                let mut expected: Vec<(Vec<u8>, u64)> = serial.counts().collect();
                expected.sort();
                assert!(expected.len() > 1 || k == 1);
                for counter in [&serial, &parallel] {
                    let mut tsv = Vec::new();
                    counter.write_tsv_sorted(&mut tsv).unwrap();
                    let rows: Vec<(Vec<u8>, u64)> = tsv
                        .split(|&byte| byte == b'\n')
                        .filter(|row| !row.is_empty())
                        .map(|row| {
                            let tab = row.iter().position(|&byte| byte == b'\t').unwrap();
                            let count = std::str::from_utf8(&row[tab + 1..]).unwrap();
                            (row[..tab].to_vec(), count.parse().unwrap())
                        })
                        .collect();
                    assert!(rows == expected, "k={k} {strand:?}");

                    let mut binary = Vec::new();
                    counter.write_binary_sorted(&mut binary).unwrap();
                    let mut reader = DatabaseReader::new(binary.as_slice()).unwrap();
                    let header = reader.header();
                    assert_eq!((header.order, header.ascending), (KeyOrder::Kmer, true));
                    let mut entries = Vec::new();
                    while let Some((key, count)) = reader.next_entry().unwrap() {
                        entries.push((key.to_vec(), count));
                    }
                    assert!(entries == expected, "k={k} {strand:?}");
                    let loaded = Counter::read_binary(binary.as_slice()).unwrap();
                    let mut loaded: Vec<_> = loaded.counts().collect();
                    loaded.sort();
                    assert!(loaded == expected, "k={k} {strand:?}");
                }
            }
        }
        // Nothing to sort is no rows, and a database of no entries.
        let empty = Counter::new(31, StrandMode::Canonical).unwrap();
        let (mut tsv, mut binary) = (Vec::new(), Vec::new());
        empty.write_tsv_sorted(&mut tsv).unwrap();
        empty.write_binary_sorted(&mut binary).unwrap();
        assert!(tsv.is_empty());
        assert!(Counter::read_binary(binary.as_slice()).unwrap().is_empty());
    }
}
