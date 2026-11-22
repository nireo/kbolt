//! Output in k-mer order.
//!
//! Counts are held by hash, so their k-mers come out in no order that means
//! anything to a reader. Putting them in k-mer order takes one more pass
//! over the distinct k-mers: each is dropped into a bucket chosen by its
//! leading bases, and then the buckets are sorted and written one after the
//! other. A bucket is small enough to sort in a core's cache whatever the
//! size of the whole, so the work grows in proportion to the number of
//! k-mers.
//!
//! In a bucket a k-mer and its count share one 64-bit word: the bases that
//! chose the bucket need not be kept, and the bits they leave hold the
//! count. Sorting the words sorts the k-mers. The rare count too large for
//! those bits is kept beside the bucket.
//!
//! A counter in memory is taken apart as it is sorted: each of its tables is
//! freed once its k-mers are in buckets, so the two together never take more
//! than the tables did. With a memory limit the buckets wait in a temporary
//! file instead, eight bytes per distinct k-mer, of which only the ones
//! being sorted are in memory.

use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

use crate::database::{self, KeyOrder, SectionWriter};
use crate::disk::Spill;
use crate::table::{self, CompactTable, PackedCounts};
use crate::{
    Counter, Counts, decode_packed128_into, format_count, write_ordered, write_packed_tsv,
    write_packed64_tsv,
};

/// Bytes of a record in a bucket.
const WORD: usize = 8;
/// Records collected before they are grouped and handed to their buckets
/// (1 MiB): few enough to group in cache, and enough to give a bucket more
/// than a handful at a time.
const BATCH: usize = 1 << 17;
/// Records a bucket should hold when everything is in memory (1 MiB): what
/// sorts in a core's cache.
const BUCKET_RECORDS: u64 = 1 << 17;
/// The most buckets, as bits of the key that choose one.
const MAX_BUCKET_BITS: u32 = 16;
/// Bits a record should have for its count where the key leaves a choice:
/// counts below 255 then need nothing more.
const COUNT_BITS: u32 = 8;
/// Bytes of a bucket collected before they are moved on.
const MIN_BLOCK: usize = 4 * 1024;
const MAX_BLOCK: usize = 256 * 1024;

fn poisoned<T>(_: T) -> io::Error {
    io::Error::other("a sorting thread failed")
}

/// The bytes of `words`, as they are kept in the temporary file.
fn bytes(words: &[u64]) -> &[u8] {
    // SAFETY: words are eight initialized bytes each.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), std::mem::size_of_val(words)) }
}

fn bytes_mut(words: &mut [u64]) -> &mut [u8] {
    // SAFETY: as above, and every byte pattern is a valid word.
    unsafe {
        std::slice::from_raw_parts_mut(words.as_mut_ptr().cast(), std::mem::size_of_val(words))
    }
}

/// One bucket's records, in no order.
#[derive(Default)]
struct Bucket {
    /// The latest records.
    open: Vec<u64>,
    /// Earlier ones in memory, or in the file: offset and number.
    held: Vec<Vec<u64>>,
    written: Vec<(u64, u32)>,
    records: u64,
    /// The low key bits and counts of records whose count does not fit.
    large: Vec<[u64; 2]>,
}

/// Records on their way to a sorter, and the space to group them by bucket.
#[derive(Default)]
pub(crate) struct Batch {
    /// Each record's bucket, and the record.
    buckets: Vec<u32>,
    words: Vec<u64>,
    grouped: Vec<u64>,
    /// Where each bucket's records start in `grouped` (then, while they are
    /// moved there, where its next one goes).
    starts: Vec<usize>,
    /// Records with a count that does not fit, by bucket.
    large: Vec<(u32, [u64; 2])>,
}

/// Collects `(key, count)` records of packed keys, and returns them a bucket
/// of leading key bits at a time, each sorted: together, in key order.
///
/// A record is `low key bits << count_bits | count`. A count that does not
/// fit is replaced by the largest value of the field, and kept with the
/// key's low bits beside the bucket.
pub(crate) struct Sorter {
    /// Bits of a key below those that choose its bucket, and the bits left
    /// for the count.
    low_bits: u32,
    count_bits: u32,
    buckets: Vec<Mutex<Bucket>>,
    /// Records at which a bucket's open ones are moved on.
    block: usize,
    /// Where they go, unless they stay in memory.
    spill: Option<Spill>,
    /// What decides the order in which buckets are sorted at once.
    admission: Option<Admission>,
    /// Buffers of buckets that have been written, to sort the next in.
    spare: Mutex<Vec<(Vec<u64>, Vec<u64>)>>,
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
        let bucket = (sorting / 8 / threads.max(1) as u64 / WORD as u64).max(1);
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
        // The bits that choose a bucket are what a record has for its count
        // beyond the unused bits of a key: enough of them, but not the
        // whole key.
        let fewest = (COUNT_BITS + key_bits).saturating_sub(64);
        let bits = buckets
            .max(1)
            .next_power_of_two()
            .trailing_zeros()
            .clamp(fewest, MAX_BUCKET_BITS.max(fewest))
            .min(key_bits - 1);
        let low_bits = key_bits - bits;
        Self {
            low_bits,
            count_bits: 64 - low_bits,
            buckets: (0..1_usize << bits).map(|_| Mutex::default()).collect(),
            block: block / WORD,
            spill,
            admission,
            spare: Mutex::default(),
        }
    }

    /// The value of a record's count field that stands for a larger count.
    #[inline(always)]
    fn marker(&self) -> u64 {
        u64::MAX >> self.low_bits
    }

    pub(crate) fn buckets(&self) -> usize {
        self.buckets.len()
    }

    /// Add a record to `batch`, handing the batch on when it is full.
    #[inline(always)]
    pub(crate) fn push(&self, batch: &mut Batch, key: u64, count: u64) -> io::Result<()> {
        let bucket = (key >> self.low_bits) as u32;
        let low = key & (u64::MAX >> self.count_bits);
        let field = if count < self.marker() {
            count
        } else {
            batch.large.push((bucket, [low, count]));
            self.marker()
        };
        batch.buckets.push(bucket);
        batch.words.push((low << self.count_bits) | field);
        if batch.words.len() >= BATCH {
            self.flush(batch)?;
        }
        Ok(())
    }

    /// Hand the records of `batch` to their buckets, leaving it empty.
    pub(crate) fn flush(&self, batch: &mut Batch) -> io::Result<()> {
        let Batch {
            buckets,
            words,
            grouped,
            starts,
            large,
        } = batch;
        // Group the records by bucket, a counting sort, so that each bucket
        // is locked once for all of its records.
        starts.clear();
        starts.resize(self.buckets.len() + 1, 0);
        for &bucket in buckets.iter() {
            starts[bucket as usize + 1] += 1;
        }
        let mut start = 0;
        for next in starts.iter_mut() {
            start += std::mem::replace(next, start);
        }
        grouped.clear();
        grouped.resize(words.len(), 0);
        for (&bucket, &word) in buckets.iter().zip(words.iter()) {
            // (Entry `bucket + 1` holds the bucket's start until its records
            // are placed, and its end, the next one's start, after.)
            let next = &mut starts[bucket as usize + 1];
            grouped[*next] = word;
            *next += 1;
        }
        buckets.clear();
        words.clear();
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
                    for part in bucket.open.chunks(1 << 27) {
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
        for (bucket, entry) in large.drain(..) {
            let mut bucket = self.buckets[bucket as usize].lock().map_err(poisoned)?;
            bucket.large.push(entry);
        }
        Ok(())
    }

    /// Empty `bucket` into `words`, sorted, and return its large counts,
    /// sorted as well. `scratch` is space to sort in.
    fn take(
        &self,
        bucket: usize,
        words: &mut Vec<u64>,
        scratch: &mut Vec<u64>,
    ) -> io::Result<Vec<[u64; 2]>> {
        let mut bucket = std::mem::take(&mut *self.buckets[bucket].lock().map_err(poisoned)?);
        words.clear();
        words.reserve(bucket.records as usize);
        for block in &bucket.held {
            words.extend_from_slice(block);
        }
        for &(offset, length) in &bucket.written {
            let start = words.len();
            words.resize(start + length as usize, 0);
            let spill = self.spill.as_ref().expect("written blocks have a file");
            spill.read(bytes_mut(&mut words[start..]), offset)?;
        }
        words.extend_from_slice(&bucket.open);
        // The key's low bits are a word's high ones, and keys are distinct.
        sort_words(words, scratch);
        bucket.large.sort_unstable();
        Ok(bucket.large)
    }

    /// Hand every entry of `counts` to the sorter, from the calling thread.
    pub(crate) fn feed(&self, counts: &PackedCounts, batch: &mut Batch) -> io::Result<()> {
        for (key, count) in counts.entries() {
            self.push(batch, key, count)?;
        }
        self.flush(batch)
    }

    /// Hand every entry of `counts` to the sorter on `threads` threads,
    /// freeing each table as soon as its entries are in buckets: the
    /// buckets grow no faster than the tables shrink.
    fn consume(&self, counts: PackedCounts, threads: usize) -> io::Result<()> {
        let (mixer, layout) = (counts.mixer(), counts.layout());
        let tables: Vec<Mutex<Option<CompactTable>>> = counts
            .into_tables()
            .into_iter()
            .map(|table| Mutex::new(Some(table)))
            .collect();
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads.clamp(1, tables.len()))
                .map(|_| {
                    scope.spawn(|| -> io::Result<()> {
                        let mut batch = Batch::default();
                        loop {
                            let partition = next.fetch_add(1, Ordering::Relaxed);
                            let Some(table) = tables.get(partition) else {
                                return self.flush(&mut batch);
                            };
                            let Some(table) = table.lock().map_err(poisoned)?.take() else {
                                continue;
                            };
                            let base = (partition as u64) << layout.remainder_bits;
                            for (remainder, count) in table.iter() {
                                self.push(&mut batch, mixer.unmix(base | remainder), count)?;
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
    /// order, as `encode` makes of its entries.
    pub(crate) fn write(
        &self,
        threads: usize,
        writer: &mut dyn Write,
        encode: impl Fn(Sorted<'_>, &mut dyn Write) -> io::Result<()> + Sync,
    ) -> io::Result<()> {
        write_ordered(threads, self.buckets(), writer, |bucket, writer| {
            let result = (|| {
                // A bucket is sorted from one buffer into another.
                let bytes =
                    2 * self.buckets[bucket].lock().map_err(poisoned)?.records * WORD as u64;
                let _admitted = match &self.admission {
                    Some(admission) => Some(admission.enter(bucket, bytes)?),
                    None => None,
                };
                let (mut words, mut scratch) = self
                    .spare
                    .lock()
                    .map_err(poisoned)?
                    .pop()
                    .unwrap_or_default();
                let large = self.take(bucket, &mut words, &mut scratch)?;
                let entries = Sorted {
                    words: words.iter(),
                    large: large.iter(),
                    base: (bucket as u64) << self.low_bits,
                    count_bits: self.count_bits,
                    marker: self.marker(),
                };
                let result = encode(entries, writer);
                self.spare.lock().map_err(poisoned)?.push((words, scratch));
                result
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

/// Sort the distinct words of a bucket, with `scratch` as space to work in.
///
/// Two counting sorts on the words' top bits, eleven at a time, leave them
/// ordered by their top 22 bits. In a bucket those are the bases that
/// follow the ones its k-mers share, so few words agree in them, and the
/// ones that do are sorted among themselves.
fn sort_words(words: &mut [u64], scratch: &mut Vec<u64>) {
    const DIGIT: u32 = 11;
    const VALUES: usize = 1 << DIGIT;
    // Not worth two tables of counts.
    if words.len() < VALUES {
        return words.sort_unstable();
    }
    let digits = |word: u64| {
        (
            (word >> (64 - 2 * DIGIT)) as usize % VALUES,
            (word >> (64 - DIGIT)) as usize,
        )
    };
    let (mut lower, mut upper) = ([0_usize; VALUES], [0_usize; VALUES]);
    for &word in words.iter() {
        let (low, high) = digits(word);
        lower[low] += 1;
        upper[high] += 1;
    }
    // Counts become the places where each value's words start.
    for counts in [&mut lower, &mut upper] {
        let mut start = 0;
        for count in counts.iter_mut() {
            start += std::mem::replace(count, start);
        }
    }
    scratch.clear();
    scratch.resize(words.len(), 0);
    for &word in words.iter() {
        let next = &mut lower[digits(word).0];
        scratch[*next] = word;
        *next += 1;
    }
    for &word in scratch.iter() {
        let next = &mut upper[digits(word).1];
        words[*next] = word;
        *next += 1;
    }
    let prefix = |word: u64| word >> (64 - 2 * DIGIT);
    let mut start = 0;
    while start < words.len() {
        let mut end = start + 1;
        while end < words.len() && prefix(words[end]) == prefix(words[start]) {
            end += 1;
        }
        if end - start > 1 {
            words[start..end].sort_unstable();
        }
        start = end;
    }
}

/// The `(key, count)` entries of a sorted bucket, in key order.
pub(crate) struct Sorted<'a> {
    words: std::slice::Iter<'a, u64>,
    /// The counts that did not fit their records, in the same order.
    large: std::slice::Iter<'a, [u64; 2]>,
    /// The key bits shared by the bucket.
    base: u64,
    count_bits: u32,
    marker: u64,
}

impl Iterator for Sorted<'_> {
    type Item = (u64, u64);

    #[inline(always)]
    fn next(&mut self) -> Option<(u64, u64)> {
        let word = *self.words.next()?;
        let (low, field) = (word >> self.count_bits, word & self.marker);
        let count = if field == self.marker {
            let &[key, count] = self.large.next().expect("a large count for every marker");
            debug_assert_eq!(key, low);
            count
        } else {
            field
        };
        Some((self.base | low, count))
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

/// Write the entries of packed keys, sorted, as rows of text.
pub(crate) fn tsv_rows(k: usize) -> impl Fn(Sorted<'_>, &mut dyn Write) -> io::Result<()> + Sync {
    move |entries, writer| write_packed64_tsv(entries, k, writer)
}

/// Write the entries of packed keys, sorted, as sections of a database in
/// k-mer order.
pub(crate) fn binary_sections(entries: Sorted<'_>, writer: &mut dyn Write) -> io::Result<()> {
    let mut sections = SectionWriter::new(writer);
    for (key, count) in entries {
        sections.ascending(key, count)?;
    }
    sections.finish()
}

impl Counter {
    /// Write headerless `KMER<TAB>COUNT` rows in ascending order of the
    /// k-mers (A < C < G < T).
    ///
    /// The counter is consumed: for k <= 32 its tables are freed as their
    /// k-mers are sorted, so that sorting takes no more memory than the
    /// counter did. Does not flush the writer; write failures may leave
    /// partial output.
    pub fn write_tsv_sorted(self, mut writer: impl Write) -> io::Result<()> {
        self.write_sorted(&mut writer, false)
    }

    /// Write a database (see [`crate::database`]) in ascending order of the
    /// k-mers; otherwise as [`Counter::write_tsv_sorted`].
    pub fn write_binary_sorted(self, mut writer: impl Write) -> io::Result<()> {
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

    fn write_sorted(self, writer: &mut dyn Write, binary: bool) -> io::Result<()> {
        let Counter {
            counts,
            partitions,
            k,
            output_threads: threads,
            ..
        } = self;
        let key_bytes = k.div_ceil(4);
        let tables = std::iter::once(counts).chain(partitions);
        if k <= 32 {
            let Some(Counts::Packed(counts)) = tables.into_iter().next() else {
                unreachable!("keys of at most 32 bases are packed")
            };
            let sorter = Sorter::in_memory(table::key_bits(k), counts.len() as u64);
            sorter.consume(counts, threads)?;
            return if binary {
                sorter.write(threads, writer, binary_sections)
            } else {
                sorter.write(threads, writer, tsv_rows(k))
            };
        }
        // Wider keys are sorted in one piece, taken out of their maps.
        if k <= 64 {
            let mut entries: Vec<((u64, u64), u64)> = Vec::new();
            for table in tables {
                let Counts::Packed128(counts) = table else {
                    unreachable!("every table holds keys of the same width")
                };
                entries.extend(counts);
            }
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
        let mut entries: Vec<(Vec<u8>, u64)> = Vec::new();
        for table in tables {
            let Counts::Bytes(counts) = table else {
                unreachable!("every table holds keys of the same width")
            };
            entries.extend(counts);
        }
        entries.sort_unstable();
        if !binary {
            let mut suffix = [0; 22];
            for (key, count) in entries {
                writer.write_all(&key)?;
                writer.write_all(format_count(count, &mut suffix))?;
            }
            return Ok(());
        }
        let mut sections = SectionWriter::new(writer);
        let mut packed = vec![0; key_bytes];
        for (key, count) in entries {
            database::pack_bases(&key, &mut packed);
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

    /// Everything a sorter writes, as `(key, count)` pairs.
    fn written(sorter: &Sorter) -> Vec<(u64, u64)> {
        let mut output = Vec::new();
        sorter
            .write(3, &mut output, |entries, writer| {
                for (key, count) in entries {
                    writer.write_all(&key.to_le_bytes())?;
                    writer.write_all(&count.to_le_bytes())?;
                }
                Ok(())
            })
            .unwrap();
        output
            .chunks_exact(16)
            .map(|record| {
                (
                    u64::from_le_bytes(record[..8].try_into().unwrap()),
                    u64::from_le_bytes(record[8..].try_into().unwrap()),
                )
            })
            .collect()
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
            // Mostly small counts, and some of every size up to the largest.
            let expected: Vec<(u64, u64)> = keys
                .iter()
                .map(|&key| match key % 23 {
                    0 => (key, u64::MAX >> (key % 64)),
                    1 => (key, 254 + key % 3),
                    _ => (key, key % 200 + 1),
                })
                .collect();
            let sorters = [
                Sorter::in_memory(key_bits, expected.len() as u64),
                // Many buckets and small blocks: nearly everything is written.
                Sorter::on_disk(key_bits, expected.len() as u64, 1 << 20, 3, &directory).unwrap(),
                // A limit so small that it allows the fewest buckets.
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
                assert!(
                    written(sorter) == expected,
                    "key_bits={key_bits} sorter={index}"
                );
                if index == 1 && records > 1000 {
                    assert!(sorter.buckets() >= 32);
                }
            }
        }
    }

    #[test]
    fn words_sort_whether_spread_out_or_alike_in_their_top_bits() {
        let mut random = Random(17);
        let mut scratch = Vec::new();
        for (length, mask) in [
            (0, u64::MAX),
            (1, u64::MAX),
            (2047, u64::MAX),
            (2048, u64::MAX),
            (100_000, u64::MAX),
            // All alike in their top 22 bits, or in all but a few of them.
            (50_000, u64::MAX >> 22),
            (50_000, u64::MAX >> 19),
            (50_000, 0xffff),
        ] {
            let mut words: Vec<u64> = (0..length).map(|_| random.next() & mask).collect();
            let mut expected = words.clone();
            expected.sort_unstable();
            sort_words(&mut words, &mut scratch);
            assert!(words == expected, "length={length} mask={mask:x}");
        }
    }

    #[test]
    fn records_leave_room_for_a_count_at_every_key_length() {
        for key_bits in (2..=64).step_by(2) {
            for records in [0, 1, 1 << 17, 1 << 30, u64::MAX] {
                let sorter = Sorter::in_memory(key_bits, records);
                // A bucket per value of the key's high bits, at least one
                // low bit, and a count field of two bits or more: a small
                // count, and the marker.
                assert_eq!(sorter.buckets(), 1 << (key_bits - sorter.low_bits));
                assert!((1..=62).contains(&sorter.low_bits), "key_bits={key_bits}");
                assert_eq!(sorter.count_bits + sorter.low_bits, 64);
                // Keys at both ends, with counts around the marker.
                let last = u64::MAX >> (64 - key_bits);
                let marker = sorter.marker();
                let expected = [
                    (0, marker - 1),
                    (1, marker),
                    (last - 1, 1),
                    (last, marker + 1),
                ];
                let mut batch = Batch::default();
                for &(key, count) in expected.iter().rev() {
                    sorter.push(&mut batch, key, count).unwrap();
                }
                sorter.flush(&mut batch).unwrap();
                if key_bits > 2 {
                    assert!(written(&sorter) == expected, "key_bits={key_bits}");
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
        let result = sorter.write(4, &mut Vec::new(), |mut entries, writer| {
            let (key, _) = entries.next().unwrap();
            if key >> sorter.low_bits == 2 {
                return Err(io::Error::other("injected"));
            }
            writer.write_all(&key.to_le_bytes())
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
                // Sorting consumes a counter: one of each kind per output.
                let counter = |parallel: bool| {
                    if parallel {
                        let mut counter = ParallelCounter::new(k, strand, 3).unwrap();
                        counter.add_record(&sequence).unwrap();
                        counter.finish().unwrap()
                    } else {
                        let mut counter = Counter::new(k, strand).unwrap();
                        counter.add_record(&sequence).unwrap();
                        counter
                    }
                };
                let mut expected: Vec<(Vec<u8>, u64)> = counter(false).counts().collect();
                expected.sort();
                assert!(expected.len() > 1 || k == 1);
                for parallel in [false, true] {
                    let mut tsv = Vec::new();
                    counter(parallel).write_tsv_sorted(&mut tsv).unwrap();
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
                    counter(parallel).write_binary_sorted(&mut binary).unwrap();
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
        let empty = || Counter::new(31, StrandMode::Canonical).unwrap();
        let (mut tsv, mut binary) = (Vec::new(), Vec::new());
        empty().write_tsv_sorted(&mut tsv).unwrap();
        empty().write_binary_sorted(&mut binary).unwrap();
        assert!(tsv.is_empty());
        assert!(Counter::read_binary(binary.as_slice()).unwrap().is_empty());
    }
}
