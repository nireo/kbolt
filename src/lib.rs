use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::io::{self, Write};

pub mod database;
mod parallel;
mod table;

pub use parallel::{ParallelCounter, ParallelInput, ParallelStats};
use table::PackedCounts;

// Store the high and low words separately to avoid u128 alignment padding in map entries.
type PackedCounts128 = HashMap<(u64, u64), u64, foldhash::fast::RandomState>;

/// Partition bits for packed (k <= 32) tables. Partitions bound growth pauses
/// and leave count bits in each compact slot; see `table`.
const PACKED_PARTITION_BITS: u32 = 10;

const DNA_BASES: [u8; 256] = {
    let mut codes = [u8::MAX; 256];
    codes[b'A' as usize] = 0;
    codes[b'a' as usize] = 0;
    codes[b'C' as usize] = 1;
    codes[b'c' as usize] = 1;
    codes[b'G' as usize] = 2;
    codes[b'g' as usize] = 2;
    codes[b'T' as usize] = 3;
    codes[b't' as usize] = 3;
    codes
};

/// Whether reverse-complement pairs share a count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrandMode {
    /// Use the lexicographically smaller of a k-mer and its reverse complement.
    Canonical,
    /// Count the observed sequence without combining reverse complements.
    Forward,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    InvalidK,
    CountOverflow,
    ReserveFailed,
    IncompatibleCounters,
    InvalidThreads,
    WorkerFailed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidK => f.write_str("k must be greater than zero"),
            Self::CountOverflow => f.write_str("k-mer count exceeds u64::MAX"),
            Self::ReserveFailed => f.write_str("unable to reserve k-mer table capacity"),
            Self::IncompatibleCounters => {
                f.write_str("counters must have the same k and strand mode")
            }
            Self::InvalidThreads => {
                f.write_str("worker count must be greater than zero and representable")
            }
            Self::WorkerFailed => f.write_str("unable to start or complete a counting worker"),
        }
    }
}

impl std::error::Error for Error {}

// Keep the representation private so wider packed keys can be added without
// changing the public output format.
enum Counts {
    /// k <= 32: compact, self-partitioned tables (never in `Counter::partitions`).
    Packed(PackedCounts),
    Packed128(PackedCounts128),
    Bytes(HashMap<Vec<u8>, u64>),
}

/// An exact, in-memory DNA k-mer counter.
///
/// Output keys are uppercase ASCII DNA sequences; counts accumulate across records.
/// Rolling 2-bit keys use 64 bits for k <= 32 and 128 bits for k <= 64.
/// Larger k uses byte-string storage, with no maximum other than practical memory limits.
/// For k <= 32, keys are stored in compact partitioned tables addressed by a seeded,
/// invertible mixing function; wider keys use FoldHash. Neither is intended to resist
/// adaptive collision attacks.
pub struct Counter {
    counts: Counts,
    // For wider keys from `ParallelCounter`, partition zero is `counts` and the
    // remaining partitions retain their own maps. Packed counts partition themselves.
    partitions: Vec<Counts>,
    partition_hash: Option<foldhash::fast::RandomState>,
    // Threads that may encode independent partitions during output. Results
    // from `ParallelCounter` inherit its worker count; other counters use one.
    output_threads: usize,
    strand: StrandMode,
    k: usize,
}

/// Bytes an output encoder buffers before handing a chunk to the writer.
const OUTPUT_CHUNK: usize = 1024 * 1024;

/// Messages from an output encoder: a chunk, the end of a table (`None`), or
/// the encoder's own error.
type EncodedChunk = io::Result<Option<Vec<u8>>>;

// An in-memory writer that streams fixed-size chunks to the output thread.
// The final chunk of a table is followed by `None`.
struct ChunkSink<'a> {
    buffer: Vec<u8>,
    sender: &'a crossbeam_channel::Sender<EncodedChunk>,
    recycled: &'a crossbeam_channel::Receiver<Vec<u8>>,
}

impl ChunkSink<'_> {
    fn send(&mut self) -> io::Result<()> {
        let replacement = self.recycled.try_recv().unwrap_or_default();
        let chunk = std::mem::replace(&mut self.buffer, replacement);
        // A disconnected receiver means the writer stopped after an error;
        // it reports that error itself.
        self.sender
            .send(Ok(Some(chunk)))
            .map_err(|_| io::Error::from(io::ErrorKind::BrokenPipe))
    }

    fn finish(mut self) -> io::Result<()> {
        if !self.buffer.is_empty() {
            self.send()?;
        }
        self.sender
            .send(Ok(None))
            .map_err(|_| io::Error::from(io::ErrorKind::BrokenPipe))
    }
}

/// One independently encodable piece of a counter's output.
#[derive(Clone, Copy)]
enum OutputTable<'a> {
    /// One partition of compact packed tables.
    Packed(&'a PackedCounts, usize),
    /// A whole map for wider keys.
    Map(&'a Counts),
}

impl Write for ChunkSink<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(bytes);
        if self.buffer.len() >= OUTPUT_CHUNK {
            self.send()?;
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Counter {
    /// Create an empty counter. Reject k = 0 with `Error::InvalidK`.
    pub fn new(k: usize, strand: StrandMode) -> Result<Self, Error> {
        Self::with_capacity(k, strand, 0)
    }

    /// Create an empty counter with space for at least `capacity` distinct k-mers.
    ///
    /// This is an allocation hint, not a limit; the table grows as needed.
    /// Reject k = 0 with `Error::InvalidK`. Return `Error::ReserveFailed` if the
    /// requested capacity overflows or the allocation fails.
    pub fn with_capacity(k: usize, strand: StrandMode, capacity: usize) -> Result<Self, Error> {
        if k == 0 {
            return Err(Error::InvalidK);
        }

        Ok(Self {
            counts: if k <= 32 {
                Counts::Packed(PackedCounts::with_capacity(
                    k,
                    PACKED_PARTITION_BITS,
                    capacity,
                )?)
            } else if k <= 64 {
                let mut counts = PackedCounts128::default();
                counts
                    .try_reserve(capacity)
                    .map_err(|_| Error::ReserveFailed)?;
                Counts::Packed128(counts)
            } else {
                let mut counts = HashMap::new();
                counts
                    .try_reserve(capacity)
                    .map_err(|_| Error::ReserveFailed)?;
                Counts::Bytes(counts)
            },
            partitions: Vec::new(),
            partition_hash: None,
            output_threads: 1,
            strand,
            k,
        })
    }

    /// Count windows within one complete read or FASTA record.
    ///
    /// A/C/G/T are case-insensitive. Any other byte breaks a valid run.
    /// Windows never cross record boundaries. Short or empty records add nothing.
    /// Return `Error::CountOverflow` instead of wrapping an overflowing count.
    /// On overflow, earlier windows in this call remain counted.
    pub fn add_record(&mut self, sequence: &[u8]) -> Result<(), Error> {
        if sequence.len() < self.k {
            return Ok(());
        }

        if let Counts::Packed(counts) = &mut self.counts {
            let (k, strand) = (self.k, self.strand);
            let mut adder = counts.adder();
            visit_packed(sequence, k, strand, |key| adder.push(key))?;
            return adder.finish();
        }
        if !self.partitions.is_empty() {
            let (k, strand) = (self.k, self.strand);
            return match &self.counts {
                Counts::Packed(_) => unreachable!("packed counts partition themselves"),
                Counts::Packed128(_) => visit_packed128(sequence, k, strand, |key| {
                    let Counts::Packed128(map) = self.partition_mut(&key) else {
                        unreachable!()
                    };
                    add_delta(map, key, 1)
                }),
                Counts::Bytes(_) => visit_bytes(sequence, k, strand, |key| {
                    let Counts::Bytes(map) = self.partition_mut(key) else {
                        unreachable!()
                    };
                    add_byte_delta(map, key, 1)
                }),
            };
        }
        match &mut self.counts {
            Counts::Packed(_) => unreachable!("handled above"),
            Counts::Packed128(counts) => match self.strand {
                StrandMode::Forward => add_packed128_record::<false>(counts, sequence, self.k),
                StrandMode::Canonical => add_packed128_record::<true>(counts, sequence, self.k),
            },
            Counts::Bytes(counts) => add_byte_record(counts, sequence, self.k, self.strand),
        }
    }

    /// Count several complete records, exactly as calling [`Counter::add_record`]
    /// for each in order would.
    ///
    /// Batching short records (such as reads) lets the counter overlap memory
    /// accesses across record boundaries. Windows never cross records. On
    /// overflow, earlier windows, including those in earlier records, remain
    /// counted, and later windows and records are not counted.
    pub fn add_records<'a>(
        &mut self,
        records: impl IntoIterator<Item = &'a [u8]>,
    ) -> Result<(), Error> {
        let (k, strand) = (self.k, self.strand);
        if let Counts::Packed(counts) = &mut self.counts {
            let mut adder = counts.adder();
            for record in records {
                if record.len() >= k {
                    visit_packed(record, k, strand, |key| adder.push(key))?;
                }
            }
            return adder.finish();
        }
        records
            .into_iter()
            .try_for_each(|record| self.add_record(record))
    }

    /// Consume another counter with the same k and strand mode, adding its counts.
    /// Keys remain packed; no DNA decoding or per-key allocation is needed for k <= 64.
    /// On overflow, earlier entries may have been merged; the overflowing count is unchanged.
    pub fn merge(&mut self, other: Self) -> Result<(), Error> {
        if self.k != other.k || self.strand != other.strand {
            return Err(Error::IncompatibleCounters);
        }
        if let (Counts::Packed(target), Counts::Packed(source)) = (&mut self.counts, &other.counts)
        {
            // Keys are routed by this counter's own mixer and partitions.
            return source
                .entries()
                .try_for_each(|(key, delta)| target.add_key(key, delta));
        }
        for source in std::iter::once(other.counts).chain(other.partitions) {
            if self.partitions.is_empty() {
                match (&mut self.counts, source) {
                    (Counts::Packed128(target), Counts::Packed128(source)) => {
                        merge_counts(target, source)?
                    }
                    (Counts::Bytes(target), Counts::Bytes(source)) => merge_counts(target, source)?,
                    _ => unreachable!("matching k selects matching representations"),
                }
            } else {
                match source {
                    Counts::Packed(_) => {
                        unreachable!("matching k selects matching representations")
                    }
                    Counts::Packed128(source) => {
                        for (key, delta) in source {
                            let Counts::Packed128(target) = self.partition_mut(&key) else {
                                unreachable!()
                            };
                            add_delta(target, key, delta)?;
                        }
                    }
                    Counts::Bytes(source) => {
                        for (key, delta) in source {
                            let Counts::Bytes(target) = self.partition_mut(&key) else {
                                unreachable!()
                            };
                            add_delta(target, key, delta)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn from_partitions(
        k: usize,
        strand: StrandMode,
        partitions: Vec<Counts>,
        hash: foldhash::fast::RandomState,
        output_threads: usize,
    ) -> Self {
        debug_assert!(partitions.len().is_power_of_two());
        let mut partitions = partitions.into_iter();
        Self {
            counts: partitions.next().unwrap(),
            partitions: partitions.collect(),
            partition_hash: Some(hash),
            output_threads: output_threads.max(1),
            k,
            strand,
        }
    }

    pub(crate) fn from_packed(
        k: usize,
        strand: StrandMode,
        counts: PackedCounts,
        output_threads: usize,
    ) -> Self {
        Self {
            counts: Counts::Packed(counts),
            partitions: Vec::new(),
            partition_hash: None,
            output_threads: output_threads.max(1),
            k,
            strand,
        }
    }

    fn tables(&self) -> impl Iterator<Item = &Counts> {
        std::iter::once(&self.counts).chain(&self.partitions)
    }

    /// Encode every table with `encode`, writing the bytes in table order.
    ///
    /// Partitioned results encode tables concurrently on up to
    /// `output_threads` scoped threads. Each encoder streams bounded chunks
    /// through its own small queue, and the calling thread writes them in
    /// order, so buffered output stays proportional to the thread count
    /// (about 17 MiB per encoder) rather than the table size.
    fn write_tables(
        &self,
        writer: &mut dyn Write,
        encode: impl Fn(OutputTable<'_>, &mut dyn Write) -> io::Result<()> + Sync,
    ) -> io::Result<()> {
        let tables: Vec<OutputTable<'_>> = match &self.counts {
            Counts::Packed(counts) => (0..counts.tables().len())
                .map(|partition| OutputTable::Packed(counts, partition))
                .collect(),
            _ => self.tables().map(OutputTable::Map).collect(),
        };
        let threads = self.output_threads.min(tables.len());
        if threads <= 1 {
            return tables
                .into_iter()
                .try_for_each(|table| encode(table, &mut *writer));
        }
        std::thread::scope(|scope| {
            let (recycle, recycled) = crossbeam_channel::unbounded::<Vec<u8>>();
            let mut streams = Vec::with_capacity(threads);
            for first in 0..threads {
                // Each encoder may run this many chunks ahead of the writer;
                // enough to buffer a typical partition while others are written.
                let (sender, receiver) = crossbeam_channel::bounded(16);
                streams.push(receiver);
                let (tables, encode, recycled) = (&tables, &encode, recycled.clone());
                std::thread::Builder::new()
                    .name(format!("output-{first}"))
                    .spawn_scoped(scope, move || {
                        // Encoder `first` owns tables first, first + threads, ...
                        for &table in tables.iter().skip(first).step_by(threads) {
                            let mut sink = ChunkSink {
                                buffer: recycled.try_recv().unwrap_or_default(),
                                sender: &sender,
                                recycled: &recycled,
                            };
                            if let Err(error) =
                                encode(table, &mut sink).and_then(|()| sink.finish())
                            {
                                // Forward the encoder's own error; this fails
                                // harmlessly if the writer already stopped.
                                let _ = sender.send(Err(error));
                                return;
                            }
                        }
                    })?;
            }
            drop(recycled);
            for index in 0..tables.len() {
                // Returning early drops every receiver, releasing blocked encoders.
                while let Some(mut chunk) = streams[index % threads]
                    .recv()
                    .map_err(|_| io::Error::other("output encoder failed"))??
                {
                    writer.write_all(&chunk)?;
                    chunk.clear();
                    let _ = recycle.send(chunk);
                }
            }
            Ok(())
        })
    }

    fn partition_mut<K: Hash + ?Sized>(&mut self, key: &K) -> &mut Counts {
        let index =
            self.partition_hash.as_ref().unwrap().hash_one(key) as usize & self.partitions.len();
        if index == 0 {
            &mut self.counts
        } else {
            &mut self.partitions[index - 1]
        }
    }

    /// Return the number of distinct k-mers counted so far.
    pub fn len(&self) -> usize {
        self.tables()
            .map(|table| match table {
                Counts::Packed(counts) => counts.len(),
                Counts::Packed128(map) => map.len(),
                Counts::Bytes(map) => map.len(),
            })
            .sum()
    }

    /// Return whether no valid k-mers have been counted.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write headerless `KMER<TAB>COUNT` rows, in unspecified order.
    ///
    /// Reuses stack buffers for packed keys and decimal counts; larger keys are
    /// borrowed from the table. No per-row heap allocation is needed. Wrap files
    /// or stdout in a [`io::BufWriter`] to batch writes. This method does not flush
    /// the writer; write failures may leave partial output.
    pub fn write_tsv(&self, mut writer: impl Write) -> io::Result<()> {
        self.write_tables(&mut writer, |table, writer| match table {
            OutputTable::Packed(counts, partition) => {
                write_packed64_tsv(counts.partition_entries(partition), self.k, writer)
            }
            OutputTable::Map(Counts::Packed(_)) => unreachable!("packed tables are split"),
            OutputTable::Map(Counts::Packed128(counts)) => write_packed_tsv::<64, _>(
                counts.iter().map(|(&key, &count)| (key, count)),
                self.k,
                writer,
                decode_packed128_into,
            ),
            OutputTable::Map(Counts::Bytes(counts)) => {
                // A u64 has at most 20 decimal digits, plus a tab and newline.
                let mut suffix = [0; 22];
                for (key, &count) in counts {
                    writer.write_all(key)?;
                    writer.write_all(format_count(count, &mut suffix))?;
                }
                Ok(())
            }
        })
    }

    /// Iterate over distinct uppercase DNA keys and their counts, in unspecified order.
    ///
    /// Each yielded key is an owned byte vector. Packed keys are decoded lazily;
    /// collecting or formatting output is separate from counting.
    ///
    /// ```
    /// use kbolt::{Counter, StrandMode};
    /// use std::collections::HashMap;
    ///
    /// let mut counter = Counter::new(3, StrandMode::Canonical).unwrap();
    /// counter.add_record(b"ACGT").unwrap();
    /// let counts: HashMap<Vec<u8>, u64> = counter.counts().collect();
    /// assert_eq!(counts[b"ACG".as_slice()], 2);
    /// ```
    pub fn counts(&self) -> impl Iterator<Item = (Vec<u8>, u64)> + '_ {
        self.tables().flat_map(move |table| {
            let (packed, wide, bytes) = match table {
                Counts::Packed(counts) => (Some(counts.entries()), None, None),
                Counts::Packed128(counts) => (None, Some(counts.iter()), None),
                Counts::Bytes(counts) => (None, None, Some(counts.iter())),
            };
            packed
                .into_iter()
                .flatten()
                .map(|(key, count)| (decode_packed(key, self.k), count))
                .chain(wide.into_iter().flatten().map(|(&key, &count)| {
                    let mut dna = vec![0; self.k];
                    decode_packed128_into(key, &mut dna);
                    (dna, count)
                }))
                .chain(
                    bytes
                        .into_iter()
                        .flatten()
                        .map(|(key, &count)| (key.clone(), count)),
                )
        })
    }
}

fn merge_counts<K: Eq + std::hash::Hash, S: std::hash::BuildHasher>(
    target: &mut HashMap<K, u64, S>,
    source: HashMap<K, u64, S>,
) -> Result<(), Error> {
    target
        .try_reserve(source.len())
        .map_err(|_| Error::ReserveFailed)?;
    for (key, delta) in source {
        let count = target.entry(key).or_insert(0);
        *count = count.checked_add(delta).ok_or(Error::CountOverflow)?;
    }
    Ok(())
}

// Assemble many rows directly in one buffer, avoiding a small memcpy and writer
// call for every distinct key. MAX is the backend's maximum number of bases.
fn write_packed_tsv<const MAX: usize, K>(
    entries: impl Iterator<Item = (K, u64)>,
    k: usize,
    writer: &mut (impl Write + ?Sized),
    decode: impl Fn(K, &mut [u8]),
) -> io::Result<()> {
    let mut batch = [0; 16 * 1024];
    let mut offset = 0;
    for (key, count) in entries {
        if offset + MAX + 22 > batch.len() {
            writer.write_all(&batch[..offset])?;
            offset = 0;
        }
        decode(key, &mut batch[offset..offset + k]);
        offset = push_count(&mut batch, offset + k, count);
    }
    writer.write_all(&batch[..offset])
}

/// `write_packed_tsv` for keys of at most 32 bases, decoding a whole key at
/// once. Each row first stores 32 decoded bytes; the count then overwrites
/// any bases beyond k.
fn write_packed64_tsv(
    entries: impl Iterator<Item = (u64, u64)>,
    k: usize,
    writer: &mut (impl Write + ?Sized),
) -> io::Result<()> {
    let mut batch = [0; 16 * 1024];
    let mut offset = 0;
    for (key, count) in entries {
        if offset + 32 + 22 > batch.len() {
            writer.write_all(&batch[..offset])?;
            offset = 0;
        }
        batch[offset..offset + 32].copy_from_slice(&decode_packed32(key, k));
        offset = push_count(&mut batch, offset + k, count);
    }
    writer.write_all(&batch[..offset])
}

/// Write `\tCOUNT\n` at `offset`, returning the end. Needs 22 bytes of room.
#[inline(always)]
fn push_count(batch: &mut [u8], offset: usize, count: u64) -> usize {
    // Most k-mers in large references occur fewer than ten times.
    if count < 10 {
        batch[offset..offset + 3].copy_from_slice(&[b'\t', b'0' + count as u8, b'\n']);
        return offset + 3;
    }
    let mut suffix = [0; 22];
    let suffix = format_count(count, &mut suffix);
    batch[offset..offset + suffix.len()].copy_from_slice(suffix);
    offset + suffix.len()
}

/// Decode the k <= 32 bases of a packed key into the first k bytes of the
/// result; the remaining bytes are unspecified.
#[inline(always)]
fn decode_packed32(key: u64, k: usize) -> [u8; 32] {
    debug_assert!((1..=32).contains(&k));
    // Left-align the key: the first base occupies the top two bits, so the
    // big-endian bytes hold four bases each, in order.
    let aligned = key << (64 - 2 * k);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is mandatory on aarch64; all loads and stores are in bounds.
    unsafe {
        use core::arch::aarch64::*;
        const LOW: [u8; 16] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
        const HIGH: [u8; 16] = [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7];
        const SHIFTS: [i8; 16] = [-6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0];
        let bytes = vcreate_u8(aligned.swap_bytes());
        let bytes = vcombine_u8(bytes, bytes);
        let shifts = vld1q_s8(SHIFTS.as_ptr());
        let mask = vdupq_n_u8(3);
        let letters = vld1q_u8(b"ACGTACGTACGTACGT".as_ptr());
        let expand = |indices: [u8; 16]| {
            let replicated = vqtbl1q_u8(bytes, vld1q_u8(indices.as_ptr()));
            let codes = vandq_u8(vshlq_u8(replicated, shifts), mask);
            vqtbl1q_u8(letters, codes)
        };
        let mut dna = [0; 32];
        vst1q_u8(dna.as_mut_ptr(), expand(LOW));
        vst1q_u8(dna.as_mut_ptr().add(16), expand(HIGH));
        dna
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut dna = [0; 32];
        decode_packed_into(aligned, &mut dna);
        dna
    }
}

// Specialize by strand once per record, leaving no reverse-complement work in
// the forward-only loop. Callers guarantee 1 <= k <= 32.
fn visit_packed(
    sequence: &[u8],
    k: usize,
    strand: StrandMode,
    emit: impl FnMut(u64) -> Result<(), Error>,
) -> Result<(), Error> {
    match strand {
        StrandMode::Forward => visit_packed_record::<false>(sequence, k, emit),
        StrandMode::Canonical => visit_packed_record::<true>(sequence, k, emit),
    }
}

fn visit_packed_record<const CANONICAL: bool>(
    sequence: &[u8],
    k: usize,
    mut emit: impl FnMut(u64) -> Result<(), Error>,
) -> Result<(), Error> {
    // Right-shifting an all-ones mask also handles k = 32 without a shift by 64.
    let mask = u64::MAX >> (64 - 2 * k);
    let high_shift = 2 * (k - 1);
    let (mut forward, mut reverse) = (0_u64, 0_u64);
    let mut valid_run = 0;

    for &byte in sequence {
        let base = DNA_BASES[byte as usize];
        if base == u8::MAX {
            forward = 0;
            reverse = 0;
            valid_run = 0;
            continue;
        }
        let base = u64::from(base);
        forward = ((forward << 2) | base) & mask;
        if CANONICAL {
            reverse = (reverse >> 2) | ((base ^ 3) << high_shift);
        }
        valid_run = (valid_run + 1).min(k);
        if valid_run < k {
            continue;
        }

        // A=0, C=1, G=2, T=3 makes numeric and lexicographic order agree.
        let key = if CANONICAL {
            forward.min(reverse)
        } else {
            forward
        };
        emit(key)?;
    }
    Ok(())
}

fn add_packed128_record<const CANONICAL: bool>(
    counts: &mut PackedCounts128,
    sequence: &[u8],
    k: usize,
) -> Result<(), Error> {
    visit_packed128_record::<CANONICAL>(sequence, k, |key| {
        let count = counts.entry(key).or_insert(0);
        *count = count.checked_add(1).ok_or(Error::CountOverflow)?;
        Ok(())
    })
}

fn visit_packed128(
    sequence: &[u8],
    k: usize,
    strand: StrandMode,
    emit: impl FnMut((u64, u64)) -> Result<(), Error>,
) -> Result<(), Error> {
    match strand {
        StrandMode::Forward => visit_packed128_record::<false>(sequence, k, emit),
        StrandMode::Canonical => visit_packed128_record::<true>(sequence, k, emit),
    }
}

fn visit_packed128_record<const CANONICAL: bool>(
    sequence: &[u8],
    k: usize,
    mut emit: impl FnMut((u64, u64)) -> Result<(), Error>,
) -> Result<(), Error> {
    // Right-shifting an all-ones mask also handles k = 64 without a shift by 128.
    let mask = u128::MAX >> (128 - 2 * k);
    let high_shift = 2 * (k - 1);
    let (mut forward, mut reverse) = (0_u128, 0_u128);
    let mut valid_run = 0;

    for &byte in sequence {
        let base = DNA_BASES[byte as usize];
        if base == u8::MAX {
            forward = 0;
            reverse = 0;
            valid_run = 0;
            continue;
        }
        let base = u128::from(base);
        forward = ((forward << 2) | base) & mask;
        if CANONICAL {
            reverse = (reverse >> 2) | ((base ^ 3) << high_shift);
        }
        valid_run = (valid_run + 1).min(k);
        if valid_run < k {
            continue;
        }

        // A=0, C=1, G=2, T=3 makes numeric and lexicographic order agree.
        let key = if CANONICAL {
            forward.min(reverse)
        } else {
            forward
        };
        emit(((key >> 64) as u64, key as u64))?;
    }
    Ok(())
}

fn decode_packed128_into(key: (u64, u64), dna: &mut [u8]) {
    let split = dna.len() - 32;
    let (high, low) = dna.split_at_mut(split);
    decode_packed_into(key.0, high);
    decode_packed_into(key.1, low);
}

fn decode_packed(key: u64, k: usize) -> Vec<u8> {
    let mut dna = vec![0; k];
    decode_packed_into(key, &mut dna);
    dna
}

// Decode four bases per lookup, using a 1 KiB table shared by all packed widths.
const DECODE_BASES: [[u8; 4]; 256] = {
    let mut table = [[0; 4]; 256];
    let mut value = 0;
    while value < 256 {
        let mut base = 0;
        while base < 4 {
            table[value][base] = b"ACGT"[(value >> (2 * (3 - base))) & 3];
            base += 1;
        }
        value += 1;
    }
    table
};
fn decode_packed_into(mut key: u64, dna: &mut [u8]) {
    let mut chunks = dna.rchunks_exact_mut(4);
    for chunk in &mut chunks {
        chunk.copy_from_slice(&DECODE_BASES[(key & 255) as usize]);
        key >>= 8;
    }
    for base in chunks.into_remainder().iter_mut().rev() {
        *base = b"ACGT"[(key & 3) as usize];
        key >>= 2;
    }
}

fn format_count(mut count: u64, buffer: &mut [u8; 22]) -> &[u8] {
    buffer[21] = b'\n';
    let mut start = 21;
    loop {
        start -= 1;
        buffer[start] = b'0' + (count % 10) as u8;
        count /= 10;
        if count == 0 {
            break;
        }
    }
    start -= 1;
    buffer[start] = b'\t';
    &buffer[start..]
}

// Byte-string fallback for k > 64.
fn add_byte_record(
    counts: &mut HashMap<Vec<u8>, u64>,
    sequence: &[u8],
    k: usize,
    strand: StrandMode,
) -> Result<(), Error> {
    visit_bytes(sequence, k, strand, |key| add_byte_delta(counts, key, 1))
}

fn visit_bytes(
    sequence: &[u8],
    k: usize,
    strand: StrandMode,
    mut emit: impl FnMut(&[u8]) -> Result<(), Error>,
) -> Result<(), Error> {
    let sequence = sequence.to_ascii_uppercase();
    let mut reverse_complement = if strand == StrandMode::Canonical {
        vec![0; k]
    } else {
        Vec::new()
    };

    for kmer in sequence.windows(k) {
        if !kmer
            .iter()
            .all(|base| matches!(base, b'A' | b'C' | b'G' | b'T'))
        {
            continue;
        }

        let key = if strand == StrandMode::Canonical {
            for (complement, base) in reverse_complement.iter_mut().zip(kmer.iter().rev()) {
                *complement = match base {
                    b'A' => b'T',
                    b'C' => b'G',
                    b'G' => b'C',
                    b'T' => b'A',
                    _ => unreachable!("bases were validated above"),
                };
            }
            kmer.min(reverse_complement.as_slice())
        } else {
            kmer
        };

        emit(key)?;
    }
    Ok(())
}

fn add_delta<K: Eq + Hash, S: BuildHasher>(
    map: &mut HashMap<K, u64, S>,
    key: K,
    delta: u64,
) -> Result<(), Error> {
    if map.len() == map.capacity() {
        map.try_reserve(1).map_err(|_| Error::ReserveFailed)?;
    }
    let count = map.entry(key).or_insert(0);
    *count = count.checked_add(delta).ok_or(Error::CountOverflow)?;
    Ok(())
}

fn add_byte_delta(map: &mut HashMap<Vec<u8>, u64>, key: &[u8], delta: u64) -> Result<(), Error> {
    if let Some(count) = map.get_mut(key) {
        *count = count.checked_add(delta).ok_or(Error::CountOverflow)?;
        Ok(())
    } else {
        map.try_reserve(1).map_err(|_| Error::ReserveFailed)?;
        map.insert(key.to_vec(), delta);
        Ok(())
    }
}

/// Replace every stored count; tests use this to construct overflow cases.
#[cfg(test)]
impl Counts {
    fn set_counts(&mut self, count: u64) {
        match self {
            Counts::Packed(counts) => counts
                .tables_mut()
                .iter_mut()
                .for_each(|table| table.set_counts(count)),
            Counts::Packed128(map) => map.values_mut().for_each(|value| *value = count),
            Counts::Bytes(map) => map.values_mut().for_each(|value| *value = count),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_counts(counter: &Counter, expected: &[(&str, u64)]) {
        let expected: HashMap<Vec<u8>, u64> = expected
            .iter()
            .map(|(key, count)| (key.as_bytes().to_vec(), *count))
            .collect();
        assert_eq!(counter.counts().collect::<HashMap<_, _>>(), expected);
    }

    #[test]
    fn packed_decoder_matches_simple_loop_for_every_length() {
        let mut state = 42_u64;
        for _ in 0..1024 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            for len in 0..=32 {
                let mut actual = [0; 32];
                decode_packed_into(state, &mut actual[..len]);
                let expected: Vec<u8> = (0..len)
                    .map(|index| b"ACGT"[((state >> (2 * (len - 1 - index))) & 3) as usize])
                    .collect();
                assert_eq!(&actual[..len], expected);
            }
        }
    }

    #[test]
    fn whole_key_decoder_matches_base_decoder_for_every_length() {
        let mut state = 42_u64;
        for _ in 0..1024 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            for k in 1..=32 {
                let key = state & (u64::MAX >> (64 - 2 * k));
                let mut expected = vec![0; k];
                decode_packed_into(key, &mut expected);
                assert_eq!(&decode_packed32(key, k)[..k], expected, "k={k}");
            }
        }
    }

    #[test]
    fn full_width_128_bit_keys_preserve_high_bits_and_leading_as() {
        let mut counter = Counter::new(64, StrandMode::Forward).unwrap();
        counter.add_record(&[b'T'; 65]).unwrap();
        counter.add_record(&[b'A'; 64]).unwrap();
        let leading_as = format!("{}C", "A".repeat(63));
        counter.add_record(leading_as.as_bytes()).unwrap();
        assert_counts(
            &counter,
            &[(&"T".repeat(64), 2), (&"A".repeat(64), 1), (&leading_as, 1)],
        );

        let mut canonical = Counter::new(64, StrandMode::Canonical).unwrap();
        canonical.add_record(&[b'T'; 65]).unwrap();
        canonical.add_record(&[b'A'; 64]).unwrap();
        assert_counts(&canonical, &[(&"A".repeat(64), 3)]);
    }

    #[test]
    fn every_invalid_byte_breaks_wide_packed_runs() {
        for k in [33, 64] {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                for byte in 0..=u8::MAX {
                    if b"ACGTacgt".contains(&byte) {
                        continue;
                    }
                    let mut record = vec![b'A'; k - 1];
                    record.push(byte);
                    record.extend(vec![b'A'; k]);
                    let mut counter = Counter::new(k, strand).unwrap();
                    counter.add_record(&record).unwrap();
                    assert_counts(&counter, &[(&"A".repeat(k), 1)]);
                }
            }
        }
    }

    #[test]
    fn capacity_hint_preserves_counts_and_allows_growth() {
        for k in [1, 3, 31, 32, 33, 63, 64, 65] {
            for strand in [StrandMode::Canonical, StrandMode::Forward] {
                let mut expected = Counter::new(k, strand).unwrap();
                let mut reserved = Counter::with_capacity(k, strand, 1).unwrap();
                assert!(reserved.is_empty());
                let capacity = match &reserved.counts {
                    Counts::Packed(counts) => counts.capacity(),
                    Counts::Packed128(counts) => counts.capacity(),
                    Counts::Bytes(counts) => counts.capacity(),
                };
                assert!(capacity >= 1);
                for record in [b"ACGTTACGAGTCAGATTCGAC".repeat(8), vec![b'T'; 80]] {
                    expected.add_record(&record).unwrap();
                    reserved.add_record(&record).unwrap();
                }
                assert!(reserved.len() > 1);
                assert_eq!(
                    reserved.counts().collect::<HashMap<_, _>>(),
                    expected.counts().collect::<HashMap<_, _>>()
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_k_and_unrepresentable_capacity() {
        assert!(matches!(
            Counter::with_capacity(0, StrandMode::Canonical, usize::MAX),
            Err(Error::InvalidK)
        ));
        for k in [31, 33, 65] {
            assert!(matches!(
                Counter::with_capacity(k, StrandMode::Canonical, usize::MAX),
                Err(Error::ReserveFailed)
            ));
        }
    }

    #[test]
    fn tsv_matches_iterator_for_both_backends_and_strands() {
        for k in 1..=66 {
            for strand in [StrandMode::Canonical, StrandMode::Forward] {
                let mut counter = Counter::new(k, strand).unwrap();
                let mut empty = Vec::new();
                counter.write_tsv(&mut empty).unwrap();
                assert!(empty.is_empty());
                for record in [
                    b"acgTTacGAGTCAGATTCGAC".repeat(8),
                    vec![b'T'; 80],
                    vec![b'A'; 80],
                    b"ACGTNACGT.--ACGT".repeat(8),
                ] {
                    counter.add_record(&record).unwrap();
                }
                let mut expected = Vec::new();
                for (key, count) in counter.counts() {
                    expected.extend_from_slice(&key);
                    writeln!(expected, "\t{count}").unwrap();
                }
                let mut actual = Vec::new();
                counter.write_tsv(&mut actual).unwrap();
                assert_eq!(actual, expected, "k={k}, strand={strand:?}");
            }
        }
    }

    #[test]
    fn tsv_formats_decimal_boundaries_and_maximum_counts() {
        for k in [1, 32, 33, 63, 64, 65] {
            let mut counter = Counter::new(k, StrandMode::Forward).unwrap();
            counter.add_record(&vec![b'A'; k]).unwrap();
            // Stored counts are never zero (see `format_count` below for 0).
            let mut values = vec![u64::MAX];
            for exponent in 0..=19 {
                let power = 10_u64.pow(exponent);
                values.extend([power - 1, power, power + 1]);
            }
            for value in values.into_iter().filter(|&value| value > 0) {
                // The counter holds exactly one key.
                counter.counts.set_counts(value);
                let mut output = Vec::new();
                counter.write_tsv(&mut output).unwrap();
                assert_eq!(output, format!("{}\t{value}\n", "A".repeat(k)).as_bytes());
            }
        }
    }

    #[test]
    fn formats_zero_and_maximum_counts() {
        let mut buffer = [0; 22];
        assert_eq!(format_count(0, &mut buffer), b"\t0\n");
        assert_eq!(
            format_count(u64::MAX, &mut buffer),
            format!("\t{}\n", u64::MAX).as_bytes()
        );
    }

    #[test]
    fn tsv_handles_short_writes_and_propagates_write_failures() {
        struct Writer {
            bytes: Vec<u8>,
            remaining: usize,
        }
        impl Write for Writer {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if self.remaining == 0 {
                    return Err(io::Error::other("injected output failure"));
                }
                let len = bytes.len().min(self.remaining).min(2);
                self.bytes.extend_from_slice(&bytes[..len]);
                self.remaining -= len;
                Ok(len)
            }
            fn flush(&mut self) -> io::Result<()> {
                panic!("write_tsv must leave flushing to the caller")
            }
        }
        for k in [3, 33] {
            let mut counter = Counter::new(k, StrandMode::Forward).unwrap();
            counter.add_record(&vec![b'A'; k]).unwrap();
            let expected = format!("{}\t1\n", "A".repeat(k));
            for remaining in [0, 2, k, k + 1, expected.len()] {
                let mut writer = Writer {
                    bytes: Vec::new(),
                    remaining,
                };
                let result = counter.write_tsv(&mut writer);
                if remaining == expected.len() {
                    result.unwrap();
                } else {
                    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::Other);
                }
                assert_eq!(writer.bytes, &expected.as_bytes()[..remaining]);
            }
        }
    }

    #[test]
    fn matches_naive_counts_on_mixed_records() {
        let mut state = 42_u64;
        let mut records: Vec<Vec<u8>> = (0..32)
            .map(|len| {
                (0..len)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        b"ACGTacgtN-"[(state % 10) as usize]
                    })
                    .collect()
            })
            .collect();

        // Exercise both backends, full-width keys, and long valid runs. Dense
        // ambiguity alone would almost never produce a valid 32-base window.
        for len in [31, 32, 33, 64, 65, 66, 129, 1024] {
            let valid: Vec<u8> = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    b"ACGTacgt"[(state % 8) as usize]
                })
                .collect();
            let mut ambiguous = valid.clone();
            for base in ambiguous.iter_mut().skip(31).step_by(97) {
                *base = b'N';
            }
            records.extend([valid, ambiguous]);
        }

        for k in 1..=66 {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let mut counter = Counter::new(k, strand).unwrap();
                let mut expected = HashMap::new();
                for record in records.iter().chain(&records) {
                    counter.add_record(record).unwrap();
                    for window in record.windows(k) {
                        let mut key = window.to_ascii_uppercase();
                        if !key.iter().all(|base| b"ACGT".contains(base)) {
                            continue;
                        }
                        if strand == StrandMode::Canonical {
                            let reverse: Vec<u8> = key
                                .iter()
                                .rev()
                                .map(|base| match base {
                                    b'A' => b'T',
                                    b'C' => b'G',
                                    b'G' => b'C',
                                    b'T' => b'A',
                                    _ => unreachable!(),
                                })
                                .collect();
                            key = key.min(reverse);
                        }
                        *expected.entry(key).or_insert(0_u64) += 1;
                    }
                    assert_eq!(
                        counter.counts().collect::<HashMap<_, _>>(),
                        expected,
                        "k={k}, strand={strand:?}, record length={}",
                        record.len()
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_zero_k() {
        assert!(matches!(
            Counter::new(0, StrandMode::Canonical),
            Err(Error::InvalidK)
        ));
    }

    #[test]
    fn reports_overflow_without_changing_the_overflowing_count() {
        let mut counter = Counter::new(3, StrandMode::Forward).unwrap();
        counter.add_record(b"ACG").unwrap();
        assert!(matches!(counter.counts, Counts::Packed(_)));
        counter.counts.set_counts(u64::MAX - 1);
        counter.add_record(b"ACG").unwrap();
        assert_eq!(counter.add_record(b"ACG"), Err(Error::CountOverflow));
        assert_counts(&counter, &[("ACG", u64::MAX)]);
    }

    #[test]
    fn overflow_keeps_earlier_windows_and_stops_before_later_ones() {
        for k in [1, 3, 32, 33, 63, 64, 65] {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let mut counter = Counter::new(k, strand).unwrap();
                counter.add_record(&vec![b'A'; k]).unwrap();
                counter.counts.set_counts(u64::MAX);
                let mut record = vec![b'C'];
                record.extend(vec![b'A'; k + 1]);
                record.push(b'G');
                assert_eq!(counter.add_record(&record), Err(Error::CountOverflow));
                let first = format!("C{}", "A".repeat(k - 1));
                assert_counts(&counter, &[(&"A".repeat(k), u64::MAX), (&first, 1)]);
            }
        }
    }

    #[test]
    fn every_invalid_byte_breaks_a_valid_run() {
        for strand in [StrandMode::Forward, StrandMode::Canonical] {
            for byte in 0..=u8::MAX {
                if b"ACGTacgt".contains(&byte) {
                    continue;
                }
                let mut counter = Counter::new(3, strand).unwrap();
                counter
                    .add_record(&[b'a', b'c', byte, b'g', b't', b'a'])
                    .unwrap();
                assert_counts(&counter, &[("GTA", 1)]);
            }
        }
    }

    #[test]
    fn full_width_keys_preserve_high_bits_and_leading_as() {
        let mut counter = Counter::new(32, StrandMode::Forward).unwrap();
        counter.add_record(&[b'T'; 33]).unwrap();
        counter.add_record(&[b'A'; 32]).unwrap();
        let leading_as = format!("{}C", "A".repeat(31));
        counter.add_record(leading_as.as_bytes()).unwrap();
        assert_counts(
            &counter,
            &[(&"T".repeat(32), 2), (&"A".repeat(32), 1), (&leading_as, 1)],
        );

        let mut canonical = Counter::new(32, StrandMode::Canonical).unwrap();
        canonical.add_record(&[b'T'; 33]).unwrap();
        canonical.add_record(&[b'A'; 32]).unwrap();
        assert_counts(&canonical, &[(&"A".repeat(32), 3)]);
    }

    #[test]
    fn full_width_windows_do_not_cross_invalid_bytes_or_records() {
        for strand in [StrandMode::Forward, StrandMode::Canonical] {
            let mut counter = Counter::new(32, strand).unwrap();
            let mut record = vec![b'T'; 31];
            record.push(b'N');
            record.extend([b'A'; 31]);
            counter.add_record(&record).unwrap();
            counter.add_record(b"C").unwrap();
            assert_counts(&counter, &[]);
            counter.add_record(&[b'A'; 32]).unwrap();
            assert_counts(&counter, &[(&"A".repeat(32), 1)]);
        }
    }

    #[test]
    fn non_ascii_and_whitespace_break_windows() {
        let mut counter = Counter::new(3, StrandMode::Forward).unwrap();
        counter.add_record(b"AC\xffACG\nTGC").unwrap();
        assert_counts(&counter, &[("ACG", 1), ("TGC", 1)]);
    }

    #[test]
    fn starts_empty() {
        let counter = Counter::new(3, StrandMode::Canonical).unwrap();
        assert!(counter.is_empty());
        assert_eq!(counter.len(), 0);
        assert_counts(&counter, &[]);
    }

    #[test]
    fn counts_overlapping_windows() {
        let mut counter = Counter::new(3, StrandMode::Forward).unwrap();
        counter.add_record(b"ACGACG").unwrap();
        assert_counts(&counter, &[("ACG", 2), ("CGA", 1), ("GAC", 1)]);
    }

    #[test]
    fn counts_repeated_windows() {
        let mut counter = Counter::new(3, StrandMode::Forward).unwrap();
        counter.add_record(b"AAAAA").unwrap();
        assert_counts(&counter, &[("AAA", 3)]);
    }

    #[test]
    fn invalid_bases_break_windows() {
        for sequence in [b"ACNACG".as_slice(), b"ACRACG", b"AC-ACG", b"ACUACG"] {
            let mut counter = Counter::new(3, StrandMode::Forward).unwrap();
            counter.add_record(sequence).unwrap();
            assert_counts(&counter, &[("ACG", 1)]);
        }
    }

    #[test]
    fn never_joins_separate_records() {
        let mut counter = Counter::new(3, StrandMode::Forward).unwrap();
        counter.add_record(b"AC").unwrap();
        counter.add_record(b"GT").unwrap();
        assert_counts(&counter, &[]);
    }

    #[test]
    fn normalizes_case() {
        let mut counter = Counter::new(3, StrandMode::Forward).unwrap();
        counter.add_record(b"acgACg").unwrap();
        assert_counts(&counter, &[("ACG", 2), ("CGA", 1), ("GAC", 1)]);
    }

    #[test]
    fn canonical_mode_combines_reverse_complements() {
        let mut counter = Counter::new(3, StrandMode::Canonical).unwrap();
        counter.add_record(b"ACG").unwrap();
        counter.add_record(b"cgt").unwrap();
        assert_counts(&counter, &[("ACG", 2)]);
    }

    #[test]
    fn forward_mode_keeps_reverse_complements_separate() {
        let mut counter = Counter::new(3, StrandMode::Forward).unwrap();
        counter.add_record(b"ACG").unwrap();
        counter.add_record(b"CGT").unwrap();
        assert_counts(&counter, &[("ACG", 1), ("CGT", 1)]);
    }

    #[test]
    fn reverse_complement_palindrome_is_counted_once_per_window() {
        let mut counter = Counter::new(4, StrandMode::Canonical).unwrap();
        counter.add_record(b"ACGT").unwrap();
        assert_counts(&counter, &[("ACGT", 1)]);
    }

    #[test]
    fn empty_short_and_all_ambiguous_records_add_nothing() {
        let mut counter = Counter::new(3, StrandMode::Canonical).unwrap();
        for sequence in [b"".as_slice(), b"A", b"AC", b"NNNNN"] {
            counter.add_record(sequence).unwrap();
        }
        assert_counts(&counter, &[]);
    }

    #[test]
    fn exact_length_record_adds_one_window() {
        let mut counter = Counter::new(3, StrandMode::Forward).unwrap();
        counter.add_record(b"TGC").unwrap();
        assert_counts(&counter, &[("TGC", 1)]);
    }

    #[test]
    fn counts_accumulate_across_records() {
        let mut counter = Counter::new(3, StrandMode::Forward).unwrap();
        counter.add_record(b"ACG").unwrap();
        counter.add_record(b"ACG").unwrap();
        assert_counts(&counter, &[("ACG", 2)]);
    }

    #[test]
    fn supports_single_base_kmers() {
        let mut counter = Counter::new(1, StrandMode::Canonical).unwrap();
        counter.add_record(b"ACGTNacgt").unwrap();
        assert_counts(&counter, &[("A", 4), ("C", 4)]);
    }

    #[test]
    fn supports_k_above_packed_integer_limits() {
        let mut counter = Counter::new(65, StrandMode::Forward).unwrap();
        counter.add_record(&[b'A'; 66]).unwrap();
        assert!(!counter.is_empty());
        assert_eq!(counter.len(), 1);
        assert_counts(&counter, &[(&"A".repeat(65), 2)]);
    }
}
