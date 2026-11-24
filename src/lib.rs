use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::io::{self, Read, Write};
use std::path::Path;

pub mod database;
mod disk;
mod gzip;
pub mod input;
mod parallel;
mod sorted;
mod table;
mod table_wide;

pub use disk::{DiskCounter, DiskCounts};
pub use input::InputStats;
pub use parallel::{ParallelCounter, ParallelInput, ParallelStats};
use table::PackedCounts;
use table_wide::{HighKey, PackedCountsWide};

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
    /// Writing the temporary file of a [`DiskCounter`] failed; its `finish`
    /// reports the underlying I/O error.
    TemporaryFile,
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
            Self::TemporaryFile => f.write_str("unable to write the temporary bin file"),
        }
    }
}

impl std::error::Error for Error {}

pub(crate) enum WideCounts {
    /// k = 33..=40: 10 bytes/slot.
    W80(PackedCountsWide<u16>),
    /// k = 41..=48: 12 bytes/slot.
    W96(PackedCountsWide<u32>),
    /// k = 49..=64: 16 bytes/slot.
    W128(PackedCountsWide<u64>),
    /// k = 65..=96: 24 bytes/slot.
    W192(PackedCountsWide<[u64; 2]>),
    /// k = 97..=128: 32 bytes/slot.
    W256(PackedCountsWide<[u64; 3]>),
}

macro_rules! dispatch_wide {
    ($value:expr, $var:ident => $body:expr) => {
        match $value {
            $crate::WideCounts::W80($var) => $body,
            $crate::WideCounts::W96($var) => $body,
            $crate::WideCounts::W128($var) => $body,
            $crate::WideCounts::W192($var) => $body,
            $crate::WideCounts::W256($var) => $body,
        }
    };
}
pub(crate) use dispatch_wide;

impl WideCounts {
    fn with_capacity(
        k: usize,
        partition_bits: u32,
        capacity: usize,
        load: table::Load,
    ) -> Result<Self, Error> {
        Ok(match k {
            33..=40 => Self::W80(PackedCountsWide::with_capacity(
                partition_bits,
                capacity,
                load,
            )?),
            41..=48 => Self::W96(PackedCountsWide::with_capacity(
                partition_bits,
                capacity,
                load,
            )?),
            49..=64 => Self::W128(PackedCountsWide::with_capacity(
                partition_bits,
                capacity,
                load,
            )?),
            65..=96 => Self::W192(PackedCountsWide::with_capacity(
                partition_bits,
                capacity,
                load,
            )?),
            97..=128 => Self::W256(PackedCountsWide::with_capacity(
                partition_bits,
                capacity,
                load,
            )?),
            _ => unreachable!("wide tables handle 33 <= k <= 128"),
        })
    }

    pub(crate) fn table_count(&self) -> usize {
        dispatch_wide!(self, c => c.tables().len())
    }

    fn len(&self) -> usize {
        dispatch_wide!(self, c => c.len())
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        dispatch_wide!(self, c => c.capacity())
    }

    fn add_records<'a>(
        &mut self,
        records: impl IntoIterator<Item = &'a [u8]>,
        k: usize,
        strand: StrandMode,
    ) -> Result<(), Error> {
        macro_rules! add_wide {
            ($counts:expr, $visit:ident, $cast:expr) => {{
                let mut adder = $counts.adder();
                for record in records {
                    if record.len() >= k {
                        $visit(record, k, strand, |hi, lo| adder.push($cast(hi), lo))?;
                    }
                }
                adder.finish()
            }};
        }
        match self {
            Self::W80(c) => add_wide!(c, visit_packed128, |hi| hi as u16),
            Self::W96(c) => add_wide!(c, visit_packed128, |hi| hi as u32),
            Self::W128(c) => add_wide!(c, visit_packed128, |hi| hi),
            Self::W192(c) => add_wide!(c, visit_packed192, |hi| hi),
            Self::W256(c) => add_wide!(c, visit_packed256, |hi| hi),
        }
    }

    fn merge(&mut self, other: &Self) -> Result<(), Error> {
        fn merge_inner<H: HighKey>(
            target: &mut PackedCountsWide<H>,
            source: &PackedCountsWide<H>,
        ) -> Result<(), Error> {
            for ((hi, lo), delta) in source.entries() {
                target.add_key(hi, lo, delta)?;
            }
            Ok(())
        }
        match (self, other) {
            (Self::W80(t), Self::W80(s)) => merge_inner(t, s),
            (Self::W96(t), Self::W96(s)) => merge_inner(t, s),
            (Self::W128(t), Self::W128(s)) => merge_inner(t, s),
            (Self::W192(t), Self::W192(s)) => merge_inner(t, s),
            (Self::W256(t), Self::W256(s)) => merge_inner(t, s),
            _ => unreachable!("matching k selects matching wide representation"),
        }
    }

    fn write_tsv_range(
        &self,
        first: usize,
        tables: usize,
        k: usize,
        writer: &mut dyn Write,
    ) -> io::Result<()> {
        fn encode<H: HighKey>(
            counts: &PackedCountsWide<H>,
            first: usize,
            tables: usize,
            k: usize,
            writer: &mut dyn Write,
        ) -> io::Result<()> {
            let entries =
                (first..first + tables).flat_map(|partition| counts.partition_entries(partition));
            write_packed_tsv(entries, k, writer, |(hi, lo), dna| hi.decode_into(lo, dna))
        }
        dispatch_wide!(self, c => encode(c, first, tables, k, writer))
    }

    fn decoded_entries(&self, k: usize) -> Box<dyn Iterator<Item = (Vec<u8>, u64)> + '_> {
        fn decode_iter<H: HighKey>(
            counts: &PackedCountsWide<H>,
            k: usize,
        ) -> Box<dyn Iterator<Item = (Vec<u8>, u64)> + '_> {
            Box::new(counts.entries().map(move |((hi, lo), count)| {
                let mut dna = vec![0; k];
                hi.decode_into(lo, &mut dna);
                (dna, count)
            }))
        }
        dispatch_wide!(self, c => decode_iter(c, k))
    }

    #[cfg(test)]
    fn set_counts(&mut self, count: u64) {
        dispatch_wide!(self, c => c.tables_mut().iter_mut().for_each(|t| t.set_counts(count)));
    }
}

/// Byte-string count storage for k > 128, optionally partitioned across multiple
/// maps (always a power-of-two count) when produced by `ParallelCounter`.
pub(crate) struct ByteCounts {
    pub(crate) maps: Vec<HashMap<Vec<u8>, u64, foldhash::fast::RandomState>>,
    hash: foldhash::fast::RandomState,
}

impl ByteCounts {
    fn with_capacity(capacity: usize) -> Result<Self, Error> {
        let mut map = HashMap::default();
        map.try_reserve(capacity)
            .map_err(|_| Error::ReserveFailed)?;
        Ok(Self {
            maps: vec![map],
            hash: foldhash::fast::RandomState::default(),
        })
    }

    pub(crate) fn from_maps(
        maps: Vec<HashMap<Vec<u8>, u64, foldhash::fast::RandomState>>,
        hash: foldhash::fast::RandomState,
    ) -> Self {
        debug_assert!(maps.len().is_power_of_two());
        Self { maps, hash }
    }

    fn map_mut(&mut self, key: &[u8]) -> &mut HashMap<Vec<u8>, u64, foldhash::fast::RandomState> {
        let index = if self.maps.len() == 1 {
            0
        } else {
            self.hash.hash_one(key) as usize & (self.maps.len() - 1)
        };
        &mut self.maps[index]
    }

    fn len(&self) -> usize {
        self.maps.iter().map(HashMap::len).sum()
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.maps.iter().map(HashMap::capacity).sum()
    }

    fn add_record(&mut self, sequence: &[u8], k: usize, strand: StrandMode) -> Result<(), Error> {
        visit_bytes(sequence, k, strand, |key| {
            add_byte_delta(self.map_mut(key), key, 1)
        })
    }

    fn merge(&mut self, other: Self) -> Result<(), Error> {
        for source in other.maps {
            if self.maps.len() == 1 {
                merge_counts(&mut self.maps[0], source)?;
            } else {
                for (key, delta) in source {
                    let target = self.map_mut(&key);
                    add_delta(target, key, delta)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn insert_new(&mut self, key: &[u8], count: u64) -> Result<bool, Error> {
        let map = self.map_mut(key);
        if map.contains_key(key) {
            return Ok(false);
        }
        map.try_reserve(1).map_err(|_| Error::ReserveFailed)?;
        map.insert(key.to_vec(), count);
        Ok(true)
    }

    #[cfg(test)]
    fn set_counts(&mut self, count: u64) {
        for map in &mut self.maps {
            map.values_mut().for_each(|value| *value = count);
        }
    }
}

// Keep the representation private so wider packed keys can be added without
// changing the public output format.
pub(crate) enum Counts {
    /// k <= 32: compact, self-partitioned 64-bit tables.
    Packed(PackedCounts),
    /// 33 <= k <= 128: compact, self-partitioned wide tables.
    Wide(WideCounts),
    /// k > 128: byte-string hash maps, single or partitioned.
    Bytes(ByteCounts),
}

/// An exact, in-memory DNA k-mer counter.
///
/// Output keys are uppercase ASCII DNA sequences; counts accumulate across records.
/// Rolling 2-bit keys use compact partitioned tables for k <= 128 (8 bytes per
/// slot for k <= 32, and 10, 12, 16, 24, or 32 bytes per slot for k up to 40,
/// 48, 64, 96, and 128). Larger k uses byte-string storage, with no maximum
/// other than practical memory limits. Neither table family is intended to
/// resist adaptive collision attacks.
pub struct Counter {
    counts: Counts,
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
    /// Consecutive partitions of compact 64-bit packed tables: the first, and how
    /// many.
    Packed(&'a PackedCounts, usize, usize),
    /// Consecutive partitions of compact wide tables: the first, and how many.
    Wide(&'a WideCounts, usize, usize),
    /// One byte-string map for k > 128.
    Bytes(&'a HashMap<Vec<u8>, u64, foldhash::fast::RandomState>),
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

        let counts = if k <= 32 {
            Counts::Packed(PackedCounts::with_capacity(
                k,
                PACKED_PARTITION_BITS,
                capacity,
                table::Load::Sparse,
            )?)
        } else if k <= 128 {
            Counts::Wide(WideCounts::with_capacity(
                k,
                PACKED_PARTITION_BITS,
                capacity,
                table::Load::Sparse,
            )?)
        } else {
            Counts::Bytes(ByteCounts::with_capacity(capacity)?)
        };
        Ok(Self::from_counts(k, strand, counts, 1))
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
        match &mut self.counts {
            Counts::Packed(counts) => {
                let mut adder = counts.adder();
                visit_packed(sequence, self.k, self.strand, |key| adder.push(key))?;
                adder.finish()
            }
            Counts::Wide(counts) => {
                counts.add_records(std::iter::once(sequence), self.k, self.strand)
            }
            Counts::Bytes(counts) => counts.add_record(sequence, self.k, self.strand),
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
        match &mut self.counts {
            Counts::Packed(counts) => {
                let mut adder = counts.adder();
                for record in records {
                    if record.len() >= k {
                        visit_packed(record, k, strand, |key| adder.push(key))?;
                    }
                }
                adder.finish()
            }
            Counts::Wide(counts) => counts.add_records(records, k, strand),
            Counts::Bytes(counts) => {
                for record in records {
                    if record.len() >= k {
                        counts.add_record(record, k, strand)?;
                    }
                }
                Ok(())
            }
        }
    }

    /// Consume another counter with the same k and strand mode, adding its counts.
    /// Keys remain packed; no DNA decoding or per-key allocation is needed for k <= 128.
    /// On overflow, earlier entries may have been merged; the overflowing count is unchanged.
    pub fn merge(&mut self, other: Self) -> Result<(), Error> {
        if self.k != other.k || self.strand != other.strand {
            return Err(Error::IncompatibleCounters);
        }
        match (&mut self.counts, other.counts) {
            (Counts::Packed(target), Counts::Packed(source)) => {
                // Keys are routed by this counter's own mixer and partitions.
                for (key, delta) in source.entries() {
                    target.add_key(key, delta)?;
                }
                Ok(())
            }
            (Counts::Wide(target), Counts::Wide(source)) => target.merge(&source),
            (Counts::Bytes(target), Counts::Bytes(source)) => target.merge(source),
            _ => unreachable!("matching k selects matching representations"),
        }
    }

    pub(crate) fn from_counts(
        k: usize,
        strand: StrandMode,
        counts: Counts,
        output_threads: usize,
    ) -> Self {
        Self {
            counts,
            output_threads: output_threads.max(1),
            strand,
            k,
        }
    }

    /// Encode every table with `encode`, writing the bytes in table order;
    /// see [`write_ordered`]. Packed tables are encoded `unit` at a time.
    fn write_tables(
        &self,
        writer: &mut dyn Write,
        unit: usize,
        encode: impl Fn(OutputTable<'_>, &mut dyn Write) -> io::Result<()> + Sync,
    ) -> io::Result<()> {
        let tables: Vec<OutputTable<'_>> = match &self.counts {
            Counts::Packed(counts) => (0..counts.tables().len())
                .step_by(unit)
                .map(|first| OutputTable::Packed(counts, first, unit))
                .collect(),
            Counts::Wide(counts) => {
                let len = counts.table_count();
                let step = (len / (self.output_threads * 8).max(1)).clamp(1, 128);
                (0..len)
                    .step_by(step)
                    .map(|first| OutputTable::Wide(counts, first, (len - first).min(step)))
                    .collect()
            }
            Counts::Bytes(counts) => counts.maps.iter().map(OutputTable::Bytes).collect(),
        };
        write_ordered(
            self.output_threads,
            tables.len(),
            writer,
            |index, writer| encode(tables[index], writer),
        )
    }

    /// The partitions of packed tables that a serial counter would hold in
    /// one: the unit in which a database is written, so that its bytes do
    /// not depend on how many partitions (and threads) counted it.
    fn serial_partition(&self, counts: &PackedCounts) -> usize {
        let serial = table::partition_bits(table::key_bits(self.k), PACKED_PARTITION_BITS);
        (counts.tables().len() >> serial).max(1)
    }

    /// Return the k-mer length of this counter.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Return the strand mode of this counter.
    pub fn strand(&self) -> StrandMode {
        self.strand
    }

    /// Return the number of distinct k-mers counted so far.
    pub fn len(&self) -> usize {
        match &self.counts {
            Counts::Packed(counts) => counts.len(),
            Counts::Wide(counts) => counts.len(),
            Counts::Bytes(counts) => counts.len(),
        }
    }

    /// Return whether no valid k-mers have been counted.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Stream FASTA or FASTQ records from `reader`, auto-detecting gzip
    /// compression from content, and add their k-mers to this counter.
    pub fn add_reader(&mut self, reader: impl Read + Send + 'static) -> io::Result<InputStats> {
        let mut stream = input::Stream::serial(self);
        stream.add_reader(reader)?;
        stream.finish()
    }

    /// Stream FASTA or FASTQ records from the file at `path`, auto-detecting
    /// gzip compression from content, and add their k-mers to this counter.
    pub fn add_file(&mut self, path: impl AsRef<Path>) -> io::Result<InputStats> {
        let mut stream = input::Stream::serial(self);
        stream.add_file(path)?;
        stream.finish()
    }

    /// Stream FASTA or FASTQ records from each file in `paths` in order,
    /// auto-detecting gzip compression from content, and add their k-mers to
    /// this counter.
    pub fn add_files<P: AsRef<Path>>(
        &mut self,
        paths: impl IntoIterator<Item = P>,
    ) -> io::Result<InputStats> {
        let mut stream = input::Stream::serial(self);
        for path in paths {
            stream.add_file(path)?;
        }
        stream.finish()
    }

    /// Write headerless `KMER<TAB>COUNT` rows. For k <= 32 they come in the
    /// order of the k-mers' hashes, the same in every run and for any number
    /// of threads; otherwise in unspecified order.
    ///
    /// Reuses stack buffers for packed keys and decimal counts; larger keys are
    /// borrowed from the table. No per-row heap allocation is needed. Wrap files
    /// or stdout in a [`io::BufWriter`] to batch writes. This method does not flush
    /// the writer; write failures may leave partial output.
    pub fn write_tsv(&self, mut writer: impl Write) -> io::Result<()> {
        self.write_tables(&mut writer, 1, |table, writer| match table {
            OutputTable::Packed(counts, first, tables) => {
                let mut slots = Vec::new();
                for partition in first..first + tables {
                    let entries = counts
                        .sorted_partition(partition, &mut slots)
                        .map(|(mixed, count)| (counts.unmix(mixed), count));
                    write_packed64_tsv(entries, self.k, writer)?;
                }
                Ok(())
            }
            OutputTable::Wide(counts, first, tables) => {
                counts.write_tsv_range(first, tables, self.k, writer)
            }
            OutputTable::Bytes(counts) => {
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
        let iter: Box<dyn Iterator<Item = (Vec<u8>, u64)> + '_> = match &self.counts {
            Counts::Packed(counts) => Box::new(
                counts
                    .entries()
                    .map(|(key, count)| (decode_packed(key, self.k), count)),
            ),
            Counts::Wide(counts) => counts.decoded_entries(self.k),
            Counts::Bytes(counts) => Box::new(
                counts
                    .maps
                    .iter()
                    .flat_map(|map| map.iter().map(|(key, &count)| (key.clone(), count))),
            ),
        };
        iter
    }
}

/// Encode `items` pieces with `encode`, writing their bytes in order.
///
/// With more than one piece and thread, pieces are encoded concurrently on
/// up to `threads` scoped threads. Each encoder streams bounded chunks
/// through its own small queue, and the calling thread writes them in order,
/// so buffered output stays proportional to the thread count (about 17 MiB
/// per encoder) rather than the size of a piece.
pub(crate) fn write_ordered(
    threads: usize,
    items: usize,
    writer: &mut dyn Write,
    encode: impl Fn(usize, &mut dyn Write) -> io::Result<()> + Sync,
) -> io::Result<()> {
    let threads = threads.min(items);
    if threads <= 1 {
        return (0..items).try_for_each(|item| encode(item, &mut *writer));
    }
    std::thread::scope(|scope| {
        let (recycle, recycled) = crossbeam_channel::unbounded::<Vec<u8>>();
        let mut streams = Vec::with_capacity(threads);
        for first in 0..threads {
            // Each encoder may run this many chunks ahead of the writer;
            // enough to buffer a typical piece while others are written.
            let (sender, receiver) = crossbeam_channel::bounded(16);
            streams.push(receiver);
            let (encode, recycled) = (&encode, recycled.clone());
            std::thread::Builder::new()
                .name(format!("output-{first}"))
                .spawn_scoped(scope, move || {
                    // Encoder `first` owns pieces first, first + threads, ...
                    for item in (first..items).step_by(threads) {
                        let mut sink = ChunkSink {
                            buffer: recycled.try_recv().unwrap_or_default(),
                            sender: &sender,
                            recycled: &recycled,
                        };
                        if let Err(error) = encode(item, &mut sink).and_then(|()| sink.finish()) {
                            // Forward the encoder's own error; this fails
                            // harmlessly if the writer already stopped.
                            let _ = sender.send(Err(error));
                            return;
                        }
                    }
                })?;
        }
        drop(recycled);
        for index in 0..items {
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
// call for every distinct key.
fn write_packed_tsv<K>(
    entries: impl Iterator<Item = (K, u64)>,
    k: usize,
    writer: &mut (impl Write + ?Sized),
    decode: impl Fn(K, &mut [u8]),
) -> io::Result<()> {
    let mut batch = [0; 16 * 1024];
    let mut offset = 0;
    for (key, count) in entries {
        if offset + k + 22 > batch.len() {
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
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE2 is part of the x86-64 baseline; the stores are in bounds.
    unsafe {
        use core::arch::x86_64::*;
        // Byte i holds bases 4i..4i + 4, the first in its top two bits.
        let bytes = _mm_cvtsi64_si128(aligned.swap_bytes() as i64);
        let two_bits = _mm_set1_epi8(3);
        // The four bases of every byte, as codes. A 16-bit shift moves bits
        // between neighbouring bytes, but never into the two bits kept.
        let first = _mm_and_si128(_mm_srli_epi16::<6>(bytes), two_bits);
        let second = _mm_and_si128(_mm_srli_epi16::<4>(bytes), two_bits);
        let third = _mm_and_si128(_mm_srli_epi16::<2>(bytes), two_bits);
        let fourth = _mm_and_si128(bytes, two_bits);
        // Interleave them back into base order.
        let leading = _mm_unpacklo_epi8(first, second);
        let trailing = _mm_unpacklo_epi8(third, fourth);
        // A, C, G and T are 65, 67, 71 and 84: 65 + 2 * code, plus 2 from G
        // on, plus 11 more for T.
        let letters = |codes| {
            let from_g = _mm_and_si128(_mm_cmpgt_epi8(codes, _mm_set1_epi8(1)), _mm_set1_epi8(2));
            let only_t = _mm_and_si128(_mm_cmpeq_epi8(codes, two_bits), _mm_set1_epi8(11));
            let ascii = _mm_add_epi8(_mm_set1_epi8(b'A' as i8), _mm_add_epi8(codes, codes));
            _mm_add_epi8(ascii, _mm_add_epi8(from_g, only_t))
        };
        let mut dna = [0; 32];
        _mm_storeu_si128(
            dna.as_mut_ptr().cast(),
            letters(_mm_unpacklo_epi16(leading, trailing)),
        );
        _mm_storeu_si128(
            dna.as_mut_ptr().add(16).cast(),
            letters(_mm_unpackhi_epi16(leading, trailing)),
        );
        dna
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
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

fn visit_packed128(
    sequence: &[u8],
    k: usize,
    strand: StrandMode,
    emit: impl FnMut(u64, u64) -> Result<(), Error>,
) -> Result<(), Error> {
    match strand {
        StrandMode::Forward => visit_packed128_record::<false>(sequence, k, emit),
        StrandMode::Canonical => visit_packed128_record::<true>(sequence, k, emit),
    }
}

fn visit_packed128_record<const CANONICAL: bool>(
    sequence: &[u8],
    k: usize,
    mut emit: impl FnMut(u64, u64) -> Result<(), Error>,
) -> Result<(), Error> {
    // k is in 33..=64, so `lo` holds all 64 low bits and `hi` holds 2..=64 high bits.
    // `hi_shift = 2*(k-1) - 64` is always in 0..=62, avoiding variable 128-bit shifts.
    let hi_mask = u64::MAX >> (128 - 2 * k);
    let hi_shift = 2 * (k - 1) - 64;
    let (mut f_hi, mut f_lo) = (0_u64, 0_u64);
    let (mut r_hi, mut r_lo) = (0_u64, 0_u64);
    let mut valid_run = 0;

    for &byte in sequence {
        let base = DNA_BASES[byte as usize];
        if base == u8::MAX {
            f_hi = 0;
            f_lo = 0;
            r_hi = 0;
            r_lo = 0;
            valid_run = 0;
            continue;
        }
        let base = u64::from(base);
        f_hi = ((f_hi << 2) | (f_lo >> 62)) & hi_mask;
        f_lo = (f_lo << 2) | base;
        if CANONICAL {
            r_lo = (r_lo >> 2) | (r_hi << 62);
            r_hi = (r_hi >> 2) | ((base ^ 3) << hi_shift);
        }
        valid_run = (valid_run + 1).min(k);
        if valid_run < k {
            continue;
        }

        // A=0, C=1, G=2, T=3 makes numeric and lexicographic order agree.
        let (hi, lo) = if CANONICAL && (r_hi, r_lo) < (f_hi, f_lo) {
            (r_hi, r_lo)
        } else {
            (f_hi, f_lo)
        };
        emit(hi, lo)?;
    }
    Ok(())
}

fn visit_packed192(
    sequence: &[u8],
    k: usize,
    strand: StrandMode,
    emit: impl FnMut([u64; 2], u64) -> Result<(), Error>,
) -> Result<(), Error> {
    match strand {
        StrandMode::Forward => visit_packed192_record::<false>(sequence, k, emit),
        StrandMode::Canonical => visit_packed192_record::<true>(sequence, k, emit),
    }
}

fn visit_packed192_record<const CANONICAL: bool>(
    sequence: &[u8],
    k: usize,
    mut emit: impl FnMut([u64; 2], u64) -> Result<(), Error>,
) -> Result<(), Error> {
    // k is in 65..=96: w0 (64b), w1 (64b), w2 (2..=64b).
    let top_mask = u64::MAX >> (192 - 2 * k);
    let top_shift = 2 * (k - 1) - 128;
    let (mut f2, mut f1, mut f0) = (0_u64, 0_u64, 0_u64);
    let (mut r2, mut r1, mut r0) = (0_u64, 0_u64, 0_u64);
    let mut valid_run = 0;

    for &byte in sequence {
        let base = DNA_BASES[byte as usize];
        if base == u8::MAX {
            (f2, f1, f0) = (0, 0, 0);
            (r2, r1, r0) = (0, 0, 0);
            valid_run = 0;
            continue;
        }
        let base = u64::from(base);
        f2 = ((f2 << 2) | (f1 >> 62)) & top_mask;
        f1 = (f1 << 2) | (f0 >> 62);
        f0 = (f0 << 2) | base;
        if CANONICAL {
            r0 = (r0 >> 2) | (r1 << 62);
            r1 = (r1 >> 2) | (r2 << 62);
            r2 = (r2 >> 2) | ((base ^ 3) << top_shift);
        }
        valid_run = (valid_run + 1).min(k);
        if valid_run < k {
            continue;
        }
        if CANONICAL && (r2, r1, r0) < (f2, f1, f0) {
            emit([r1, r2], r0)?;
        } else {
            emit([f1, f2], f0)?;
        }
    }
    Ok(())
}

fn visit_packed256(
    sequence: &[u8],
    k: usize,
    strand: StrandMode,
    emit: impl FnMut([u64; 3], u64) -> Result<(), Error>,
) -> Result<(), Error> {
    match strand {
        StrandMode::Forward => visit_packed256_record::<false>(sequence, k, emit),
        StrandMode::Canonical => visit_packed256_record::<true>(sequence, k, emit),
    }
}

fn visit_packed256_record<const CANONICAL: bool>(
    sequence: &[u8],
    k: usize,
    mut emit: impl FnMut([u64; 3], u64) -> Result<(), Error>,
) -> Result<(), Error> {
    // k is in 97..=128: w0..w2 (64b each), w3 (2..=64b).
    let top_mask = u64::MAX >> (256 - 2 * k);
    let top_shift = 2 * (k - 1) - 192;
    let (mut f3, mut f2, mut f1, mut f0) = (0_u64, 0_u64, 0_u64, 0_u64);
    let (mut r3, mut r2, mut r1, mut r0) = (0_u64, 0_u64, 0_u64, 0_u64);
    let mut valid_run = 0;

    for &byte in sequence {
        let base = DNA_BASES[byte as usize];
        if base == u8::MAX {
            (f3, f2, f1, f0) = (0, 0, 0, 0);
            (r3, r2, r1, r0) = (0, 0, 0, 0);
            valid_run = 0;
            continue;
        }
        let base = u64::from(base);
        f3 = ((f3 << 2) | (f2 >> 62)) & top_mask;
        f2 = (f2 << 2) | (f1 >> 62);
        f1 = (f1 << 2) | (f0 >> 62);
        f0 = (f0 << 2) | base;
        if CANONICAL {
            r0 = (r0 >> 2) | (r1 << 62);
            r1 = (r1 >> 2) | (r2 << 62);
            r2 = (r2 >> 2) | (r3 << 62);
            r3 = (r3 >> 2) | ((base ^ 3) << top_shift);
        }
        valid_run = (valid_run + 1).min(k);
        if valid_run < k {
            continue;
        }
        if CANONICAL && (r3, r2, r1, r0) < (f3, f2, f1, f0) {
            emit([r1, r2, r3], r0)?;
        } else {
            emit([f1, f2, f3], f0)?;
        }
    }
    Ok(())
}

fn decode_packed(key: u64, k: usize) -> Vec<u8> {
    decode_packed32(key, k)[..k].to_vec()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn decode_packed_into(mut key: u64, dna: &mut [u8]) {
    for base in dna.iter_mut().rev() {
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

fn visit_bytes(
    sequence: &[u8],
    k: usize,
    strand: StrandMode,
    mut emit: impl FnMut(&[u8]) -> Result<(), Error>,
) -> Result<(), Error> {
    if sequence.len() < k {
        return Ok(());
    }
    let mut normalized = Vec::with_capacity(sequence.len());
    let mut reverse_complement = if strand == StrandMode::Canonical {
        vec![0; k]
    } else {
        Vec::new()
    };

    for &byte in sequence {
        let base = DNA_BASES[byte as usize];
        if base == u8::MAX {
            normalized.clear();
            continue;
        }
        normalized.push(b"ACGT"[base as usize]);
        if normalized.len() < k {
            continue;
        }
        let kmer = &normalized[normalized.len() - k..];
        let key = if strand == StrandMode::Canonical {
            for (complement, &b) in reverse_complement.iter_mut().zip(kmer.iter().rev()) {
                *complement = b"TGCA"[DNA_BASES[b as usize] as usize];
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
    if let Some(count) = map.get_mut(&key) {
        *count = count.checked_add(delta).ok_or(Error::CountOverflow)?;
    } else {
        map.try_reserve(1).map_err(|_| Error::ReserveFailed)?;
        map.insert(key, delta);
    }
    Ok(())
}

fn add_byte_delta<S: BuildHasher>(
    map: &mut HashMap<Vec<u8>, u64, S>,
    key: &[u8],
    delta: u64,
) -> Result<(), Error> {
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
            Counts::Wide(counts) => counts.set_counts(count),
            Counts::Bytes(counts) => counts.set_counts(count),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Representative k values covering every backend width and boundary.
    const TEST_KS: &[usize] = &[1, 3, 4, 31, 32, 33, 40, 41, 48, 49, 64, 65, 96, 97, 128, 129];

    fn assert_counts(counter: &Counter, expected: &[(&str, u64)]) {
        let expected: HashMap<Vec<u8>, u64> = expected
            .iter()
            .map(|(key, count)| (key.as_bytes().to_vec(), *count))
            .collect();
        assert_eq!(counter.counts().collect::<HashMap<_, _>>(), expected);
    }

    #[test]
    fn whole_key_decoder_matches_simple_loop_for_every_length() {
        let mut state = 42_u64;
        for _ in 0..1024 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            for k in 1..=32 {
                let key = state & (u64::MAX >> (64 - 2 * k));
                let expected: Vec<u8> = (0..k)
                    .map(|index| b"ACGT"[((key >> (2 * (k - 1 - index))) & 3) as usize])
                    .collect();
                assert_eq!(&decode_packed32(key, k)[..k], expected, "k={k}");
            }
        }
    }

    #[test]
    fn full_width_keys_preserve_high_bits_and_leading_as() {
        for k in [32, 40, 48, 64, 96, 128] {
            let mut counter = Counter::new(k, StrandMode::Forward).unwrap();
            counter.add_record(&vec![b'T'; k + 1]).unwrap();
            counter.add_record(&vec![b'A'; k]).unwrap();
            let leading_as = format!("{}C", "A".repeat(k - 1));
            counter.add_record(leading_as.as_bytes()).unwrap();
            assert_counts(
                &counter,
                &[(&"T".repeat(k), 2), (&"A".repeat(k), 1), (&leading_as, 1)],
            );

            let mut canonical = Counter::new(k, StrandMode::Canonical).unwrap();
            canonical.add_record(&vec![b'T'; k + 1]).unwrap();
            canonical.add_record(&vec![b'A'; k]).unwrap();
            assert_counts(&canonical, &[(&"A".repeat(k), 3)]);
        }
    }

    #[test]
    fn every_invalid_byte_breaks_runs_for_all_backends() {
        for k in [3, 32, 33, 40, 48, 64, 65, 96, 97, 128, 129] {
            let mut record = Vec::new();
            for byte in 0..=u8::MAX {
                if b"ACGTacgt".contains(&byte) {
                    continue;
                }
                record.extend(std::iter::repeat_n(b'A', k - 1));
                record.push(byte);
            }
            record.extend(std::iter::repeat_n(b'A', k));
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let mut counter = Counter::new(k, strand).unwrap();
                counter.add_record(&record).unwrap();
                assert_counts(&counter, &[(&"A".repeat(k), 1)]);
            }
        }
    }

    #[test]
    fn capacity_hint_preserves_counts_and_allows_growth() {
        for &k in TEST_KS {
            for strand in [StrandMode::Canonical, StrandMode::Forward] {
                let mut expected = Counter::new(k, strand).unwrap();
                let mut reserved = Counter::with_capacity(k, strand, 1).unwrap();
                assert!(reserved.is_empty());
                let capacity = match &reserved.counts {
                    Counts::Packed(counts) => counts.capacity(),
                    Counts::Wide(counts) => counts.capacity(),
                    Counts::Bytes(counts) => counts.capacity(),
                };
                assert!(capacity >= 1);
                for record in [b"ACGTTACGAGTCAGATTCGAC".repeat(12), vec![b'T'; 160]] {
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
            Counter::new(0, StrandMode::Canonical),
            Err(Error::InvalidK)
        ));
        assert!(matches!(
            Counter::with_capacity(0, StrandMode::Canonical, usize::MAX),
            Err(Error::InvalidK)
        ));
        for k in [31, 33, 41, 49, 65, 97, 129] {
            assert!(matches!(
                Counter::with_capacity(k, StrandMode::Canonical, usize::MAX),
                Err(Error::ReserveFailed)
            ));
        }
    }

    #[test]
    fn tsv_matches_iterator_for_all_backends_and_strands() {
        for &k in TEST_KS {
            for strand in [StrandMode::Canonical, StrandMode::Forward] {
                let mut counter = Counter::new(k, strand).unwrap();
                let mut empty = Vec::new();
                counter.write_tsv(&mut empty).unwrap();
                assert!(empty.is_empty());
                for record in [
                    b"acgTTacGAGTCAGATTCGAC".repeat(10),
                    vec![b'T'; 140],
                    vec![b'A'; 140],
                    b"ACGTNACGT.--ACGT".repeat(10),
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
                let rows = |tsv: &[u8]| {
                    let mut rows: Vec<Vec<u8>> =
                        tsv.split(|&b| b == b'\n').map(Vec::from).collect();
                    rows.sort();
                    rows
                };
                assert_eq!(rows(&actual), rows(&expected), "k={k}, strand={strand:?}");
            }
        }
    }

    #[test]
    fn tsv_formats_decimal_boundaries_and_maximum_counts() {
        for k in [1, 32, 33, 64, 97, 129] {
            let mut counter = Counter::new(k, StrandMode::Forward).unwrap();
            counter.add_record(&vec![b'A'; k]).unwrap();
            let mut values = vec![u64::MAX];
            for exponent in 0..=19 {
                let power = 10_u64.pow(exponent);
                values.extend([power - 1, power, power + 1]);
            }
            for value in values.into_iter().filter(|&value| value > 0) {
                counter.counts.set_counts(value);
                let mut output = Vec::new();
                counter.write_tsv(&mut output).unwrap();
                assert_eq!(output, format!("{}\t{value}\n", "A".repeat(k)).as_bytes());
            }
        }
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
        for k in [3, 33, 129] {
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

        // Exercise every backend, full-width keys, palindromes, and long valid runs.
        records.push(b"ACGT".to_vec());
        for len in [31, 32, 33, 64, 65, 96, 97, 128, 129, 1024] {
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

        for &k in TEST_KS {
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
    fn overflow_keeps_earlier_windows_and_stops_before_later_ones() {
        for k in [1, 3, 32, 33, 64, 65, 97, 128, 129] {
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
}
