//! Counting within a memory limit, for inputs whose k-mers do not fit in RAM.
//!
//! Counting runs in two phases. In the first, workers cut every read into
//! super-k-mers: runs of consecutive k-mers that share a minimizer (the
//! smallest hash among a k-mer's canonical m-mers). A k-mer and its reverse
//! complement have the same minimizer wherever they occur, so the minimizer
//! picks one of `BINS` bins, and every occurrence of a k-mer lands in the same
//! bin. A super-k-mer is stored as its bases, two bits each, so a bin takes
//! about a seventh of the space of the k-mers it holds. Bins are kept in
//! memory as blocks while half the memory limit allows; later blocks are
//! written to one temporary file shared by all bins, their locations kept in
//! memory.
//!
//! In the second phase each bin is counted on its own, by the same compact
//! tables as in memory, and its k-mers are written out before the tables are
//! reused for the next bin. Memory then follows the bins being counted at
//! once, not the input: a bin is admitted only when its estimated tables fit
//! in the remaining budget, and a bin too large for the whole budget is
//! counted in several passes, each taking a slice of its k-mers.
//!
//! Reads overlap, so at any depth of coverage most super-k-mers occur many
//! times over. As a bin is read, equal records are merged in a hash table:
//! each distinct record is expanded once, and its k-mers counted with the
//! number of copies. Input without such repeats, a genome for instance,
//! shows it in the first bin, and most later bins skip the merging.
//!
//! A super-k-mer record is one header byte, `skip << 6 | (kmers - 1)`, then
//! `ceil((skip + kmers + k - 1) / 4)` bytes of bases, the first base in the
//! top two bits of each byte. The first `skip` bases are padding: records are
//! copied byte-aligned out of the packed read. Merging first shifts the
//! padding out and clears the unused bits, so that equal super-k-mers become
//! equal bytes.

use std::fs::File;
use std::hash::BuildHasher;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::parallel::{ParallelCounter, ParallelInput};
use crate::table::{Layout, Load, Mixer, PackedCounts};
use crate::{ChunkSink, DNA_BASES, EncodedChunk, Error, StrandMode, database};

/// Bins the input is split into; a power of two.
const BINS: usize = 512;
/// Longest minimizer. Shorter k use half their length.
const MINIMIZER: usize = 11;
/// Most k-mers in one super-k-mer, so that the count fits its header.
const MAX_RUN: usize = 64;
/// Largest block of one bin's records, the unit kept in memory or written.
const MAX_BLOCK: usize = 256 * 1024;
/// The smallest block: more than the longest record.
const MIN_BLOCK: usize = 64;
/// Bytes a worker collects for a bin before taking the bin's lock.
const LOCAL: usize = 4 * 1024;
/// Partition bits of a bin's tables. A bin's k-mers are spread over this many
/// tables so that each is small enough to be updated in cache, as in
/// `crate::parallel`.
const PARTITION_BITS: u32 = 10;
/// Updates collected per table of a group before the group is applied.
const BURST: usize = if cfg!(test) { 8 } else { 512 };
/// Slots of a counting thread's table of repeated records: within an eighth
/// of the memory limit between the threads.
const MAX_REPEATS: usize = 1 << 20;
/// Bins a thread counts without merging after one with few repeats.
const UNMERGED: usize = if cfg!(test) { 1 } else { 7 };
const MIN_REPEATS: usize = if cfg!(test) { 16 } else { 1 << 10 };
/// Table bytes assumed per distinct k-mer when admitting a bin: 64-byte
/// buckets of eight slots at most 7/8 full, plus room left by growth steps.
const KEY_BYTES: u64 = 12;

fn minimizer_length(k: usize) -> usize {
    if k >= 2 * MINIMIZER {
        MINIMIZER
    } else {
        k.div_ceil(2)
    }
}

/// A bijective 32-bit mix, giving minimizers a pseudo-random order.
#[inline(always)]
fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// The temporary file shared by all bins. Blocks are appended at offsets
/// reserved atomically, so writers never wait for each other.
struct Spill {
    file: File,
    end: AtomicU64,
    // Positional I/O needs no lock on Unix.
    #[cfg(not(unix))]
    position: Mutex<()>,
    #[cfg(not(unix))]
    path: PathBuf,
}

impl Spill {
    fn create(directory: &Path) -> io::Result<Self> {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let path: PathBuf = directory.join(format!("kbolt-{}-{unique:x}.bins", std::process::id()));
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| {
                io::Error::new(error.kind(), format!("{}: {error}", path.display()))
            })?;
        // An unlinked file disappears with the process, however it ends.
        #[cfg(unix)]
        std::fs::remove_file(&path)?;
        Ok(Self {
            file,
            end: AtomicU64::new(0),
            #[cfg(not(unix))]
            position: Mutex::new(()),
            #[cfg(not(unix))]
            path,
        })
    }

    fn append(&self, bytes: &[u8]) -> io::Result<u64> {
        let offset = self.end.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        #[cfg(unix)]
        std::os::unix::fs::FileExt::write_all_at(&self.file, bytes, offset)?;
        #[cfg(not(unix))]
        {
            let _held = self
                .position
                .lock()
                .map_err(|_| io::Error::other("poisoned"))?;
            (&self.file).seek(SeekFrom::Start(offset))?;
            (&self.file).write_all(bytes)?;
        }
        Ok(offset)
    }

    fn read(&self, bytes: &mut [u8], offset: u64) -> io::Result<()> {
        #[cfg(unix)]
        return std::os::unix::fs::FileExt::read_exact_at(&self.file, bytes, offset);
        #[cfg(not(unix))]
        {
            let _held = self
                .position
                .lock()
                .map_err(|_| io::Error::other("poisoned"))?;
            (&self.file).seek(SeekFrom::Start(offset))?;
            io::Read::read_exact(&mut &self.file, bytes)
        }
    }
}

#[cfg(not(unix))]
impl Drop for Spill {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// One bin's records, as blocks of whole records in no particular order.
#[derive(Default)]
struct Bin {
    /// The block being filled.
    open: Vec<u8>,
    /// Full blocks kept in memory.
    held: Vec<Vec<u8>>,
    /// Full blocks in the file: offset and length.
    written: Vec<(u64, u32)>,
    kmers: u64,
}

/// The first failure to write the temporary file, kept for its message.
type Failure = Arc<Mutex<Option<io::Error>>>;

/// Shared state of the splitting phase.
pub(crate) struct Bins {
    k: usize,
    minimizer: usize,
    /// Bytes in a full block.
    block: usize,
    /// Bytes of full blocks that may stay in memory, and how many do.
    limit: usize,
    held: AtomicUsize,
    spill: Spill,
    bins: Vec<Mutex<Bin>>,
    failure: Failure,
}

/// One worker's splitting state.
pub(crate) struct BinBuffers {
    /// Records not yet handed to their bin, and the k-mers they hold.
    buffers: Vec<Vec<u8>>,
    kmers: Vec<u64>,
    /// An empty block to replace the next one this worker fills.
    spare: Vec<u8>,
    /// The current run of valid bases, four per byte.
    packed: Vec<u8>,
    /// Hashes of the latest m-mers, by m-mer index modulo its length; a
    /// k-mer spans at most 32 of them.
    hashes: [u32; 32],
}

impl Bins {
    pub(crate) fn buffers(&self) -> BinBuffers {
        BinBuffers {
            buffers: (0..BINS).map(|_| Vec::new()).collect(),
            kmers: vec![0; BINS],
            spare: Vec::new(),
            packed: Vec::new(),
            hashes: [0; 32],
        }
    }

    /// Cut `record` into super-k-mers and append each to its bin.
    pub(crate) fn split(&self, record: &[u8], buffers: &mut BinBuffers) -> Result<(), Error> {
        let (k, m) = (self.k, self.minimizer);
        let window = k - m + 1;
        let mask = u32::MAX >> (32 - 2 * m);
        let high = 2 * (m - 1);
        // Bases in the current run of valid bases, and its rolling m-mer.
        let (mut length, mut forward, mut reverse) = (0_usize, 0_u32, 0_u32);
        // The smallest hash among the last `window` m-mers, and its index.
        let (mut least, mut least_at) = (u32::MAX, 0_usize);
        // The open super-k-mer: its first k-mer and its bin.
        let (mut start, mut bin) = (0_usize, 0_usize);
        buffers.packed.clear();
        for &byte in record {
            let code = DNA_BASES[byte as usize];
            if code == u8::MAX {
                if length >= k {
                    self.emit(buffers, bin, start, length - k + 1 - start)?;
                }
                buffers.packed.clear();
                (length, least, start) = (0, u32::MAX, 0);
                continue;
            }
            match length % 4 {
                0 => buffers.packed.push(code << 6),
                phase => *buffers.packed.last_mut().unwrap() |= code << (6 - 2 * phase),
            }
            let code = u32::from(code);
            forward = ((forward << 2) | code) & mask;
            reverse = (reverse >> 2) | ((code ^ 3) << high);
            length += 1;
            if length < m {
                continue;
            }
            let at = length - m;
            let hash = mix(forward.min(reverse));
            buffers.hashes[at % 32] = hash;
            if hash <= least {
                (least, least_at) = (hash, at);
            } else if least_at + window <= at {
                // The smallest hash left the window; find the next (the
                // latest among equals, so that it stays longest).
                least = u32::MAX;
                for index in (at + 1 - window..=at).rev() {
                    let hash = buffers.hashes[index % 32];
                    if hash < least {
                        (least, least_at) = (hash, index);
                    }
                }
            }
            if length < k {
                continue;
            }
            let kmer = length - k;
            // The order of hashes follows their high bits; the low bits are
            // free to spread minimizers over the bins.
            let next = least as usize & (BINS - 1);
            if kmer == start {
                bin = next;
            } else if next != bin || kmer - start == MAX_RUN {
                self.emit(buffers, bin, start, kmer - start)?;
                (start, bin) = (kmer, next);
            }
        }
        if length >= k {
            self.emit(buffers, bin, start, length - k + 1 - start)?;
        }
        Ok(())
    }

    /// Append the super-k-mer of `kmers` k-mers starting at base `start` of
    /// the current run.
    #[inline(always)]
    fn emit(
        &self,
        buffers: &mut BinBuffers,
        bin: usize,
        start: usize,
        kmers: usize,
    ) -> Result<(), Error> {
        let bytes = start / 4..(start + kmers + self.k - 1).div_ceil(4);
        if buffers.buffers[bin].len() + 1 + bytes.len() > LOCAL.min(self.block) {
            self.hand(buffers, bin)?;
        }
        let buffer = &mut buffers.buffers[bin];
        buffer.push(((start % 4) << 6 | (kmers - 1)) as u8);
        buffer.extend_from_slice(&buffers.packed[bytes]);
        buffers.kmers[bin] += kmers as u64;
        Ok(())
    }

    /// Hand a worker's records for `bin` to the bin.
    #[cold]
    fn hand(&self, buffers: &mut BinBuffers, bin: usize) -> Result<(), Error> {
        let records = &mut buffers.buffers[bin];
        let mut full = None;
        {
            let mut shared = self.bins[bin].lock().map_err(|_| Error::WorkerFailed)?;
            shared.kmers += std::mem::take(&mut buffers.kmers[bin]);
            if shared.open.len() + records.len() > self.block {
                // The block is full. Keep it while memory allows; otherwise
                // write it below, outside the lock.
                let block = std::mem::replace(&mut shared.open, std::mem::take(&mut buffers.spare));
                if self.held.fetch_add(block.len(), Ordering::Relaxed) + block.len() <= self.limit {
                    shared.held.push(block);
                } else {
                    self.held.fetch_sub(block.len(), Ordering::Relaxed);
                    full = Some(block);
                }
            }
            if shared.open.capacity() == 0 {
                shared.open.reserve_exact(self.block);
            }
            shared.open.extend_from_slice(records);
        }
        records.clear();
        if records.capacity() == 0 {
            records.reserve_exact(LOCAL.min(self.block));
        }
        let Some(mut block) = full else {
            return Ok(());
        };
        match self.spill.append(&block) {
            Ok(offset) => {
                let mut shared = self.bins[bin].lock().map_err(|_| Error::WorkerFailed)?;
                shared.written.push((offset, block.len() as u32));
                block.clear();
                buffers.spare = block;
                Ok(())
            }
            Err(error) => {
                let mut failure = self
                    .failure
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                failure.get_or_insert(error);
                Err(Error::TemporaryFile)
            }
        }
    }

    /// Hand over everything a finishing worker still holds.
    pub(crate) fn flush(&self, buffers: &mut BinBuffers) -> Result<(), Error> {
        (0..BINS).try_for_each(|bin| match buffers.buffers[bin].is_empty() {
            true => Ok(()),
            false => self.hand(buffers, bin),
        })
    }
}

/// The low bit of every two-bit base.
const ODD_BITS: u64 = 0x5555_5555_5555_5555;

/// The record at the start of `block`, header included.
#[inline(always)]
fn first_record(block: &[u8], k: usize) -> io::Result<&[u8]> {
    let (skip, kmers) = (usize::from(block[0] >> 6), usize::from(block[0] & 63) + 1);
    block
        .get(..1 + (skip + kmers + k - 1).div_ceil(4))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated record in the temporary bin file",
            )
        })
}

/// A record without padding: its header is `kmers - 1`, its bases start at
/// the first byte, and every bit after the last base is zero.
#[inline(always)]
fn unpadded(record: &[u8], k: usize) -> [u8; RECORD] {
    let (skip, kmers) = (u32::from(record[0] >> 6), usize::from(record[0] & 63) + 1);
    let mut bytes = [0_u8; 32];
    bytes[..record.len() - 1].copy_from_slice(&record[1..]);
    let word =
        |index: usize| u64::from_be_bytes(bytes[8 * index..8 * index + 8].try_into().unwrap());
    let mut unpadded = [0; RECORD];
    unpadded[0] = (kmers - 1) as u8;
    // Bits of bases still to keep, from the most significant on.
    let mut bits = 2 * (kmers + k - 1);
    for index in 0..3 {
        let shifted = match 2 * skip {
            0 => word(index),
            shift => (word(index) << shift) | (word(index + 1) >> (64 - shift)),
        };
        let kept = bits.min(64);
        let cleared = if kept == 0 {
            0
        } else {
            shifted & (u64::MAX << (64 - kept))
        };
        unpadded[1 + 8 * index..9 + 8 * index].copy_from_slice(&cleared.to_be_bytes());
        bits -= kept;
    }
    unpadded
}

/// Visit the k-mers of one super-k-mer record, padded or not.
#[inline(always)]
fn expand(
    record: &[u8],
    k: usize,
    canonical: bool,
    mut emit: impl FnMut(u64) -> Result<(), Error>,
) -> Result<(), Error> {
    let mask = u64::MAX >> (64 - 2 * k);
    let high = 2 * (k - 1);
    let (skip, kmers) = (usize::from(record[0] >> 6), usize::from(record[0] & 63) + 1);
    let data = &record[1..];
    // The first k-mer is read whole: its 2k bits start `skip` bases into the
    // record, within the first 9 bytes.
    let mut head = [0_u8; 16];
    let whole = data.len().min(16);
    head[..whole].copy_from_slice(&data[..whole]);
    let mut forward = (u128::from_be_bytes(head) >> (128 - 2 * (skip + k))) as u64 & mask;
    // Its reverse complement: complement every base, then reverse the order
    // of the two-bit bases.
    let flipped = (!forward).reverse_bits();
    let mut reverse = (((flipped >> 1) & ODD_BITS) | ((flipped & ODD_BITS) << 1)) >> (64 - 2 * k);
    for index in skip + k..skip + k - 1 + kmers {
        emit(if canonical {
            forward.min(reverse)
        } else {
            forward
        })?;
        let code = u64::from(data[index / 4] >> (6 - 2 * (index % 4))) & 3;
        forward = ((forward << 2) | code) & mask;
        reverse = (reverse >> 2) | ((code ^ 3) << high);
    }
    emit(if canonical {
        forward.min(reverse)
    } else {
        forward
    })
}

/// One bin's count tables, with the updates collected for them.
///
/// A k-mer is not counted when it is expanded. Its mixed key, with the
/// number of occurrences to add above it, joins the updates of its group of
/// tables; a full group is split by table, and each
/// table then receives its updates in one burst while it sits in cache.
/// Going through groups keeps every step to a few dozen sequential streams:
/// scattering straight to a thousand tables would miss the cache on nearly
/// every k-mer.
struct Tables {
    counts: PackedCounts,
    mixer: Mixer,
    layout: Layout,
    /// Tables in a group, and the key bits below a group's index.
    grouped: usize,
    group_shift: u32,
    /// Room for `grouped * BURST` updates per group, and how many are there.
    pending: Vec<u64>,
    filled: Vec<usize>,
    /// Scratch for splitting a group's updates by table.
    split: Vec<u64>,
    ends: Vec<usize>,
}

impl Tables {
    fn new(k: usize) -> Result<Self, Error> {
        let counts = PackedCounts::with_capacity(k, PARTITION_BITS, 0, Load::Dense)?;
        let layout = counts.layout();
        let table_bits = counts.tables().len().trailing_zeros();
        let grouped = 1_usize << (table_bits - table_bits / 2);
        let groups = counts.tables().len() / grouped;
        Ok(Self {
            mixer: counts.mixer(),
            layout,
            counts,
            grouped,
            group_shift: layout.remainder_bits + grouped.trailing_zeros(),
            pending: vec![0; groups * grouped * BURST],
            filled: vec![0; groups],
            split: Vec::new(),
            ends: Vec::new(),
        })
    }

    /// Count `weight` (at least one) occurrences of the k-mer whose mixed
    /// key is `mixed`.
    #[inline(always)]
    fn push(&mut self, mixed: u64, mut weight: u64) -> Result<(), Error> {
        let capacity = self.grouped * BURST;
        // The group index leaves at least two bits above the rest of the key
        // (a 64-bit key always has group bits), which hold the weight.
        let group = (mixed >> self.group_shift) as usize;
        let (low, heaviest) = (
            mixed ^ ((group as u64) << self.group_shift),
            u64::MAX >> self.group_shift,
        );
        loop {
            let part = weight.min(heaviest);
            let filled = self.filled[group];
            self.pending[group * capacity + filled] = (part << self.group_shift) | low;
            self.filled[group] = filled + 1;
            if filled + 1 == capacity {
                self.apply(group)?;
            }
            weight -= part;
            if weight == 0 {
                return Ok(());
            }
        }
    }

    #[inline(never)]
    fn apply(&mut self, group: usize) -> Result<(), Error> {
        let capacity = self.grouped * BURST;
        let filled = std::mem::take(&mut self.filled[group]);
        crate::table::add_split(
            &mut self.counts.tables_mut()[group * self.grouped..(group + 1) * self.grouped],
            self.layout,
            &self.pending[group * capacity..group * capacity + filled],
            &mut self.split,
            &mut self.ends,
            |table, updates| table.add_burst(updates, self.group_shift),
        )
    }

    /// Apply every collected update.
    fn finish(&mut self) -> Result<(), Error> {
        (0..self.filled.len()).try_for_each(|group| self.apply(group))
    }
}

/// Table memory shared by the bins being counted at once.
struct Budget {
    total: u64,
    free: Mutex<u64>,
    released: Condvar,
}

struct Held<'a>(&'a Budget, u64);

impl Budget {
    /// Wait until `bytes` (at most the whole budget) are free, and take them.
    fn take(&self, bytes: u64) -> Held<'_> {
        let bytes = bytes.min(self.total);
        let mut free = self
            .free
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        while *free < bytes {
            free = self
                .released
                .wait(free)
                .unwrap_or_else(|poison| poison.into_inner());
        }
        *free -= bytes;
        Held(self, bytes)
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        *self
            .0
            .free
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) += self.1;
        self.0.released.notify_all();
    }
}

/// An exact streaming counter for k <= 32 that keeps its k-mers on disk.
///
/// Input is submitted as to a [`ParallelCounter`]. Memory stays near the
/// limit given at construction however many distinct k-mers the input holds;
/// the limit is a target for the count tables and buffers, not a hard cap.
/// Up to half of it holds the split input; the rest of the input goes to a
/// temporary file, a little under one byte per input base.
pub struct DiskCounter {
    parallel: ParallelCounter,
    failure: Failure,
    memory: u64,
}

impl DiskCounter {
    /// Start `threads` workers splitting input into bins, in memory and then
    /// in a temporary file under `directory`, to be counted within about
    /// `memory` bytes.
    ///
    /// Rejects k = 0 and k > 32 with `Error::InvalidK`.
    pub fn new(
        k: usize,
        strand: StrandMode,
        threads: usize,
        memory: u64,
        directory: &Path,
    ) -> io::Result<Self> {
        if !(1..=32).contains(&k) || threads == 0 {
            return Err(io::Error::other(if threads == 0 {
                Error::InvalidThreads
            } else {
                Error::InvalidK
            }));
        }
        let failure = Failure::default();
        // Half the memory may hold bins; a quarter of that, their open blocks.
        let blocks = usize::try_from(memory / 2).unwrap_or(usize::MAX);
        let block = (blocks / BINS / 4).clamp(MIN_BLOCK, MAX_BLOCK);
        let bins = Bins {
            k,
            minimizer: minimizer_length(k),
            block,
            limit: blocks.saturating_sub(BINS * block),
            held: AtomicUsize::new(0),
            spill: Spill::create(directory)?,
            bins: (0..BINS).map(|_| Mutex::default()).collect(),
            failure: Arc::clone(&failure),
        };
        Ok(Self {
            parallel: ParallelCounter::with_bins(k, strand, threads, bins)
                .map_err(io::Error::other)?,
            failure,
            memory,
        })
    }

    /// Create an independent input producer; see [`ParallelCounter::input`].
    pub fn input(&self) -> ParallelInput<'_> {
        self.parallel.input()
    }

    /// Wait for all submitted input to be split into bins.
    pub fn finish(self) -> io::Result<DiskCounts> {
        let (k, strand, threads) = self.parallel.shape();
        match self.parallel.into_bins() {
            Ok(bins) => Ok(DiskCounts {
                k,
                strand,
                threads,
                memory: self.memory,
                spill: bins.spill,
                bins: bins.bins,
            }),
            Err(error) => Err(self
                .failure
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .take()
                .unwrap_or_else(|| io::Error::other(error))),
        }
    }
}

/// Input split into bins, ready to be counted and written out.
pub struct DiskCounts {
    k: usize,
    strand: StrandMode,
    threads: usize,
    memory: u64,
    spill: Spill,
    /// Each bin is taken, leaving it empty, by the thread that counts it.
    bins: Vec<Mutex<Bin>>,
}

impl DiskCounts {
    /// Count on `threads` threads (at least one) instead of as many as split
    /// the input; threads that were reading input are free by now.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// K-mers in the input, counting repeats.
    pub fn kmers(&self) -> u64 {
        self.bins
            .iter()
            .map(|bin| bin.lock().map_or(0, |bin| bin.kmers))
            .sum()
    }

    /// Bytes written to the temporary file.
    pub fn temporary_bytes(&self) -> u64 {
        self.spill.end.load(Ordering::Relaxed)
    }

    /// Count every bin and write headerless `KMER<TAB>COUNT` rows, in
    /// unspecified order. Returns the number of distinct k-mers. Does not
    /// flush; failures may leave partial output.
    pub fn write_tsv(self, mut writer: impl Write) -> io::Result<u64> {
        let k = self.k;
        self.count(&mut writer, |counts, writer| {
            crate::write_packed64_tsv(counts.entries(), k, writer)
        })
    }

    /// Count every bin and write a version-1 database (see
    /// [`crate::database`]). The entry count is known only once every bin has
    /// been counted, so the header is completed last and the writer must
    /// seek. Returns the number of distinct k-mers. Does not flush; failures
    /// may leave partial output.
    pub fn write_binary(self, mut writer: impl Write + Seek) -> io::Result<u64> {
        let (k, strand) = (self.k, self.strand);
        let start = writer.stream_position()?;
        writer.write_all(&database::header(k, strand, 0))?;
        let distinct = self.count(&mut writer, |counts, writer| {
            database::write_packed_binary(
                counts
                    .entries()
                    .map(|(key, count)| (key.to_le_bytes(), count)),
                k.div_ceil(4),
                writer,
            )
        })?;
        let end = writer.stream_position()?;
        writer.seek(SeekFrom::Start(start))?;
        writer.write_all(&database::header(k, strand, distinct))?;
        writer.seek(SeekFrom::Start(end))?;
        Ok(distinct)
    }

    /// Count bins on `threads` threads, encoding each with `encode` and
    /// writing the encoded chunks, in no particular order, from this thread.
    fn count(
        &self,
        writer: &mut dyn Write,
        encode: impl Fn(&PackedCounts, &mut dyn Write) -> io::Result<()> + Sync,
    ) -> io::Result<u64> {
        // Three eighths for tables, and up to an eighth for merging repeats
        // (see `count_bins`). Bins held in memory take up to half until they
        // are counted; update and output buffers come on top.
        let total = (self.memory / 8 * 3).max(1);
        let budget = Budget {
            total,
            free: Mutex::new(total),
            released: Condvar::new(),
        };
        let progress = Progress::default();
        let (sender, receiver) = crossbeam_channel::bounded::<EncodedChunk>(2 * self.threads);
        let (recycle, recycled) = crossbeam_channel::unbounded::<Vec<u8>>();
        std::thread::scope(|scope| {
            for index in 0..self.threads {
                let (sender, recycled) = (sender.clone(), recycled.clone());
                let (budget, progress, encode) = (&budget, &progress, &encode);
                std::thread::Builder::new()
                    .name(format!("bin-{index}"))
                    .spawn_scoped(scope, move || {
                        let result = self.count_bins(budget, progress, encode, &sender, &recycled);
                        if let Err(error) = result {
                            // Fails harmlessly if the writer already stopped.
                            let _ = sender.send(Err(error));
                        }
                    })?;
            }
            drop((sender, recycled));
            // Returning early drops the receiver, which stops the workers.
            for chunk in receiver {
                if let Some(mut chunk) = chunk? {
                    writer.write_all(&chunk)?;
                    chunk.clear();
                    let _ = recycle.send(chunk);
                }
            }
            Ok(())
        })
        .map(|()| progress.distinct.load(Ordering::Relaxed))
    }

    /// One counting thread: take bins until none are left.
    fn count_bins(
        &self,
        budget: &Budget,
        progress: &Progress,
        encode: &(impl Fn(&PackedCounts, &mut dyn Write) -> io::Result<()> + Sync),
        sender: &crossbeam_channel::Sender<EncodedChunk>,
        recycled: &crossbeam_channel::Receiver<Vec<u8>>,
    ) -> io::Result<()> {
        let mut tables = Tables::new(self.k).map_err(io::Error::other)?;
        // An eighth of the memory, between the threads, merges equal records.
        let slots = usize::try_from(self.memory / 8).unwrap_or(usize::MAX)
            / self.threads
            / std::mem::size_of::<Repeat>();
        let slots = (slots / 2 + 1)
            .next_power_of_two()
            .clamp(MIN_REPEATS, MAX_REPEATS);
        let mut repeats = Repeats::new(slots);
        // Bins this thread still counts without merging; see below.
        let mut unmerged = 0;
        let mut block = Vec::new();
        while let Some(bin) = self.bins.get(progress.next.fetch_add(1, Ordering::Relaxed)) {
            // Taking the bin frees its blocks once it has been counted.
            let bin = std::mem::take(&mut *bin.lock().map_err(|_| io::Error::other("poisoned"))?);
            if bin.kmers == 0 {
                continue;
            }
            let needed = progress.estimate(bin.kmers).saturating_mul(KEY_BYTES);
            // A bin too large for the whole budget is counted in passes, each
            // taking the k-mers whose mixed key falls in one residue class.
            let passes = needed.div_ceil(budget.total);
            let needed = needed / passes;
            if tables.counts.bytes() as u64 > needed.saturating_mul(2) {
                // Tables grown for a much larger bin; start again small.
                tables = Tables::new(self.k).map_err(io::Error::other)?;
            }
            let _held = budget.take(needed.max(tables.counts.bytes() as u64));
            let mixer = tables.mixer;
            for pass in 0..passes {
                let mut count = |record: &[u8], copies: u64| {
                    expand(
                        record,
                        self.k,
                        self.strand == StrandMode::Canonical,
                        |key| {
                            let mixed = mixer.mix(key);
                            if passes == 1 || mixed % passes == pass {
                                tables.push(mixed, copies)?;
                            }
                            Ok(())
                        },
                    )
                };
                let mut read = |block: &[u8]| match unmerged {
                    0 => repeats.add(block, self.k, &mut count),
                    _ => Repeats::pass(block, self.k, &mut count),
                };
                for block in bin.held.iter().chain(std::iter::once(&bin.open)) {
                    read(block)?;
                }
                for &(offset, len) in &bin.written {
                    block.resize(len as usize, 0);
                    self.spill.read(&mut block, offset)?;
                    read(&block)?;
                }
                repeats
                    .drain(self.k, &mut count)
                    .map_err(io::Error::other)?;
                tables.finish().map_err(io::Error::other)?;
                let mut sink = ChunkSink {
                    buffer: recycled.try_recv().unwrap_or_default(),
                    sender,
                    recycled,
                };
                encode(&tables.counts, &mut sink)?;
                sink.finish()?;
                progress
                    .distinct
                    .fetch_add(tables.counts.len() as u64, Ordering::Relaxed);
                tables.counts.clear();
            }
            progress.kmers.fetch_add(bin.kmers, Ordering::Relaxed);
            // Bins are alike: where under a quarter of a bin's records were
            // copies, merging costs more than it saves, so the next bins go
            // without it until one is tried again.
            if unmerged > 0 {
                unmerged -= 1;
            } else if repeats.copies * 4 < repeats.added {
                unmerged = UNMERGED;
            }
            (repeats.added, repeats.copies) = (0, 0);
        }
        Ok(())
    }
}

/// One distinct super-k-mer record and how many copies of it were seen.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Repeat {
    /// Zero marks an empty slot.
    copies: u32,
    /// The unpadded record, header included, followed by zeros.
    record: [u8; RECORD],
}

/// Room for the longest unpadded record: a header, and `MAX_RUN` k-mers of
/// 32 bases in three 64-bit words.
const RECORD: usize = 28;
const _: () = assert!(2 * (MAX_RUN + 31) <= 3 * 64);

/// A hash table that merges equal records before their k-mers are counted.
///
/// Records are added as they are read. When the table is three quarters
/// full, and at the end of a bin, every distinct record is visited once with
/// its number of copies, and the table starts again empty; copies on either
/// side of that are merely counted separately.
struct Repeats {
    slots: Vec<Repeat>,
    filled: usize,
    hasher: foldhash::fast::FixedState,
    /// Records added, and those that were copies, since last reset.
    added: u64,
    copies: u64,
}

impl Repeats {
    /// A table of `slots` slots, a power of two.
    fn new(slots: usize) -> Self {
        Self {
            slots: vec![
                Repeat {
                    copies: 0,
                    record: [0; RECORD],
                };
                slots
            ],
            filled: 0,
            hasher: foldhash::fast::FixedState::default(),
            added: 0,
            copies: 0,
        }
    }

    /// Visit every record of `block` as it comes, without merging.
    fn pass(
        block: &[u8],
        k: usize,
        mut visit: impl FnMut(&[u8], u64) -> Result<(), Error>,
    ) -> io::Result<()> {
        let mut rest = block;
        while !rest.is_empty() {
            let record = first_record(rest, k)?;
            rest = &rest[record.len()..];
            visit(record, 1).map_err(io::Error::other)?;
        }
        Ok(())
    }

    /// Add every record of `block`.
    fn add(
        &mut self,
        block: &[u8],
        k: usize,
        mut visit: impl FnMut(&[u8], u64) -> Result<(), Error>,
    ) -> io::Result<()> {
        /// Records hashed, and their slots fetched, before any is looked up.
        const AHEAD: usize = 16;
        let mask = self.slots.len() - 1;
        let mut batch = [([0_u8; RECORD], 0_usize); AHEAD];
        let mut rest = block;
        while !rest.is_empty() {
            let mut records = 0;
            while records < AHEAD && !rest.is_empty() {
                let record = first_record(rest, k)?;
                rest = &rest[record.len()..];
                let record = unpadded(record, k);
                let slot = self.hasher.hash_one(record) as usize & mask;
                crate::table::prefetch(&raw const self.slots[slot]);
                batch[records] = (record, slot);
                records += 1;
            }
            self.added += records as u64;
            for &(record, mut slot) in &batch[..records] {
                loop {
                    let repeat = &mut self.slots[slot];
                    if repeat.copies == 0 {
                        *repeat = Repeat { copies: 1, record };
                        self.filled += 1;
                        // Keep a quarter of the slots empty, so that every
                        // search ends.
                        if self.filled * 4 >= self.slots.len() * 3 {
                            self.drain(k, &mut visit).map_err(io::Error::other)?;
                        }
                        break;
                    }
                    if repeat.record == record {
                        if repeat.copies == u32::MAX {
                            // Count this many now and start again.
                            let length = 1 + (usize::from(record[0]) + k).div_ceil(4);
                            visit(&record[..length], u64::from(u32::MAX))
                                .map_err(io::Error::other)?;
                            repeat.copies = 0;
                        }
                        repeat.copies += 1;
                        self.copies += 1;
                        break;
                    }
                    slot = (slot + 1) & mask;
                }
            }
        }
        Ok(())
    }

    /// Visit each distinct record with its copies, and empty the table.
    fn drain(
        &mut self,
        k: usize,
        mut visit: impl FnMut(&[u8], u64) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if self.filled == 0 {
            return Ok(());
        }
        for repeat in &mut self.slots {
            if repeat.copies != 0 {
                let length = 1 + (usize::from(repeat.record[0]) + k).div_ceil(4);
                visit(&repeat.record[..length], u64::from(repeat.copies))?;
                repeat.copies = 0;
            }
        }
        self.filled = 0;
        Ok(())
    }
}

/// Bins handed out and counted so far.
#[derive(Default)]
struct Progress {
    next: AtomicUsize,
    kmers: AtomicU64,
    distinct: AtomicU64,
}

impl Progress {
    /// Distinct k-mers expected among `kmers`, from the ratio in the bins
    /// counted so far with a quarter added; all of them before any is known.
    fn estimate(&self, kmers: u64) -> u64 {
        let counted = self.kmers.load(Ordering::Relaxed);
        if counted == 0 {
            return kmers.max(1);
        }
        let distinct = self.distinct.load(Ordering::Relaxed) as f64 * 1.25;
        ((kmers as f64 * (distinct / counted as f64).min(1.0)) as u64).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Counter;
    use std::collections::BTreeMap;

    /// A deterministic pseudo-random sequence with repeats and invalid bases.
    fn sequence(length: usize, mut state: u64) -> Vec<u8> {
        let mut bases = Vec::with_capacity(length);
        while bases.len() < length {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let value = state >> 33;
            if value.is_multiple_of(97) {
                bases.push(b'N');
            } else if value.is_multiple_of(53) && bases.len() >= 40 {
                // Repeat an earlier stretch, so that k-mers recur.
                let from = (value as usize / 53) % (bases.len() - 39);
                bases.extend_from_within(from..from + 40);
            } else {
                bases.push(b"ACGTacgt"[(value % 8) as usize]);
            }
        }
        bases.truncate(length);
        bases
    }

    fn serial(records: &[Vec<u8>], k: usize, strand: StrandMode) -> BTreeMap<Vec<u8>, u64> {
        let mut counter = Counter::new(k, strand).unwrap();
        for record in records {
            counter.add_record(record).unwrap();
        }
        counter.counts().collect()
    }

    fn on_disk(records: &[Vec<u8>], k: usize, strand: StrandMode, memory: u64) -> DiskCounts {
        let counter = DiskCounter::new(k, strand, 3, memory, &std::env::temp_dir()).unwrap();
        let mut input = counter.input();
        for record in records {
            input.add_record(record).unwrap();
        }
        input.finish().unwrap();
        counter.finish().unwrap()
    }

    fn rows(tsv: &[u8]) -> BTreeMap<Vec<u8>, u64> {
        let mut rows = BTreeMap::new();
        for line in tsv
            .split(|&byte| byte == b'\n')
            .filter(|line| !line.is_empty())
        {
            let tab = line.iter().position(|&byte| byte == b'\t').unwrap();
            let count = std::str::from_utf8(&line[tab + 1..])
                .unwrap()
                .parse()
                .unwrap();
            assert!(
                rows.insert(line[..tab].to_vec(), count).is_none(),
                "duplicate key"
            );
        }
        rows
    }

    #[test]
    fn matches_serial_counts_for_every_k_and_both_strands() {
        let records: Vec<_> = (0..40)
            .map(|seed| sequence(50 + 37 * seed, seed as u64))
            .collect();
        for k in [1, 2, 5, 11, 21, 22, 31, 32] {
            for strand in [StrandMode::Canonical, StrandMode::Forward] {
                let expected = serial(&records, k, strand);
                // Everything in memory; part in the file; all of it in the
                // file in tiny blocks, with several passes per bin.
                for memory in [1 << 30, 80 << 10, 4096] {
                    let counts = on_disk(&records, k, strand, memory);
                    let kmers = counts.kmers();
                    assert_eq!(kmers, expected.values().sum::<u64>(), "k={k}");
                    let mut tsv = Vec::new();
                    let distinct = counts.write_tsv(&mut tsv).unwrap();
                    assert_eq!(distinct, expected.len() as u64);
                    assert_eq!(rows(&tsv), expected, "k={k} {strand:?} memory={memory}");
                }
            }
        }
    }

    #[test]
    fn binary_output_is_a_complete_database() {
        let records: Vec<_> = (0..20).map(|seed| sequence(400, 100 + seed)).collect();
        let expected = serial(&records, 31, StrandMode::Canonical);
        let mut database = io::Cursor::new(Vec::new());
        let distinct = on_disk(&records, 31, StrandMode::Canonical, 1 << 30)
            .write_binary(&mut database)
            .unwrap();
        assert_eq!(distinct, expected.len() as u64);
        let loaded = Counter::read_binary(database.get_ref().as_slice()).unwrap();
        assert_eq!(loaded.counts().collect::<BTreeMap<_, _>>(), expected);
    }

    #[test]
    fn long_runs_of_one_base_and_empty_input_are_supported() {
        let records = vec![vec![b'A'; 5000], b"ACGTN".repeat(300), Vec::new()];
        let expected = serial(&records, 21, StrandMode::Canonical);
        let mut tsv = Vec::new();
        on_disk(&records, 21, StrandMode::Canonical, 4096)
            .write_tsv(&mut tsv)
            .unwrap();
        assert_eq!(rows(&tsv), expected);
        let mut tsv = Vec::new();
        let empty = on_disk(&[], 21, StrandMode::Canonical, 4096);
        assert_eq!((empty.kmers(), empty.temporary_bytes()), (0, 0));
        assert_eq!(empty.write_tsv(&mut tsv).unwrap(), 0);
        assert!(tsv.is_empty());
    }

    #[test]
    fn repeated_reads_are_merged_and_counted_with_their_copies() {
        // Hundreds of copies of a few reads, in both orientations and cut at
        // different places: more copies than one update can carry.
        let reads: Vec<_> = (0..4).map(|seed| sequence(120, 900 + seed)).collect();
        let complement = |read: &[u8]| -> Vec<u8> {
            let bases = read.iter().rev();
            bases
                .map(|&base| {
                    b"TGCAN"[b"ACGTN"
                        .iter()
                        .position(|&b| b == base.to_ascii_uppercase())
                        .unwrap()]
                })
                .collect()
        };
        let mut records = Vec::new();
        for copy in 0..300 {
            let read = &reads[copy % reads.len()];
            records.push(read[copy % 7..].to_vec());
            records.push(complement(&read[..read.len() - copy % 5]));
        }
        for k in [9, 31, 32] {
            let expected = serial(&records, k, StrandMode::Canonical);
            assert!(expected.values().any(|&count| count > 127));
            for memory in [1 << 30, 4096] {
                let mut tsv = Vec::new();
                on_disk(&records, k, StrandMode::Canonical, memory)
                    .write_tsv(&mut tsv)
                    .unwrap();
                assert_eq!(rows(&tsv), expected, "k={k} memory={memory}");
            }
        }
    }

    #[test]
    fn unpadding_makes_equal_runs_equal_records() {
        // The same 40 bases packed from each of the four alignments.
        let bases: Vec<u8> = (0..43).map(|index| (index * 7 % 4) as u8).collect();
        let pack = |skip: usize| {
            let mut record = vec![(skip << 6 | (10 - 1)) as u8];
            // Stray bases before and after the run must not matter.
            let run = (0..skip)
                .map(|_| 3)
                .chain(bases[..40].iter().copied())
                .chain([1, 2, 3]);
            let run: Vec<u8> = run.collect();
            for four in run.chunks(4) {
                let byte = four
                    .iter()
                    .enumerate()
                    .fold(0, |byte, (at, &base)| byte | base << (6 - 2 * at));
                record.push(byte);
            }
            record.truncate(1 + (skip + 40).div_ceil(4));
            record
        };
        let expected = unpadded(&pack(0), 31);
        assert_eq!(expected[0], 9);
        assert_eq!(&expected[11..], &[0; RECORD - 11]);
        for skip in 1..4 {
            assert_eq!(unpadded(&pack(skip), 31), expected, "skip={skip}");
        }
    }

    #[test]
    fn rejects_unsupported_k() {
        let directory = std::env::temp_dir();
        for k in [0, 33] {
            assert!(DiskCounter::new(k, StrandMode::Canonical, 2, 1 << 20, &directory).is_err());
        }
    }
}
