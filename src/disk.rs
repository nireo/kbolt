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
//! In the second phase each bin is counted on its own, by sorting rather
//! than by hash tables (see [`Updates`]): every k-mer of the bin becomes an
//! update, the updates are sorted by mixed key, and equal keys are summed as
//! they are written out, already in the order the output wants. Memory then
//! follows the bins being counted at once, not the input: a bin has at most
//! one update per k-mer, it is admitted only when that many fit in the
//! remaining budget, and a bin too large for the whole budget is counted in
//! several passes, each taking a slice of its k-mers.
//!
//! Large inputs set three choices here. Bins are many, so that each thread's
//! updates are a small part of the budget and every thread can count at
//! once; the largest are counted first, so that none is left for the end. A
//! thread that would still wait for memory writes the blocks kept in memory
//! of a bin not yet counted to the file instead, which frees their memory
//! for updates. And the file is read back with a hint to the OS to read
//! ahead, so that the threads rarely wait for the disk.
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

use crate::database::{self, KeyOrder};
use crate::parallel::{ParallelCounter, ParallelInput};
use crate::sorted::{self, Sorter};
use crate::table::{Buckets, Mixer, key_bits};
use crate::{ChunkSink, DNA_BASES, EncodedChunk, Error, StrandMode};

/// Bins the input is split into; a power of two. With 512, a human genome
/// at 4 GiB left most threads waiting for memory.
const BINS: usize = 2048;
/// Longest minimizer. Shorter k use half their length.
const MINIMIZER: usize = 11;
/// Most k-mers in one super-k-mer, so that the count fits its header.
const MAX_RUN: usize = 64;
/// Largest block of one bin's records, the unit kept in memory or written.
const MAX_BLOCK: usize = 256 * 1024;
/// The smallest block: more than the longest record.
const MIN_BLOCK: usize = 64;
/// Bytes a worker collects for a bin before taking the bin's lock.
const LOCAL: usize = 1024;
/// Slots of a counting thread's table of repeated records: within an eighth
/// of the memory limit between the threads.
const MAX_REPEATS: usize = 1 << 20;
/// Bins a thread counts without merging after one with few repeats.
const UNMERGED: usize = if cfg!(test) { 1 } else { 7 };
const MIN_REPEATS: usize = if cfg!(test) { 16 } else { 1 << 10 };

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
pub(crate) struct Spill {
    file: File,
    end: AtomicU64,
    // Positional I/O needs no lock on Unix.
    #[cfg(not(unix))]
    position: Mutex<()>,
    #[cfg(not(unix))]
    path: PathBuf,
}

impl Spill {
    pub(crate) fn create(directory: &Path) -> io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
        let path: PathBuf = directory.join(format!(
            "kbolt-{}-{unique:x}-{sequence}.bins",
            std::process::id()
        ));
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

    pub(crate) fn append(&self, bytes: &[u8]) -> io::Result<u64> {
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

    /// Ask the OS to start reading `len` bytes at `offset` into its cache,
    /// so that a later [`Spill::read`] of them need not wait for the disk.
    /// Only a hint: failures are ignored, and some systems ignore it.
    pub(crate) fn advise(&self, offset: u64, len: u32) {
        #[cfg(target_os = "macos")]
        {
            use std::os::fd::AsRawFd;
            let advice = libc::radvisory {
                ra_offset: offset as libc::off_t,
                ra_count: len as libc::c_int,
            };
            // SAFETY: F_RDADVISE reads the struct and nothing else.
            unsafe { libc::fcntl(self.file.as_raw_fd(), libc::F_RDADVISE, &advice) };
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: a plain advisory call on an open descriptor.
            unsafe {
                libc::posix_fadvise(
                    self.file.as_raw_fd(),
                    offset as libc::off_t,
                    libc::off_t::from(len),
                    libc::POSIX_FADV_WILLNEED,
                )
            };
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        let _ = (offset, len);
    }

    pub(crate) fn read(&self, bytes: &mut [u8], offset: u64) -> io::Result<()> {
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

/// One block of super-k-mer records, backed by [`Buckets`] so large blocks
/// return their memory directly to the OS when a bin is dropped.
struct Block {
    data: Buckets<u8>,
    len: usize,
}

impl Default for Block {
    fn default() -> Self {
        Self {
            data: Buckets::new(),
            len: 0,
        }
    }
}

impl Block {
    fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> usize {
        self.data.len()
    }

    fn as_slice(&self) -> &[u8] {
        &self.data[..self.len]
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn ensure_capacity(&mut self, capacity: usize) -> Result<(), Error> {
        if self.data.is_empty() {
            self.data = Buckets::zeroed(capacity)?;
        }
        Ok(())
    }

    fn extend_from_slice(&mut self, bytes: &[u8]) {
        self.data[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
    }
}

/// One bin's records, as blocks of whole records in no particular order.
#[derive(Default)]
struct Bin {
    /// The block being filled.
    open: Block,
    /// Full blocks kept in memory.
    held: Vec<Block>,
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
    spare: Block,
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
            spare: Block::default(),
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
                if self.held.fetch_add(block.capacity(), Ordering::Relaxed) + block.capacity()
                    <= self.limit
                {
                    shared.held.push(block);
                } else {
                    self.held.fetch_sub(block.capacity(), Ordering::Relaxed);
                    full = Some(block);
                }
            }
            if shared.open.capacity() == 0 {
                shared.open.ensure_capacity(self.block)?;
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
        match self.spill.append(block.as_slice()) {
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
        for bin in 0..BINS {
            if !buffers.buffers[bin].is_empty() {
                self.hand(buffers, bin)?;
            }
        }
        Ok(())
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
    let length = data.len();
    let head = if length >= 16 {
        u128::from_be_bytes(data[..16].try_into().unwrap())
    } else if length >= 8 {
        // Two loads that overlap, the second shifted so that only the bytes
        // past the first eight remain: no copy of variable length.
        let high = u64::from_be_bytes(data[..8].try_into().unwrap());
        let low = u64::from_be_bytes(data[length - 8..].try_into().unwrap());
        (u128::from(high) << 64) | u128::from(low.checked_shl(8 * (16 - length) as u32).unwrap_or(0))
    } else {
        let mut head = [0_u8; 16];
        head[..length].copy_from_slice(data);
        u128::from_be_bytes(head)
    };
    let mut forward = (head >> (128 - 2 * (skip + k))) as u64 & mask;
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

/// One bin's k-mers, counted by sorting.
///
/// A k-mer is not counted when it is expanded. Its mixed key, with the
/// number of occurrences to add below it, is appended to one of `GROUPS`
/// lists by the key's top bits: a few dozen sequential streams. Once the bin
/// has been read, each list is sorted while it sits in cache, and equal keys
/// are summed as they are visited in ascending order, the order in which the
/// output wants them. In a bin whose repeated records are merged, most keys
/// arrive once or twice, and sorting a few updates per key costs less than
/// probing a hash table for each and sorting the table again for output.
struct Updates {
    mixer: Mixer,
    /// Updates by group: the key bits below the group index, then the
    /// weight. Group `g` fills `arena[g * room..(g + 1) * room]`, and any
    /// more go to `overflow[g]`. The arena is mapped, so its memory returns
    /// to the OS as soon as it is dropped.
    arena: Buckets<u64>,
    room: usize,
    filled: Vec<usize>,
    overflow: Vec<Vec<u64>>,
    /// Key bits below the group index, and the bits left for the weight.
    low_bits: u32,
    weight_bits: u32,
    /// A group's updates once sorted, and the bucket counts for sorting.
    sorted: Buckets<u64>,
    buckets: Vec<u32>,
}

/// Groups of updates of one bin; a power of two.
const GROUPS: usize = 64;
/// Updates per bucket when a group is sorted, on average.
const UPDATES_PER_BUCKET: usize = 4;

impl Updates {
    fn new(k: usize) -> Self {
        let bits = key_bits(k);
        // Every key keeps at least one bit below its group, and the weight
        // at least one bit.
        let group_bits = GROUPS.trailing_zeros().min(bits - 1);
        let low_bits = bits - group_bits;
        Self {
            mixer: Mixer::new(bits, crate::table::SEED),
            arena: Buckets::new(),
            room: 0,
            filled: vec![0; 1 << group_bits],
            overflow: (0..1 << group_bits).map(|_| Vec::new()).collect(),
            low_bits,
            weight_bits: 64 - low_bits,
            sorted: Buckets::new(),
            buckets: Vec::new(),
        }
    }

    /// Make room for `updates` updates, spread evenly over the groups.
    fn reserve(&mut self, updates: u64) -> io::Result<()> {
        let groups = self.filled.len();
        let room = usize::try_from(updates / groups as u64 * 9 / 8 + 64).unwrap_or(usize::MAX);
        if room > self.room {
            self.arena = Buckets::new();
            self.arena = Buckets::zeroed(room * groups).map_err(io::Error::other)?;
            self.room = room;
        }
        Ok(())
    }

    /// Count `weight` (at least one) occurrences of the k-mer whose mixed
    /// key is `mixed`.
    #[inline(always)]
    fn push(&mut self, mixed: u64, mut weight: u64) {
        let group = (mixed >> self.low_bits) as usize;
        // The group bits leave at the top.
        let key = mixed << self.weight_bits;
        let heaviest = u64::MAX >> self.low_bits;
        while weight > heaviest {
            self.push_one(group, key | heaviest);
            weight -= heaviest;
        }
        self.push_one(group, key | weight);
    }

    #[inline(always)]
    fn push_one(&mut self, group: usize, update: u64) {
        let filled = self.filled[group];
        if filled < self.room {
            self.arena[group * self.room + filled] = update;
            self.filled[group] = filled + 1;
        } else {
            self.overflow[group].push(update);
        }
    }

    /// Visit every distinct k-mer with its count, in ascending order of the
    /// mixed key, and empty the bin. Returns the number visited.
    fn drain(&mut self, mut visit: impl FnMut(u64, u64) -> io::Result<()>) -> io::Result<u64> {
        let (low_bits, weight_bits) = (self.low_bits, self.weight_bits);
        let weight_mask = u64::MAX >> low_bits;
        let mut distinct = 0;
        for group in 0..self.filled.len() {
            let start = group * self.room;
            let (stored, extra) = (
                &self.arena[start..start + self.filled[group]],
                &self.overflow[group],
            );
            let length = stored.len() + extra.len();
            if self.sorted.len() < length {
                self.sorted = Buckets::new();
                self.sorted = Buckets::zeroed(length + length / 4).map_err(io::Error::other)?;
            }
            let list = &mut self.sorted[..length];
            sort_updates(stored, extra, list, &mut self.buckets);
            let base = (group as u64) << low_bits;
            let mut at = 0;
            while at < list.len() {
                let key = list[at] >> weight_bits;
                let mut count = list[at] & weight_mask;
                at += 1;
                while at < list.len() && list[at] >> weight_bits == key {
                    count += list[at] & weight_mask;
                    at += 1;
                }
                visit(base | key, count)?;
                distinct += 1;
            }
        }
        self.clear();
        Ok(distinct)
    }

    fn clear(&mut self) {
        self.filled.fill(0);
        for overflow in &mut self.overflow {
            // Overflow is rare; do not keep its memory.
            *overflow = Vec::new();
        }
    }

    /// Bytes allocated for updates and scratch.
    fn bytes(&self) -> u64 {
        let overflow: usize = self.overflow.iter().map(Vec::capacity).sum();
        (8 * (self.arena.len() + self.sorted.len() + overflow) + 4 * self.buckets.capacity())
            as u64
    }

    /// Bytes to reserve for a bin of `updates` updates at most.
    fn needed(updates: u64) -> u64 {
        // The updates with room to spare, and the largest group sorted.
        updates.saturating_mul(9) + updates / GROUPS as u64 * 12 + (64 << 10)
    }
}

/// Sort the updates of `stored` and `extra` into `sorted`, as long as both:
/// by their top bits into buckets of a few updates each, then by insertion,
/// which on nearly sorted updates costs about one comparison each. A bucket
/// far larger than the rest (many copies of one k-mer with different
/// weights) is sorted on its own.
fn sort_updates(stored: &[u64], extra: &[u64], sorted: &mut [u64], counts: &mut Vec<u32>) {
    let length = sorted.len();
    if length <= 32 {
        sorted[..stored.len()].copy_from_slice(stored);
        sorted[stored.len()..].copy_from_slice(extra);
        insertion_sort(sorted);
        return;
    }
    let bits = (length / UPDATES_PER_BUCKET).next_power_of_two().trailing_zeros().min(20);
    let shift = 64 - bits;
    counts.clear();
    counts.resize((1 << bits) + 1, 0);
    for &update in stored.iter().chain(extra) {
        counts[(update >> shift) as usize + 1] += 1;
    }
    let largest = counts.iter().copied().max().unwrap_or(0) as usize;
    for index in 0..1 << bits {
        counts[index + 1] += counts[index];
    }
    for &update in stored.iter().chain(extra) {
        let bucket = (update >> shift) as usize;
        sorted[counts[bucket] as usize] = update;
        counts[bucket] += 1;
    }
    if largest <= 64 {
        insertion_sort(sorted);
        return;
    }
    // `counts[bucket]` is now where the bucket ends.
    let mut begin = 0;
    for &end in &counts[..1 << bits] {
        sorted[begin..end as usize].sort_unstable();
        begin = end as usize;
    }
}

#[inline(always)]
fn insertion_sort(values: &mut [u64]) {
    for index in 1..values.len() {
        let value = values[index];
        let mut at = index;
        while at > 0 && values[at - 1] > value {
            values[at] = values[at - 1];
            at -= 1;
        }
        values[at] = value;
    }
}

struct BudgetState {
    /// Bytes currently available for active and pooled updates. Grows as
    /// in-memory bin blocks are freed during pass 2.
    limit: u64,
    /// Bytes reserved by currently active workers.
    reserved: u64,
    /// Emptied updates awaiting reuse by the next worker. Keeping them in a
    /// shared pool prevents idle threads waiting for budget from holding onto
    /// grown allocations.
    pool: Vec<Updates>,
}

/// Memory for updates, and reusable [`Updates`], shared by the counting
/// workers.
struct Budget {
    max_limit: u64,
    state: Mutex<BudgetState>,
    released: Condvar,
}

/// An active worker's leased [`Updates`] and its reservation in [`Budget`].
struct Lease<'a> {
    budget: &'a Budget,
    updates: Option<Updates>,
    reserved: u64,
}

impl Budget {
    fn new(initial_limit: u64, max_limit: u64) -> Self {
        let max_limit = max_limit.max(1);
        let initial_limit = initial_limit.clamp(1, max_limit);
        Self {
            max_limit,
            state: Mutex::new(BudgetState {
                limit: initial_limit,
                reserved: 0,
                pool: Vec::new(),
            }),
            released: Condvar::new(),
        }
    }

    /// Wait until room for `needed` bytes is free, returning leased
    /// [`Updates`] (reused from the pool when they fit, or new). Evicts
    /// pooled ones before blocking so idle memory is returned to the OS.
    /// Before blocking, `free` is called without the lock held; it may make
    /// room (returning true, and crediting what it freed) instead of waiting.
    fn acquire(
        &self,
        k: usize,
        needed: u64,
        mut free: impl FnMut() -> io::Result<bool>,
    ) -> io::Result<Lease<'_>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut can_free = true;
        let (reused, reserved) = loop {
            let total_pool: u64 = state.pool.iter().map(Updates::bytes).sum();
            let candidate = state
                .pool
                .iter()
                .enumerate()
                .filter(|(_, t)| {
                    let bytes = t.bytes();
                    bytes <= needed.saturating_mul(2) && bytes <= state.limit
                })
                .max_by_key(|(_, t)| t.bytes())
                .map(|(index, _)| index);
            let (want, other_pool) = match candidate {
                Some(index) => {
                    let bytes = state.pool[index].bytes();
                    (
                        needed.max(bytes).min(state.limit),
                        total_pool.saturating_sub(bytes),
                    )
                }
                None => (needed.min(state.limit), total_pool),
            };
            if state.reserved + other_pool + want <= state.limit {
                let updates = candidate.map(|index| state.pool.swap_remove(index));
                state.reserved += want;
                break (updates, want);
            }
            if state.reserved == 0 {
                // Always allow at least one worker to make progress even under
                // tiny memory limits, reusing idle empty updates if present.
                let updates = if state.pool.len() == 1 && state.pool[0].bytes() <= 16 * 1024 {
                    Some(state.pool.swap_remove(0))
                } else {
                    None
                };
                if updates.is_some() || state.pool.is_empty() {
                    state.reserved += want;
                    break (updates, want);
                }
            }
            if !state.pool.is_empty() {
                // Evict the largest idle updates not selected as our candidate
                // (or the candidate itself if nothing else remains).
                let evict = state
                    .pool
                    .iter()
                    .enumerate()
                    .filter(|&(index, _)| Some(index) != candidate)
                    .max_by_key(|(_, t)| t.bytes())
                    .map(|(index, _)| index)
                    .unwrap_or_else(|| candidate.expect("non-empty pool"));
                let evicted = state.pool.swap_remove(evict);
                drop(state);
                drop(evicted);
                state = self
                    .state
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                continue;
            }
            if can_free {
                // Whether or not this frees anything, look again under the
                // lock: a worker may have released its updates meanwhile.
                drop(state);
                can_free = free()?;
                state = self
                    .state
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                continue;
            }
            state = self
                .released
                .wait(state)
                .unwrap_or_else(|poison| poison.into_inner());
        };
        drop(state);
        let updates = match reused {
            Some(updates) => updates,
            None => Updates::new(k),
        };
        Ok(Lease {
            budget: self,
            updates: Some(updates),
            reserved,
        })
    }

    fn release_raw(&self, updates: Option<Updates>, reserved: u64, freed_bin_bytes: u64) {
        let mut evicted = None;
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            state.reserved = state.reserved.saturating_sub(reserved);
            state.limit = state
                .limit
                .saturating_add(freed_bin_bytes)
                .min(self.max_limit);
            if let Some(mut updates) = updates {
                updates.clear();
                let total_pool: u64 = state.pool.iter().map(Updates::bytes).sum();
                let tiny_fallback = state.pool.is_empty() && updates.bytes() <= 16 * 1024;
                if state.reserved + total_pool + updates.bytes() <= state.limit || tiny_fallback {
                    state.pool.push(updates);
                } else {
                    evicted = Some(updates);
                }
            }
            self.released.notify_all();
        }
        drop(evicted);
    }

    fn credit_bin_bytes(&self, freed_bin_bytes: u64) {
        if freed_bin_bytes == 0 {
            return;
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.limit = state
            .limit
            .saturating_add(freed_bin_bytes)
            .min(self.max_limit);
        self.released.notify_all();
    }
}

impl Lease<'_> {
    fn updates(&mut self) -> &mut Updates {
        self.updates.as_mut().expect("active lease holds updates")
    }

    /// Return the updates to the pool (or drop them if over budget) and
    /// credit any freed in-memory bin bytes to the budget limit.
    fn release(mut self, freed_bin_bytes: u64) {
        let updates = self.updates.take();
        let reserved = std::mem::replace(&mut self.reserved, 0);
        self.budget.release_raw(updates, reserved, freed_bin_bytes);
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        if self.reserved > 0 || self.updates.is_some() {
            let updates = self.updates.take();
            let reserved = std::mem::replace(&mut self.reserved, 0);
            self.budget.release_raw(updates, reserved, 0);
        }
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
    directory: PathBuf,
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
            directory: directory.to_owned(),
        })
    }

    /// Create an independent input producer; see [`ParallelCounter::input`].
    pub fn input(&self) -> ParallelInput<'_> {
        self.parallel.input()
    }

    /// Stream FASTA or FASTQ records from `reader`, auto-detecting gzip
    /// compression from content, and submit them for binning.
    pub fn add_reader(
        &self,
        reader: impl io::Read + Send + 'static,
    ) -> io::Result<crate::InputStats> {
        let mut stream = crate::input::Stream::parallel(self.input());
        stream.add_reader(reader)?;
        stream.finish()
    }

    /// Stream FASTA or FASTQ records from the file at `path`, auto-detecting
    /// gzip compression from content, and submit them for binning.
    pub fn add_file(&self, path: impl AsRef<Path>) -> io::Result<crate::InputStats> {
        let mut stream = crate::input::Stream::parallel(self.input());
        stream.add_file(path)?;
        stream.finish()
    }

    /// Stream FASTA or FASTQ records from `paths` using up to `readers`
    /// concurrent file-reading threads (when all paths are regular files),
    /// auto-detecting gzip compression from content.
    pub fn add_files<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
        readers: usize,
    ) -> io::Result<crate::InputStats> {
        crate::input::count_parallel_files(|| self.input(), paths, readers, None)
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
                directory: self.directory,
                sorted: false,
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
    /// Where temporary files go.
    directory: PathBuf,
    /// Whether to write k-mers in their own order; see `crate::sorted`.
    sorted: bool,
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

    /// Write k-mers in ascending order (A < C < G < T) instead of bin by
    /// bin. This takes another temporary file, of 8 bytes per distinct
    /// k-mer, and one more pass over them.
    pub fn sorted(mut self, sorted: bool) -> Self {
        self.sorted = sorted;
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
        if self.sorted {
            return self.count_sorted(&mut writer, sorted::tsv_rows(k), |_, _| Ok(()));
        }
        self.count(&mut writer, 0, |updates, writer| {
            let mixer = updates.mixer;
            let mut batch = [0; 16 * 1024];
            let mut offset = 0;
            let distinct = updates.drain(|mixed, count| {
                if offset + 32 + 22 > batch.len() {
                    writer.write_all(&batch[..offset])?;
                    offset = 0;
                }
                batch[offset..offset + 32].copy_from_slice(&crate::decode_packed32(mixer.unmix(mixed), k));
                offset = crate::push_count(&mut batch, offset + k, count);
                Ok(())
            })?;
            writer.write_all(&batch[..offset])?;
            Ok(distinct)
        })
    }

    /// Count every bin and write a version-2 database (see
    /// [`crate::database`]), each bin in hash order. The entry count is
    /// known only once every bin has been counted, so the header is
    /// completed last and the writer must seek. Returns the number of
    /// distinct k-mers. Does not flush; failures may leave partial output.
    ///
    /// After [`DiskCounts::sorted`] the database is in k-mer order
    /// throughout instead.
    pub fn write_binary(self, mut writer: impl Write + Seek) -> io::Result<u64> {
        let (k, strand) = (self.k, self.strand);
        if self.sorted {
            return self.count_sorted(&mut writer, sorted::binary_sections, |distinct, writer| {
                writer.write_all(&database::header(k, strand, KeyOrder::Kmer, true, distinct))
            });
        }
        let start = writer.stream_position()?;
        let header = |entries| database::header(k, strand, KeyOrder::Hash, false, entries);
        writer.write_all(&header(0))?;
        let distinct = self.count(&mut writer, 0, |updates, writer| {
            let mut sections = database::SectionWriter::new(writer);
            let distinct = updates.drain(|mixed, count| sections.ascending(mixed, count))?;
            sections.finish()?;
            Ok(distinct)
        })?;
        let end = writer.stream_position()?;
        writer.seek(SeekFrom::Start(start))?;
        writer.write_all(&header(distinct))?;
        writer.seek(SeekFrom::Start(end))?;
        Ok(distinct)
    }

    /// Count every bin into a sorter, then write its buckets, each as
    /// `encode` makes of its records, after `begin` has written what comes
    /// before them, given the number of distinct k-mers.
    ///
    /// The sorter's records wait in a second temporary file, 8 bytes per
    /// distinct k-mer; sorting them takes about half the memory limit.
    fn count_sorted(
        &self,
        writer: &mut dyn Write,
        encode: impl Fn(sorted::Sorted<'_>, &mut dyn Write) -> io::Result<()> + Sync,
        begin: impl FnOnce(u64, &mut dyn Write) -> io::Result<()>,
    ) -> io::Result<u64> {
        // Every k-mer of the input may be a distinct one.
        let sorter = Sorter::on_disk(
            key_bits(self.k),
            self.kmers(),
            self.memory,
            self.threads,
            &self.directory,
        )?;
        let distinct = self.count(writer, self.memory / 8, |updates, _| {
            let mixer = updates.mixer;
            let mut batch = sorted::Batch::default();
            let distinct =
                updates.drain(|mixed, count| sorter.push(&mut batch, mixer.unmix(mixed), count))?;
            sorter.flush(&mut batch)?;
            Ok(distinct)
        })?;
        begin(distinct, writer)?;
        sorter.write(self.threads, writer, encode)?;
        Ok(distinct)
    }

    /// Write the blocks still in memory of the last bin in order not yet
    /// taken to the temporary file, and give their memory to the updates.
    /// Returns false once no bin is left to spill.
    ///
    /// Blocks kept in memory spare the disk only while the updates have room:
    /// counting a large input needs every byte for updates, and a thread that
    /// would wait for memory does better to write blocks out than to idle.
    fn spill_held(&self, progress: &Progress, budget: &Budget) -> io::Result<bool> {
        while let Ok(spilled) =
            progress
                .spilled
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |spilled| {
                    (spilled < progress.order.len()).then_some(spilled + 1)
                })
        {
            let position = progress.order.len() - 1 - spilled;
            if position < progress.next.load(Ordering::Relaxed) {
                break;
            }
            let mut bin = self.bins[progress.order[position]]
                .lock()
                .map_err(|_| io::Error::other("poisoned"))?;
            let freed = Self::bin_bytes(&bin);
            if freed == 0 {
                continue;
            }
            let Bin {
                open,
                held,
                written,
                ..
            } = &mut *bin;
            let open = std::mem::take(open);
            for block in held.drain(..).chain(std::iter::once(open)) {
                if block.len() > 0 {
                    let offset = self.spill.append(block.as_slice())?;
                    written.push((offset, block.len() as u32));
                }
            }
            drop(bin);
            budget.credit_bin_bytes(freed);
            return Ok(true);
        }
        Ok(false)
    }

    /// Bytes currently held in RAM by one bin's in-memory blocks.
    fn bin_bytes(bin: &Bin) -> u64 {
        let held: usize = bin.held.iter().map(Block::capacity).sum();
        (bin.open.capacity() + held) as u64
    }

    /// Count bins on `threads` threads, encoding each with `encode` and
    /// writing the encoded chunks, in no particular order, from this thread.
    /// `encode` drains a bin's updates and returns its distinct k-mers.
    fn count(
        &self,
        writer: &mut dyn Write,
        extra_reserved: u64,
        encode: impl Fn(&mut Updates, &mut dyn Write) -> io::Result<u64> + Sync,
    ) -> io::Result<u64> {
        // An eighth of the memory, between the threads, merges equal records.
        let repeat_slots = usize::try_from(self.memory / 8).unwrap_or(usize::MAX)
            / self.threads
            / std::mem::size_of::<Repeat>();
        let repeat_slots = (1_usize << repeat_slots.max(1).ilog2()).clamp(MIN_REPEATS, MAX_REPEATS);
        let repeats_bytes =
            (repeat_slots * std::mem::size_of::<Repeat>() * self.threads) as u64;
        // Bounded output channel plus one in-flight buffer per worker.
        let output_bytes = (3 * self.threads * crate::OUTPUT_CHUNK) as u64;
        // Reserve headroom for allocator retention (e.g. pass-1 buffers) and
        // transient resize spikes, clamped so both small and large limits behave well.
        let headroom = (self.memory / 10).clamp(64 << 20, 256 << 20);
        let held_bytes: u64 = self
            .bins
            .iter()
            .map(|bin| {
                bin.lock()
                    .map_or(0, |bin| Self::bin_bytes(&bin))
            })
            .sum();
        let max_limit = self
            .memory
            .saturating_sub(repeats_bytes + output_bytes + headroom + extra_reserved)
            .max(self.memory * 3 / 8)
            .max(1);
        let initial_limit = max_limit
            .saturating_sub(held_bytes)
            .max(self.memory / 8)
            .min(max_limit)
            .max(1);
        let budget = Budget::new(initial_limit, max_limit);
        // The largest bins first, so that no large one is left for the end
        // while the other threads idle.
        let mut order: Vec<(u64, usize)> = self
            .bins
            .iter()
            .enumerate()
            .map(|(index, bin)| (bin.lock().map_or(0, |bin| bin.kmers), index))
            .collect();
        order.sort_unstable_by(|a, b| b.cmp(a));
        let progress = Progress {
            order: order.into_iter().map(|(_, index)| index).collect(),
            ..Progress::default()
        };
        let (sender, receiver) = crossbeam_channel::bounded::<EncodedChunk>(2 * self.threads);
        let (recycle, recycled) = crossbeam_channel::unbounded::<Vec<u8>>();
        std::thread::scope(|scope| {
            for index in 0..self.threads {
                let (sender, recycled) = (sender.clone(), recycled.clone());
                let (budget, progress, encode) = (&budget, &progress, &encode);
                std::thread::Builder::new()
                    .name(format!("bin-{index}"))
                    .spawn_scoped(scope, move || {
                        let result = self.count_bins(
                            budget,
                            progress,
                            repeat_slots,
                            encode,
                            &sender,
                            &recycled,
                        );
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
        repeat_slots: usize,
        encode: &(impl Fn(&mut Updates, &mut dyn Write) -> io::Result<u64> + Sync),
        sender: &crossbeam_channel::Sender<EncodedChunk>,
        recycled: &crossbeam_channel::Receiver<Vec<u8>>,
    ) -> io::Result<()> {
        let mut repeats = Repeats::new(repeat_slots)?;
        // Bins this thread still counts without merging; see below.
        let mut unmerged = 0;
        let mut block = Vec::new();
        while let Some(&index) = progress.order.get(progress.next.fetch_add(1, Ordering::Relaxed)) {
            let bin = &self.bins[index];
            // Taking the bin frees its blocks once it has been counted.
            let bin = std::mem::take(&mut *bin.lock().map_err(|_| io::Error::other("poisoned"))?);
            let freed_bin_bytes = Self::bin_bytes(&bin);
            if bin.kmers == 0 {
                drop(bin);
                budget.credit_bin_bytes(freed_bin_bytes);
                continue;
            }
            let bin_kmers = bin.kmers;
            // A bin has at most one update per k-mer. One too large for the
            // whole budget is counted in passes, each taking the k-mers whose
            // mixed key falls in one residue class.
            let passes = Updates::needed(bin_kmers).div_ceil(budget.max_limit);
            let updates_per_pass = bin_kmers.div_ceil(passes);
            let needed = Updates::needed(updates_per_pass);
            let mut lease =
                budget.acquire(self.k, needed, || self.spill_held(progress, budget))?;
            // Its blocks in the file are read while those in memory are counted.
            for &(offset, len) in &bin.written {
                self.spill.advise(offset, len);
            }
            let updates = lease.updates();
            updates.reserve(updates_per_pass)?;
            let mixer = updates.mixer;
            for pass in 0..passes {
                let mut count = |record: &[u8], copies: u64| {
                    expand(
                        record,
                        self.k,
                        self.strand == StrandMode::Canonical,
                        |key| {
                            let mixed = mixer.mix(key);
                            if passes == 1 || mixed % passes == pass {
                                updates.push(mixed, copies);
                            }
                            Ok(())
                        },
                    )
                };
                let mut read = |block: &[u8]| {
                    if unmerged == 0 {
                        repeats.add(block, self.k, &mut count)
                    } else {
                        Repeats::pass(block, self.k, &mut count)
                    }
                };
                for block in bin.held.iter().chain(std::iter::once(&bin.open)) {
                    read(block.as_slice())?;
                }
                for &(offset, len) in &bin.written {
                    block.resize(len as usize, 0);
                    self.spill.read(&mut block, offset)?;
                    read(&block)?;
                }
                repeats
                    .drain(self.k, &mut count)
                    .map_err(io::Error::other)?;
                let mut sink = ChunkSink {
                    buffer: recycled.try_recv().unwrap_or_default(),
                    sender,
                    recycled,
                };
                let distinct = encode(updates, &mut sink)?;
                sink.finish()?;
                progress.distinct.fetch_add(distinct, Ordering::Relaxed);
            }
            drop(bin);
            lease.release(freed_bin_bytes);
            progress.kmers.fetch_add(bin_kmers, Ordering::Relaxed);
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
    slots: Buckets<Repeat>,
    filled: usize,
    hasher: foldhash::fast::FixedState,
    /// Records added, and those that were copies, since last reset.
    added: u64,
    copies: u64,
}

impl Repeats {
    /// A table of `slots` slots, a power of two.
    fn new(slots: usize) -> io::Result<Self> {
        Ok(Self {
            slots: Buckets::zeroed(slots).map_err(io::Error::other)?,
            filled: 0,
            hasher: foldhash::fast::FixedState::default(),
            added: 0,
            copies: 0,
        })
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
        for repeat in self.slots.iter_mut() {
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
    /// Bins in the order they are counted, and how many have been taken.
    order: Vec<usize>,
    next: AtomicUsize,
    /// Bins, from the last in order, whose blocks have been written out.
    spilled: AtomicUsize,
    kmers: AtomicU64,
    distinct: AtomicU64,
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
    fn blocks_kept_in_memory_are_written_out_when_tables_need_the_room() {
        let records: Vec<_> = (0..8)
            .map(|seed| sequence(60_000, 100 + seed as u64))
            .collect();
        let k = 31;
        let expected = serial(&records, k, StrandMode::Canonical);
        let counts = on_disk(&records, k, StrandMode::Canonical, 512 << 10);
        let held: u64 = counts
            .bins
            .iter()
            .map(|bin| DiskCounts::bin_bytes(&bin.lock().unwrap()))
            .sum();
        assert!(held > 0, "the first phase kept no blocks in memory");
        let before = counts.temporary_bytes();
        let mut tsv = Vec::new();
        let distinct = counts
            .count(&mut tsv, 0, |updates, writer| {
                let mixer = updates.mixer;
                let mut rows = Vec::new();
                let distinct = updates.drain(|mixed, count| {
                    rows.push((mixer.unmix(mixed), count));
                    Ok(())
                })?;
                crate::write_packed64_tsv(rows.into_iter(), k, writer)?;
                Ok(distinct)
            })
            .unwrap();
        assert!(
            counts.temporary_bytes() > before,
            "no blocks were written out while counting"
        );
        assert_eq!(distinct, expected.len() as u64);
        assert_eq!(rows(&tsv), expected);
    }

    #[test]
    fn sorted_output_is_in_kmer_order_whatever_the_memory() {
        let records: Vec<_> = (0..40)
            .map(|seed| sequence(50 + 37 * seed, 7 + seed as u64))
            .collect();
        for k in [1, 5, 21, 31, 32] {
            for strand in [StrandMode::Canonical, StrandMode::Forward] {
                // A map lists its keys in order.
                let expected: Vec<(Vec<u8>, u64)> =
                    serial(&records, k, strand).into_iter().collect();
                // Buckets in memory, mostly in the file, and (with a limit
                // below any bucket) admitted one at a time.
                for memory in [1 << 30, 80 << 10, 4096] {
                    let mut tsv = Vec::new();
                    let distinct = on_disk(&records, k, strand, memory)
                        .sorted(true)
                        .write_tsv(&mut tsv)
                        .unwrap();
                    assert_eq!(distinct, expected.len() as u64);
                    let rows: Vec<(Vec<u8>, u64)> = tsv
                        .split(|&byte| byte == b'\n')
                        .filter(|row| !row.is_empty())
                        .map(|row| {
                            let tab = row.iter().position(|&byte| byte == b'\t').unwrap();
                            let count = std::str::from_utf8(&row[tab + 1..]).unwrap();
                            (row[..tab].to_vec(), count.parse().unwrap())
                        })
                        .collect();
                    assert!(rows == expected, "k={k} {strand:?} memory={memory}");

                    let mut database = io::Cursor::new(Vec::new());
                    let distinct = on_disk(&records, k, strand, memory)
                        .sorted(true)
                        .write_binary(&mut database)
                        .unwrap();
                    assert_eq!(distinct, expected.len() as u64);
                    let mut reader =
                        database::DatabaseReader::new(database.get_ref().as_slice()).unwrap();
                    let header = reader.header();
                    assert_eq!((header.order, header.ascending), (KeyOrder::Kmer, true));
                    assert_eq!(header.entries, distinct);
                    let mut entries = Vec::new();
                    while let Some((key, count)) = reader.next_entry().unwrap() {
                        entries.push((key.to_vec(), count));
                    }
                    assert!(entries == expected, "k={k} {strand:?} memory={memory}");
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
