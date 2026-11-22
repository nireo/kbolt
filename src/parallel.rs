//! Parallel extraction and buffered updates to exclusively owned key partitions.
//! Partition locks are acquired once per batch, never once per k-mer, and
//! updates to large tables are queued and applied one cache-sized table at a
//! time.

use crossbeam_channel as channel;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::disk::{BinBuffers, Bins};
use crate::table::{self, CompactTable, Layout, Load, Mixer, Overflow, PackedCounts};
use crate::{
    Counter, Counts, Error, StrandMode, add_delta, visit_bytes, visit_packed, visit_packed128,
};

const BATCH_BASES: usize = 1024 * 1024;
const INITIAL_BATCH_BASES: usize = 64 * 1024;
const BYTE_UPDATE_BYTES: usize = 256 * 1024;

#[derive(Default)]
struct Batch {
    bases: Vec<u8>,
    ends: Vec<usize>,
}

impl Batch {
    fn records(&self) -> impl Iterator<Item = &[u8]> {
        let mut start = 0;
        self.ends.iter().map(move |&end| {
            let record = &self.bases[start..end];
            start = end;
            record
        })
    }

    fn clear(&mut self) {
        self.bases.clear();
        self.ends.clear();
    }
}

// Keep unrelated partition locks off the same cache line.
#[repr(align(128))]
struct Shard<T>(Mutex<T>);

/// Apply each nonempty worker-local partition buffer to its shard.
///
/// Different workers start at different partitions, and a partition held by
/// another worker is skipped and revisited instead of waited on. Only when
/// every remaining partition is busy does a worker block on one. Ownership
/// follows the partition, not the worker or input batch.
fn apply_partitions<T, U>(
    shards: &[Shard<T>],
    partitions: &mut [Vec<U>],
    pending: &mut Vec<usize>,
    offset: usize,
    drain: impl Fn(&mut T, &mut Vec<U>) -> Result<(), Error>,
) -> Result<(), Error> {
    pending.clear();
    let mask = shards.len() - 1;
    pending.extend(
        (0..shards.len())
            .map(|step| (step + offset) & mask)
            .filter(|&index| !partitions[index].is_empty()),
    );
    while !pending.is_empty() {
        let before = pending.len();
        let mut result = Ok(());
        pending.retain(|&index| {
            if result.is_err() {
                return true;
            }
            match shards[index].0.try_lock() {
                Ok(mut table) => {
                    result = drain(&mut table, &mut partitions[index]);
                    false
                }
                Err(std::sync::TryLockError::WouldBlock) => true,
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    result = Err(Error::WorkerFailed);
                    true
                }
            }
        });
        result?;
        if pending.len() == before {
            let index = pending.swap_remove(0);
            let mut table = shards[index].0.lock().map_err(|_| Error::WorkerFailed)?;
            drain(&mut table, &mut partitions[index])?;
        }
    }
    Ok(())
}

/// Partition bits for parallel packed counting: this many top bits of a mixed
/// key select its table. With thousands of tables, each one stays small enough
/// to sit in a core's cache while a burst of updates is applied to it.
/// Partition bits also leave count bits in each slot (see `crate::table`).
const PARALLEL_PARTITION_BITS: u32 = 13;
/// Of the partition bits, how many select a group. A group is the unit of
/// buffering and locking: workers scatter updates into one local buffer per
/// group, and one worker at a time applies a group's queued updates.
const GROUP_BITS: u32 = 6;
/// A group's queue is applied once it holds one update per table bucket
/// (cache line): every table of the group is then read into cache once per
/// visit and receives enough updates to pay for that. Queued updates then
/// take an eighth of the tables' memory. These bounds apply to that threshold,
/// in updates: the lower one keeps small tables from being visited for a
/// handful of updates, the upper one caps queued memory at 32 MiB per group.
// Two updates per bucket took 8% less time on uncompressed reads for 12% more
// memory, but no less on gzip input, where decompression sets the pace.
// Tests shrink the thresholds so that small inputs reach every path.
const MIN_PENDING: usize = if cfg!(test) { 64 } else { 4 * 1024 };
const MAX_PENDING: usize = 4 * 1024 * 1024;
/// A group whose tables total at most this many buckets (512 KiB) is updated
/// directly from each worker's buffer: all groups together then fit in cache,
/// and queueing and splitting their updates would only add work.
const DIRECT_BUCKETS: usize = if cfg!(test) { 2 } else { 8192 };
/// Prefetch this far ahead within a run of updates.
const AHEAD: usize = 16;
/// A worker that finds a group busy leaves its updates queued, unless the
/// queue has grown to this many thresholds (and at least `MIN_BACKLOG`
/// updates, 2 MiB); then it waits for the group.
const BACKLOG: usize = 4;
const MIN_BACKLOG: usize = if cfg!(test) { 256 } else { 256 * 1024 };

/// One group of compact tables and the updates queued for them.
// Keep unrelated groups' locks off the same cache line.
#[repr(align(128))]
struct Group {
    /// Queued updates in arrival order; see `PackedShared::push` for the
    /// entry format. Held only to append a buffer or to take the whole queue.
    pending: Mutex<Vec<u64>>,
    /// The group's tables, in sub-partition order. Whoever holds this lock
    /// applies queued updates.
    tables: Mutex<Vec<CompactTable>>,
    /// Queue length at which applying is worthwhile; follows the table sizes.
    threshold: AtomicUsize,
}

/// Shared compact tables for packed keys (k <= 32); see `crate::table`.
struct PackedShared {
    mixer: Mixer,
    layout: Layout,
    /// Bits of a mixed key below its group index: sub-partition and remainder.
    entry_bits: u32,
    groups: Vec<Group>,
}

/// Worker-local state for packed keys.
#[derive(Default)]
struct PackedUpdates {
    /// Updates not yet queued, one buffer per group.
    local: Vec<Vec<u64>>,
    /// A group's queue while this worker applies it.
    input: Vec<u64>,
    /// The same updates grouped by table.
    split: Vec<u64>,
    /// Where each table's updates end (then, while scattering, the cursors).
    ends: Vec<usize>,
    /// Small groups found busy during a pass, to revisit at its end.
    busy: Vec<usize>,
    /// The mixed key this worker buffered last.
    previous: u64,
}

/// The queue length at which a group with these tables is applied, or zero
/// while the group is small enough to be updated directly.
fn pending_threshold(tables: &[CompactTable]) -> usize {
    let buckets: usize = tables.iter().map(CompactTable::buckets).sum();
    if buckets <= DIRECT_BUCKETS {
        0
    } else {
        buckets.clamp(MIN_PENDING, MAX_PENDING)
    }
}

impl PackedShared {
    fn new(k: usize, capacity: usize) -> Result<Self, Error> {
        let counts =
            PackedCounts::with_capacity(k, PARALLEL_PARTITION_BITS, capacity, Load::Dense)?;
        let (mixer, layout) = (counts.mixer(), counts.layout());
        let partition_bits = table::key_bits(k) - layout.remainder_bits;
        let group_bits = GROUP_BITS.min(partition_bits);
        let per_group = 1_usize << (partition_bits - group_bits);
        let mut tables = counts.into_tables().into_iter();
        let groups = (0..1_usize << group_bits)
            .map(|_| {
                let tables: Vec<_> = tables.by_ref().take(per_group).collect();
                Group {
                    pending: Mutex::new(Vec::new()),
                    threshold: AtomicUsize::new(pending_threshold(&tables)),
                    tables: Mutex::new(tables),
                }
            })
            .collect();
        Ok(Self {
            mixer,
            layout,
            entry_bits: table::key_bits(k) - group_bits,
            groups,
        })
    }

    fn updates(&self) -> PackedUpdates {
        PackedUpdates {
            local: (0..self.groups.len()).map(|_| Vec::new()).collect(),
            ..PackedUpdates::default()
        }
    }

    /// Buffer one occurrence of `key`. An entry holds the mixed key's bits
    /// below its group index, with a run of repeated increments above them.
    #[inline(always)]
    fn push(&self, updates: &mut PackedUpdates, key: u64) {
        let mixed = self.mixer.mix(key);
        let entry_mask = u64::MAX >> (64 - self.entry_bits);
        let entry = mixed & entry_mask;
        let run = 1_u64 << self.entry_bits;
        // Run aggregation handles hot keys and short motifs cheaply. A repeat
        // is the same mixed key as the one before, a comparison of two
        // registers; only then is the last entry of its buffer read back. That
        // entry is the earlier occurrence unless the buffer has been handed
        // on since, which the checks below cover.
        let repeated = mixed == updates.previous;
        updates.previous = mixed;
        let buffer = &mut updates.local[(mixed >> self.entry_bits) as usize];
        if repeated
            && let Some(last) = buffer.last_mut()
            && *last & entry_mask == entry
            && *last >> self.entry_bits < !entry_mask >> self.entry_bits
        {
            *last += run;
            return;
        }
        buffer.push(run | entry);
    }

    /// Hand this worker's buffered updates to their groups.
    ///
    /// Different workers start at different groups. A small group that
    /// another worker is updating is skipped and revisited after the others,
    /// and a large one simply keeps the queued updates for that worker's next
    /// visit, so no worker waits on a table unless a queue has grown far past
    /// its threshold.
    fn submit(&self, updates: &mut PackedUpdates, offset: usize) -> Result<(), Error> {
        let mask = self.groups.len() - 1;
        updates.busy.clear();
        for step in 0..self.groups.len() {
            let index = (step + offset) & mask;
            if !updates.local[index].is_empty() && !self.submit_group(index, updates, false)? {
                updates.busy.push(index);
            }
        }
        for position in 0..updates.busy.len() {
            let index = updates.busy[position];
            self.submit_group(index, updates, true)?;
        }
        Ok(())
    }

    /// Hand this worker's updates for group `index` to it: applied at once to
    /// a small group, queued (and applied once enough have collected) for a
    /// large one. A small group that is busy is left alone, returning `false`,
    /// unless `queue_if_busy`.
    fn submit_group(
        &self,
        index: usize,
        updates: &mut PackedUpdates,
        queue_if_busy: bool,
    ) -> Result<bool, Error> {
        let group = &self.groups[index];
        let threshold = group.threshold.load(Ordering::Relaxed);
        if threshold == 0 {
            match group.tables.try_lock() {
                Ok(mut tables) => {
                    // First anything queued while the group was busy, then
                    // this worker's updates straight from its buffer.
                    self.apply(group, &mut tables, updates)?;
                    self.add_direct(&mut tables, &updates.local[index])?;
                    updates.local[index].clear();
                    group
                        .threshold
                        .store(pending_threshold(&tables), Ordering::Relaxed);
                    return Ok(true);
                }
                Err(std::sync::TryLockError::WouldBlock) if !queue_if_busy => return Ok(false),
                Err(std::sync::TryLockError::WouldBlock) => {}
                Err(std::sync::TryLockError::Poisoned(_)) => return Err(Error::WorkerFailed),
            }
        }
        // A small group that stays busy queues like a large one.
        let threshold = threshold.max(MIN_PENDING);
        let queued = {
            let mut pending = group.pending.lock().map_err(|_| Error::WorkerFailed)?;
            let needed = pending.len() + updates.local[index].len();
            if needed > pending.capacity() {
                // Size the queue for its threshold rather than doubling it.
                let target = needed.max(threshold + threshold / 4);
                let additional = target - pending.len();
                pending.reserve_exact(additional);
            }
            pending.extend_from_slice(&updates.local[index]);
            pending.len()
        };
        updates.local[index].clear();
        if queued < threshold {
            return Ok(true);
        }
        let mut tables = match group.tables.try_lock() {
            Ok(tables) => tables,
            Err(std::sync::TryLockError::WouldBlock) => {
                if queued < threshold.saturating_mul(BACKLOG).max(MIN_BACKLOG) {
                    return Ok(true);
                }
                group.tables.lock().map_err(|_| Error::WorkerFailed)?
            }
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(Error::WorkerFailed),
        };
        self.apply(group, &mut tables, updates)?;
        Ok(true)
    }

    /// Apply `entries` to a small group's tables in arrival order.
    fn add_direct(&self, tables: &mut [CompactTable], entries: &[u64]) -> Result<(), Error> {
        let layout = self.layout;
        let (mask, run_shift) = (layout.remainder_mask(), self.entry_bits);
        let last = tables.len() - 1;
        let table_of = |entry: u64| (entry >> layout.remainder_bits) as usize & last;
        for index in 0..entries.len() {
            if let Some(&ahead) = entries.get(index + AHEAD) {
                tables[table_of(ahead)].prefetch(ahead & mask);
            }
            let entry = entries[index];
            tables[table_of(entry)].add(entry & mask, entry >> run_shift, Overflow::Reject)?;
        }
        Ok(())
    }

    /// Apply every update queued for `group` to its tables.
    fn apply(
        &self,
        group: &Group,
        tables: &mut [CompactTable],
        updates: &mut PackedUpdates,
    ) -> Result<(), Error> {
        let PackedUpdates {
            input, split, ends, ..
        } = updates;
        input.clear();
        // Trade buffers: the queue keeps this worker's capacity for reuse.
        std::mem::swap(
            &mut *group.pending.lock().map_err(|_| Error::WorkerFailed)?,
            input,
        );
        if input.is_empty() {
            return Ok(());
        }
        self.add_split(tables, input, split, ends)?;
        group
            .threshold
            .store(pending_threshold(tables), Ordering::Relaxed);
        Ok(())
    }

    /// Apply `input` to a group's tables, one table at a time; see
    /// [`table::add_split`].
    fn add_split(
        &self,
        tables: &mut [CompactTable],
        input: &[u64],
        split: &mut Vec<u64>,
        ends: &mut Vec<usize>,
    ) -> Result<(), Error> {
        let run_shift = self.entry_bits;
        table::add_split(tables, self.layout, input, split, ends, |table, entries| {
            table.warm(entries.len());
            table.add_entries(entries, run_shift)
        })
    }

    /// Apply whatever is still queued, on `threads` threads, and return the
    /// tables.
    fn into_counts(self, threads: usize) -> Result<PackedCounts, Error> {
        let next = AtomicUsize::new(0);
        thread::scope(|scope| {
            let workers: Vec<_> = (0..threads.clamp(1, self.groups.len()))
                .map(|_| {
                    scope.spawn(|| -> Result<(), Error> {
                        let mut updates = PackedUpdates::default();
                        while let Some(group) =
                            self.groups.get(next.fetch_add(1, Ordering::Relaxed))
                        {
                            let mut tables =
                                group.tables.lock().map_err(|_| Error::WorkerFailed)?;
                            self.apply(group, &mut tables, &mut updates)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            workers
                .into_iter()
                .try_for_each(|worker| worker.join().map_err(|_| Error::WorkerFailed)?)
        })?;
        let mut tables = Vec::new();
        for group in self.groups {
            tables.extend(group.tables.into_inner().map_err(|_| Error::WorkerFailed)?);
        }
        Ok(PackedCounts::from_tables(self.mixer, self.layout, tables))
    }
}

struct Partitioned<K, S> {
    shards: Vec<Shard<HashMap<K, u64, S>>>,
    hash: foldhash::fast::RandomState,
}

impl<K: Eq + Hash, S: BuildHasher + Default> Partitioned<K, S> {
    fn new(partitions: usize, capacity: usize) -> Result<Self, Error> {
        let mut shards = Vec::new();
        shards
            .try_reserve_exact(partitions)
            .map_err(|_| Error::ReserveFailed)?;
        for index in 0..partitions {
            let mut map = HashMap::with_hasher(S::default());
            map.try_reserve(capacity / partitions + usize::from(index < capacity % partitions))
                .map_err(|_| Error::ReserveFailed)?;
            shards.push(Shard(Mutex::new(map)));
        }
        Ok(Self {
            shards,
            hash: foldhash::fast::RandomState::default(),
        })
    }

    fn apply(&self, buffers: &mut Updates<K>, offset: usize) -> Result<(), Error> {
        apply_partitions(
            &self.shards,
            &mut buffers.partitions,
            &mut buffers.pending,
            offset,
            |map: &mut HashMap<K, u64, S>, updates: &mut Vec<(K, u64)>| {
                updates
                    .drain(..)
                    .try_for_each(|(key, delta)| add_delta(map, key, delta))
            },
        )?;
        buffers.buffered_bytes = 0;
        Ok(())
    }

    fn into_counts(
        self,
        wrap: impl Fn(HashMap<K, u64, S>) -> Counts,
    ) -> Result<(Vec<Counts>, foldhash::fast::RandomState), Error> {
        let counts = self
            .shards
            .into_iter()
            .map(|shard| {
                shard
                    .0
                    .into_inner()
                    .map(&wrap)
                    .map_err(|_| Error::WorkerFailed)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((counts, self.hash))
    }
}

struct Updates<K> {
    partitions: Vec<Vec<(K, u64)>>,
    // Reused list of nonempty partitions still waiting for their lock.
    pending: Vec<usize>,
    buffered_bytes: usize,
    #[cfg(test)]
    peak_buffered_bytes: usize,
}

impl<K: Eq + Hash> Updates<K> {
    fn new(partitions: usize) -> Self {
        Self {
            partitions: (0..partitions).map(|_| Vec::new()).collect(),
            pending: Vec::with_capacity(partitions),
            buffered_bytes: 0,
            #[cfg(test)]
            peak_buffered_bytes: 0,
        }
    }

    fn push(&mut self, hash: &foldhash::fast::RandomState, key: K) -> Result<(), Error> {
        let index = hash.hash_one(&key) as usize & (self.partitions.len() - 1);
        let buffer = &mut self.partitions[index];
        // Run aggregation within a partition handles hot keys and small motifs
        // without building another hash table or sorting ordinary random input.
        if let Some((previous, delta)) = buffer.last_mut()
            && *previous == key
        {
            *delta = delta.checked_add(1).ok_or(Error::CountOverflow)?;
        } else {
            buffer.push((key, 1));
        }
        Ok(())
    }
}

impl Updates<Vec<u8>> {
    fn push_bytes(&mut self, hash: &foldhash::fast::RandomState, key: &[u8]) -> Result<(), Error> {
        let index = hash.hash_one(key) as usize & (self.partitions.len() - 1);
        let buffer = &mut self.partitions[index];
        if let Some((previous, delta)) = buffer.last_mut()
            && previous.as_slice() == key
        {
            *delta = delta.checked_add(1).ok_or(Error::CountOverflow)?;
        } else {
            buffer.push((key.to_vec(), 1));
            self.buffered_bytes = self
                .buffered_bytes
                .saturating_add(key.len())
                .saturating_add(std::mem::size_of::<(Vec<u8>, u64)>());
            #[cfg(test)]
            {
                self.peak_buffered_bytes = self.peak_buffered_bytes.max(self.buffered_bytes);
            }
        }
        Ok(())
    }
}

enum SharedCounts {
    Packed(PackedShared),
    Packed128(Partitioned<(u64, u64), foldhash::fast::RandomState>),
    Bytes(Partitioned<Vec<u8>, std::collections::hash_map::RandomState>),
    /// Nothing is counted: input is split into bins on disk; see `crate::disk`.
    Bins(Bins),
}

enum Buffers {
    Packed(PackedUpdates),
    Packed128(Updates<(u64, u64)>),
    Bytes(Updates<Vec<u8>>),
    Bins(BinBuffers),
}

impl SharedCounts {
    fn new(k: usize, partitions: usize, capacity: usize) -> Result<Self, Error> {
        Ok(if k <= 32 {
            Self::Packed(PackedShared::new(k, capacity)?)
        } else if k <= 64 {
            Self::Packed128(Partitioned::new(partitions, capacity)?)
        } else {
            Self::Bytes(Partitioned::new(partitions, capacity)?)
        })
    }

    fn buffers(&self) -> Buffers {
        match self {
            Self::Packed(table) => Buffers::Packed(table.updates()),
            Self::Packed128(table) => Buffers::Packed128(Updates::new(table.shards.len())),
            Self::Bytes(table) => Buffers::Bytes(Updates::new(table.shards.len())),
            Self::Bins(bins) => Buffers::Bins(bins.buffers()),
        }
    }

    /// Hand over whatever a finishing worker still holds in `buffers`.
    fn flush(&self, buffers: &mut Buffers) -> Result<(), Error> {
        match (self, buffers) {
            (Self::Bins(bins), Buffers::Bins(buffers)) => bins.flush(buffers),
            // Counting workers submit their updates after every batch.
            _ => Ok(()),
        }
    }

    fn count(
        &self,
        batch: &Batch,
        buffers: &mut Buffers,
        k: usize,
        strand: StrandMode,
        offset: usize,
    ) -> Result<(), Error> {
        match (self, buffers) {
            (Self::Packed(table), Buffers::Packed(buffers)) => {
                for record in batch.records() {
                    visit_packed(record, k, strand, |key| {
                        table.push(buffers, key);
                        Ok(())
                    })?;
                }
                table.submit(buffers, offset)
            }
            (Self::Packed128(table), Buffers::Packed128(buffers)) => {
                for record in batch.records() {
                    visit_packed128(record, k, strand, |key| buffers.push(&table.hash, key))?;
                }
                table.apply(buffers, offset)
            }
            (Self::Bytes(table), Buffers::Bytes(buffers)) => {
                for record in batch.records() {
                    if record.len() < k {
                        continue;
                    }
                    visit_bytes(record, k, strand, |key| {
                        buffers.push_bytes(&table.hash, key)?;
                        // Large exact keys must not accumulate for an entire
                        // sequence batch. Limit owned key payload per worker,
                        // allowing at most one key to cross the byte budget.
                        if buffers.buffered_bytes >= BYTE_UPDATE_BYTES {
                            table.apply(buffers, offset)?;
                        }
                        Ok(())
                    })?;
                }
                table.apply(buffers, offset)
            }
            (Self::Bins(bins), Buffers::Bins(buffers)) => batch
                .records()
                .try_for_each(|record| bins.split(record, buffers)),
            _ => unreachable!("worker buffers match the shared key representation"),
        }
    }

    fn finish(self, k: usize, strand: StrandMode, threads: usize) -> Result<Counter, Error> {
        let (partitions, hash) = match self {
            Self::Packed(table) => {
                return Ok(Counter::from_packed(
                    k,
                    strand,
                    table.into_counts(threads)?,
                    threads,
                ));
            }
            Self::Packed128(table) => table.into_counts(Counts::Packed128)?,
            Self::Bytes(table) => table.into_counts(Counts::Bytes)?,
            Self::Bins(_) => unreachable!("bins are finished by `DiskCounter`"),
        };
        Ok(Counter::from_partitions(
            k, strand, partitions, hash, threads,
        ))
    }
}

/// Wall-clock phases measured from construction to completion, excluding output.
#[derive(Debug, Clone, Copy)]
pub struct ParallelStats {
    /// Input submission, extraction, and counting, including waiting for workers.
    pub counting: Duration,
    /// Applying updates still queued when input ends; no whole-table merging
    /// is performed. The field name is retained for API and CLI timing
    /// compatibility.
    pub merging: Duration,
}

/// An exact streaming counter with parallel extraction and key-owned partitions.
///
/// Bounded reusable sequence batches preserve record boundaries and split long
/// records with k−1 context bases. Workers aggregate updates into reusable local
/// buffers, one per group of count tables, and hand each buffer to its group
/// after a batch. For k <= 32 a large group queues updates until it holds
/// about one per table cache line; one worker then splits the queue by table
/// and applies it, so each table is read into cache once and updated in a
/// burst rather than missing the cache on every k-mer. Any worker can apply a
/// group; updates never lock once per k-mer.
/// Count tables remain partitioned through finalization, iteration, and output.
/// Input buffering is bounded by worker count and batch size, and queued
/// updates normally stay within an eighth of the tables' memory; the
/// distinct-key tables remain unbounded. Drop closes and joins workers.
pub struct ParallelCounter {
    input: Batcher,
    workers: Vec<JoinHandle<Result<(), Error>>>,
    failure: Arc<Mutex<Option<Error>>>,
    shared: Option<Arc<SharedCounts>>,
    threads: usize,
    k: usize,
    strand: StrandMode,
    started: Instant,
}

impl ParallelCounter {
    /// Start `threads` workers. The caller submits input separately.
    pub fn new(k: usize, strand: StrandMode, threads: usize) -> Result<Self, Error> {
        Self::with_capacity(k, strand, threads, 0)
    }

    /// Divide a total distinct-key capacity hint between key partitions.
    /// The hint is not a limit; keys are stored in exactly one partition.
    pub fn with_capacity(
        k: usize,
        strand: StrandMode,
        threads: usize,
        capacity: usize,
    ) -> Result<Self, Error> {
        if k == 0 {
            return Err(Error::InvalidK);
        }
        // Partitions for wider keys (k > 32). Enough that workers rarely find a
        // partition busy, few enough to keep per-worker buffers cache friendly.
        let partitions = (threads.min(32) * 32).next_power_of_two().clamp(16, 1024);
        Self::start(
            k,
            strand,
            threads,
            SharedCounts::new(k, partitions, capacity)?,
        )
    }

    /// Start workers that split input into `bins` instead of counting it.
    pub(crate) fn with_bins(
        k: usize,
        strand: StrandMode,
        threads: usize,
        bins: Bins,
    ) -> Result<Self, Error> {
        Self::start(k, strand, threads, SharedCounts::Bins(bins))
    }

    fn start(
        k: usize,
        strand: StrandMode,
        threads: usize,
        shared: SharedCounts,
    ) -> Result<Self, Error> {
        let started = Instant::now();
        let queued = threads
            .checked_mul(2)
            .filter(|&n| n > 0)
            .ok_or(Error::InvalidThreads)?;
        let shared = Arc::new(shared);
        let mut workers = Vec::new();
        workers
            .try_reserve(threads)
            .map_err(|_| Error::ReserveFailed)?;
        let (sender, receiver) = channel::bounded::<Batch>(queued);
        let (recycle, recycled) = channel::unbounded();
        let failure = Arc::new(Mutex::new(None));
        let mut result = Self {
            input: Batcher {
                sender: Some(sender),
                recycled,
                failure: Arc::clone(&failure),
                batch: Batch::default(),
                record_start: None,
                continuation: Vec::new(),
                batch_bases: INITIAL_BATCH_BASES,
                warm_batches: queued,
                k,
            },
            workers,
            failure,
            shared: Some(Arc::clone(&shared)),
            threads,
            k,
            strand,
            started,
        };
        for index in 0..threads {
            let shared = Arc::clone(&shared);
            let receiver = receiver.clone();
            let recycle = recycle.clone();
            let failure = Arc::clone(&result.failure);
            let worker = thread::Builder::new()
                .name(format!("count-{index}"))
                .spawn(move || {
                    // Turn panics into a shared failure so other receivers stop and
                    // a producer blocked on a full queue is eventually released.
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut buffers = shared.buffers();
                        let mut offset = index;
                        while let Ok(mut batch) = receiver.recv() {
                            if let Some(error) = *failure.lock().map_err(|_| Error::WorkerFailed)? {
                                return Err(error);
                            }
                            shared.count(&batch, &mut buffers, k, strand, offset)?;
                            offset = offset.wrapping_add(1);
                            batch.clear();
                            if recycle.send(batch).is_err() {
                                break;
                            }
                        }
                        shared.flush(&mut buffers)
                    }))
                    .unwrap_or(Err(Error::WorkerFailed));
                    if let Err(error) = result {
                        let mut failure =
                            failure.lock().unwrap_or_else(|poison| poison.into_inner());
                        failure.get_or_insert(error);
                    }
                    result
                })
                .map_err(|_| Error::WorkerFailed)?;
            result.workers.push(worker);
        }
        Ok(result)
    }

    /// Submit a complete record. Worker errors may be reported by `finish`.
    pub fn add_record(&mut self, sequence: &[u8]) -> Result<(), Error> {
        self.input.add_record(sequence)
    }

    /// Begin a new streaming record; see [`ParallelInput::start_record`].
    pub fn start_record(&mut self) {
        self.input.start_record();
    }

    /// Append a fragment of the current streaming record.
    pub fn add_sequence(&mut self, sequence: &[u8]) -> Result<(), Error> {
        self.input.add_sequence(sequence)
    }

    /// Create an independent input producer for a scoped reader thread.
    /// Producers share the bounded work queue and recycle pool. Their borrow
    /// prevents finishing the counter until every producer has been released.
    pub fn input(&self) -> ParallelInput<'_> {
        ParallelInput {
            batcher: Batcher {
                sender: self.input.sender.clone(),
                recycled: self.input.recycled.clone(),
                failure: Arc::clone(&self.failure),
                batch: self.input.recycled.try_recv().unwrap_or_default(),
                record_start: None,
                continuation: Vec::new(),
                batch_bases: INITIAL_BATCH_BASES,
                warm_batches: self.input.sender.as_ref().unwrap().capacity().unwrap(),
                k: self.k,
            },
            _counter: std::marker::PhantomData,
        }
    }

    /// Wait for counting and retain the final key partitions without reinsertion.
    pub fn finish(self) -> Result<Counter, Error> {
        self.finish_with_stats().map(|(counter, _)| counter)
    }

    /// K-mer length, strand mode and worker count.
    pub(crate) fn shape(&self) -> (usize, StrandMode, usize) {
        (self.k, self.strand, self.threads)
    }

    /// Submit remaining input, wait for the workers, and take the shared state.
    fn join(&mut self) -> Result<SharedCounts, Error> {
        self.input.submit()?;
        self.input.sender.take();
        while let Some(worker) = self.workers.pop() {
            worker.join().map_err(|_| Error::WorkerFailed)??;
        }
        Arc::try_unwrap(self.shared.take().unwrap()).map_err(|_| Error::WorkerFailed)
    }

    /// Wait for the workers of a counter started by `with_bins`.
    pub(crate) fn into_bins(mut self) -> Result<Bins, Error> {
        match self.join()? {
            SharedCounts::Bins(bins) => Ok(bins),
            _ => unreachable!("only `with_bins` counters are finished as bins"),
        }
    }

    /// Finish counting and report counting/finalization durations.
    pub fn finish_with_stats(mut self) -> Result<(Counter, ParallelStats), Error> {
        let shared = self.join()?;
        let counting = self.started.elapsed();
        let merging = Instant::now();
        let counter = shared.finish(self.k, self.strand, self.threads)?;
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
        self.input.sender.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
struct Batcher {
    sender: Option<channel::Sender<Batch>>,
    recycled: channel::Receiver<Batch>,
    failure: Arc<Mutex<Option<Error>>>,
    batch: Batch,
    record_start: Option<usize>,
    continuation: Vec<u8>,
    batch_bases: usize,
    warm_batches: usize,
    k: usize,
}

impl Batcher {
    /// Submit a complete record. Worker errors may be reported by `finish`.
    pub fn add_record(&mut self, sequence: &[u8]) -> Result<(), Error> {
        self.start_record();
        if sequence.len() < self.k {
            return Ok(());
        }
        self.add_sequence(sequence)?;
        self.start_record();
        Ok(())
    }

    /// Begin a new streaming record. Subsequent `add_sequence` fragments belong
    /// to this record until the next `start_record` or independent `add_record`.
    /// This resets context without flushing short records into separate batches.
    pub fn start_record(&mut self) {
        self.record_start = None;
        self.continuation.clear();
    }

    /// Append a sequence fragment to the current record, preserving windows
    /// across fragments. Data is copied directly into reusable worker batches;
    /// only k−1 context bases are copied again at batch boundaries.
    /// A fresh counter implicitly starts its first streaming record.
    pub fn add_sequence(&mut self, mut sequence: &[u8]) -> Result<(), Error> {
        while !sequence.is_empty() {
            let limit = self.batch_bases.saturating_add(self.k - 1);
            if self.record_start.is_none() {
                self.record_start = Some(self.batch.bases.len());
                self.batch.bases.extend_from_slice(&self.continuation);
                self.continuation.clear();
                self.batch.ends.push(self.batch.bases.len());
            }
            let take = sequence.len().min(limit - self.batch.bases.len());
            self.batch.bases.extend_from_slice(&sequence[..take]);
            *self.batch.ends.last_mut().unwrap() = self.batch.bases.len();
            sequence = &sequence[take..];
            if self.batch.bases.len() == limit {
                let start = self.record_start.unwrap();
                let keep = (self.k - 1).min(self.batch.bases.len() - start);
                let tail = self.batch.bases.len() - keep;
                self.continuation
                    .extend_from_slice(&self.batch.bases[tail..]);
                self.record_start = None;
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
        })?;
        // Fill the worker pool quickly on small inputs, then amortize routing
        // and synchronization with larger batches on sustained streams.
        if self.warm_batches > 0 {
            self.warm_batches -= 1;
        } else {
            self.batch_bases = (self.batch_bases * 2).min(BATCH_BASES);
        }
        Ok(())
    }
}

/// An independent, bounded-memory producer for a [`ParallelCounter`].
/// Call `finish` to submit its last partial batch and report submission errors.
/// Drop also attempts submission; worker errors remain visible on the counter.
/// With P simultaneous producers, each may retain a partial batch plus k−1
/// context bases in addition to the globally bounded work queue.
#[must_use = "finish the producer to report submission errors"]
pub struct ParallelInput<'a> {
    batcher: Batcher,
    _counter: std::marker::PhantomData<&'a ParallelCounter>,
}

impl ParallelInput<'_> {
    /// Submit an independent complete record; never joins adjacent calls.
    pub fn add_record(&mut self, sequence: &[u8]) -> Result<(), Error> {
        self.batcher.add_record(sequence)
    }

    /// Begin a new record. Fragments added with `add_sequence` share context
    /// until the next `start_record` or independent `add_record` call.
    pub fn start_record(&mut self) {
        self.batcher.start_record();
    }

    /// Append a sequence fragment, copying directly into reusable batches.
    /// A new producer implicitly starts its first streaming record.
    pub fn add_sequence(&mut self, sequence: &[u8]) -> Result<(), Error> {
        self.batcher.add_sequence(sequence)
    }

    /// Submit the last partial batch and release this producer.
    pub fn finish(mut self) -> Result<(), Error> {
        self.batcher.submit()
    }
}

impl Drop for ParallelInput<'_> {
    fn drop(&mut self) {
        let _ = self.batcher.submit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn table(counter: &Counter) -> BTreeMap<Vec<u8>, u64> {
        counter.counts().collect()
    }

    #[test]
    fn byte_keys_flush_mid_record_with_a_fixed_payload_budget() {
        let mut state = 43_u64;
        let genome: Vec<u8> = (0..4096)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                b"ACGT"[(state & 3) as usize]
            })
            .collect();
        let sequence = genome.repeat(16);
        for k in [65, 257] {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let shared = SharedCounts::new(k, 32, 0).unwrap();
                let mut buffers = shared.buffers();
                let batch = Batch {
                    bases: sequence.clone(),
                    ends: vec![sequence.len()],
                };
                shared.count(&batch, &mut buffers, k, strand, 0).unwrap();
                let Buffers::Bytes(updates) = &buffers else {
                    unreachable!()
                };
                assert!(updates.peak_buffered_bytes >= BYTE_UPDATE_BYTES);
                assert!(
                    updates.peak_buffered_bytes
                        <= BYTE_UPDATE_BYTES + k + std::mem::size_of::<(Vec<u8>, u64)>()
                );
                assert_eq!(updates.buffered_bytes, 0);
                let result = shared.finish(k, strand, 1).unwrap();
                let mut expected = Counter::new(k, strand).unwrap();
                expected.add_record(&sequence).unwrap();
                assert_eq!(table(&result), table(&expected), "k={k} {strand:?}");
            }
        }
    }

    #[test]
    fn streaming_fragments_and_independent_producers_preserve_windows() {
        let mut long = b"acGTTGCAGATTCGAC".repeat(BATCH_BASES / 8 + 13);
        for index in [BATCH_BASES - 1, BATCH_BASES + 29, 2 * BATCH_BASES] {
            long[index] = b'N';
        }
        for k in [1, 31, 32, 33, 64, 65] {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let mut expected = Counter::new(k, strand).unwrap();
                for record in [&long[..], b"AC", b"GT", &long, b"ACGTTGCA"] {
                    expected.add_record(record).unwrap();
                }
                let parallel = ParallelCounter::new(k, strand, 3).unwrap();
                let long = &long;
                thread::scope(|scope| {
                    let mut first = parallel.input();
                    let mut second = parallel.input();
                    scope.spawn(|| {
                        // The first record is implicit; empty fragments preserve it.
                        let mut offset = 0;
                        for size in [1, 17, 65_536, BATCH_BASES + 31].into_iter().cycle() {
                            let end = (offset + size).min(long.len());
                            first.add_sequence(&long[offset..end]).unwrap();
                            first.add_sequence(b"").unwrap();
                            offset = end;
                            if offset == long.len() {
                                break;
                            }
                        }
                        first.start_record();
                        first.add_sequence(b"AC").unwrap();
                        first.start_record();
                        first.add_sequence(b"GT").unwrap();
                        first.finish().unwrap();
                    });
                    scope.spawn(move || {
                        second.add_record(long).unwrap();
                        second.add_record(b"ACGTTGCA").unwrap();
                        // Dropping a producer also submits its partial batch.
                    });
                });
                let result = parallel.finish().unwrap();
                assert_eq!(table(&result), table(&expected), "k={k} {strand:?}");
            }
        }
    }

    #[test]
    fn partitioned_results_support_mutation_merging_and_both_outputs() {
        for k in [1, 31, 32, 33, 64, 65] {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let record = b"acGTTGCAGATTCGAC".repeat(300);
                let mut parallel = ParallelCounter::new(k, strand, 3).unwrap();
                parallel.add_record(&record).unwrap();
                let mut result = parallel.finish().unwrap();
                // Packed keys partition internally; wider keys keep separate maps.
                assert!(match &result.counts {
                    Counts::Packed(counts) => counts.tables().len() > 1,
                    _ => !result.partitions.is_empty(),
                });
                result.add_record(&record).unwrap();
                let mut other = ParallelCounter::new(k, strand, 2).unwrap();
                other.add_record(&record).unwrap();
                // Each result has a different partition count and hash seed.
                result.merge(other.finish().unwrap()).unwrap();
                let mut expected = Counter::new(k, strand).unwrap();
                for _ in 0..3 {
                    expected.add_record(&record).unwrap();
                }
                assert_eq!(table(&result), table(&expected));
                let mut binary = Vec::new();
                result.write_binary(&mut binary).unwrap();
                assert_eq!(
                    table(&Counter::read_binary(binary.as_slice()).unwrap()),
                    table(&expected)
                );
                let mut tsv = Vec::new();
                result.write_tsv(&mut tsv).unwrap();
                let mut dumped = Vec::new();
                crate::database::DatabaseReader::new(binary.as_slice())
                    .unwrap()
                    .write_tsv(&mut dumped)
                    .unwrap();
                assert_eq!(tsv, dumped);
                let mut plain = Counter::new(k, strand).unwrap();
                plain.merge(result).unwrap();
                assert_eq!(table(&plain), table(&expected));

                let mut partitioned = ParallelCounter::new(k, strand, 2).unwrap();
                partitioned.add_record(&vec![b'A'; k]).unwrap();
                let mut partitioned = partitioned.finish().unwrap();
                for counts in
                    std::iter::once(&mut partitioned.counts).chain(&mut partitioned.partitions)
                {
                    counts.set_counts(u64::MAX);
                }
                let before = table(&partitioned);
                assert_eq!(
                    partitioned.add_record(&vec![b'A'; k]),
                    Err(Error::CountOverflow)
                );
                assert_eq!(table(&partitioned), before);
                let mut delta = Counter::new(k, strand).unwrap();
                delta.add_record(&vec![b'A'; k]).unwrap();
                assert_eq!(partitioned.merge(delta), Err(Error::CountOverflow));
                assert_eq!(table(&partitioned), before);
            }
        }
    }

    #[test]
    fn worker_overflow_releases_a_producer_blocked_on_a_full_queue() {
        let (send, receive) = std::sync::mpsc::channel();
        let handle = thread::spawn(move || {
            let mut counter = ParallelCounter::new(31, StrandMode::Forward, 3).unwrap();
            let SharedCounts::Packed(table) = counter.shared.as_ref().unwrap().as_ref() else {
                unreachable!()
            };
            let (partition, remainder) = table::route(table.mixer, table.layout, 0);
            let per_group = 1 << (table.entry_bits - table.layout.remainder_bits);
            table.groups[partition / per_group].tables.lock().unwrap()[partition % per_group]
                .add(remainder, u64::MAX, Overflow::Reject)
                .unwrap();
            let result = counter.add_record(&vec![b'A'; BATCH_BASES * 32]);
            let result = result.and_then(|()| counter.finish().map(|_| ()));
            send.send(result).unwrap();
        });
        assert_eq!(
            receive.recv_timeout(Duration::from_secs(10)).unwrap(),
            Err(Error::CountOverflow)
        );
        handle.join().unwrap();
    }

    #[test]
    fn queued_and_direct_groups_match_serial_counts() {
        // Enough distinct keys that groups outgrow direct updates and queue,
        // with repeats so that queued runs and existing keys are both hit.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let genome: Vec<u8> = (0..150_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                b"ACGT"[(state >> 33 & 3) as usize]
            })
            .collect();
        let mut sequence = genome.clone();
        sequence.extend_from_slice(&genome[..60_000]);
        sequence.extend_from_slice(&vec![b'A'; 5_000]);
        for k in [7, 16, 31, 32] {
            for threads in [1, 4] {
                let mut parallel = ParallelCounter::new(k, StrandMode::Canonical, threads).unwrap();
                let mut serial = Counter::new(k, StrandMode::Canonical).unwrap();
                for record in sequence.chunks(70_001) {
                    parallel.add_record(record).unwrap();
                    serial.add_record(record).unwrap();
                }
                if k == 31 {
                    let SharedCounts::Packed(shared) = parallel.shared.as_ref().unwrap().as_ref()
                    else {
                        unreachable!()
                    };
                    assert!(
                        shared.groups.len() > 1 && shared.entry_bits > shared.layout.remainder_bits
                    );
                }
                let parallel = parallel.finish().unwrap();
                assert_eq!(table(&parallel), table(&serial), "k={k} threads={threads}");
            }
        }
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
                overflowing.counts.set_counts(u64::MAX);
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
        disconnected.input.sender = Some(sender);
        assert_eq!(
            disconnected.add_record(&vec![b'A'; BATCH_BASES + 31]),
            Err(Error::WorkerFailed)
        );
    }
}
