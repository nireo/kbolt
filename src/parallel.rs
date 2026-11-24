//! Parallel extraction and buffered updates to exclusively owned key partitions.
//! Partition locks are acquired once per batch, never once per k-mer, and
//! updates to large tables are queued and applied one cache-sized table at a
//! time.

use crossbeam_channel as channel;
use std::collections::HashMap;
use std::hash::BuildHasher;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::disk::{BinBuffers, Bins};
use crate::table::{self, CompactTable, Layout, Load, Mixer, Overflow, PackedCounts};
use crate::table_wide::{
    BucketWide, CompactTableWide, HighKey, LayoutWide, PackedCountsWide, QueuedEntry, mix_lo,
};
use crate::{
    ByteCounts, Counter, Counts, Error, StrandMode, WideCounts, add_delta, visit_bytes,
    visit_packed, visit_packed128, visit_packed192, visit_packed256,
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
struct Group<T, E> {
    /// Queued updates in arrival order. Held only to append a buffer or to
    /// take the whole queue.
    pending: Mutex<Vec<E>>,
    /// The group's tables, in sub-partition order. Whoever holds this lock
    /// applies queued updates.
    tables: Mutex<Vec<T>>,
    /// Queue length at which applying is worthwhile; follows the table sizes.
    threshold: AtomicUsize,
}

/// Worker-local buffers shared by packed and wide group backends.
struct GroupUpdates<E> {
    /// Updates not yet queued, one buffer per group.
    local: Vec<Vec<E>>,
    /// A group's queue while this worker applies it.
    input: Vec<E>,
    /// The same updates grouped by table.
    split: Vec<E>,
    /// Where each table's updates end (then, while scattering, the cursors).
    ends: Vec<usize>,
    /// Small groups found busy during a pass, to revisit at its end.
    busy: Vec<usize>,
}

impl<E> Default for GroupUpdates<E> {
    fn default() -> Self {
        Self {
            local: Vec::new(),
            input: Vec::new(),
            split: Vec::new(),
            ends: Vec::new(),
            busy: Vec::new(),
        }
    }
}

impl<E> GroupUpdates<E> {
    fn for_groups(groups: usize) -> Self {
        Self {
            local: (0..groups).map(|_| Vec::new()).collect(),
            ..Self::default()
        }
    }
}

/// Common group-queueing and submission behavior for 64-bit and wide compact
/// tables, monomorphized so hot update loops stay fully specialized.
trait GroupBackend: Sync {
    type Table: Send;
    type Entry: Copy + Default + Send;

    fn groups(&self) -> &[Group<Self::Table, Self::Entry>];
    fn into_groups(self) -> Vec<Group<Self::Table, Self::Entry>>;
    fn threshold(&self, tables: &[Self::Table]) -> usize;
    fn add_direct(&self, tables: &mut [Self::Table], entries: &[Self::Entry]) -> Result<(), Error>;
    fn add_split(
        &self,
        tables: &mut [Self::Table],
        input: &[Self::Entry],
        split: &mut Vec<Self::Entry>,
        ends: &mut Vec<usize>,
    ) -> Result<(), Error>;

    /// Hand this worker's buffered updates to their groups.
    ///
    /// Different workers start at different groups. A small group that
    /// another worker is updating is skipped and revisited after the others,
    /// and a large one simply keeps the queued updates for that worker's next
    /// visit, so no worker waits on a table unless a queue has grown far past
    /// its threshold.
    fn submit(&self, updates: &mut GroupUpdates<Self::Entry>, offset: usize) -> Result<(), Error> {
        let len = self.groups().len();
        let mask = len - 1;
        updates.busy.clear();
        for step in 0..len {
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
        updates: &mut GroupUpdates<Self::Entry>,
        queue_if_busy: bool,
    ) -> Result<bool, Error> {
        let group = &self.groups()[index];
        let threshold = group.threshold.load(Ordering::Relaxed);
        if threshold == 0 {
            match group.tables.try_lock() {
                Ok(mut tables) => {
                    // First anything queued while the group was busy, then
                    // this worker's updates straight from its buffer.
                    self.apply_group(group, &mut tables, updates)?;
                    self.add_direct(&mut tables, &updates.local[index])?;
                    updates.local[index].clear();
                    group
                        .threshold
                        .store(self.threshold(&tables), Ordering::Relaxed);
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
        self.apply_group(group, &mut tables, updates)?;
        Ok(true)
    }

    /// Apply every update queued for `group` to its tables.
    fn apply_group(
        &self,
        group: &Group<Self::Table, Self::Entry>,
        tables: &mut [Self::Table],
        updates: &mut GroupUpdates<Self::Entry>,
    ) -> Result<(), Error> {
        let GroupUpdates {
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
            .store(self.threshold(tables), Ordering::Relaxed);
        Ok(())
    }

    /// Apply whatever is still queued, on `threads` threads, and return all
    /// tables in partition order.
    fn finish_tables(self, threads: usize) -> Result<Vec<Self::Table>, Error>
    where
        Self: Sized,
    {
        let next = AtomicUsize::new(0);
        thread::scope(|scope| {
            let workers: Vec<_> = (0..threads.clamp(1, self.groups().len()))
                .map(|_| {
                    scope.spawn(|| -> Result<(), Error> {
                        let mut updates = GroupUpdates::default();
                        while let Some(group) =
                            self.groups().get(next.fetch_add(1, Ordering::Relaxed))
                        {
                            let mut tables =
                                group.tables.lock().map_err(|_| Error::WorkerFailed)?;
                            self.apply_group(group, &mut tables, &mut updates)?;
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
        for group in self.into_groups() {
            tables.extend(group.tables.into_inner().map_err(|_| Error::WorkerFailed)?);
        }
        Ok(tables)
    }
}

fn make_groups<T, E>(
    tables: Vec<T>,
    group_bits: u32,
    per_group: usize,
    threshold: impl Fn(&[T]) -> usize,
) -> Vec<Group<T, E>> {
    let mut tables = tables.into_iter();
    (0..1_usize << group_bits)
        .map(|_| {
            let group_tables: Vec<_> = tables.by_ref().take(per_group).collect();
            Group {
                pending: Mutex::new(Vec::new()),
                threshold: AtomicUsize::new(threshold(&group_tables)),
                tables: Mutex::new(group_tables),
            }
        })
        .collect()
}

/// Shared compact tables for packed keys (k <= 32); see `crate::table`.
struct PackedShared {
    mixer: Mixer,
    layout: Layout,
    /// Bits of a mixed key below its group index: sub-partition and remainder.
    entry_bits: u32,
    groups: Vec<Group<CompactTable, u64>>,
}

/// Worker-local state for packed keys.
#[derive(Default)]
struct PackedUpdates {
    group: GroupUpdates<u64>,
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
        let groups = make_groups(counts.into_tables(), group_bits, per_group, pending_threshold);
        Ok(Self {
            mixer,
            layout,
            entry_bits: table::key_bits(k) - group_bits,
            groups,
        })
    }

    fn updates(&self) -> PackedUpdates {
        PackedUpdates {
            group: GroupUpdates::for_groups(self.groups.len()),
            previous: 0,
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
        let buffer = &mut updates.group.local[(mixed >> self.entry_bits) as usize];
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

    fn into_counts(self, threads: usize) -> Result<PackedCounts, Error> {
        let (mixer, layout) = (self.mixer, self.layout);
        Ok(PackedCounts::from_tables(
            mixer,
            layout,
            self.finish_tables(threads)?,
        ))
    }
}

impl GroupBackend for PackedShared {
    type Table = CompactTable;
    type Entry = u64;

    fn groups(&self) -> &[Group<CompactTable, u64>] {
        &self.groups
    }

    fn into_groups(self) -> Vec<Group<CompactTable, u64>> {
        self.groups
    }

    fn threshold(&self, tables: &[CompactTable]) -> usize {
        pending_threshold(tables)
    }

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

    fn add_split(
        &self,
        tables: &mut [CompactTable],
        input: &[u64],
        split: &mut Vec<u64>,
        ends: &mut Vec<usize>,
    ) -> Result<(), Error> {
        let (shift, mask, run_shift) = (self.layout.remainder_bits, tables.len() - 1, self.entry_bits);
        table::add_split(
            tables,
            input,
            split,
            ends,
            |entry| (entry >> shift) as usize & mask,
            |table, entries| {
                table.warm(entries.len());
                table.add_entries(entries, run_shift)
            },
        )
    }
}

/// Shared compact tables for wide packed keys (`33 <= k <= 128`); see `crate::table_wide`.
struct PackedWideShared<H: HighKey> {
    layout: LayoutWide,
    /// Bits of `mixed_l` below its group index (`64 - GROUP_BITS = 58`).
    entry_bits: u32,
    groups: Vec<Group<CompactTableWide<H>, QueuedEntry<H>>>,
}

struct PackedWideUpdates<H: HighKey> {
    group: GroupUpdates<QueuedEntry<H>>,
    previous_l: u64,
    previous_h: H,
}

fn pending_threshold_wide<H: HighKey>(tables: &[CompactTableWide<H>]) -> usize {
    let bucket_bytes = std::mem::size_of::<BucketWide<H>>();
    let direct_buckets = if cfg!(test) {
        2
    } else {
        (512 * 1024) / bucket_bytes
    };
    let max_pending = (32 * 1024 * 1024) / std::mem::size_of::<QueuedEntry<H>>();
    let buckets: usize = tables.iter().map(CompactTableWide::buckets).sum();
    if buckets <= direct_buckets {
        0
    } else {
        buckets.clamp(MIN_PENDING, max_pending)
    }
}

impl<H: HighKey> PackedWideShared<H> {
    fn new(capacity: usize) -> Result<Self, Error> {
        let counts =
            PackedCountsWide::<H>::with_capacity(PARALLEL_PARTITION_BITS, capacity, Load::Dense)?;
        let layout = counts.layout();
        let partition_bits = layout.partition_bits;
        let group_bits = GROUP_BITS.min(partition_bits);
        let per_group = 1_usize << (partition_bits - group_bits);
        let groups = make_groups(
            counts.into_tables(),
            group_bits,
            per_group,
            pending_threshold_wide,
        );
        Ok(Self {
            layout,
            entry_bits: 64 - group_bits,
            groups,
        })
    }

    fn updates(&self) -> PackedWideUpdates<H> {
        PackedWideUpdates {
            group: GroupUpdates::for_groups(self.groups.len()),
            previous_l: 0,
            previous_h: H::default(),
        }
    }

    #[inline(always)]
    fn push(&self, updates: &mut PackedWideUpdates<H>, hi: H, lo: u64) {
        let l = mix_lo(hi, lo);
        let entry_mask = u64::MAX >> (64 - self.entry_bits);
        let entry_l = l & entry_mask;
        let run = 1_u64 << self.entry_bits;
        let repeated = l == updates.previous_l && hi == updates.previous_h;
        updates.previous_l = l;
        updates.previous_h = hi;
        let buffer = &mut updates.group.local[(l >> self.entry_bits) as usize];
        if repeated
            && let Some(last) = buffer.last_mut()
            && last.l() & entry_mask == entry_l
            && last.h() == hi
            && last.l() >> self.entry_bits < !entry_mask >> self.entry_bits
        {
            *last = QueuedEntry::new(last.l() + run, hi);
            return;
        }
        buffer.push(QueuedEntry::new(run | entry_l, hi));
    }

    fn into_counts(self, threads: usize) -> Result<PackedCountsWide<H>, Error> {
        let layout = self.layout;
        Ok(PackedCountsWide::from_tables(
            layout,
            self.finish_tables(threads)?,
        ))
    }
}

impl<H: HighKey> GroupBackend for PackedWideShared<H> {
    type Table = CompactTableWide<H>;
    type Entry = QueuedEntry<H>;

    fn groups(&self) -> &[Group<CompactTableWide<H>, QueuedEntry<H>>] {
        &self.groups
    }

    fn into_groups(self) -> Vec<Group<CompactTableWide<H>, QueuedEntry<H>>> {
        self.groups
    }

    fn threshold(&self, tables: &[CompactTableWide<H>]) -> usize {
        pending_threshold_wide(tables)
    }

    fn add_direct(
        &self,
        tables: &mut [CompactTableWide<H>],
        entries: &[QueuedEntry<H>],
    ) -> Result<(), Error> {
        let layout = self.layout;
        let (mask, run_shift) = (layout.l_rem_mask(), self.entry_bits);
        let last = tables.len() - 1;
        let table_of = |l_entry: u64| (l_entry >> layout.l_rem_bits) as usize & last;
        for index in 0..entries.len() {
            if let Some(ahead) = entries.get(index + AHEAD) {
                let ahead_l = ahead.l();
                tables[table_of(ahead_l)].prefetch(ahead_l & mask);
            }
            let entry = entries[index];
            let l_entry = entry.l();
            tables[table_of(l_entry)].add(
                l_entry & mask,
                entry.h(),
                l_entry >> run_shift,
                Overflow::Reject,
            )?;
        }
        Ok(())
    }

    fn add_split(
        &self,
        tables: &mut [CompactTableWide<H>],
        input: &[QueuedEntry<H>],
        split: &mut Vec<QueuedEntry<H>>,
        ends: &mut Vec<usize>,
    ) -> Result<(), Error> {
        let (shift, mask, run_shift) = (self.layout.l_rem_bits, tables.len() - 1, self.entry_bits);
        table::add_split(
            tables,
            input,
            split,
            ends,
            |entry| (entry.l() >> shift) as usize & mask,
            |table, entries| {
                table.warm(entries.len());
                table.add_entries(entries, run_shift)
            },
        )
    }
}

enum WideShared {
    W80(PackedWideShared<u16>),
    W96(PackedWideShared<u32>),
    W128(PackedWideShared<u64>),
    W192(PackedWideShared<[u64; 2]>),
    W256(PackedWideShared<[u64; 3]>),
}

enum WideUpdates {
    W80(PackedWideUpdates<u16>),
    W96(PackedWideUpdates<u32>),
    W128(PackedWideUpdates<u64>),
    W192(PackedWideUpdates<[u64; 2]>),
    W256(PackedWideUpdates<[u64; 3]>),
}

impl WideShared {
    fn new(k: usize, capacity: usize) -> Result<Self, Error> {
        Ok(match k {
            33..=40 => Self::W80(PackedWideShared::new(capacity)?),
            41..=48 => Self::W96(PackedWideShared::new(capacity)?),
            49..=64 => Self::W128(PackedWideShared::new(capacity)?),
            65..=96 => Self::W192(PackedWideShared::new(capacity)?),
            97..=128 => Self::W256(PackedWideShared::new(capacity)?),
            _ => unreachable!("wide shared tables handle 33 <= k <= 128"),
        })
    }

    fn updates(&self) -> WideUpdates {
        match self {
            Self::W80(s) => WideUpdates::W80(s.updates()),
            Self::W96(s) => WideUpdates::W96(s.updates()),
            Self::W128(s) => WideUpdates::W128(s.updates()),
            Self::W192(s) => WideUpdates::W192(s.updates()),
            Self::W256(s) => WideUpdates::W256(s.updates()),
        }
    }

    fn count(
        &self,
        batch: &Batch,
        buffers: &mut WideUpdates,
        k: usize,
        strand: StrandMode,
        offset: usize,
    ) -> Result<(), Error> {
        macro_rules! count_wide {
            ($table:expr, $buffers:expr, $visit:ident, $cast:expr) => {{
                for record in batch.records() {
                    $visit(record, k, strand, |hi, lo| {
                        $table.push($buffers, $cast(hi), lo);
                        Ok(())
                    })?;
                }
                $table.submit(&mut $buffers.group, offset)
            }};
        }
        match (self, buffers) {
            (Self::W80(t), WideUpdates::W80(b)) => count_wide!(t, b, visit_packed128, |hi| hi as u16),
            (Self::W96(t), WideUpdates::W96(b)) => count_wide!(t, b, visit_packed128, |hi| hi as u32),
            (Self::W128(t), WideUpdates::W128(b)) => count_wide!(t, b, visit_packed128, |hi| hi),
            (Self::W192(t), WideUpdates::W192(b)) => count_wide!(t, b, visit_packed192, |hi| hi),
            (Self::W256(t), WideUpdates::W256(b)) => count_wide!(t, b, visit_packed256, |hi| hi),
            _ => unreachable!("wide worker buffers match shared wide tables"),
        }
    }

    fn into_counts(self, threads: usize) -> Result<WideCounts, Error> {
        Ok(match self {
            Self::W80(t) => WideCounts::W80(t.into_counts(threads)?),
            Self::W96(t) => WideCounts::W96(t.into_counts(threads)?),
            Self::W128(t) => WideCounts::W128(t.into_counts(threads)?),
            Self::W192(t) => WideCounts::W192(t.into_counts(threads)?),
            Self::W256(t) => WideCounts::W256(t.into_counts(threads)?),
        })
    }
}

type ByteMap = HashMap<Vec<u8>, u64, foldhash::fast::RandomState>;

struct ByteShared {
    shards: Vec<Shard<ByteMap>>,
    hash: foldhash::fast::RandomState,
}

impl ByteShared {
    fn new(partitions: usize, capacity: usize) -> Result<Self, Error> {
        let mut shards = Vec::new();
        shards
            .try_reserve_exact(partitions)
            .map_err(|_| Error::ReserveFailed)?;
        for index in 0..partitions {
            let mut map = ByteMap::default();
            map.try_reserve(capacity / partitions + usize::from(index < capacity % partitions))
                .map_err(|_| Error::ReserveFailed)?;
            shards.push(Shard(Mutex::new(map)));
        }
        Ok(Self {
            shards,
            hash: foldhash::fast::RandomState::default(),
        })
    }

    /// Apply each nonempty worker-local partition buffer to its shard.
    ///
    /// Different workers start at different partitions, and a partition held by
    /// another worker is skipped and revisited instead of waited on. Only when
    /// every remaining partition is busy does a worker block on one.
    fn apply(&self, buffers: &mut ByteUpdates, offset: usize) -> Result<(), Error> {
        let drain = |map: &mut ByteMap, updates: &mut Vec<(Vec<u8>, u64)>| {
            updates
                .drain(..)
                .try_for_each(|(key, delta)| add_delta(map, key, delta))
        };
        buffers.pending.clear();
        let mask = self.shards.len() - 1;
        buffers.pending.extend(
            (0..self.shards.len())
                .map(|step| (step + offset) & mask)
                .filter(|&index| !buffers.partitions[index].is_empty()),
        );
        while !buffers.pending.is_empty() {
            let before = buffers.pending.len();
            let mut result = Ok(());
            buffers.pending.retain(|&index| {
                if result.is_err() {
                    return true;
                }
                match self.shards[index].0.try_lock() {
                    Ok(mut table) => {
                        result = drain(&mut table, &mut buffers.partitions[index]);
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
            if buffers.pending.len() == before {
                let index = buffers.pending.swap_remove(0);
                let mut table = self.shards[index]
                    .0
                    .lock()
                    .map_err(|_| Error::WorkerFailed)?;
                drain(&mut table, &mut buffers.partitions[index])?;
            }
        }
        buffers.buffered_bytes = 0;
        Ok(())
    }

    fn into_counts(self) -> Result<ByteCounts, Error> {
        let maps = self
            .shards
            .into_iter()
            .map(|shard| shard.0.into_inner().map_err(|_| Error::WorkerFailed))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ByteCounts::from_maps(maps, self.hash))
    }
}

struct ByteUpdates {
    partitions: Vec<Vec<(Vec<u8>, u64)>>,
    // Reused list of nonempty partitions still waiting for their lock.
    pending: Vec<usize>,
    buffered_bytes: usize,
    #[cfg(test)]
    peak_buffered_bytes: usize,
}

impl ByteUpdates {
    fn new(partitions: usize) -> Self {
        Self {
            partitions: (0..partitions).map(|_| Vec::new()).collect(),
            pending: Vec::with_capacity(partitions),
            buffered_bytes: 0,
            #[cfg(test)]
            peak_buffered_bytes: 0,
        }
    }

    fn push(&mut self, hash: &foldhash::fast::RandomState, key: &[u8]) -> Result<(), Error> {
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
    Wide(WideShared),
    Bytes(ByteShared),
    /// Nothing is counted: input is split into bins on disk; see `crate::disk`.
    Bins(Bins),
}

enum Buffers {
    Packed(PackedUpdates),
    Wide(WideUpdates),
    Bytes(ByteUpdates),
    Bins(BinBuffers),
}

impl SharedCounts {
    fn new(k: usize, partitions: usize, capacity: usize) -> Result<Self, Error> {
        Ok(if k <= 32 {
            Self::Packed(PackedShared::new(k, capacity)?)
        } else if k <= 128 {
            Self::Wide(WideShared::new(k, capacity)?)
        } else {
            Self::Bytes(ByteShared::new(partitions, capacity)?)
        })
    }

    fn buffers(&self) -> Buffers {
        match self {
            Self::Packed(table) => Buffers::Packed(table.updates()),
            Self::Wide(table) => Buffers::Wide(table.updates()),
            Self::Bytes(table) => Buffers::Bytes(ByteUpdates::new(table.shards.len())),
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
                table.submit(&mut buffers.group, offset)
            }
            (Self::Wide(table), Buffers::Wide(buffers)) => {
                table.count(batch, buffers, k, strand, offset)
            }
            (Self::Bytes(table), Buffers::Bytes(buffers)) => {
                for record in batch.records() {
                    if record.len() < k {
                        continue;
                    }
                    visit_bytes(record, k, strand, |key| {
                        buffers.push(&table.hash, key)?;
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
        let counts = match self {
            Self::Packed(table) => Counts::Packed(table.into_counts(threads)?),
            Self::Wide(table) => Counts::Wide(table.into_counts(threads)?),
            Self::Bytes(table) => Counts::Bytes(table.into_counts()?),
            Self::Bins(_) => unreachable!("bins are finished by `DiskCounter`"),
        };
        Ok(Counter::from_counts(k, strand, counts, threads))
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

    /// Stream FASTA or FASTQ records from `reader`, auto-detecting gzip
    /// compression from content, and submit them to this counter's workers.
    pub fn add_reader(
        &self,
        reader: impl std::io::Read + Send + 'static,
    ) -> std::io::Result<crate::InputStats> {
        let mut stream = crate::input::Stream::parallel(self.input());
        stream.add_reader(reader)?;
        stream.finish()
    }

    /// Stream FASTA or FASTQ records from the file at `path`, auto-detecting
    /// gzip compression from content, and submit them to this counter's
    /// workers.
    pub fn add_file(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<crate::InputStats> {
        let mut stream = crate::input::Stream::parallel(self.input());
        stream.add_file(path)?;
        stream.finish()
    }

    /// Stream FASTA or FASTQ records from `paths` using up to `readers`
    /// concurrent file-reading threads (when all paths are regular files),
    /// auto-detecting gzip compression from content.
    pub fn add_files<P: AsRef<std::path::Path> + Sync>(
        &self,
        paths: &[P],
        readers: usize,
    ) -> std::io::Result<crate::InputStats> {
        crate::input::count_parallel_files(|| self.input(), paths, readers, None)
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
    fn add_record(&mut self, sequence: &[u8]) -> Result<(), Error> {
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
    fn start_record(&mut self) {
        self.record_start = None;
        self.continuation.clear();
    }

    /// Append a sequence fragment to the current record, preserving windows
    /// across fragments. Data is copied directly into reusable worker batches;
    /// only k−1 context bases are copied again at batch boundaries.
    /// A fresh counter implicitly starts its first streaming record.
    fn add_sequence(&mut self, mut sequence: &[u8]) -> Result<(), Error> {
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
        for k in [129, 257] {
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
        for k in [1, 31, 32, 33, 64, 65, 97, 128, 129] {
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
        for k in [1, 31, 32, 33, 40, 48, 64, 65, 96, 97, 128, 129] {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let record = b"acGTTGCAGATTCGAC".repeat(300);
                let mut parallel = ParallelCounter::new(k, strand, 3).unwrap();
                parallel.add_record(&record).unwrap();
                let mut result = parallel.finish().unwrap();
                // Packed keys partition internally; wider keys keep separate maps.
                assert!(match &result.counts {
                    Counts::Packed(counts) => counts.tables().len() > 1,
                    Counts::Wide(counts) => counts.table_count() > 1,
                    Counts::Bytes(counts) => counts.maps.len() > 1,
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
                partitioned.counts.set_counts(u64::MAX);
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
        for k in [1, 3, 31, 32, 33, 63, 64, 65, 97, 128, 129] {
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
        for k in [1, 32, 33, 64, 65, 97, 128, 129] {
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
