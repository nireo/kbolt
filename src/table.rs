//! Compact exact count tables for packed k-mers (k <= 32).
//!
//! Each 2k-bit key is first passed through an invertible mixing function, so
//! the mixed value `h` identifies the key exactly and `unmix(h)` recovers it.
//! The top bits of `h` select one of many partitions; a slot stores only the
//! remaining *quotient* bits plus a small count, all in one `u64`:
//!
//! ```text
//! slot = (remainder << count_bits) | small_count     (0 = empty)
//! ```
//!
//! For k = 31 with 1024 partitions this leaves 52 key bits and 12 count bits.
//! A small count equal to the field's maximum means the exact count lives in
//! a per-table overflow map, so counts remain exact, checked `u64` values.
//!
//! Slots are grouped into 64-byte buckets of eight, aligned to cache lines. A
//! key's home bucket is chosen by multiplying its remainder by the bucket
//! count (a monotone "fastrange"), and collisions spill to the next bucket.
//! Matching a whole bucket is one cache line and a few vector compares, so
//! callers can software-prefetch a key's bucket well before updating it.
//! Because the home bucket is monotone in the remainder, growing the table
//! visits old buckets in order and writes new buckets nearly sequentially.

use std::collections::HashMap;

use crate::Error;

const C1: u64 = 0xbf58_476d_1ce4_e5b9;
const C2: u64 = 0x94d0_49bb_1331_11eb;

/// Multiplicative inverse modulo 2^64 of an odd constant (Newton iteration).
const fn inverse(value: u64) -> u64 {
    let mut inverse = value;
    let mut step = 0;
    while step < 6 {
        inverse = inverse.wrapping_mul(2_u64.wrapping_sub(value.wrapping_mul(inverse)));
        step += 1;
    }
    inverse
}

const C1_INVERSE: u64 = inverse(C1);
const C2_INVERSE: u64 = inverse(C2);

/// Bits of a packed key's two-bit bases.
pub(crate) fn key_bits(k: usize) -> u32 {
    debug_assert!((1..=32).contains(&k));
    2 * k as u32
}

/// An invertible mixing function on `bits`-bit values.
///
/// Each step is a bijection on `bits` bits: XOR with a seed, an xorshift by at
/// least half the width (its own inverse), and multiplication by an odd
/// constant modulo 2^bits.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mixer {
    mask: u64,
    shift: u32,
    seed: u64,
}

impl Mixer {
    pub(crate) fn new(bits: u32, seed: u64) -> Self {
        let mask = u64::MAX >> (64 - bits);
        Self {
            mask,
            shift: bits.div_ceil(2),
            seed: seed & mask,
        }
    }

    #[inline(always)]
    pub(crate) fn mix(&self, key: u64) -> u64 {
        let mut x = key ^ self.seed;
        x ^= x >> self.shift;
        x = x.wrapping_mul(C1) & self.mask;
        x ^= x >> self.shift;
        x = x.wrapping_mul(C2) & self.mask;
        x ^ (x >> self.shift)
    }

    #[inline(always)]
    pub(crate) fn unmix(&self, mixed: u64) -> u64 {
        let mut x = mixed ^ (mixed >> self.shift);
        x = x.wrapping_mul(C2_INVERSE) & self.mask;
        x ^= x >> self.shift;
        x = x.wrapping_mul(C1_INVERSE) & self.mask;
        x ^= x >> self.shift;
        x ^ self.seed
    }
}

/// Bit widths shared by every table of one partitioned counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    /// Quotient bits stored in a slot.
    pub(crate) remainder_bits: u32,
    /// Small-count bits stored in a slot (`64 - remainder_bits`).
    pub(crate) count_bits: u32,
}

impl Layout {
    #[inline(always)]
    fn count_marker(self) -> u64 {
        u64::MAX >> self.remainder_bits
    }

    #[inline(always)]
    pub(crate) fn remainder_mask(self) -> u64 {
        u64::MAX >> self.count_bits
    }
}

/// Choose partition bits for `bits`-bit keys, as close to `desired` as the
/// constraints allow: at least 10 small-count bits per slot, and at least
/// half the key bits left as the stored quotient.
pub(crate) fn partition_bits(bits: u32, desired: u32) -> u32 {
    desired.max(bits.saturating_sub(54)).min(bits / 2)
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Bucket([u64; 8]);

/// Allocate `count` empty buckets. Zeroed allocation lets large tables take
/// fresh zero pages from the OS instead of writing every byte up front.
fn zeroed_buckets(count: usize) -> Result<Vec<Bucket>, Error> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let layout = std::alloc::Layout::array::<Bucket>(count).map_err(|_| Error::ReserveFailed)?;
    // SAFETY: the layout has nonzero size. An all-zero `Bucket` is valid (and
    // empty), and the allocation matches the layout `Vec<Bucket>` frees with.
    unsafe {
        let pointer = std::alloc::alloc_zeroed(layout).cast::<Bucket>();
        if pointer.is_null() {
            return Err(Error::ReserveFailed);
        }
        Ok(Vec::from_raw_parts(pointer, count, count))
    }
}
const SLOTS: usize = 8;
/// Grow before more than three quarters of the slots are occupied.
const LOAD_NUMERATOR: usize = 3;
const LOAD_DENOMINATOR: usize = 4;

/// Byte-per-slot masks (0xff or 0x00) of slots equal to `key` after clearing
/// their count bits, and of empty slots.
#[inline(always)]
fn bucket_masks(bucket: &Bucket, key: u64, key_mask: u64) -> (u64, u64) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is mandatory on aarch64, and the bucket holds 8 u64s.
    unsafe {
        use core::arch::aarch64::*;
        let slots = vld1q_u64_x4(bucket.0.as_ptr());
        let key = vdupq_n_u64(key);
        let key_mask = vdupq_n_u64(key_mask);
        let matches = |slots| vceqq_u64(vandq_u64(slots, key_mask), key);
        let narrow = |a, b, c, d| {
            let low = vcombine_u32(vmovn_u64(a), vmovn_u64(b));
            let high = vcombine_u32(vmovn_u64(c), vmovn_u64(d));
            let bytes = vmovn_u16(vcombine_u16(vmovn_u32(low), vmovn_u32(high)));
            vget_lane_u64(vreinterpret_u64_u8(bytes), 0)
        };
        (
            narrow(
                matches(slots.0),
                matches(slots.1),
                matches(slots.2),
                matches(slots.3),
            ),
            narrow(
                vceqzq_u64(slots.0),
                vceqzq_u64(slots.1),
                vceqzq_u64(slots.2),
                vceqzq_u64(slots.3),
            ),
        )
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let (mut matches, mut empty) = (0, 0);
        for (index, &slot) in bucket.0.iter().enumerate() {
            matches |= u64::from(slot & key_mask == key) * (0xff << (8 * index));
            empty |= u64::from(slot == 0) * (0xff << (8 * index));
        }
        (matches, empty)
    }
}

/// Hint that the cache line at `pointer` will soon be written.
#[inline(always)]
pub(crate) fn prefetch<T>(pointer: *const T) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: a prefetch is a hint; it never faults or changes memory.
    unsafe {
        core::arch::asm!(
            "prfm pstl1keep, [{0}]",
            in(reg) pointer,
            options(nostack, readonly, preserves_flags)
        );
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: as above; SSE is part of the x86_64 baseline.
    unsafe {
        core::arch::x86_64::_mm_prefetch(pointer.cast::<i8>(), core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = pointer;
}

/// How a count update that would exceed `u64::MAX` is applied.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Overflow {
    /// Leave the count unchanged.
    Reject,
    /// Apply the representable part of a run of single increments, leaving
    /// `u64::MAX`, exactly as adding them one at a time would.
    Saturate,
}

/// One partition's compact table.
pub(crate) struct CompactTable {
    buckets: Vec<Bucket>,
    len: usize,
    grow_at: usize,
    layout: Layout,
    // Exact counts for slots whose small count holds the marker.
    overflow: HashMap<u64, u64, foldhash::fast::RandomState>,
}

impl CompactTable {
    pub(crate) fn new(layout: Layout) -> Self {
        Self {
            buckets: Vec::new(),
            len: 0,
            grow_at: 0,
            layout,
            overflow: HashMap::default(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Distinct keys this table can hold before growing.
    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.grow_at
    }

    #[inline(always)]
    fn home(&self, remainder: u64) -> usize {
        let aligned = remainder << self.layout.count_bits;
        ((u128::from(aligned) * self.buckets.len() as u128) >> 64) as usize
    }

    /// Prefetch the home bucket of `remainder`, if the table is allocated.
    #[inline(always)]
    pub(crate) fn prefetch(&self, remainder: u64) {
        if !self.buckets.is_empty() {
            prefetch(self.buckets.as_ptr().wrapping_add(self.home(remainder)));
        }
    }

    /// Ensure room for `additional` more distinct keys without growing.
    pub(crate) fn try_reserve(&mut self, additional: usize) -> Result<(), Error> {
        let needed = self
            .len
            .checked_add(additional)
            .ok_or(Error::ReserveFailed)?;
        if needed <= self.grow_at {
            return Ok(());
        }
        let slots = needed
            .checked_mul(LOAD_DENOMINATOR)
            .ok_or(Error::ReserveFailed)?
            .div_ceil(LOAD_NUMERATOR);
        self.resize(slots.div_ceil(SLOTS).max(2))
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self) -> Result<(), Error> {
        let buckets = self
            .buckets
            .len()
            .checked_mul(2)
            .ok_or(Error::ReserveFailed)?
            .max(2);
        self.resize(buckets)
    }

    fn resize(&mut self, buckets: usize) -> Result<(), Error> {
        // Allocate everything before replacing the old buckets, so a failed
        // allocation leaves the table unchanged.
        let mut filled = Vec::new();
        filled
            .try_reserve_exact(buckets)
            .map_err(|_| Error::ReserveFailed)?;
        filled.resize(buckets, 0_u8);
        let old = std::mem::replace(&mut self.buckets, zeroed_buckets(buckets)?);
        self.grow_at = buckets * SLOTS / LOAD_DENOMINATOR * LOAD_NUMERATOR;
        // Old buckets are visited in order, so their new homes are nearly
        // sequential; buckets fill left to right, so stop at the first empty.
        // Keys are distinct, so each needs only the next free slot at or after
        // its home, tracked per bucket (in `filled`) instead of searched for.
        for bucket in &old {
            for &slot in bucket.0.iter().take_while(|&&slot| slot != 0) {
                let mut index = self.home(slot >> self.layout.count_bits);
                while filled[index] == SLOTS as u8 {
                    index += 1;
                    if index == buckets {
                        index = 0;
                    }
                }
                self.buckets[index].0[usize::from(filled[index])] = slot;
                filled[index] += 1;
            }
        }
        Ok(())
    }

    /// Locate `remainder`: `(true, slot)` if present, otherwise
    /// `(false, first empty slot)` on its probe path.
    #[inline(always)]
    fn find(&self, remainder: u64) -> (bool, usize) {
        let key = remainder << self.layout.count_bits;
        let key_mask = !self.layout.count_marker();
        let mut bucket = self.home(remainder);
        loop {
            // SAFETY: `home` is below the bucket count, and the wrap below
            // keeps `bucket` in bounds; the load factor guarantees an empty slot.
            let (matches, empty) =
                bucket_masks(unsafe { self.buckets.get_unchecked(bucket) }, key, key_mask);
            // Slots fill left to right and are never removed, so a stored key
            // precedes every empty slot; clear empties that compare equal to 0.
            let matches = matches & !empty;
            if matches != 0 {
                return (
                    true,
                    bucket * SLOTS + (matches.trailing_zeros() / 8) as usize,
                );
            }
            if empty != 0 {
                return (
                    false,
                    bucket * SLOTS + (empty.trailing_zeros() / 8) as usize,
                );
            }
            bucket += 1;
            if bucket == self.buckets.len() {
                bucket = 0;
            }
        }
    }

    /// # Safety
    /// `index` must be below `buckets.len() * SLOTS`.
    #[inline(always)]
    unsafe fn slot_mut(&mut self, index: usize) -> &mut u64 {
        // SAFETY: guaranteed by the caller.
        unsafe {
            self.buckets
                .get_unchecked_mut(index / SLOTS)
                .0
                .get_unchecked_mut(index % SLOTS)
        }
    }

    /// Add `delta` (at least 1) to the count of `remainder`.
    #[inline(always)]
    pub(crate) fn add(
        &mut self,
        remainder: u64,
        delta: u64,
        overflow: Overflow,
    ) -> Result<(), Error> {
        debug_assert!(delta > 0 && remainder <= self.layout.remainder_mask());
        if self.len >= self.grow_at {
            self.grow()?;
        }
        let marker = self.layout.count_marker();
        let (found, index) = self.find(remainder);
        // SAFETY: `find` returns an in-bounds slot.
        let slot = unsafe { *self.slot_mut(index) };
        let small = slot & marker;
        if found && small != marker && delta < marker - small {
            // SAFETY: as above.
            unsafe { *self.slot_mut(index) = slot + delta };
            return Ok(());
        }
        if !found && delta < marker {
            // SAFETY: as above.
            unsafe { *self.slot_mut(index) = (remainder << self.layout.count_bits) | delta };
            self.len += 1;
            return Ok(());
        }
        self.add_large(remainder, index, found, delta, overflow)
    }

    // Counts at or above the small-count marker live in the overflow map.
    #[cold]
    #[inline(never)]
    fn add_large(
        &mut self,
        remainder: u64,
        index: usize,
        found: bool,
        delta: u64,
        mode: Overflow,
    ) -> Result<(), Error> {
        let marker = self.layout.count_marker();
        // SAFETY: `index` came from `find`.
        let slot = unsafe { *self.slot_mut(index) };
        if found && slot & marker == marker {
            let count = self.overflow.get_mut(&remainder).expect("marked count");
            return match count.checked_add(delta) {
                Some(sum) => {
                    *count = sum;
                    Ok(())
                }
                None => {
                    if mode == Overflow::Saturate {
                        *count = u64::MAX;
                    }
                    Err(Error::CountOverflow)
                }
            };
        }
        let current = if found { slot & marker } else { 0 };
        let (count, result) = match current.checked_add(delta) {
            Some(sum) => (sum, Ok(())),
            None if mode == Overflow::Saturate => (u64::MAX, Err(Error::CountOverflow)),
            None => return Err(Error::CountOverflow),
        };
        self.overflow
            .try_reserve(1)
            .map_err(|_| Error::ReserveFailed)?;
        self.overflow.insert(remainder, count);
        // SAFETY: as above.
        unsafe { *self.slot_mut(index) = (remainder << self.layout.count_bits) | marker };
        if !found {
            self.len += 1;
        }
        result
    }

    /// Insert a key known to the caller only by remainder, failing with
    /// `Ok(false)` instead of adding if it is already present.
    pub(crate) fn insert_new(&mut self, remainder: u64, count: u64) -> Result<bool, Error> {
        if self.len >= self.grow_at {
            self.grow()?;
        }
        if self.find(remainder).0 {
            return Ok(false);
        }
        self.add(remainder, count, Overflow::Reject)?;
        Ok(true)
    }

    /// Iterate over `(remainder, count)` pairs in bucket order.
    pub(crate) fn iter(&self) -> Entries<'_> {
        Entries {
            buckets: self.buckets.iter(),
            slots: &[],
            table: self,
        }
    }

    /// Replace every count with `count`; used to construct overflow tests.
    #[cfg(test)]
    pub(crate) fn set_counts(&mut self, count: u64) {
        assert!(count > 0, "stored counts are positive");
        let entries: Vec<_> = self.iter().map(|(remainder, _)| remainder).collect();
        self.buckets
            .iter_mut()
            .for_each(|bucket| *bucket = Bucket([0; SLOTS]));
        self.overflow.clear();
        self.len = 0;
        for remainder in entries {
            self.add(remainder, count, Overflow::Reject).unwrap();
        }
    }
}

/// Iterator over a table's `(remainder, count)` pairs in bucket order.
pub(crate) struct Entries<'a> {
    buckets: std::slice::Iter<'a, Bucket>,
    // Remaining slots of the current bucket.
    slots: &'a [u64],
    table: &'a CompactTable,
}

impl Iterator for Entries<'_> {
    type Item = (u64, u64);

    #[inline]
    fn next(&mut self) -> Option<(u64, u64)> {
        loop {
            // Buckets fill left to right, so the first empty slot ends a bucket.
            if let Some((&slot, rest)) = self.slots.split_first()
                && slot != 0
            {
                self.slots = rest;
                let layout = self.table.layout;
                let remainder = slot >> layout.count_bits;
                let small = slot & layout.count_marker();
                let count = if small == layout.count_marker() {
                    self.table.overflow[&remainder]
                } else {
                    small
                };
                return Some((remainder, count));
            }
            self.slots = &self.buckets.next()?.0;
        }
    }
}

/// A complete partitioned counter for packed keys.
pub(crate) struct PackedCounts {
    mixer: Mixer,
    layout: Layout,
    tables: Vec<CompactTable>,
    // Reusable buffer for `OrderedAdder`.
    scratch: Vec<(u64, u64)>,
}

impl PackedCounts {
    /// Create `2^partition_bits(..)` empty tables with room for `capacity`
    /// distinct keys in total.
    pub(crate) fn with_capacity(
        k: usize,
        desired_partition_bits: u32,
        capacity: usize,
    ) -> Result<Self, Error> {
        let bits = key_bits(k);
        let partition_bits = partition_bits(bits, desired_partition_bits);
        let remainder_bits = bits - partition_bits;
        let layout = Layout {
            remainder_bits,
            count_bits: 64 - remainder_bits,
        };
        let partitions = 1_usize << partition_bits;
        let mut tables = Vec::new();
        tables
            .try_reserve_exact(partitions)
            .map_err(|_| Error::ReserveFailed)?;
        for index in 0..partitions {
            let mut table = CompactTable::new(layout);
            let share = capacity / partitions + usize::from(index < capacity % partitions);
            table.try_reserve(share)?;
            tables.push(table);
        }
        Ok(Self {
            mixer: Mixer::new(bits, random_seed()),
            layout,
            tables,
            scratch: Vec::new(),
        })
    }

    pub(crate) fn from_tables(mixer: Mixer, layout: Layout, tables: Vec<CompactTable>) -> Self {
        Self {
            mixer,
            layout,
            tables,
            scratch: Vec::new(),
        }
    }

    pub(crate) fn mixer(&self) -> Mixer {
        self.mixer
    }

    pub(crate) fn layout(&self) -> Layout {
        self.layout
    }

    pub(crate) fn into_tables(self) -> Vec<CompactTable> {
        self.tables
    }

    pub(crate) fn tables(&self) -> &[CompactTable] {
        &self.tables
    }

    #[cfg(test)]
    pub(crate) fn tables_mut(&mut self) -> &mut [CompactTable] {
        &mut self.tables
    }

    pub(crate) fn len(&self) -> usize {
        self.tables.iter().map(CompactTable::len).sum()
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.tables.iter().map(CompactTable::capacity).sum()
    }

    /// Partition index and stored remainder of `key`.
    #[inline(always)]
    pub(crate) fn route(&self, key: u64) -> (usize, u64) {
        route(self.mixer, self.layout, key)
    }

    /// Recover the key stored as `remainder` in partition `partition`.
    #[inline(always)]
    pub(crate) fn key(&self, partition: usize, remainder: u64) -> u64 {
        self.mixer
            .unmix(((partition as u64) << self.layout.remainder_bits) | remainder)
    }

    /// Add `delta` to `key`, leaving its count unchanged on overflow.
    pub(crate) fn add_key(&mut self, key: u64, delta: u64) -> Result<(), Error> {
        let (partition, remainder) = self.route(key);
        self.tables[partition].add(remainder, delta, Overflow::Reject)
    }

    /// Insert `key` unless present; returns whether it was inserted.
    pub(crate) fn insert_new(&mut self, key: u64, count: u64) -> Result<bool, Error> {
        let (partition, remainder) = self.route(key);
        self.tables[partition].insert_new(remainder, count)
    }

    /// Iterate over one partition's `(key, count)` pairs.
    pub(crate) fn partition_entries(
        &self,
        partition: usize,
    ) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.tables[partition]
            .iter()
            .map(move |(remainder, count)| (self.key(partition, remainder), count))
    }

    /// Iterate over every `(key, count)` pair.
    pub(crate) fn entries(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        (0..self.tables.len()).flat_map(|partition| self.partition_entries(partition))
    }

    /// Start counting a sequence of keys in order; see [`OrderedAdder`].
    pub(crate) fn adder(&mut self) -> OrderedAdder<'_> {
        let mut pending = std::mem::take(&mut self.scratch);
        pending.clear();
        OrderedAdder {
            mixer: self.mixer,
            counts: self,
            pending,
        }
    }
}

/// Prefetch distance for buffered updates: enough outstanding prefetches to
/// hide DRAM latency, few enough that prefetched lines are still cached.
const AHEAD: usize = 16;
/// Keys buffered by an `OrderedAdder` before they are applied.
const ORDERED_CHUNK: usize = 64 * 1024;

/// Counts keys in submission order. Keys are first mixed into a buffer, with
/// consecutive equal keys combined, then applied in order while prefetching
/// each key's bucket `AHEAD` updates in advance. Keeping extraction and table
/// updates in separate tight loops lets each one optimize well.
///
/// On overflow, keys before the overflowing one remain counted, that key
/// saturates exactly as single increments would, and later keys (including
/// any still buffered when the adder is dropped) are not counted. Call
/// `finish` to apply buffered keys.
pub(crate) struct OrderedAdder<'a> {
    mixer: Mixer,
    counts: &'a mut PackedCounts,
    // (mixed key, consecutive increments), in submission order.
    pending: Vec<(u64, u64)>,
}

impl OrderedAdder<'_> {
    #[inline(always)]
    pub(crate) fn push(&mut self, key: u64) -> Result<(), Error> {
        let mixed = self.mixer.mix(key);
        match self.pending.last_mut() {
            Some(last) if last.0 == mixed => last.1 += 1,
            _ => {
                if self.pending.len() == ORDERED_CHUNK {
                    self.apply()?;
                }
                self.pending.push((mixed, 1));
            }
        }
        Ok(())
    }

    #[inline(never)]
    fn apply(&mut self) -> Result<(), Error> {
        let layout = self.counts.layout;
        let tables = &mut self.counts.tables;
        let split = |mixed: u64| {
            (
                (mixed >> layout.remainder_bits) as usize,
                mixed & layout.remainder_mask(),
            )
        };
        let result = (0..self.pending.len()).try_for_each(|index| {
            if let Some(&(ahead, _)) = self.pending.get(index + AHEAD) {
                let (partition, remainder) = split(ahead);
                tables[partition].prefetch(remainder);
            }
            let (mixed, delta) = self.pending[index];
            let (partition, remainder) = split(mixed);
            tables[partition].add(remainder, delta, Overflow::Saturate)
        });
        self.pending.clear();
        result
    }

    pub(crate) fn finish(mut self) -> Result<(), Error> {
        self.apply()
    }
}

impl Drop for OrderedAdder<'_> {
    fn drop(&mut self) {
        // Return the buffer for reuse; unapplied keys are discarded.
        self.counts.scratch = std::mem::take(&mut self.pending);
    }
}

#[inline(always)]
pub(crate) fn route(mixer: Mixer, layout: Layout, key: u64) -> (usize, u64) {
    let mixed = mixer.mix(key);
    (
        (mixed >> layout.remainder_bits) as usize,
        mixed & layout.remainder_mask(),
    )
}

/// A per-counter seed so table layouts are not predictable across runs.
fn random_seed() -> u64 {
    use std::hash::BuildHasher;
    std::collections::hash_map::RandomState::new().hash_one(0x006b_626f_6c74_u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixing_is_invertible_for_every_width() {
        let mut state = 7_u64;
        for bits in (2..=64).step_by(2) {
            let mixer = Mixer::new(bits, state);
            let mask = u64::MAX >> (64 - bits);
            for _ in 0..2000 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let key = state & mask;
                let mixed = mixer.mix(key);
                assert!(mixed <= mask);
                assert_eq!(mixer.unmix(mixed), key, "bits={bits}");
            }
        }
        // Small widths are permutations.
        for bits in [2, 4, 6, 8] {
            let mixer = Mixer::new(bits, 3);
            let mut seen = vec![false; 1 << bits];
            for key in 0..1_u64 << bits {
                seen[mixer.mix(key) as usize] = true;
            }
            assert!(seen.into_iter().all(|seen| seen));
        }
    }

    #[test]
    fn tables_count_exactly_through_growth_and_large_counts() {
        for (k, partition_bits) in [(1, 10), (5, 3), (16, 10), (31, 10), (32, 10), (32, 0)] {
            let mut counts = PackedCounts::with_capacity(k, partition_bits, 0).unwrap();
            let marker = counts.layout().count_marker();
            let mut expected = HashMap::new();
            let mask = u64::MAX >> (64 - key_bits(k));
            let mut state = 99_u64;
            for step in 0..60_000_u64 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let key = state & mask & if step % 3 == 0 { 0xff } else { u64::MAX };
                // Occasionally exceed the small-count field (bounded so that
                // 60 such updates cannot overflow u64 for k = 1).
                let delta = if step % 1000 == 0 {
                    (marker + 1).min(1 << 40) + step
                } else {
                    1
                };
                counts.add_key(key, delta).unwrap();
                *expected.entry(key).or_insert(0) += delta;
            }
            assert_eq!(counts.len(), expected.len());
            let actual: HashMap<_, _> = counts.entries().collect();
            assert_eq!(actual, expected, "k={k}");
        }
    }

    #[test]
    fn overflow_rejects_or_saturates() {
        let mut counts = PackedCounts::with_capacity(31, 10, 0).unwrap();
        counts.add_key(5, u64::MAX - 1).unwrap();
        assert_eq!(counts.add_key(5, 2), Err(Error::CountOverflow));
        assert_eq!(counts.entries().collect::<Vec<_>>(), [(5, u64::MAX - 1)]);
        let mut adder = counts.adder();
        let result = [9, 5, 5, 5, 9]
            .into_iter()
            .try_for_each(|key| adder.push(key))
            .and_then(|()| adder.finish());
        assert_eq!(result, Err(Error::CountOverflow));
        let mut entries = counts.entries().collect::<Vec<_>>();
        entries.sort_unstable();
        assert_eq!(entries, [(5, u64::MAX), (9, 1)]);
        assert!(!counts.insert_new(5, 1).unwrap());
        assert!(counts.insert_new(6, 1).unwrap());
    }
}
