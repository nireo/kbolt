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

/// Bucket storage that returns its memory to the OS as soon as it is dropped.
///
/// Tables are replaced each time they grow. The general allocator keeps freed
/// blocks of that size for reuse, so every outgrown table would stay resident
/// (and a reused block must be cleared by hand). Large tables are therefore
/// mapped directly: fresh mappings are zero pages, and unmapping frees them.
struct Buckets {
    pointer: std::ptr::NonNull<Bucket>,
    len: usize,
}

/// Bucket counts from this size up are mapped directly (16 KiB). With
/// thousands of tables, even outgrown tables of a few pages add up.
const MAPPED_BUCKETS: usize = 256;

// SAFETY: `Buckets` owns its allocation exclusively, like a `Vec<Bucket>`.
unsafe impl Send for Buckets {}
// SAFETY: as above; shared access only reads.
unsafe impl Sync for Buckets {}

impl Buckets {
    const fn new() -> Self {
        Self {
            pointer: std::ptr::NonNull::dangling(),
            len: 0,
        }
    }

    fn mapped(len: usize) -> bool {
        cfg!(unix) && len >= MAPPED_BUCKETS
    }

    /// Allocate `len` empty (all-zero) buckets.
    fn zeroed(len: usize) -> Result<Self, Error> {
        if len == 0 {
            return Ok(Self::new());
        }
        let layout = std::alloc::Layout::array::<Bucket>(len).map_err(|_| Error::ReserveFailed)?;
        #[cfg(unix)]
        if Self::mapped(len) {
            // SAFETY: an anonymous private mapping of nonzero size; the result
            // is checked. Mappings are page aligned, which satisfies `Bucket`.
            let pointer = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    layout.size(),
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            if pointer == libc::MAP_FAILED {
                return Err(Error::ReserveFailed);
            }
            let pointer = std::ptr::NonNull::new(pointer.cast()).ok_or(Error::ReserveFailed)?;
            return Ok(Self { pointer, len });
        }
        // SAFETY: the layout has nonzero size, and an all-zero `Bucket` is
        // valid (and empty).
        let pointer = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<Bucket>();
        let pointer = std::ptr::NonNull::new(pointer).ok_or(Error::ReserveFailed)?;
        Ok(Self { pointer, len })
    }
}

impl Drop for Buckets {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        let layout = std::alloc::Layout::array::<Bucket>(self.len).expect("allocated layout");
        #[cfg(unix)]
        if Self::mapped(self.len) {
            // SAFETY: this is exactly the mapping made in `zeroed`.
            unsafe { libc::munmap(self.pointer.as_ptr().cast(), layout.size()) };
            return;
        }
        // SAFETY: allocated in `zeroed` with this layout.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr().cast(), layout) };
    }
}

impl std::ops::Deref for Buckets {
    type Target = [Bucket];

    #[inline(always)]
    fn deref(&self) -> &[Bucket] {
        // SAFETY: `pointer` is valid for `len` initialized buckets (or
        // dangling and aligned when `len` is zero).
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }
}

impl std::ops::DerefMut for Buckets {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [Bucket] {
        // SAFETY: as in `deref`, with exclusive access through `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.len) }
    }
}

const SLOTS: usize = 8;
/// How full a table may get before it grows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Load {
    /// Three quarters, for tables updated one key at a time in arbitrary
    /// order: each extra bucket probed is likely another cache miss.
    Sparse,
    /// Seven eighths, for tables that receive their updates in bursts while
    /// they sit in cache, where longer probes cost little and memory matters.
    Dense,
}

impl Load {
    /// The limit in sixteenths of the slots.
    const fn sixteenths(self) -> usize {
        match self {
            Self::Sparse => 12,
            Self::Dense => 14,
        }
    }
}

/// Tables double until they reach this many buckets, then grow by half.
/// Home buckets are a fastrange, so any bucket count works, and a smaller step
/// leaves less of a large table empty: its fill stays within two thirds of
/// the limit instead of dropping to half of it.
const STEADY_BUCKETS: usize = 1024;

/// Bits per slot in the masks of `bucket_masks`: slot `i` owns bits
/// `i * MASK_BITS..(i + 1) * MASK_BITS`, all of them set or all clear. NEON
/// narrows each comparison to a byte; x86 extracts one bit per slot.
#[cfg(target_arch = "x86_64")]
const MASK_BITS: u32 = 1;
#[cfg(not(target_arch = "x86_64"))]
const MASK_BITS: u32 = 8;

/// Per-slot masks (see `MASK_BITS`) of slots equal to `key` after clearing
/// their count bits, and of empty slots.
///
/// On x86-64 the widest instruction set enabled for the build is used:
/// AVX-512 compares a whole bucket at once, AVX2 half of it, and the SSE2
/// baseline a quarter. Build with `-C target-cpu=native` (or `x86-64-v3`,
/// `x86-64-v4`) to select the wider ones; the choice is made at compile time
/// so that the comparison stays inlined in every update loop.
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    // SAFETY: AVX-512F is enabled for this build, and the bucket is 64 bytes
    // aligned to 64.
    unsafe {
        use core::arch::x86_64::*;
        let slots = _mm512_load_si512(bucket.0.as_ptr().cast());
        let matches = _mm512_cmpeq_epi64_mask(
            _mm512_and_si512(slots, _mm512_set1_epi64(key_mask as i64)),
            _mm512_set1_epi64(key as i64),
        );
        let empty = _mm512_testn_epi64_mask(slots, slots);
        (u64::from(matches), u64::from(empty))
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    // SAFETY: AVX2 is enabled for this build, and the bucket holds two
    // aligned 256-bit halves.
    unsafe {
        use core::arch::x86_64::*;
        let pointer = bucket.0.as_ptr().cast::<__m256i>();
        let (low, high) = (
            _mm256_load_si256(pointer),
            _mm256_load_si256(pointer.add(1)),
        );
        let key = _mm256_set1_epi64x(key as i64);
        let key_mask = _mm256_set1_epi64x(key_mask as i64);
        let zero = _mm256_setzero_si256();
        // One bit per 64-bit lane.
        let bits = |equal| _mm256_movemask_pd(_mm256_castsi256_pd(equal)) as u32 as u64;
        let matches = |slots| bits(_mm256_cmpeq_epi64(_mm256_and_si256(slots, key_mask), key));
        let empty = |slots| bits(_mm256_cmpeq_epi64(slots, zero));
        (
            matches(low) | (matches(high) << 4),
            empty(low) | (empty(high) << 4),
        )
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    // SAFETY: SSE2 is part of the x86-64 baseline, and the bucket holds four
    // aligned 128-bit quarters.
    unsafe {
        use core::arch::x86_64::*;
        let pointer = bucket.0.as_ptr().cast::<__m128i>();
        let key = _mm_set1_epi64x(key as i64);
        let key_mask = _mm_set1_epi64x(key_mask as i64);
        let zero = _mm_setzero_si128();
        // SSE2 compares 32-bit lanes: a slot is equal when both of its halves
        // are. One bit per 64-bit lane.
        let equal = |a, b| {
            let halves = _mm_cmpeq_epi32(a, b);
            let both = _mm_and_si128(halves, _mm_shuffle_epi32::<0b1011_0001>(halves));
            _mm_movemask_pd(_mm_castsi128_pd(both)) as u32 as u64
        };
        let (mut matches, mut empty) = (0, 0);
        for quarter in 0..4 {
            let slots = _mm_load_si128(pointer.add(quarter));
            matches |= equal(_mm_and_si128(slots, key_mask), key) << (2 * quarter);
            empty |= equal(slots, zero) << (2 * quarter);
        }
        (matches, empty)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
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
    buckets: Buckets,
    len: usize,
    grow_at: usize,
    layout: Layout,
    // Fraction (in 65536ths) of a growth step by which this table's sizes are
    // offset. Partitions hold nearly equal numbers of keys, so with identical
    // sizes they would all grow at once and all be mostly empty together;
    // spread out, their total stays near the average at every moment.
    stagger: u16,
    load: Load,
    // Exact counts for slots whose small count holds the marker.
    overflow: HashMap<u64, u64, foldhash::fast::RandomState>,
}

impl CompactTable {
    /// An empty table for partition `index` of `partitions`.
    pub(crate) fn staggered(layout: Layout, load: Load, index: usize, partitions: usize) -> Self {
        debug_assert!(index < partitions);
        Self {
            buckets: Buckets::new(),
            len: 0,
            grow_at: 0,
            layout,
            stagger: ((index as u128 * 65536) / partitions as u128) as u16,
            load,
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

    /// Number of 64-byte buckets.
    pub(crate) fn buckets(&self) -> usize {
        self.buckets.len()
    }

    /// Read the table into cache in address order ahead of `updates` updates,
    /// when it is large enough to have left the cache since its last visit
    /// and the updates will touch most of it anyway. A sequential read is
    /// several times cheaper per line than the scattered misses it replaces.
    pub(crate) fn warm(&self, updates: usize) {
        const WARM_BUCKETS: usize = 512;
        if self.buckets.len() >= WARM_BUCKETS && updates >= self.buckets.len() / 4 {
            self.read();
        }
    }

    /// Read every bucket into cache, in address order.
    fn read(&self) {
        let touched = self
            .buckets
            .iter()
            .fold(0_u64, |sum, bucket| sum.wrapping_add(bucket.0[0]));
        std::hint::black_box(touched);
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
            .checked_mul(16)
            .ok_or(Error::ReserveFailed)?
            .div_ceil(self.load.sixteenths());
        self.resize(slots.div_ceil(SLOTS).max(2))
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self) -> Result<(), Error> {
        let current = self.buckets.len();
        let buckets = if current < STEADY_BUCKETS / 2 {
            (current * 2).max(2)
        } else if current < STEADY_BUCKETS {
            // Enter the steady phase part of the way through a growth step.
            STEADY_BUCKETS + (((STEADY_BUCKETS / 2) * usize::from(self.stagger)) >> 16)
        } else {
            current
                .checked_add(current / 2)
                .ok_or(Error::ReserveFailed)?
        };
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
        let old = std::mem::replace(&mut self.buckets, Buckets::zeroed(buckets)?);
        self.grow_at = buckets * SLOTS / 16 * self.load.sixteenths();
        // Old buckets are visited in order, so their new homes are nearly
        // sequential; buckets fill left to right, so stop at the first empty.
        // Keys are distinct, so each needs only the next free slot at or after
        // its home, tracked per bucket (in `filled`) instead of searched for.
        for bucket in old.iter() {
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
                    bucket * SLOTS + (matches.trailing_zeros() / MASK_BITS) as usize,
                );
            }
            if empty != 0 {
                return (
                    false,
                    bucket * SLOTS + (empty.trailing_zeros() / MASK_BITS) as usize,
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

    /// Add `entry >> delta_shift` (at least 1) to the count of the remainder
    /// in the low bits of each entry, in order, leaving a count unchanged
    /// (and stopping) on overflow.
    ///
    /// This is `add` for a run of updates to a table that is in cache. The
    /// table's fields stay in registers across the run, each key's bucket is
    /// prefetched a few updates ahead, and finding a key and adding one take
    /// the same instructions, so that a mix of the two does not mislead the
    /// branch predictor.
    #[inline(always)]
    pub(crate) fn add_entries(&mut self, entries: &[u64], delta_shift: u32) -> Result<(), Error> {
        let layout = self.layout;
        let marker = layout.count_marker();
        let (key_mask, remainder_mask) = (!marker, layout.remainder_mask());
        let mut index = 0;
        'run: while index < entries.len() {
            if self.len >= self.grow_at {
                self.grow()?;
            }
            // New keys the table takes before it must grow, and how many of
            // them this run has added.
            let room = self.grow_at - self.len;
            let mut added = 0;
            let (buckets, count) = (self.buckets.as_mut_ptr(), self.buckets.len());
            let home = |key: u64| ((u128::from(key) * count as u128) >> 64) as usize;
            while index < entries.len() && added < room {
                if let Some(&ahead) = entries.get(index + AHEAD) {
                    prefetch(
                        buckets.wrapping_add(home((ahead & remainder_mask) << layout.count_bits)),
                    );
                }
                let entry = entries[index];
                index += 1;
                let key = (entry & remainder_mask) << layout.count_bits;
                let delta = entry >> delta_shift;
                let mut bucket = home(key);
                loop {
                    // SAFETY: `home` is below the bucket count, and the wrap
                    // below keeps `bucket` in bounds; the load factor
                    // guarantees an empty slot.
                    let slots = unsafe { &mut *buckets.add(bucket) };
                    let (matches, empty) = bucket_masks(slots, key, key_mask);
                    let matches = matches & !empty;
                    // One test covers both ways a search ends, at the key or
                    // at the first empty slot. Which of the two it was is
                    // data here, not a branch: in a burst, whether a key is
                    // new is as hard to predict as the input.
                    let found = matches != 0;
                    let ends = if found { matches } else { empty };
                    if ends != 0 {
                        let at = (ends.trailing_zeros() / MASK_BITS) as usize;
                        // SAFETY: a mask has bits for eight slots only.
                        let slot = unsafe { slots.0.get_unchecked_mut(at) };
                        // An empty slot is zero, so one update serves a new
                        // key and a present one: the key bits are set
                        // (again) and the small count grows. A count at the
                        // marker leaves no room for any delta.
                        if delta < marker - (*slot & marker) {
                            *slot = (*slot | key) + delta;
                            added += usize::from(!found);
                            break;
                        }
                        // A large count: the general path takes over, and
                        // the run starts again after it.
                        self.len += added;
                        self.add_large(
                            entry & remainder_mask,
                            bucket * SLOTS + at,
                            found,
                            delta,
                            Overflow::Reject,
                        )?;
                        continue 'run;
                    }
                    bucket += 1;
                    if bucket == count {
                        bucket = 0;
                    }
                }
            }
            self.len += added;
        }
        Ok(())
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

    /// Add `entry >> delta_shift` for each entry, whose low bits are a
    /// remainder of this table, for a table that has left the cache since
    /// its last burst whatever its size (see `crate::disk`). A burst that
    /// will touch most of the table reads it into cache first; a sparser one
    /// relies on each bucket being prefetched ahead of its update.
    pub(crate) fn add_burst(&mut self, entries: &[u64], delta_shift: u32) -> Result<(), Error> {
        if entries.len() >= self.buckets.len() / 4 {
            self.read();
        }
        self.add_entries(entries, delta_shift)
    }

    /// Remove every key, keeping the allocated buckets for reuse.
    pub(crate) fn clear(&mut self) {
        self.buckets.fill(Bucket([0; SLOTS]));
        self.overflow.clear();
        self.len = 0;
    }

    /// The remainder and count held by an occupied slot.
    #[inline(always)]
    fn entry(&self, slot: u64) -> (u64, u64) {
        let remainder = slot >> self.layout.count_bits;
        let small = slot & self.layout.count_marker();
        let count = if small == self.layout.count_marker() {
            self.overflow[&remainder]
        } else {
            small
        };
        (remainder, count)
    }

    /// Fill `slots` with the occupied slots in ascending order of their
    /// remainders.
    ///
    /// The table is nearly in that order already: a key's home bucket rises
    /// with its remainder, and a key sits in its home bucket or a little
    /// after it. Each bucket is sorted by itself, without branches; what is
    /// left out of place are the keys that spilled into a later bucket, a
    /// few places from their own. Only keys that were turned away at the
    /// end of the table are far from their place, at its start, and those
    /// are set aside first.
    pub(crate) fn sorted_slots(&self, slots: &mut Vec<u64>) {
        slots.clear();
        // Whole buckets are stored, of which only the occupied part counts.
        slots.reserve(self.len + SLOTS);
        let mut wrapped = Vec::new();
        // A key that wrapped around passed only full buckets, so such keys
        // are found before, or in, the first bucket with an empty slot.
        let mut rest = self.buckets.iter().enumerate();
        for (index, bucket) in rest.by_ref() {
            for &slot in bucket.0.iter().take_while(|&&slot| slot != 0) {
                if self.home(slot >> self.layout.count_bits) > index {
                    wrapped.push(slot);
                } else {
                    slots.push(slot);
                }
            }
            if bucket.0[SLOTS - 1] == 0 {
                break;
            }
        }
        let mut length = slots.len();
        for (_, bucket) in rest {
            // SAFETY: fewer than `self.len` slots have been stored, each an
            // occupied one, and there is room for a bucket more than that.
            // Empty slots are stored last and overwritten by the next
            // bucket, or left beyond the final length.
            unsafe {
                let sorted = sort_bucket(&bucket.0);
                slots
                    .as_mut_ptr()
                    .add(length)
                    .cast::<[u64; SLOTS]>()
                    .write_unaligned(sorted);
                length += bucket.0.iter().filter(|&&slot| slot != 0).count();
                slots.set_len(length);
            }
        }
        slots.extend_from_slice(&wrapped);
        sort_nearly_sorted(slots);
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

/// A bucket's slots in ascending order, the empty (zero) ones last.
///
/// This is the smallest sorting network for eight values: nineteen
/// exchanges, each a minimum and a maximum, with no branch on the data.
#[inline(always)]
fn sort_bucket(bucket: &[u64; SLOTS]) -> [u64; SLOTS] {
    const EXCHANGES: [(usize, usize); 19] = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (1, 2),
        (5, 6),
        (0, 4),
        (3, 7),
        (1, 5),
        (2, 6),
        (1, 4),
        (3, 6),
        (2, 4),
        (3, 5),
        (3, 4),
    ];
    // One less makes zero the largest value and keeps the order of the rest.
    let mut values = bucket.map(|slot| slot.wrapping_sub(1));
    for (low, high) in EXCHANGES {
        let (a, b) = (values[low], values[high]);
        values[low] = a.min(b);
        values[high] = a.max(b);
    }
    values.map(|value| value.wrapping_add(1))
}

/// Sort values of which each is at most a few places from its own.
///
/// An insertion sort does that in time proportional to the distances moved.
/// Should they add up to far more than expected, an ordinary sort finishes.
fn sort_nearly_sorted(values: &mut [u64]) {
    let mut allowance = 32 * values.len() + 1024;
    for index in 1..values.len() {
        let value = values[index];
        let mut at = index;
        while at > 0 && values[at - 1] > value {
            values[at] = values[at - 1];
            at -= 1;
        }
        values[at] = value;
        let moved = index - at;
        if moved > allowance {
            return values.sort_unstable();
        }
        allowance -= moved;
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
                return Some(self.table.entry(slot));
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
        load: Load,
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
            let mut table = CompactTable::staggered(layout, load, index, partitions);
            let share = capacity / partitions + usize::from(index < capacity % partitions);
            table.try_reserve(share)?;
            tables.push(table);
        }
        Ok(Self {
            mixer: Mixer::new(bits, SEED),
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

    /// One partition's `(mixed key, count)` pairs in ascending order of the
    /// mixed key, which is also the order of partitions. `slots` is scratch
    /// space.
    pub(crate) fn sorted_partition<'a>(
        &'a self,
        partition: usize,
        slots: &'a mut Vec<u64>,
    ) -> impl Iterator<Item = (u64, u64)> + 'a {
        let table = &self.tables[partition];
        table.sorted_slots(slots);
        let base = (partition as u64) << self.layout.remainder_bits;
        slots.iter().map(move |&slot| {
            let (remainder, count) = table.entry(slot);
            (base | remainder, count)
        })
    }

    /// Recover the key from its mixed key.
    #[inline(always)]
    pub(crate) fn unmix(&self, mixed: u64) -> u64 {
        self.mixer.unmix(mixed)
    }

    /// Iterate over every `(key, count)` pair.
    pub(crate) fn entries(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        (0..self.tables.len()).flat_map(|partition| self.partition_entries(partition))
    }

    /// Remove every key, keeping the tables' allocations for reuse.
    pub(crate) fn clear(&mut self) {
        self.tables.iter_mut().for_each(CompactTable::clear);
    }

    pub(crate) fn tables_mut(&mut self) -> &mut [CompactTable] {
        &mut self.tables
    }

    /// Bytes of bucket storage currently allocated.
    pub(crate) fn bytes(&self) -> usize {
        self.tables
            .iter()
            .map(|table| table.buckets() * std::mem::size_of::<Bucket>())
            .sum()
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

/// Apply `input` to `tables` (a power-of-two number of them), one table at a
/// time.
///
/// An entry's table is given by the bits above its remainder, modulo the
/// number of tables. The entries are first split by table, a sequential
/// pass, so that `apply` then gives each table all of its updates in one
/// burst while it is cached.
pub(crate) fn add_split(
    tables: &mut [CompactTable],
    layout: Layout,
    input: &[u64],
    split: &mut Vec<u64>,
    ends: &mut Vec<usize>,
    apply: impl Fn(&mut CompactTable, &[u64]) -> Result<(), Error>,
) -> Result<(), Error> {
    if tables.len() == 1 {
        return apply(&mut tables[0], input);
    }
    let table_of = |entry: u64| (entry >> layout.remainder_bits) as usize & (tables.len() - 1);
    // A counting sort by table: count, turn counts into start offsets, then
    // scatter (advancing each start to its end).
    ends.clear();
    ends.resize(tables.len(), 0);
    for &entry in input {
        ends[table_of(entry)] += 1;
    }
    let mut start = 0;
    for end in ends.iter_mut() {
        start += std::mem::replace(end, start);
    }
    split.clear();
    split.resize(input.len(), 0);
    for &entry in input {
        let cursor = &mut ends[table_of(entry)];
        split[*cursor] = entry;
        *cursor += 1;
    }
    let mut start = 0;
    for (table, &end) in tables.iter_mut().zip(ends.iter()) {
        apply(table, &split[start..end])?;
        start = end;
    }
    Ok(())
}

#[inline(always)]
pub(crate) fn route(mixer: Mixer, layout: Layout, key: u64) -> (usize, u64) {
    let mixed = mixer.mix(key);
    (
        (mixed >> layout.remainder_bits) as usize,
        mixed & layout.remainder_mask(),
    )
}

/// The seed of every counter's mixer. It is the same in every run, so that
/// the order of mixed keys, in which tables are written out, is the same in
/// every output of the same k-mers (see `crate::database`).
pub(crate) const SEED: u64 = 0x6b62_6f6c_7420_7632;

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
    fn bucket_masks_match_a_scalar_search() {
        let mut state = 11_u64;
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // All bits of one slot's part of a mask.
        let slot_mask = u64::MAX >> (64 - MASK_BITS);
        for count_bits in [10_u32, 12, 15, 32, 59] {
            let key_mask = u64::MAX << count_bits;
            for _ in 0..4000 {
                let key = random() & key_mask;
                let mut bucket = Bucket([0; SLOTS]);
                for slot in &mut bucket.0 {
                    *slot = match random() % 4 {
                        0 => 0,
                        // The key with an arbitrary count.
                        1 => key | (random() & !key_mask),
                        // Another key.
                        2 => (random() & key_mask) | 1,
                        // The key with a single bit changed, in either half
                        // of the word.
                        _ => {
                            (key ^ (1 << (count_bits + (random() % 64) as u32 % (64 - count_bits))))
                                | 1
                        }
                    };
                }
                let (matches, empty) = bucket_masks(&bucket, key, key_mask);
                for (index, &slot) in bucket.0.iter().enumerate() {
                    let shift = index as u32 * MASK_BITS;
                    let expected = |set: bool| if set { slot_mask } else { 0 };
                    assert_eq!(
                        (matches >> shift) & slot_mask,
                        expected(slot & key_mask == key),
                        "match in slot {index}, count_bits={count_bits}"
                    );
                    assert_eq!(
                        (empty >> shift) & slot_mask,
                        expected(slot == 0),
                        "empty in slot {index}, count_bits={count_bits}"
                    );
                }
                // Nothing beyond the eight slots.
                let beyond = SLOTS as u32 * MASK_BITS;
                assert_eq!(matches.checked_shr(beyond).unwrap_or(0), 0);
                assert_eq!(empty.checked_shr(beyond).unwrap_or(0), 0);
            }
        }
    }

    #[test]
    fn tables_count_exactly_through_growth_and_large_counts() {
        for (k, partition_bits) in [(1, 10), (5, 3), (16, 10), (31, 10), (32, 10), (32, 0)] {
            let mut counts =
                PackedCounts::with_capacity(k, partition_bits, 0, Load::Sparse).unwrap();
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
    fn sorted_partitions_ascend_through_growth_wrapping_and_large_counts() {
        let mut state = 5_u64;
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut slots = Vec::new();
        for (k, partition_bits, keys) in [
            (31, 10, 300_000),
            (31, 2, 70_000),
            (9, 3, 100_000),
            (3, 1, 500),
        ] {
            let mut counts =
                PackedCounts::with_capacity(k, partition_bits, 0, Load::Dense).unwrap();
            let mask = u64::MAX >> (64 - key_bits(k));
            let mut wrapped = 0;
            for step in 0..keys {
                // Counts beyond the slot's field now and then.
                let delta = if step % 1000 == 0 { 1 << 20 } else { 1 };
                counts.add_key(random() & mask, delta).unwrap();
                // Small tables are checked at every size they pass through.
                if step % 997 != 0 && step + 1 != keys {
                    continue;
                }
                let mut expected: Vec<(u64, u64)> = counts
                    .entries()
                    .map(|(key, count)| (counts.mixer.mix(key), count))
                    .collect();
                expected.sort_unstable();
                let mut actual = Vec::new();
                for partition in 0..counts.tables.len() {
                    let table = &counts.tables[partition];
                    // Keys turned away at the table's end sit at its start.
                    wrapped += table
                        .iter()
                        .take(8)
                        .filter(|&(remainder, _)| table.home(remainder) + 1 == table.buckets())
                        .count();
                    actual.extend(counts.sorted_partition(partition, &mut slots));
                }
                assert!(actual == expected, "k={k} step={step}");
            }
            assert!(k == 3 || wrapped > 0, "k={k}: no table wrapped around");
        }
        // A bucket sorts whatever it holds, empty slots last.
        for _ in 0..2000 {
            let filled = (random() % 9) as usize;
            let mut bucket = [0; SLOTS];
            for slot in &mut bucket[..filled] {
                *slot = random() | 1;
            }
            let mut expected = bucket[..filled].to_vec();
            expected.sort_unstable();
            expected.resize(SLOTS, 0);
            assert_eq!(sort_bucket(&bucket).as_slice(), expected);
        }
        // Values far from their places still end up sorted.
        let mut reversed: Vec<u64> = (0..5000).rev().collect();
        sort_nearly_sorted(&mut reversed);
        assert!(reversed.is_sorted());
    }

    #[test]
    fn overflow_rejects_or_saturates() {
        let mut counts = PackedCounts::with_capacity(31, 10, 0, Load::Sparse).unwrap();
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
