//! Compact exact count tables for wide packed k-mers (33 <= k <= 128).
//!
//! A wide key is split into its low 64 bits (`lo: u64`) and its remaining
//! `2k - 64` high bits (`hi: H`), sized tightly to `k`:
//!
//! - `k = 33..=40`: `H = u16` (10 bytes/slot, 80-byte buckets)
//! - `k = 41..=48`: `H = u32` (12 bytes/slot, 96-byte buckets)
//! - `k = 49..=64`: `H = u64` (16 bytes/slot, 128-byte buckets)
//! - `k = 65..=96`: `H = [u64; 2]` (24 bytes/slot, 192-byte buckets)
//! - `k = 97..=128`: `H = [u64; 3]` (32 bytes/slot, 256-byte buckets)
//!
//! `hi` is kept verbatim while `lo` passes through a bijective 64-bit mixer
//! keyed by a fast fold of `hi`:
//!
//! ```text
//! l  = mix64(lo ^ fold(hi) ^ SEED)
//! lo = unmix64(l) ^ fold(hi) ^ SEED
//! ```
//!
//! The top `p` bits of `l` (`10 <= p <= 13`) select the partition; the
//! remaining `64 - p` bits occupy the top of `bucket.l[i]` alongside `p`
//! inline count bits (0 = empty), while `hi` sits unshifted in `bucket.h[i]`.
//! Probing uses the same 8-way SIMD scan over `bucket.l` as `crate::table` and
//! checks `bucket.h` only on a 51..54-bit `l` match.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::Hash;

use crate::table::{Load, MASK_BITS, Overflow, bucket_masks_8_unaligned, prefetch};
use crate::{Error, decode_packed32};

const M1: u64 = 0xbf58_476d_1ce4_e5b9;
const M2: u64 = 0x94d0_49bb_1331_11eb;
const M3: u64 = 0x9e37_79b9_7f4a_7c15;

const fn inverse64(value: u64) -> u64 {
    let mut inv = value;
    let mut step = 0;
    while step < 6 {
        inv = inv.wrapping_mul(2_u64.wrapping_sub(value.wrapping_mul(inv)));
        step += 1;
    }
    inv
}

const M1_INV: u64 = inverse64(M1);
const M2_INV: u64 = inverse64(M2);

const _: () = assert!(M1.wrapping_mul(M1_INV) == 1);
const _: () = assert!(M2.wrapping_mul(M2_INV) == 1);

const SEED: u64 = 0x6b62_6f6c_7420_7632;

#[inline(always)]
pub(crate) fn mix_lo<H: HighKey>(hi: H, lo: u64) -> u64 {
    let mut x = lo ^ hi.fold() ^ SEED;
    x ^= x >> 32;
    x = x.wrapping_mul(M1);
    x ^= x >> 32;
    x = x.wrapping_mul(M2);
    x ^ (x >> 32)
}

#[inline(always)]
pub(crate) fn unmix_lo<H: HighKey>(hi: H, l: u64) -> u64 {
    let mut x = l ^ (l >> 32);
    x = x.wrapping_mul(M2_INV);
    x ^= x >> 32;
    x = x.wrapping_mul(M1_INV);
    x ^= x >> 32;
    x ^ hi.fold() ^ SEED
}

/// Extract the top `bucket_bits` (`1..=16`) bits of a key whose most
/// significant word `top` holds `rem` (`2..=64`) valid bits above a full
/// 64-bit word `below`.
#[inline(always)]
fn extract_top_bits(top: u64, below: u64, rem: u32, bucket_bits: u32) -> usize {
    if rem >= bucket_bits {
        (top >> (rem - bucket_bits)) as usize
    } else {
        let need = bucket_bits - rem;
        ((top << need) | (below >> (64 - need))) as usize
    }
}

/// High-bit storage for a wide k-mer (`33 <= k <= 128`).
pub(crate) trait HighKey: Copy + Eq + Hash + Default + Send + Sync + 'static {
    /// Fold all bits into 64 bits for mixing with `lo`.
    fn fold(self) -> u64;
    /// Append `lo` followed by `self` in little-endian byte order (`key_bytes` bytes).
    fn push_le(self, lo: u64, key_bytes: usize, out: &mut Vec<u8>);
    /// Read `(self, lo)` from little-endian key bytes.
    fn read_le(bytes: &[u8]) -> (Self, u64);
    /// Decode `(self, lo)` of length `dna.len()` into uppercase ASCII DNA.
    fn decode_into(self, lo: u64, dna: &mut [u8]);
    /// Extract the top `bucket_bits` (`1..=16`) of `(self, lo)` for radix bucketing.
    fn top_bits(self, lo: u64, k: usize, bucket_bits: u32) -> usize;
    /// Lexicographic ordering of `(self, lo)` as unsigned base-4 integers.
    fn cmp_key(self, lo: u64, other_hi: Self, other_lo: u64) -> Ordering;
}

impl HighKey for u16 {
    #[inline(always)]
    fn fold(self) -> u64 {
        u64::from(self).wrapping_mul(M1)
    }

    #[inline(always)]
    fn push_le(self, lo: u64, key_bytes: usize, out: &mut Vec<u8>) {
        let end = out.len() + key_bytes;
        out.extend_from_slice(&lo.to_le_bytes());
        out.extend_from_slice(&self.to_le_bytes());
        out.truncate(end);
    }

    #[inline(always)]
    fn read_le(bytes: &[u8]) -> (Self, u64) {
        let mut buf = [0_u8; 10];
        buf[..bytes.len()].copy_from_slice(bytes);
        let lo = u64::from_le_bytes(buf[..8].try_into().unwrap());
        let hi = u16::from_le_bytes(buf[8..10].try_into().unwrap());
        (hi, lo)
    }

    #[inline(always)]
    fn decode_into(self, lo: u64, dna: &mut [u8]) {
        let split = dna.len() - 32;
        let high = decode_packed32(u64::from(self), split);
        let low = decode_packed32(lo, 32);
        dna[..split].copy_from_slice(&high[..split]);
        dna[split..].copy_from_slice(&low);
    }

    #[inline(always)]
    fn top_bits(self, lo: u64, k: usize, bucket_bits: u32) -> usize {
        extract_top_bits(u64::from(self), lo, (2 * k - 64) as u32, bucket_bits)
    }

    #[inline(always)]
    fn cmp_key(self, lo: u64, other_hi: Self, other_lo: u64) -> Ordering {
        (self, lo).cmp(&(other_hi, other_lo))
    }
}

impl HighKey for u32 {
    #[inline(always)]
    fn fold(self) -> u64 {
        u64::from(self).wrapping_mul(M1)
    }

    #[inline(always)]
    fn push_le(self, lo: u64, key_bytes: usize, out: &mut Vec<u8>) {
        let end = out.len() + key_bytes;
        out.extend_from_slice(&lo.to_le_bytes());
        out.extend_from_slice(&self.to_le_bytes());
        out.truncate(end);
    }

    #[inline(always)]
    fn read_le(bytes: &[u8]) -> (Self, u64) {
        let mut buf = [0_u8; 12];
        buf[..bytes.len()].copy_from_slice(bytes);
        let lo = u64::from_le_bytes(buf[..8].try_into().unwrap());
        let hi = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        (hi, lo)
    }

    #[inline(always)]
    fn decode_into(self, lo: u64, dna: &mut [u8]) {
        let split = dna.len() - 32;
        let high = decode_packed32(u64::from(self), split);
        let low = decode_packed32(lo, 32);
        dna[..split].copy_from_slice(&high[..split]);
        dna[split..].copy_from_slice(&low);
    }

    #[inline(always)]
    fn top_bits(self, lo: u64, k: usize, bucket_bits: u32) -> usize {
        extract_top_bits(u64::from(self), lo, (2 * k - 64) as u32, bucket_bits)
    }

    #[inline(always)]
    fn cmp_key(self, lo: u64, other_hi: Self, other_lo: u64) -> Ordering {
        (self, lo).cmp(&(other_hi, other_lo))
    }
}

impl HighKey for u64 {
    #[inline(always)]
    fn fold(self) -> u64 {
        self.wrapping_mul(M1)
    }

    #[inline(always)]
    fn push_le(self, lo: u64, key_bytes: usize, out: &mut Vec<u8>) {
        let end = out.len() + key_bytes;
        out.extend_from_slice(&lo.to_le_bytes());
        out.extend_from_slice(&self.to_le_bytes());
        out.truncate(end);
    }

    #[inline(always)]
    fn read_le(bytes: &[u8]) -> (Self, u64) {
        let mut buf = [0_u8; 16];
        buf[..bytes.len()].copy_from_slice(bytes);
        let lo = u64::from_le_bytes(buf[..8].try_into().unwrap());
        let hi = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        (hi, lo)
    }

    #[inline(always)]
    fn decode_into(self, lo: u64, dna: &mut [u8]) {
        let split = dna.len() - 32;
        let high = decode_packed32(self, split);
        let low = decode_packed32(lo, 32);
        dna[..split].copy_from_slice(&high[..split]);
        dna[split..].copy_from_slice(&low);
    }

    #[inline(always)]
    fn top_bits(self, lo: u64, k: usize, bucket_bits: u32) -> usize {
        extract_top_bits(self, lo, (2 * k - 64) as u32, bucket_bits)
    }

    #[inline(always)]
    fn cmp_key(self, lo: u64, other_hi: Self, other_lo: u64) -> Ordering {
        (self, lo).cmp(&(other_hi, other_lo))
    }
}

impl HighKey for [u64; 2] {
    #[inline(always)]
    fn fold(self) -> u64 {
        self[0].wrapping_mul(M1) ^ self[1].wrapping_mul(M2)
    }

    #[inline(always)]
    fn push_le(self, lo: u64, key_bytes: usize, out: &mut Vec<u8>) {
        let end = out.len() + key_bytes;
        out.extend_from_slice(&lo.to_le_bytes());
        out.extend_from_slice(&self[0].to_le_bytes());
        out.extend_from_slice(&self[1].to_le_bytes());
        out.truncate(end);
    }

    #[inline(always)]
    fn read_le(bytes: &[u8]) -> (Self, u64) {
        let mut buf = [0_u8; 24];
        buf[..bytes.len()].copy_from_slice(bytes);
        let lo = u64::from_le_bytes(buf[..8].try_into().unwrap());
        let w1 = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let w2 = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        ([w1, w2], lo)
    }

    #[inline(always)]
    fn decode_into(self, lo: u64, dna: &mut [u8]) {
        let split = dna.len() - 64;
        let w2 = decode_packed32(self[1], split);
        let w1 = decode_packed32(self[0], 32);
        let w0 = decode_packed32(lo, 32);
        dna[..split].copy_from_slice(&w2[..split]);
        dna[split..split + 32].copy_from_slice(&w1);
        dna[split + 32..].copy_from_slice(&w0);
    }

    #[inline(always)]
    fn top_bits(self, _lo: u64, k: usize, bucket_bits: u32) -> usize {
        extract_top_bits(self[1], self[0], (2 * k - 128) as u32, bucket_bits)
    }

    #[inline(always)]
    fn cmp_key(self, lo: u64, other_hi: Self, other_lo: u64) -> Ordering {
        (self[1], self[0], lo).cmp(&(other_hi[1], other_hi[0], other_lo))
    }
}

impl HighKey for [u64; 3] {
    #[inline(always)]
    fn fold(self) -> u64 {
        self[0].wrapping_mul(M1) ^ self[1].wrapping_mul(M2) ^ self[2].wrapping_mul(M3)
    }

    #[inline(always)]
    fn push_le(self, lo: u64, key_bytes: usize, out: &mut Vec<u8>) {
        let end = out.len() + key_bytes;
        out.extend_from_slice(&lo.to_le_bytes());
        out.extend_from_slice(&self[0].to_le_bytes());
        out.extend_from_slice(&self[1].to_le_bytes());
        out.extend_from_slice(&self[2].to_le_bytes());
        out.truncate(end);
    }

    #[inline(always)]
    fn read_le(bytes: &[u8]) -> (Self, u64) {
        let mut buf = [0_u8; 32];
        buf[..bytes.len()].copy_from_slice(bytes);
        let lo = u64::from_le_bytes(buf[..8].try_into().unwrap());
        let w1 = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let w2 = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let w3 = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        ([w1, w2, w3], lo)
    }

    #[inline(always)]
    fn decode_into(self, lo: u64, dna: &mut [u8]) {
        let split = dna.len() - 96;
        let w3 = decode_packed32(self[2], split);
        let w2 = decode_packed32(self[1], 32);
        let w1 = decode_packed32(self[0], 32);
        let w0 = decode_packed32(lo, 32);
        dna[..split].copy_from_slice(&w3[..split]);
        dna[split..split + 32].copy_from_slice(&w2);
        dna[split + 32..split + 64].copy_from_slice(&w1);
        dna[split + 64..].copy_from_slice(&w0);
    }

    #[inline(always)]
    fn top_bits(self, _lo: u64, k: usize, bucket_bits: u32) -> usize {
        extract_top_bits(self[2], self[1], (2 * k - 192) as u32, bucket_bits)
    }

    #[inline(always)]
    fn cmp_key(self, lo: u64, other_hi: Self, other_lo: u64) -> Ordering {
        (self[2], self[1], self[0], lo).cmp(&(other_hi[2], other_hi[1], other_hi[0], other_lo))
    }
}

/// Bit widths shared by every table of one partitioned wide counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LayoutWide {
    pub(crate) partition_bits: u32,
    pub(crate) l_rem_bits: u32,
}

impl LayoutWide {
    pub(crate) fn new(partition_bits: u32) -> Self {
        debug_assert!((10..=20).contains(&partition_bits));
        Self {
            partition_bits,
            l_rem_bits: 64 - partition_bits,
        }
    }

    #[inline(always)]
    pub(crate) fn count_bits(self) -> u32 {
        self.partition_bits
    }

    #[inline(always)]
    pub(crate) fn count_marker(self) -> u64 {
        u64::MAX >> self.l_rem_bits
    }

    #[inline(always)]
    pub(crate) fn l_key_mask(self) -> u64 {
        !self.count_marker()
    }

    #[inline(always)]
    pub(crate) fn l_rem_mask(self) -> u64 {
        u64::MAX >> self.partition_bits
    }
}

const SLOTS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct BucketWide<H: HighKey> {
    l: [u64; SLOTS],
    h: [H; SLOTS],
}

#[cfg(test)]
impl<H: HighKey> BucketWide<H> {
    fn empty() -> Self {
        Self {
            l: [0; SLOTS],
            h: [H::default(); SLOTS],
        }
    }
}

struct BucketsWide<H: HighKey> {
    pointer: std::ptr::NonNull<BucketWide<H>>,
    len: usize,
}

/// Map bucket arrays directly once they reach 16 KiB so outgrown tables return
/// memory to the OS immediately.
const MAPPED_BYTES: usize = 16 * 1024;

// SAFETY: `BucketsWide` owns its allocation exclusively.
unsafe impl<H: HighKey> Send for BucketsWide<H> {}
// SAFETY: shared access only reads.
unsafe impl<H: HighKey> Sync for BucketsWide<H> {}

impl<H: HighKey> BucketsWide<H> {
    const fn new() -> Self {
        Self {
            pointer: std::ptr::NonNull::dangling(),
            len: 0,
        }
    }

    const fn mapped_buckets() -> usize {
        MAPPED_BYTES.div_ceil(std::mem::size_of::<BucketWide<H>>())
    }

    fn mapped(len: usize) -> bool {
        cfg!(unix) && len >= Self::mapped_buckets()
    }

    fn zeroed(len: usize) -> Result<Self, Error> {
        if len == 0 {
            return Ok(Self::new());
        }
        let layout =
            std::alloc::Layout::array::<BucketWide<H>>(len).map_err(|_| Error::ReserveFailed)?;
        #[cfg(unix)]
        if Self::mapped(len) {
            // SAFETY: anonymous private mapping of nonzero size; page-aligned.
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
        // SAFETY: nonzero layout; all-zero `BucketWide<H>` is valid and empty.
        let pointer = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<BucketWide<H>>();
        let pointer = std::ptr::NonNull::new(pointer).ok_or(Error::ReserveFailed)?;
        Ok(Self { pointer, len })
    }
}

impl<H: HighKey> Drop for BucketsWide<H> {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        let layout =
            std::alloc::Layout::array::<BucketWide<H>>(self.len).expect("allocated layout");
        #[cfg(unix)]
        if Self::mapped(self.len) {
            // SAFETY: matches mapping created in `zeroed`.
            unsafe { libc::munmap(self.pointer.as_ptr().cast(), layout.size()) };
            return;
        }
        // SAFETY: matches allocation created in `zeroed`.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr().cast(), layout) };
    }
}

impl<H: HighKey> std::ops::Deref for BucketsWide<H> {
    type Target = [BucketWide<H>];

    #[inline(always)]
    fn deref(&self) -> &[BucketWide<H>] {
        // SAFETY: valid for `len` initialized buckets.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }
}

impl<H: HighKey> std::ops::DerefMut for BucketsWide<H> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [BucketWide<H>] {
        // SAFETY: exclusive access through `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.len) }
    }
}

/// Packed queued update entry: zero alignment padding (`8 + size_of::<H>()`
/// bytes) so workers push into one `Vec` per group instead of two.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub(crate) struct QueuedEntry<H: HighKey> {
    pub(crate) l: u64,
    pub(crate) h: H,
}

impl<H: HighKey> QueuedEntry<H> {
    #[inline(always)]
    pub(crate) fn new(l: u64, h: H) -> Self {
        Self { l, h }
    }

    #[inline(always)]
    pub(crate) fn l(self) -> u64 {
        self.l
    }

    #[inline(always)]
    pub(crate) fn h(self) -> H {
        self.h
    }
}

/// Prefetch every cache line covered by `bucket_ptr`. Buckets whose size does
/// not divide the cache-line size (80B, 96B) or exceeds one line (192B, 256B)
/// can span two lines; prefetching both ends hides both misses.
#[inline(always)]
fn prefetch_bucket<H: HighKey>(bucket_ptr: *const BucketWide<H>) {
    let base = bucket_ptr.cast::<u8>();
    prefetch(base);
    let last = std::mem::size_of::<BucketWide<H>>() - 1;
    prefetch(base.wrapping_add(last));
}

const fn load_sixteenths(load: Load) -> usize {
    match load {
        Load::Sparse => 12,
        Load::Dense => 14,
    }
}

/// One partition's compact wide table.
pub(crate) struct CompactTableWide<H: HighKey> {
    buckets: BucketsWide<H>,
    len: usize,
    grow_at: usize,
    layout: LayoutWide,
    stagger: u16,
    load: Load,
    overflow: HashMap<(u64, H), u64, foldhash::fast::RandomState>,
}

impl<H: HighKey> CompactTableWide<H> {
    const STEADY_BUCKETS: usize = (64 * 1024) / std::mem::size_of::<BucketWide<H>>();

    pub(crate) fn staggered(
        layout: LayoutWide,
        load: Load,
        index: usize,
        partitions: usize,
    ) -> Self {
        debug_assert!(index < partitions);
        Self {
            buckets: BucketsWide::new(),
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

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.grow_at
    }

    pub(crate) fn buckets(&self) -> usize {
        self.buckets.len()
    }

    #[inline(always)]
    fn home(&self, l_key: u64) -> usize {
        ((u128::from(l_key) * self.buckets.len() as u128) >> 64) as usize
    }

    pub(crate) fn warm(&self, updates: usize) {
        const WARM_BYTES: usize = 32 * 1024;
        let warm_buckets = WARM_BYTES / std::mem::size_of::<BucketWide<H>>();
        if self.buckets.len() >= warm_buckets && updates >= self.buckets.len() / 4 {
            let touched = self
                .buckets
                .iter()
                .fold(0_u64, |sum, bucket| sum.wrapping_add(bucket.l[0]));
            std::hint::black_box(touched);
        }
    }

    #[inline(always)]
    pub(crate) fn prefetch(&self, l_rem: u64) {
        if !self.buckets.is_empty() {
            let l_key = l_rem << self.layout.count_bits();
            prefetch_bucket(self.buckets.as_ptr().wrapping_add(self.home(l_key)));
        }
    }

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
            .div_ceil(load_sixteenths(self.load));
        self.resize(slots.div_ceil(SLOTS).max(2))
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self) -> Result<(), Error> {
        let current = self.buckets.len();
        let steady = Self::STEADY_BUCKETS.max(64);
        let buckets = if current < steady / 2 {
            (current * 2).max(2)
        } else if current < steady {
            steady + (((steady / 2) * usize::from(self.stagger)) >> 16)
        } else {
            current
                .checked_add(current / 2)
                .ok_or(Error::ReserveFailed)?
        };
        self.resize(buckets)
    }

    fn resize(&mut self, buckets: usize) -> Result<(), Error> {
        let mut filled = Vec::new();
        filled
            .try_reserve_exact(buckets)
            .map_err(|_| Error::ReserveFailed)?;
        filled.resize(buckets, 0_u8);
        let old = std::mem::replace(&mut self.buckets, BucketsWide::zeroed(buckets)?);
        self.grow_at = buckets
            .checked_mul(SLOTS)
            .and_then(|s| s.checked_mul(load_sixteenths(self.load)))
            .ok_or(Error::ReserveFailed)?
            / 16;
        let l_key_mask = self.layout.l_key_mask();
        for bucket in old.iter() {
            for slot in 0..SLOTS {
                let l = bucket.l[slot];
                if l == 0 {
                    break;
                }
                let h = bucket.h[slot];
                let mut index = self.home(l & l_key_mask);
                while filled[index] == SLOTS as u8 {
                    index += 1;
                    if index == buckets {
                        index = 0;
                    }
                }
                let at = usize::from(filled[index]);
                self.buckets[index].l[at] = l;
                self.buckets[index].h[at] = h;
                filled[index] += 1;
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn find(&self, l_key: u64, h: H) -> (bool, usize) {
        let l_key_mask = self.layout.l_key_mask();
        let mut bucket = self.home(l_key);
        loop {
            // SAFETY: `bucket < buckets.len()`, and the load factor guarantees an empty slot.
            let slots = unsafe { self.buckets.get_unchecked(bucket) };
            let (matches, empty) = bucket_masks_8_unaligned(&slots.l, l_key, l_key_mask);
            let matches = matches & !empty;
            if matches != 0 {
                let at = (matches.trailing_zeros() / MASK_BITS) as usize;
                // SAFETY: mask encodes 8 slots only.
                if unsafe { *slots.h.get_unchecked(at) } == h {
                    return (true, bucket * SLOTS + at);
                }
                return self.find_collision(bucket, matches, empty, l_key, h);
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

    #[cold]
    #[inline(never)]
    fn find_collision(
        &self,
        mut bucket: usize,
        mut matches: u64,
        mut empty: u64,
        l_key: u64,
        h: H,
    ) -> (bool, usize) {
        let l_key_mask = self.layout.l_key_mask();
        loop {
            let slots = unsafe { self.buckets.get_unchecked(bucket) };
            while matches != 0 {
                let bit = matches.trailing_zeros();
                let at = (bit / MASK_BITS) as usize;
                if unsafe { *slots.h.get_unchecked(at) } == h {
                    return (true, bucket * SLOTS + at);
                }
                matches &= !((u64::MAX >> (64 - MASK_BITS)) << bit);
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
            let next = unsafe { self.buckets.get_unchecked(bucket) };
            let (m, e) = bucket_masks_8_unaligned(&next.l, l_key, l_key_mask);
            matches = m & !e;
            empty = e;
        }
    }

    #[inline(always)]
    pub(crate) fn add(
        &mut self,
        l_rem: u64,
        h: H,
        delta: u64,
        overflow: Overflow,
    ) -> Result<(), Error> {
        debug_assert!(delta > 0 && l_rem <= self.layout.l_rem_mask());
        if self.len >= self.grow_at {
            self.grow()?;
        }
        let marker = self.layout.count_marker();
        let l_key = l_rem << self.layout.count_bits();
        let (found, index) = self.find(l_key, h);
        // SAFETY: `find` returns an in-bounds slot index.
        let bucket = unsafe { self.buckets.get_unchecked_mut(index / SLOTS) };
        let at = index % SLOTS;
        let l = bucket.l[at];
        let small = l & marker;
        if found && small != marker && delta < marker - small {
            bucket.l[at] = l + delta;
            return Ok(());
        }
        if !found && delta < marker {
            bucket.l[at] = l_key | delta;
            bucket.h[at] = h;
            self.len += 1;
            return Ok(());
        }
        self.add_large(l_rem, h, index, found, delta, overflow)
    }

    /// Burst-update a cache-resident table with packed `QueuedEntry<H>` entries.
    #[inline(always)]
    pub(crate) fn add_entries(
        &mut self,
        entries: &[QueuedEntry<H>],
        delta_shift: u32,
    ) -> Result<(), Error> {
        let layout = self.layout;
        let pbits = layout.partition_bits;
        let marker = layout.count_marker();
        let l_key_mask = layout.l_key_mask();
        let mut index = 0;
        'run: while index < entries.len() {
            if self.len >= self.grow_at {
                self.grow()?;
            }
            let room = self.grow_at - self.len;
            let mut added = 0;
            let (buckets, count) = (self.buckets.as_mut_ptr(), self.buckets.len());
            let home = |l_key: u64| ((u128::from(l_key) * count as u128) >> 64) as usize;
            while index < entries.len() && added < room {
                if let Some(&ahead) = entries.get(index + AHEAD) {
                    prefetch_bucket(buckets.wrapping_add(home(ahead.l() << pbits)));
                }
                let entry = entries[index];
                index += 1;
                let l_entry = entry.l();
                let h = entry.h();
                let l_key = l_entry << pbits;
                let delta = l_entry >> delta_shift;
                let mut bucket = home(l_key);
                loop {
                    // SAFETY: `bucket < count`, and the load factor guarantees an empty slot.
                    let slots = unsafe { &mut *buckets.add(bucket) };
                    let (matches, empty) = bucket_masks_8_unaligned(&slots.l, l_key, l_key_mask);
                    let matches = matches & !empty;
                    let found = matches != 0;
                    let ends = if found { matches } else { empty };
                    if ends != 0 {
                        let at = (ends.trailing_zeros() / MASK_BITS) as usize;
                        // SAFETY: mask encodes 8 slots only.
                        let h_slot = unsafe { slots.h.get_unchecked_mut(at) };
                        if !found || *h_slot == h {
                            let l_slot = unsafe { slots.l.get_unchecked_mut(at) };
                            if delta < marker - (*l_slot & marker) {
                                *h_slot = h;
                                *l_slot = (*l_slot | l_key) + delta;
                                added += usize::from(!found);
                                break;
                            }
                            self.len += added;
                            self.add_large(
                                l_key >> pbits,
                                h,
                                bucket * SLOTS + at,
                                found,
                                delta,
                                Overflow::Reject,
                            )?;
                            continue 'run;
                        }
                        self.len += added;
                        self.add(l_key >> pbits, h, delta, Overflow::Reject)?;
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

    #[cold]
    #[inline(never)]
    fn add_large(
        &mut self,
        l_rem: u64,
        h: H,
        index: usize,
        found: bool,
        delta: u64,
        mode: Overflow,
    ) -> Result<(), Error> {
        let marker = self.layout.count_marker();
        let (b, at) = (index / SLOTS, index % SLOTS);
        // SAFETY: `index` came from `find` or `add_entries`.
        let l = unsafe { self.buckets.get_unchecked(b).l[at] };
        if found && l & marker == marker {
            let count = self.overflow.get_mut(&(l_rem, h)).expect("marked count");
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
        let current = if found { l & marker } else { 0 };
        let (count, result) = match current.checked_add(delta) {
            Some(sum) => (sum, Ok(())),
            None if mode == Overflow::Saturate => (u64::MAX, Err(Error::CountOverflow)),
            None => return Err(Error::CountOverflow),
        };
        self.overflow
            .try_reserve(1)
            .map_err(|_| Error::ReserveFailed)?;
        self.overflow.insert((l_rem, h), count);
        // SAFETY: as above.
        let bucket = unsafe { self.buckets.get_unchecked_mut(b) };
        bucket.l[at] = (l_rem << self.layout.count_bits()) | marker;
        bucket.h[at] = h;
        if !found {
            self.len += 1;
        }
        result
    }

    pub(crate) fn insert_new(&mut self, l_rem: u64, h: H, count: u64) -> Result<bool, Error> {
        if self.len >= self.grow_at {
            self.grow()?;
        }
        let l_key = l_rem << self.layout.count_bits();
        if self.find(l_key, h).0 {
            return Ok(false);
        }
        self.add(l_rem, h, count, Overflow::Reject)?;
        Ok(true)
    }

    #[inline(always)]
    fn entry(&self, l: u64, h: H) -> (u64, H, u64) {
        let l_rem = l >> self.layout.count_bits();
        let small = l & self.layout.count_marker();
        let count = if small == self.layout.count_marker() {
            self.overflow[&(l_rem, h)]
        } else {
            small
        };
        (l_rem, h, count)
    }

    pub(crate) fn iter(&self) -> EntriesWide<'_, H> {
        EntriesWide {
            buckets: self.buckets.iter(),
            bucket: None,
            slot: SLOTS,
            table: self,
        }
    }

    #[cfg(test)]
    pub(crate) fn set_counts(&mut self, count: u64) {
        assert!(count > 0, "stored counts are positive");
        let entries: Vec<_> = self.iter().map(|(l_rem, h, _)| (l_rem, h)).collect();
        self.buckets
            .iter_mut()
            .for_each(|bucket| *bucket = BucketWide::empty());
        self.overflow.clear();
        self.len = 0;
        for (l_rem, h) in entries {
            self.add(l_rem, h, count, Overflow::Reject).unwrap();
        }
    }
}

pub(crate) struct EntriesWide<'a, H: HighKey> {
    buckets: std::slice::Iter<'a, BucketWide<H>>,
    bucket: Option<&'a BucketWide<H>>,
    slot: usize,
    table: &'a CompactTableWide<H>,
}

impl<H: HighKey> Iterator for EntriesWide<'_, H> {
    /// `(l_rem, h, count)`.
    type Item = (u64, H, u64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(bucket) = self.bucket
                && self.slot < SLOTS
            {
                let l = bucket.l[self.slot];
                if l != 0 {
                    let h = bucket.h[self.slot];
                    self.slot += 1;
                    return Some(self.table.entry(l, h));
                }
            }
            self.bucket = Some(self.buckets.next()?);
            self.slot = 0;
        }
    }
}

/// Single-vector entry for `OrderedAdderWide`: one bounds check and push per k-mer.
#[derive(Clone, Copy)]
struct PendingEntry<H: HighKey> {
    l: u64,
    h: H,
    d: u32,
}

/// A complete partitioned counter for wide packed keys (`33 <= k <= 128`).
pub(crate) struct PackedCountsWide<H: HighKey> {
    layout: LayoutWide,
    tables: Vec<CompactTableWide<H>>,
    scratch: Vec<PendingEntry<H>>,
}

impl<H: HighKey> PackedCountsWide<H> {
    pub(crate) fn with_capacity(
        partition_bits: u32,
        capacity: usize,
        load: Load,
    ) -> Result<Self, Error> {
        let layout = LayoutWide::new(partition_bits);
        let partitions = 1_usize << partition_bits;
        let mut tables = Vec::new();
        tables
            .try_reserve_exact(partitions)
            .map_err(|_| Error::ReserveFailed)?;
        for index in 0..partitions {
            let mut table = CompactTableWide::staggered(layout, load, index, partitions);
            let share = capacity / partitions + usize::from(index < capacity % partitions);
            table.try_reserve(share)?;
            tables.push(table);
        }
        Ok(Self {
            layout,
            tables,
            scratch: Vec::new(),
        })
    }

    pub(crate) fn from_tables(layout: LayoutWide, tables: Vec<CompactTableWide<H>>) -> Self {
        Self {
            layout,
            tables,
            scratch: Vec::new(),
        }
    }

    pub(crate) fn layout(&self) -> LayoutWide {
        self.layout
    }

    pub(crate) fn into_tables(self) -> Vec<CompactTableWide<H>> {
        self.tables
    }

    pub(crate) fn tables(&self) -> &[CompactTableWide<H>] {
        &self.tables
    }

    #[cfg(test)]
    pub(crate) fn tables_mut(&mut self) -> &mut [CompactTableWide<H>] {
        &mut self.tables
    }

    pub(crate) fn len(&self) -> usize {
        self.tables.iter().map(CompactTableWide::len).sum()
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.tables.iter().map(CompactTableWide::capacity).sum()
    }

    #[inline(always)]
    pub(crate) fn route(&self, hi: H, lo: u64) -> (usize, u64) {
        let l = mix_lo(hi, lo);
        (
            (l >> self.layout.l_rem_bits) as usize,
            l & self.layout.l_rem_mask(),
        )
    }

    #[inline(always)]
    pub(crate) fn key(&self, partition: usize, l_rem: u64, h: H) -> (H, u64) {
        let l = ((partition as u64) << self.layout.l_rem_bits) | l_rem;
        (h, unmix_lo(h, l))
    }

    pub(crate) fn add_key(&mut self, hi: H, lo: u64, delta: u64) -> Result<(), Error> {
        let (partition, l_rem) = self.route(hi, lo);
        self.tables[partition].add(l_rem, hi, delta, Overflow::Reject)
    }

    pub(crate) fn insert_new(&mut self, hi: H, lo: u64, count: u64) -> Result<bool, Error> {
        let (partition, l_rem) = self.route(hi, lo);
        self.tables[partition].insert_new(l_rem, hi, count)
    }

    pub(crate) fn partition_entries(
        &self,
        partition: usize,
    ) -> impl Iterator<Item = ((H, u64), u64)> + '_ {
        self.tables[partition]
            .iter()
            .map(move |(l_rem, h, count)| (self.key(partition, l_rem, h), count))
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = ((H, u64), u64)> + '_ {
        (0..self.tables.len()).flat_map(|partition| self.partition_entries(partition))
    }

    pub(crate) fn adder(&mut self) -> OrderedAdderWide<'_, H> {
        let mut pending = std::mem::take(&mut self.scratch);
        pending.clear();
        OrderedAdderWide {
            counts: self,
            pending,
        }
    }
}

const AHEAD: usize = 16;
const ORDERED_CHUNK: usize = 32 * 1024;

pub(crate) struct OrderedAdderWide<'a, H: HighKey> {
    counts: &'a mut PackedCountsWide<H>,
    pending: Vec<PendingEntry<H>>,
}

impl<H: HighKey> OrderedAdderWide<'_, H> {
    #[inline(always)]
    pub(crate) fn push(&mut self, hi: H, lo: u64) -> Result<(), Error> {
        let l = mix_lo(hi, lo);
        match self.pending.last_mut() {
            Some(last) if last.l == l && last.h == hi && last.d < u32::MAX => last.d += 1,
            _ => {
                if self.pending.len() == ORDERED_CHUNK {
                    self.apply()?;
                }
                self.pending.push(PendingEntry { l, h: hi, d: 1 });
            }
        }
        Ok(())
    }

    #[inline(never)]
    fn apply(&mut self) -> Result<(), Error> {
        let layout = self.counts.layout;
        let l_rem_bits = layout.l_rem_bits;
        let l_rem_mask = layout.l_rem_mask();
        let tables = &mut self.counts.tables;
        let result = (0..self.pending.len()).try_for_each(|index| {
            if let Some(ahead) = self.pending.get(index + AHEAD) {
                tables[(ahead.l >> l_rem_bits) as usize].prefetch(ahead.l & l_rem_mask);
            }
            let entry = self.pending[index];
            tables[(entry.l >> l_rem_bits) as usize].add(
                entry.l & l_rem_mask,
                entry.h,
                u64::from(entry.d),
                Overflow::Saturate,
            )
        });
        self.pending.clear();
        result
    }

    pub(crate) fn finish(mut self) -> Result<(), Error> {
        self.apply()
    }
}

impl<H: HighKey> Drop for OrderedAdderWide<'_, H> {
    fn drop(&mut self) {
        self.counts.scratch = std::mem::take(&mut self.pending);
    }
}

/// Counting-sort packed `QueuedEntry<H>` entries by sub-partition table and
/// apply each table's burst.
pub(crate) fn add_split<H: HighKey>(
    tables: &mut [CompactTableWide<H>],
    layout: LayoutWide,
    input: &[QueuedEntry<H>],
    split: &mut Vec<QueuedEntry<H>>,
    ends: &mut Vec<usize>,
    apply: impl Fn(&mut CompactTableWide<H>, &[QueuedEntry<H>]) -> Result<(), Error>,
) -> Result<(), Error> {
    if tables.len() == 1 {
        return apply(&mut tables[0], input);
    }
    let mask = tables.len() - 1;
    let shift = layout.l_rem_bits;
    let table_of = |l_entry: u64| (l_entry >> shift) as usize & mask;
    ends.clear();
    ends.resize(tables.len(), 0);
    for entry in input {
        ends[table_of(entry.l())] += 1;
    }
    let mut start = 0;
    for end in ends.iter_mut() {
        start += std::mem::replace(end, start);
    }
    split.clear();
    split.resize(input.len(), QueuedEntry::new(0, H::default()));
    for &entry in input {
        let cursor = &mut ends[table_of(entry.l())];
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

#[cfg(test)]
mod tests {
    use super::*;

    fn next_rand(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn mixing_is_invertible_for_every_wide_type() {
        let mut state = 7_u64;
        for _ in 0..2000 {
            let lo = next_rand(&mut state);
            let h16 = next_rand(&mut state) as u16;
            assert_eq!(unmix_lo(h16, mix_lo(h16, lo)), lo);
            let h32 = next_rand(&mut state) as u32;
            assert_eq!(unmix_lo(h32, mix_lo(h32, lo)), lo);
            let h64 = next_rand(&mut state);
            assert_eq!(unmix_lo(h64, mix_lo(h64, lo)), lo);
            let h128 = [next_rand(&mut state), next_rand(&mut state)];
            assert_eq!(unmix_lo(h128, mix_lo(h128, lo)), lo);
            let h192 = [
                next_rand(&mut state),
                next_rand(&mut state),
                next_rand(&mut state),
            ];
            assert_eq!(unmix_lo(h192, mix_lo(h192, lo)), lo);
        }
    }

    #[test]
    fn tables_handle_l_key_collisions_and_large_counts() {
        let layout = LayoutWide::new(13);
        let mut table = CompactTableWide::<u16>::staggered(layout, Load::Dense, 0, 1);
        for h in 0_u16..40 {
            table
                .add(12345, h, u64::from(h) + 1, Overflow::Reject)
                .unwrap();
        }
        for h in 0_u16..40 {
            table.add(12345, h, 10, Overflow::Reject).unwrap();
        }
        assert_eq!(table.len(), 40);
        let actual: HashMap<_, _> = table.iter().map(|(l, h, c)| ((l, h), c)).collect();
        for h in 0_u16..40 {
            assert_eq!(actual[&(12345, h)], u64::from(h) + 11);
        }
    }

    #[test]
    fn wide_tables_count_exactly_through_growth_and_overflow() {
        let mut state = 99_u64;
        let mut counts = PackedCountsWide::<u16>::with_capacity(10, 0, Load::Sparse).unwrap();
        let marker = counts.layout().count_marker();
        let mut expected = HashMap::new();
        for step in 0..40_000_u64 {
            let hi = (next_rand(&mut state) & 0xffff) as u16;
            let lo = if step % 3 == 0 {
                next_rand(&mut state) & 0xff
            } else {
                next_rand(&mut state)
            };
            let delta = if step % 1000 == 0 {
                marker + 1 + step
            } else {
                1
            };
            counts.add_key(hi, lo, delta).unwrap();
            *expected.entry((hi, lo)).or_insert(0) += delta;
        }
        assert_eq!(counts.len(), expected.len());
        let actual: HashMap<_, _> = counts.entries().collect();
        assert_eq!(actual, expected);

        let mut c2 = PackedCountsWide::<[u64; 2]>::with_capacity(10, 0, Load::Sparse).unwrap();
        c2.add_key([1, 2], 5, u64::MAX - 1).unwrap();
        assert_eq!(c2.add_key([1, 2], 5, 2), Err(Error::CountOverflow));
        let mut adder = c2.adder();
        let res = [([0, 0], 9), ([1, 2], 5), ([1, 2], 5), ([0, 0], 9)]
            .into_iter()
            .try_for_each(|(h, l)| adder.push(h, l))
            .and_then(|()| adder.finish());
        assert_eq!(res, Err(Error::CountOverflow));
    }

    #[test]
    fn bucket_sizes_are_tightly_packed() {
        assert_eq!(std::mem::size_of::<BucketWide<u16>>(), 80);
        assert_eq!(std::mem::size_of::<BucketWide<u32>>(), 96);
        assert_eq!(std::mem::size_of::<BucketWide<u64>>(), 128);
        assert_eq!(std::mem::size_of::<BucketWide<[u64; 2]>>(), 192);
        assert_eq!(std::mem::size_of::<BucketWide<[u64; 3]>>(), 256);
    }
}
