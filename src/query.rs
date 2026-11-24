//! Point and batch queries, count histograms, top-k extraction, and filtered
//! iteration/export on [`Counter`].

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::io::{self, Write};

use crate::table::PackedCounts;
use crate::table_wide::{HighKey, PackedCountsWide};
use crate::{
    Counter, Counts, DNA_BASES, OutputTable, StrandMode, WideCounts, decode_packed, dispatch_wide,
    format_count, write_packed_tsv, write_packed64_tsv,
};

/// Prefetch distance for batched queries.
const QUERY_AHEAD: usize = 16;
/// Queries encoded and routed per prefetch batch.
const QUERY_CHUNK: usize = 256;
/// Counts below this threshold accumulate in a direct-indexed array when
/// computing histograms.
const DIRECT_HISTOGRAM: usize = 4096;

/// Inclusive count bounds used to filter k-mer iteration and export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CountFilter {
    /// Minimum count to include (inclusive).
    pub min_count: u64,
    /// Optional maximum count to include (inclusive).
    pub max_count: Option<u64>,
}

impl Default for CountFilter {
    fn default() -> Self {
        Self {
            min_count: 1,
            max_count: None,
        }
    }
}

impl CountFilter {
    /// Include k-mers with count at least `min_count`.
    pub const fn at_least(min_count: u64) -> Self {
        Self {
            min_count,
            max_count: None,
        }
    }

    /// Include k-mers with `min_count <= count <= max_count`.
    pub const fn between(min_count: u64, max_count: u64) -> Self {
        Self {
            min_count,
            max_count: Some(max_count),
        }
    }

    /// Return whether `count` falls within these bounds.
    #[inline(always)]
    pub const fn matches(self, count: u64) -> bool {
        if count < self.min_count {
            return false;
        }
        match self.max_count {
            Some(max) => count <= max,
            None => true,
        }
    }

    /// Return whether every stored (positive) count satisfies this filter.
    #[inline(always)]
    pub(crate) const fn is_all(self) -> bool {
        self.min_count <= 1 && self.max_count.is_none()
    }

    /// Return whether no positive count can satisfy this filter.
    #[inline(always)]
    pub(crate) const fn is_empty(self) -> bool {
        match self.max_count {
            Some(max) => max == 0 || max < self.min_count,
            None => false,
        }
    }
}

/// Reverse complement of a packed `k`-mer (`1 <= k <= 32`).
#[inline(always)]
fn revcomp64(key: u64, k: usize) -> u64 {
    debug_assert!((1..=32).contains(&k));
    let mut x = !key;
    x = ((x & 0x3333_3333_3333_3333) << 2) | ((x >> 2) & 0x3333_3333_3333_3333);
    x = ((x & 0x0f0f_0f0f_0f0f_0f0f) << 4) | ((x >> 4) & 0x0f0f_0f0f_0f0f_0f0f);
    x = x.swap_bytes();
    x >> (64 - 2 * k)
}

#[inline(always)]
fn canonical64(key: u64, k: usize, strand: StrandMode) -> u64 {
    match strand {
        StrandMode::Forward => key,
        StrandMode::Canonical => key.min(revcomp64(key, k)),
    }
}

#[inline]
fn pack_kmer64(kmer: &[u8], strand: StrandMode) -> Option<u64> {
    let mut forward = 0_u64;
    for &byte in kmer {
        let base = DNA_BASES[byte as usize];
        if base == u8::MAX {
            return None;
        }
        forward = (forward << 2) | u64::from(base);
    }
    Some(canonical64(forward, kmer.len(), strand))
}

fn pack_kmer128(kmer: &[u8], strand: StrandMode) -> Option<(u64, u64)> {
    let hi_shift = 2 * (kmer.len() - 1) - 64;
    let (mut f_hi, mut f_lo) = (0_u64, 0_u64);
    let (mut r_hi, mut r_lo) = (0_u64, 0_u64);
    let canonical = strand == StrandMode::Canonical;
    for &byte in kmer {
        let base = DNA_BASES[byte as usize];
        if base == u8::MAX {
            return None;
        }
        let base = u64::from(base);
        f_hi = (f_hi << 2) | (f_lo >> 62);
        f_lo = (f_lo << 2) | base;
        if canonical {
            r_lo = (r_lo >> 2) | (r_hi << 62);
            r_hi = (r_hi >> 2) | ((base ^ 3) << hi_shift);
        }
    }
    if canonical && (r_hi, r_lo) < (f_hi, f_lo) {
        Some((r_hi, r_lo))
    } else {
        Some((f_hi, f_lo))
    }
}

fn pack_kmer192(kmer: &[u8], strand: StrandMode) -> Option<([u64; 2], u64)> {
    let top_shift = 2 * (kmer.len() - 1) - 128;
    let (mut f2, mut f1, mut f0) = (0_u64, 0_u64, 0_u64);
    let (mut r2, mut r1, mut r0) = (0_u64, 0_u64, 0_u64);
    let canonical = strand == StrandMode::Canonical;
    for &byte in kmer {
        let base = DNA_BASES[byte as usize];
        if base == u8::MAX {
            return None;
        }
        let base = u64::from(base);
        f2 = (f2 << 2) | (f1 >> 62);
        f1 = (f1 << 2) | (f0 >> 62);
        f0 = (f0 << 2) | base;
        if canonical {
            r0 = (r0 >> 2) | (r1 << 62);
            r1 = (r1 >> 2) | (r2 << 62);
            r2 = (r2 >> 2) | ((base ^ 3) << top_shift);
        }
    }
    if canonical && (r2, r1, r0) < (f2, f1, f0) {
        Some(([r1, r2], r0))
    } else {
        Some(([f1, f2], f0))
    }
}

fn pack_kmer256(kmer: &[u8], strand: StrandMode) -> Option<([u64; 3], u64)> {
    let top_shift = 2 * (kmer.len() - 1) - 192;
    let (mut f3, mut f2, mut f1, mut f0) = (0_u64, 0_u64, 0_u64, 0_u64);
    let (mut r3, mut r2, mut r1, mut r0) = (0_u64, 0_u64, 0_u64, 0_u64);
    let canonical = strand == StrandMode::Canonical;
    for &byte in kmer {
        let base = DNA_BASES[byte as usize];
        if base == u8::MAX {
            return None;
        }
        let base = u64::from(base);
        f3 = (f3 << 2) | (f2 >> 62);
        f2 = (f2 << 2) | (f1 >> 62);
        f1 = (f1 << 2) | (f0 >> 62);
        f0 = (f0 << 2) | base;
        if canonical {
            r0 = (r0 >> 2) | (r1 << 62);
            r1 = (r1 >> 2) | (r2 << 62);
            r2 = (r2 >> 2) | (r3 << 62);
            r3 = (r3 >> 2) | ((base ^ 3) << top_shift);
        }
    }
    if canonical && (r3, r2, r1, r0) < (f3, f2, f1, f0) {
        Some(([r1, r2, r3], r0))
    } else {
        Some(([f1, f2, f3], f0))
    }
}

fn normalize_bytes<'a>(
    kmer: &[u8],
    strand: StrandMode,
    fwd: &'a mut Vec<u8>,
    rev: &'a mut Vec<u8>,
) -> Option<&'a [u8]> {
    fwd.clear();
    for &byte in kmer {
        let base = DNA_BASES[byte as usize];
        if base == u8::MAX {
            return None;
        }
        fwd.push(b"ACGT"[base as usize]);
    }
    if strand == StrandMode::Canonical {
        rev.clear();
        for &byte in fwd.iter().rev() {
            rev.push(b"TGCA"[DNA_BASES[byte as usize] as usize]);
        }
        if rev.as_slice() < fwd.as_slice() {
            return Some(rev.as_slice());
        }
    }
    Some(fwd.as_slice())
}

fn get_many_packed<'a>(
    counts: &PackedCounts,
    k: usize,
    strand: StrandMode,
    kmers: impl IntoIterator<Item = &'a [u8]>,
) -> Vec<u64> {
    let tables = counts.tables();
    let mut iter = kmers.into_iter();
    let mut out = Vec::with_capacity(iter.size_hint().0);
    let mut routed = [None; QUERY_CHUNK];
    loop {
        let mut len = 0;
        while len < QUERY_CHUNK {
            let Some(kmer) = iter.next() else { break };
            routed[len] = if kmer.len() == k {
                pack_kmer64(kmer, strand).map(|key| counts.route(key))
            } else {
                None
            };
            len += 1;
        }
        if len == 0 {
            break;
        }
        for i in 0..len {
            if let Some(Some((partition, remainder))) = routed.get(i + QUERY_AHEAD).copied() {
                tables[partition].prefetch(remainder);
            }
            let count = match routed[i] {
                Some((partition, remainder)) => tables[partition].get(remainder),
                None => 0,
            };
            out.push(count);
        }
    }
    out
}

fn get_many_wide<'a, H: HighKey>(
    counts: &PackedCountsWide<H>,
    k: usize,
    strand: StrandMode,
    pack: impl Fn(&[u8], StrandMode) -> Option<(H, u64)>,
    kmers: impl IntoIterator<Item = &'a [u8]>,
) -> Vec<u64> {
    let tables = counts.tables();
    let mut iter = kmers.into_iter();
    let mut out = Vec::with_capacity(iter.size_hint().0);
    let mut routed: Vec<Option<(usize, u64, H)>> = Vec::with_capacity(QUERY_CHUNK);
    loop {
        routed.clear();
        for _ in 0..QUERY_CHUNK {
            let Some(kmer) = iter.next() else { break };
            let entry = if kmer.len() == k {
                pack(kmer, strand).map(|(hi, lo)| {
                    let (partition, l_rem) = counts.route(hi, lo);
                    (partition, l_rem, hi)
                })
            } else {
                None
            };
            routed.push(entry);
        }
        if routed.is_empty() {
            break;
        }
        for i in 0..routed.len() {
            if let Some(Some((partition, l_rem, _))) = routed.get(i + QUERY_AHEAD).copied() {
                tables[partition].prefetch(l_rem);
            }
            let count = match routed[i] {
                Some((partition, l_rem, hi)) => tables[partition].get(l_rem, hi),
                None => 0,
            };
            out.push(count);
        }
    }
    out
}

/// Accumulator for a k-mer count histogram.
struct HistAcc {
    direct: Vec<u64>,
    large: HashMap<u64, u64, foldhash::fast::RandomState>,
}

impl HistAcc {
    fn new() -> Self {
        Self {
            direct: vec![0; DIRECT_HISTOGRAM],
            large: HashMap::default(),
        }
    }

    #[inline(always)]
    fn add(&mut self, count: u64) {
        if count < DIRECT_HISTOGRAM as u64 {
            self.direct[count as usize] += 1;
        } else {
            *self.large.entry(count).or_insert(0) += 1;
        }
    }

    fn merge(&mut self, other: Self) {
        for (dst, src) in self.direct.iter_mut().zip(other.direct) {
            *dst += src;
        }
        for (count, freq) in other.large {
            *self.large.entry(count).or_insert(0) += freq;
        }
    }

    fn into_sorted(self) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        for (count, freq) in self.direct.into_iter().enumerate().skip(1) {
            if freq > 0 {
                out.push((count as u64, freq));
            }
        }
        if !self.large.is_empty() {
            let mut large: Vec<(u64, u64)> = self.large.into_iter().collect();
            large.sort_unstable_by_key(|&(count, _)| count);
            out.extend(large);
        }
        out
    }
}

fn parallel_reduce<T: Send>(
    threads: usize,
    items: usize,
    run_slice: impl Fn(usize, usize) -> T + Sync,
    mut combine: impl FnMut(&mut T, T),
) -> Option<T> {
    if items == 0 {
        return None;
    }
    let workers = threads.min(items).max(1);
    if workers == 1 {
        return Some(run_slice(0, items));
    }
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let start = worker * items / workers;
            let end = (worker + 1) * items / workers;
            let run_slice = &run_slice;
            handles.push(scope.spawn(move || run_slice(start, end)));
        }
        let mut total: Option<T> = None;
        for handle in handles {
            let part = handle.join().expect("query worker panicked");
            match &mut total {
                Some(acc) => combine(acc, part),
                None => total = Some(part),
            }
        }
        total
    })
}

/// Entry for a bounded max-heap that keeps the *worst* of the current top-k on
/// top: lower count is worse; on a tie, lexicographically larger key is worse.
struct WorstFirst<K> {
    count: u64,
    key: K,
}

impl<K: Eq> PartialEq for WorstFirst<K> {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.key == other.key
    }
}

impl<K: Eq> Eq for WorstFirst<K> {}

impl<K: Ord> PartialOrd for WorstFirst<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord> Ord for WorstFirst<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .count
            .cmp(&self.count)
            .then_with(|| self.key.cmp(&other.key))
    }
}

/// Wrapper giving wide keys `(H, u64)` their DNA lexicographic `Ord`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct OrderedWide<H: HighKey>(H, u64);

impl<H: HighKey> PartialOrd for OrderedWide<H> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<H: HighKey> Ord for OrderedWide<H> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp_key(self.1, other.0, other.1)
    }
}

fn push_top_k<K: Ord>(
    heap: &mut BinaryHeap<WorstFirst<K>>,
    n: usize,
    count: u64,
    make_key: impl FnOnce() -> K,
) {
    if heap.len() < n {
        heap.push(WorstFirst {
            count,
            key: make_key(),
        });
        return;
    }
    let Some(mut worst) = heap.peek_mut() else {
        return;
    };
    if count < worst.count {
        return;
    }
    let key = make_key();
    if count > worst.count || key < worst.key {
        *worst = WorstFirst { count, key };
    }
}

impl Counter {
    /// Return the count of `kmer`, or `0` if it is absent, has length other
    /// than `self.k()`, or contains any non-ACGT byte.
    ///
    /// A/C/G/T are case-insensitive, and canonical counters automatically
    /// canonicalize the queried k-mer.
    pub fn get(&self, kmer: &[u8]) -> u64 {
        if kmer.len() != self.k {
            return 0;
        }
        match &self.counts {
            Counts::Packed(counts) => {
                let Some(key) = pack_kmer64(kmer, self.strand) else {
                    return 0;
                };
                counts.get(key)
            }
            Counts::Wide(counts) => match counts {
                WideCounts::W80(c) => pack_kmer128(kmer, self.strand)
                    .map_or(0, |(hi, lo)| c.get(hi as u16, lo)),
                WideCounts::W96(c) => pack_kmer128(kmer, self.strand)
                    .map_or(0, |(hi, lo)| c.get(hi as u32, lo)),
                WideCounts::W128(c) => {
                    pack_kmer128(kmer, self.strand).map_or(0, |(hi, lo)| c.get(hi, lo))
                }
                WideCounts::W192(c) => {
                    pack_kmer192(kmer, self.strand).map_or(0, |(hi, lo)| c.get(hi, lo))
                }
                WideCounts::W256(c) => {
                    pack_kmer256(kmer, self.strand).map_or(0, |(hi, lo)| c.get(hi, lo))
                }
            },
            Counts::Bytes(counts) => {
                let mut fwd = Vec::with_capacity(self.k);
                let mut rev = if self.strand == StrandMode::Canonical {
                    Vec::with_capacity(self.k)
                } else {
                    Vec::new()
                };
                let Some(key) = normalize_bytes(kmer, self.strand, &mut fwd, &mut rev) else {
                    return 0;
                };
                counts.map(key).get(key).copied().unwrap_or(0)
            }
        }
    }

    /// Look up many k-mers at once, returning a `Vec<u64>` with one count per
    /// queried k-mer (in input order).
    ///
    /// For `k <= 128`, queries are routed in batches and software-prefetched
    /// ahead of probing to hide memory latency on large tables.
    pub fn get_many<'a>(&self, kmers: impl IntoIterator<Item = &'a [u8]>) -> Vec<u64> {
        match &self.counts {
            Counts::Packed(counts) => get_many_packed(counts, self.k, self.strand, kmers),
            Counts::Wide(counts) => match counts {
                WideCounts::W80(c) => get_many_wide(
                    c,
                    self.k,
                    self.strand,
                    |kmer, strand| pack_kmer128(kmer, strand).map(|(hi, lo)| (hi as u16, lo)),
                    kmers,
                ),
                WideCounts::W96(c) => get_many_wide(
                    c,
                    self.k,
                    self.strand,
                    |kmer, strand| pack_kmer128(kmer, strand).map(|(hi, lo)| (hi as u32, lo)),
                    kmers,
                ),
                WideCounts::W128(c) => get_many_wide(c, self.k, self.strand, pack_kmer128, kmers),
                WideCounts::W192(c) => get_many_wide(c, self.k, self.strand, pack_kmer192, kmers),
                WideCounts::W256(c) => get_many_wide(c, self.k, self.strand, pack_kmer256, kmers),
            },
            Counts::Bytes(counts) => {
                let iter = kmers.into_iter();
                let mut out = Vec::with_capacity(iter.size_hint().0);
                let mut fwd = Vec::with_capacity(self.k);
                let mut rev = if self.strand == StrandMode::Canonical {
                    Vec::with_capacity(self.k)
                } else {
                    Vec::new()
                };
                for kmer in iter {
                    let count = if kmer.len() == self.k {
                        normalize_bytes(kmer, self.strand, &mut fwd, &mut rev)
                            .and_then(|key| counts.map(key).get(key).copied())
                            .unwrap_or(0)
                    } else {
                        0
                    };
                    out.push(count);
                }
                out
            }
        }
    }

    /// Look up a 2-bit packed k-mer (`A=0, C=1, G=2, T=3`, first base most
    /// significant).
    ///
    /// Returns `None` if `self.k() > 32` or if `key` has nonzero bits above
    /// `2 * self.k()`. Canonical counters automatically canonicalize `key`.
    pub fn get_packed(&self, key: u64) -> Option<u64> {
        let Counts::Packed(counts) = &self.counts else {
            return None;
        };
        if key >> 1 >> (2 * self.k - 1) != 0 {
            return None;
        }
        Some(counts.get(canonical64(key, self.k, self.strand)))
    }

    /// Look up a slice of 2-bit packed k-mers with software prefetching.
    ///
    /// Returns `None` if `self.k() > 32`. Keys with nonzero bits above
    /// `2 * self.k()` yield `0`. Canonical counters automatically canonicalize
    /// each key.
    pub fn get_packed_many(&self, keys: &[u64]) -> Option<Vec<u64>> {
        let Counts::Packed(counts) = &self.counts else {
            return None;
        };
        let tables = counts.tables();
        let high_shift = 2 * self.k - 1;
        let mut out = Vec::with_capacity(keys.len());
        let mut routed = [None; QUERY_CHUNK];
        for chunk in keys.chunks(QUERY_CHUNK) {
            for (slot, &key) in routed.iter_mut().zip(chunk) {
                *slot = if key >> 1 >> high_shift == 0 {
                    Some(counts.route(canonical64(key, self.k, self.strand)))
                } else {
                    None
                };
            }
            for i in 0..chunk.len() {
                if let Some(Some((partition, remainder))) = routed.get(i + QUERY_AHEAD).copied() {
                    tables[partition].prefetch(remainder);
                }
                let count = match routed[i] {
                    Some((partition, remainder)) => tables[partition].get(remainder),
                    None => 0,
                };
                out.push(count);
            }
        }
        Some(out)
    }

    /// Compute the k-mer count histogram (spectrum): a sorted list of
    /// `(count, frequency)` pairs giving how many distinct k-mers occurred
    /// `count` times.
    ///
    /// Scans table counts directly without unmixing or decoding keys, using
    /// multiple threads when this counter was built by [`crate::ParallelCounter`].
    pub fn histogram(&self) -> Vec<(u64, u64)> {
        let acc = match &self.counts {
            Counts::Packed(counts) => {
                let tables = counts.tables();
                parallel_reduce(
                    self.output_threads,
                    tables.len(),
                    |start, end| {
                        let mut acc = HistAcc::new();
                        for table in &tables[start..end] {
                            table.for_each_count(|count| acc.add(count));
                        }
                        acc
                    },
                    HistAcc::merge,
                )
            }
            Counts::Wide(counts) => {
                fn hist_wide<H: HighKey>(
                    counts: &PackedCountsWide<H>,
                    threads: usize,
                ) -> Option<HistAcc> {
                    let tables = counts.tables();
                    parallel_reduce(
                        threads,
                        tables.len(),
                        |start, end| {
                            let mut acc = HistAcc::new();
                            for table in &tables[start..end] {
                                table.for_each_count(|count| acc.add(count));
                            }
                            acc
                        },
                        HistAcc::merge,
                    )
                }
                dispatch_wide!(counts, c => hist_wide(c, self.output_threads))
            }
            Counts::Bytes(counts) => {
                let maps = &counts.maps;
                parallel_reduce(
                    self.output_threads,
                    maps.len(),
                    |start, end| {
                        let mut acc = HistAcc::new();
                        for map in &maps[start..end] {
                            for &count in map.values() {
                                acc.add(count);
                            }
                        }
                        acc
                    },
                    HistAcc::merge,
                )
            }
        };
        acc.map_or_else(Vec::new, HistAcc::into_sorted)
    }

    /// Return the number of distinct k-mers whose count satisfies `filter`.
    ///
    /// Scans counts directly without unmixing or decoding keys.
    pub fn len_filtered(&self, filter: CountFilter) -> usize {
        if filter.is_all() {
            return self.len();
        }
        if filter.is_empty() || self.is_empty() {
            return 0;
        }
        let total = match &self.counts {
            Counts::Packed(counts) => {
                let tables = counts.tables();
                parallel_reduce(
                    self.output_threads,
                    tables.len(),
                    |start, end| {
                        let mut matched = 0_usize;
                        for table in &tables[start..end] {
                            table.for_each_count(|count| {
                                matched += usize::from(filter.matches(count));
                            });
                        }
                        matched
                    },
                    |acc, part| *acc += part,
                )
            }
            Counts::Wide(counts) => {
                fn count_wide<H: HighKey>(
                    counts: &PackedCountsWide<H>,
                    threads: usize,
                    filter: CountFilter,
                ) -> Option<usize> {
                    let tables = counts.tables();
                    parallel_reduce(
                        threads,
                        tables.len(),
                        |start, end| {
                            let mut matched = 0_usize;
                            for table in &tables[start..end] {
                                table.for_each_count(|count| {
                                    matched += usize::from(filter.matches(count));
                                });
                            }
                            matched
                        },
                        |acc, part| *acc += part,
                    )
                }
                dispatch_wide!(counts, c => count_wide(c, self.output_threads, filter))
            }
            Counts::Bytes(counts) => {
                let maps = &counts.maps;
                parallel_reduce(
                    self.output_threads,
                    maps.len(),
                    |start, end| {
                        let mut matched = 0_usize;
                        for map in &maps[start..end] {
                            for &count in map.values() {
                                matched += usize::from(filter.matches(count));
                            }
                        }
                        matched
                    },
                    |acc, part| *acc += part,
                )
            }
        };
        total.unwrap_or(0)
    }

    /// Return up to `n` most abundant k-mers, ordered by count descending with
    /// ties broken alphabetically by k-mer (`A < C < G < T`).
    ///
    /// Keys below the current `n`-th largest count are skipped without
    /// unmixing or decoding.
    pub fn top_k(&self, n: usize) -> Vec<(Vec<u8>, u64)> {
        if n == 0 || self.is_empty() {
            return Vec::new();
        }
        let k = self.k;
        match &self.counts {
            Counts::Packed(counts) => {
                let mut heap: BinaryHeap<WorstFirst<u64>> = BinaryHeap::new();
                for (partition, table) in counts.tables().iter().enumerate() {
                    for (remainder, count) in table.iter() {
                        push_top_k(&mut heap, n, count, || counts.key(partition, remainder));
                    }
                }
                let mut result: Vec<(u64, u64)> = heap
                    .into_iter()
                    .map(|entry| (entry.key, entry.count))
                    .collect();
                result.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                result
                    .into_iter()
                    .map(|(key, count)| (decode_packed(key, k), count))
                    .collect()
            }
            Counts::Wide(counts) => {
                fn top_wide<H: HighKey>(
                    counts: &PackedCountsWide<H>,
                    k: usize,
                    n: usize,
                ) -> Vec<(Vec<u8>, u64)> {
                    let mut heap: BinaryHeap<WorstFirst<OrderedWide<H>>> = BinaryHeap::new();
                    for (partition, table) in counts.tables().iter().enumerate() {
                        for (l_rem, h, count) in table.iter() {
                            push_top_k(&mut heap, n, count, || {
                                let (hi, lo) = counts.key(partition, l_rem, h);
                                OrderedWide(hi, lo)
                            });
                        }
                    }
                    let mut result: Vec<(OrderedWide<H>, u64)> = heap
                        .into_iter()
                        .map(|entry| (entry.key, entry.count))
                        .collect();
                    result.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                    result
                        .into_iter()
                        .map(|(OrderedWide(hi, lo), count)| {
                            let mut dna = vec![0; k];
                            hi.decode_into(lo, &mut dna);
                            (dna, count)
                        })
                        .collect()
                }
                dispatch_wide!(counts, c => top_wide(c, k, n))
            }
            Counts::Bytes(counts) => {
                let mut heap: BinaryHeap<WorstFirst<&[u8]>> = BinaryHeap::new();
                for map in &counts.maps {
                    for (key, &count) in map {
                        push_top_k(&mut heap, n, count, || key.as_slice());
                    }
                }
                let mut result: Vec<(&[u8], u64)> = heap
                    .into_iter()
                    .map(|entry| (entry.key, entry.count))
                    .collect();
                result.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
                result
                    .into_iter()
                    .map(|(key, count)| (key.to_vec(), count))
                    .collect()
            }
        }
    }

    /// Iterate over distinct uppercase DNA keys and counts matching `filter`,
    /// in unspecified order.
    ///
    /// Non-matching entries are skipped before unmixing or decoding their keys.
    pub fn counts_filtered(
        &self,
        filter: CountFilter,
    ) -> impl Iterator<Item = (Vec<u8>, u64)> + '_ {
        let iter: Box<dyn Iterator<Item = (Vec<u8>, u64)> + '_> = if filter.is_empty() {
            Box::new(std::iter::empty())
        } else {
            match &self.counts {
                Counts::Packed(counts) => Box::new(
                    counts
                        .filtered_entries(move |count| filter.matches(count))
                        .map(|(key, count)| (decode_packed(key, self.k), count)),
                ),
                Counts::Wide(counts) => {
                    fn decode_filtered<H: HighKey>(
                        counts: &PackedCountsWide<H>,
                        k: usize,
                        filter: CountFilter,
                    ) -> Box<dyn Iterator<Item = (Vec<u8>, u64)> + '_> {
                        Box::new(
                            counts
                                .filtered_entries(move |count| filter.matches(count))
                                .map(move |((hi, lo), count)| {
                                    let mut dna = vec![0; k];
                                    hi.decode_into(lo, &mut dna);
                                    (dna, count)
                                }),
                        )
                    }
                    dispatch_wide!(counts, c => decode_filtered(c, self.k, filter))
                }
                Counts::Bytes(counts) => Box::new(counts.maps.iter().flat_map(move |map| {
                    map.iter()
                        .filter(move |&(_, &count)| filter.matches(count))
                        .map(|(key, &count)| (key.clone(), count))
                })),
            }
        };
        iter
    }

    /// Iterate over raw 2-bit packed `(key, count)` pairs (`A=0, C=1, G=2, T=3`,
    /// first base most significant) without ASCII decoding or per-key
    /// allocation.
    ///
    /// Returns `None` if `self.k() > 32`.
    pub fn packed_counts(&self) -> Option<impl Iterator<Item = (u64, u64)> + '_> {
        let Counts::Packed(counts) = &self.counts else {
            return None;
        };
        Some(counts.entries())
    }

    /// Iterate over raw 2-bit packed `(key, count)` pairs whose counts match
    /// `filter`, checking counts before unmixing keys.
    ///
    /// Returns `None` if `self.k() > 32`.
    pub fn packed_counts_filtered(
        &self,
        filter: CountFilter,
    ) -> Option<impl Iterator<Item = (u64, u64)> + '_> {
        let Counts::Packed(counts) = &self.counts else {
            return None;
        };
        Some(counts.filtered_entries(move |count| filter.matches(count)))
    }

    /// Write headerless `KMER<TAB>COUNT` rows for k-mers whose counts match
    /// `filter`. Behavior and ordering otherwise match [`Counter::write_tsv`].
    pub fn write_tsv_filtered(
        &self,
        filter: CountFilter,
        mut writer: impl Write,
    ) -> io::Result<()> {
        if filter.is_all() {
            return self.write_tsv(writer);
        }
        if filter.is_empty() || self.is_empty() {
            return Ok(());
        }
        self.write_tables(&mut writer, 1, |table, writer| match table {
            OutputTable::Packed(counts, first, tables) => {
                let mut slots = Vec::new();
                for partition in first..first + tables {
                    let entries = counts
                        .sorted_partition(partition, &mut slots)
                        .filter(|&(_, count)| filter.matches(count))
                        .map(|(mixed, count)| (counts.unmix(mixed), count));
                    write_packed64_tsv(entries, self.k, writer)?;
                }
                Ok(())
            }
            OutputTable::Wide(counts, first, tables) => {
                fn encode_wide<H: HighKey>(
                    counts: &PackedCountsWide<H>,
                    first: usize,
                    tables: usize,
                    k: usize,
                    filter: CountFilter,
                    writer: &mut dyn Write,
                ) -> io::Result<()> {
                    let entries = (first..first + tables).flat_map(|partition| {
                        counts.tables()[partition]
                            .iter()
                            .filter(move |&(_, _, count)| filter.matches(count))
                            .map(move |(l_rem, h, count)| (counts.key(partition, l_rem, h), count))
                    });
                    write_packed_tsv(entries, k, writer, |(hi, lo), dna| hi.decode_into(lo, dna))
                }
                dispatch_wide!(counts, c => encode_wide(c, first, tables, self.k, filter, writer))
            }
            OutputTable::Bytes(counts) => {
                let mut suffix = [0; 22];
                for (key, &count) in counts {
                    if filter.matches(count) {
                        writer.write_all(key)?;
                        writer.write_all(format_count(count, &mut suffix))?;
                    }
                }
                Ok(())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{ParallelCounter, database::DatabaseReader};

    use super::*;

    const TEST_KS: &[usize] = &[1, 3, 4, 31, 32, 33, 40, 41, 48, 49, 64, 65, 96, 97, 128, 129];

    fn revcomp_bytes(kmer: &[u8]) -> Vec<u8> {
        kmer.iter()
            .rev()
            .map(|&b| match b {
                b'A' | b'a' => b'T',
                b'C' | b'c' => b'G',
                b'G' | b'g' => b'C',
                b'T' | b't' => b'A',
                _ => unreachable!(),
            })
            .collect()
    }

    #[test]
    fn revcomp64_matches_byte_revcomp_for_all_lengths() {
        let mut state = 0x1234_5678_9abc_def1_u64;
        for k in 1..=32 {
            let mask = u64::MAX >> (64 - 2 * k);
            for _ in 0..128 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let key = state & mask;
                let dna = decode_packed(key, k);
                let rc_dna = revcomp_bytes(&dna);
                let expected = pack_kmer64(&rc_dna, StrandMode::Forward).unwrap();
                assert_eq!(revcomp64(key, k), expected, "k={k}");
            }
        }
    }

    #[test]
    fn get_and_get_many_match_all_backends_strands_and_edge_cases() {
        let mut state = 99_u64;
        let record: Vec<u8> = (0..600)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                b"ACGTacgt"[(state % 8) as usize]
            })
            .collect();

        for &k in TEST_KS {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                // Empty counter returns 0 for every query without crashing.
                let empty = Counter::new(k, strand).unwrap();
                assert_eq!(empty.get(&vec![b'A'; k]), 0);
                assert_eq!(empty.get_many([vec![b'A'; k].as_slice()]), vec![0]);

                let mut counter = Counter::new(k, strand).unwrap();
                counter.add_record(&record).unwrap();
                counter.add_record(&record[..record.len() / 2]).unwrap();
                let expected: HashMap<Vec<u8>, u64> = counter.counts().collect();

                // Build queries: present keys (upper + lower), reverse complements,
                // absent keys, wrong lengths, and invalid bases.
                let mut queries: Vec<Vec<u8>> = Vec::new();
                let mut expected_counts: Vec<u64> = Vec::new();
                for (key, &count) in expected.iter().take(64) {
                    queries.push(key.clone());
                    expected_counts.push(count);

                    let lower: Vec<u8> = key.iter().map(u8::to_ascii_lowercase).collect();
                    queries.push(lower);
                    expected_counts.push(count);

                    let rc = revcomp_bytes(key);
                    let rc_expected = match strand {
                        StrandMode::Canonical => count,
                        StrandMode::Forward => expected.get(&rc).copied().unwrap_or(0),
                    };
                    queries.push(rc);
                    expected_counts.push(rc_expected);
                }

                // Absent, wrong length, and invalid-base queries.
                let absent = vec![b'A'; k];
                let absent_expected = expected.get(&absent).copied().unwrap_or(0);
                queries.push(absent);
                expected_counts.push(absent_expected);

                let mut with_n = vec![b'A'; k];
                with_n[k / 2] = b'N';
                queries.push(with_n);
                expected_counts.push(0);

                queries.push(Vec::new());
                expected_counts.push(0);

                if k > 1 {
                    queries.push(vec![b'A'; k - 1]);
                    expected_counts.push(0);
                }
                queries.push(vec![b'A'; k + 1]);
                expected_counts.push(0);

                for (query, &exp) in queries.iter().zip(&expected_counts) {
                    assert_eq!(counter.get(query), exp, "k={k}, strand={strand:?}");
                }
                let batch = counter.get_many(queries.iter().map(Vec::as_slice));
                assert_eq!(batch, expected_counts, "batch k={k}, strand={strand:?}");

                // Packed queries for k <= 32, None for k > 32.
                if k <= 32 {
                    let mut packed_keys = Vec::new();
                    let mut packed_expected = Vec::new();
                    for (key, &count) in expected.iter().take(64) {
                        let fwd = pack_kmer64(key, StrandMode::Forward).unwrap();
                        packed_keys.push(fwd);
                        packed_expected.push(count);
                        assert_eq!(counter.get_packed(fwd), Some(count));
                        if strand == StrandMode::Canonical {
                            let rc = revcomp64(fwd, k);
                            packed_keys.push(rc);
                            packed_expected.push(count);
                            assert_eq!(counter.get_packed(rc), Some(count));
                        }
                    }
                    if k < 32 {
                        let out_of_range = 1_u64 << (2 * k);
                        assert_eq!(counter.get_packed(out_of_range), None);
                        packed_keys.push(out_of_range);
                        packed_expected.push(0);
                    }
                    assert_eq!(counter.get_packed_many(&packed_keys), Some(packed_expected));
                } else {
                    assert_eq!(counter.get_packed(0), None);
                    assert_eq!(counter.get_packed_many(&[0]), None);
                }
            }
        }
    }

    #[test]
    fn get_handles_overflow_counts_across_backends() {
        for k in [3, 32, 33, 49, 65, 97, 129] {
            let mut counter = Counter::new(k, StrandMode::Forward).unwrap();
            let kmer = vec![b'C'; k];
            counter.add_record(&kmer).unwrap();
            counter.counts.set_counts(100_000);
            assert_eq!(counter.get(&kmer), 100_000);
            assert_eq!(counter.get_many([kmer.as_slice()]), vec![100_000]);
            if k <= 32 {
                let packed = pack_kmer64(&kmer, StrandMode::Forward).unwrap();
                assert_eq!(counter.get_packed(packed), Some(100_000));
            }
        }
    }

    #[test]
    fn histogram_top_k_and_filters_match_reference_for_all_backends() {
        let mut sequence = Vec::new();
        // Repeat segments different numbers of times so we get varied counts
        // including counts above DIRECT_HISTOGRAM.
        let motifs = [
            b"ACGTACGTACGTACGT".repeat(10),
            b"TGCATGCATGCATGCA".repeat(10),
            b"AAAACCCCGGGGTTTT".repeat(10),
        ];
        for (i, motif) in motifs.iter().enumerate() {
            for _ in 0..=(i + 1) * 3 {
                sequence.push(motif.clone());
            }
        }
        // Add a high-count homopolymer exceeding DIRECT_HISTOGRAM.
        sequence.push(vec![b'A'; DIRECT_HISTOGRAM + 200]);

        for &k in &[3, 31, 32, 33, 49, 65, 97, 129] {
            for threads in [1, 3] {
                let counter = if threads == 1 {
                    let mut c = Counter::new(k, StrandMode::Canonical).unwrap();
                    for rec in &sequence {
                        c.add_record(rec).unwrap();
                    }
                    c
                } else {
                    let mut p = ParallelCounter::new(k, StrandMode::Canonical, threads).unwrap();
                    for rec in &sequence {
                        p.add_record(rec).unwrap();
                    }
                    p.finish().unwrap()
                };

                let all: BTreeMap<Vec<u8>, u64> = counter.counts().collect();

                // Histogram check.
                let mut expected_hist: BTreeMap<u64, u64> = BTreeMap::new();
                for &count in all.values() {
                    *expected_hist.entry(count).or_insert(0) += 1;
                }
                let expected_hist: Vec<(u64, u64)> = expected_hist.into_iter().collect();
                assert_eq!(counter.histogram(), expected_hist, "k={k}, threads={threads}");

                // Top-k check (including 0, partial, and > len).
                assert!(counter.top_k(0).is_empty());
                let mut sorted_all: Vec<(Vec<u8>, u64)> =
                    all.iter().map(|(key, &c)| (key.clone(), c)).collect();
                sorted_all.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                for n in [1, 5, 20, all.len() + 5] {
                    let expected_top: Vec<_> = sorted_all.iter().take(n).cloned().collect();
                    assert_eq!(counter.top_k(n), expected_top, "top_k({n}) k={k}");
                }

                // Filter checks: len_filtered, counts_filtered, packed_counts_filtered,
                // write_tsv_filtered, write_binary_filtered.
                for filter in [
                    CountFilter::default(),
                    CountFilter::at_least(5),
                    CountFilter::between(2, 10),
                    CountFilter::between(10, 5), // empty range
                    CountFilter::between(0, 0),  // empty range
                    CountFilter::at_least(u64::MAX),
                ] {
                    let expected_filtered: BTreeMap<Vec<u8>, u64> = all
                        .iter()
                        .filter(|&(_, &c)| filter.matches(c))
                        .map(|(key, &c)| (key.clone(), c))
                        .collect();

                    assert_eq!(counter.len_filtered(filter), expected_filtered.len());
                    assert_eq!(
                        counter.counts_filtered(filter).collect::<BTreeMap<_, _>>(),
                        expected_filtered
                    );

                    if k <= 32 {
                        let packed: BTreeMap<Vec<u8>, u64> = counter
                            .packed_counts_filtered(filter)
                            .unwrap()
                            .map(|(key, count)| (decode_packed(key, k), count))
                            .collect();
                        assert_eq!(packed, expected_filtered);
                    } else {
                        assert!(counter.packed_counts().is_none());
                        assert!(counter.packed_counts_filtered(filter).is_none());
                    }

                    // write_tsv_filtered
                    let mut tsv = Vec::new();
                    counter.write_tsv_filtered(filter, &mut tsv).unwrap();
                    let parsed_tsv: BTreeMap<Vec<u8>, u64> = String::from_utf8(tsv)
                        .unwrap()
                        .lines()
                        .map(|line| {
                            let (key, count) = line.split_once('\t').unwrap();
                            (key.as_bytes().to_vec(), count.parse().unwrap())
                        })
                        .collect();
                    assert_eq!(parsed_tsv, expected_filtered, "tsv filter k={k}");

                    // write_binary_filtered
                    let mut bin = Vec::new();
                    counter.write_binary_filtered(filter, &mut bin).unwrap();
                    let mut reader = DatabaseReader::new(bin.as_slice()).unwrap();
                    assert_eq!(reader.header().entries, expected_filtered.len() as u64);
                    let mut parsed_bin = BTreeMap::new();
                    while let Some((key, count)) = reader.next_entry().unwrap() {
                        parsed_bin.insert(key.to_vec(), count);
                    }
                    assert_eq!(parsed_bin, expected_filtered, "bin filter k={k}");
                }
            }
        }
    }
}
