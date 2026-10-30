use std::collections::HashMap;
use std::fmt;
use std::io::{self, Write};

pub mod database;

type PackedCounts = HashMap<u64, u64, foldhash::fast::RandomState>;

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
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidK => f.write_str("k must be greater than zero"),
            Self::CountOverflow => f.write_str("k-mer count exceeds u64::MAX"),
            Self::ReserveFailed => f.write_str("unable to reserve k-mer table capacity"),
        }
    }
}

impl std::error::Error for Error {}

// Keep the representation private so wider packed keys can be added without
// changing the public output format.
enum Counts {
    Packed(PackedCounts),
    Bytes(HashMap<Vec<u8>, u64>),
}

/// An exact, in-memory DNA k-mer counter.
///
/// Output keys are uppercase ASCII DNA sequences; counts accumulate across records.
/// For k <= 32, counting uses rolling 2-bit keys without allocating individual DNA keys.
/// Larger k uses byte-string storage, with no maximum other than practical memory limits.
/// Packed-key hashing uses noncryptographic FoldHash, not intended to resist adaptive
/// collision attacks.
pub struct Counter {
    counts: Counts,
    strand: StrandMode,
    k: usize,
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
                let mut counts = PackedCounts::default();
                counts
                    .try_reserve(capacity)
                    .map_err(|_| Error::ReserveFailed)?;
                Counts::Packed(counts)
            } else {
                let mut counts = HashMap::new();
                counts
                    .try_reserve(capacity)
                    .map_err(|_| Error::ReserveFailed)?;
                Counts::Bytes(counts)
            },
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

        match &mut self.counts {
            Counts::Packed(counts) => match self.strand {
                StrandMode::Forward => add_packed_record::<false>(counts, sequence, self.k),
                StrandMode::Canonical => add_packed_record::<true>(counts, sequence, self.k),
            },
            Counts::Bytes(counts) => add_byte_record(counts, sequence, self.k, self.strand),
        }
    }

    /// Return the number of distinct k-mers counted so far.
    pub fn len(&self) -> usize {
        match &self.counts {
            Counts::Packed(counts) => counts.len(),
            Counts::Bytes(counts) => counts.len(),
        }
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
        // A u64 has at most 20 decimal digits, plus a tab and newline.
        let mut suffix = [0; 22];
        match &self.counts {
            Counts::Packed(counts) => {
                let mut row = [0; 32 + 22];
                for (&key, &count) in counts {
                    decode_packed_into(key, &mut row[..self.k]);
                    let suffix = format_count(count, &mut suffix);
                    let end = self.k + suffix.len();
                    row[self.k..end].copy_from_slice(suffix);
                    writer.write_all(&row[..end])?;
                }
            }
            Counts::Bytes(counts) => {
                for (key, &count) in counts {
                    writer.write_all(key)?;
                    writer.write_all(format_count(count, &mut suffix))?;
                }
            }
        }
        Ok(())
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
        let (packed, bytes) = match &self.counts {
            Counts::Packed(counts) => (Some(counts.iter()), None),
            Counts::Bytes(counts) => (None, Some(counts.iter())),
        };
        packed
            .into_iter()
            .flatten()
            .map(|(&key, &count)| (decode_packed(key, self.k), count))
            .chain(
                bytes
                    .into_iter()
                    .flatten()
                    .map(|(key, &count)| (key.clone(), count)),
            )
    }
}

// Specialize by strand once per record, leaving no reverse-complement work in
// the forward-only loop. Callers guarantee 1 <= k <= 32.
fn add_packed_record<const CANONICAL: bool>(
    counts: &mut PackedCounts,
    sequence: &[u8],
    k: usize,
) -> Result<(), Error> {
    // Right-shifting an all-ones mask also handles k = 32 without a shift by 64.
    let mask = u64::MAX >> (64 - 2 * k);
    let high_shift = 2 * (k - 1);
    let (mut forward, mut reverse) = (0_u64, 0_u64);
    let mut valid_run = 0;

    for &byte in sequence {
        let base = match byte {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' => 3,
            _ => {
                forward = 0;
                reverse = 0;
                valid_run = 0;
                continue;
            }
        };
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
        let count = counts.entry(key).or_insert(0);
        *count = count.checked_add(1).ok_or(Error::CountOverflow)?;
    }
    Ok(())
}

fn decode_packed(key: u64, k: usize) -> Vec<u8> {
    let mut dna = vec![0; k];
    decode_packed_into(key, &mut dna);
    dna
}

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

// Existing arbitrary-k implementation; a multiword packed backend can replace
// this independently of the single-word counting loop and public iterator.
fn add_byte_record(
    counts: &mut HashMap<Vec<u8>, u64>,
    sequence: &[u8],
    k: usize,
    strand: StrandMode,
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

        // Existing keys can be looked up by a borrowed slice, without allocating.
        if let Some(count) = counts.get_mut(key) {
            *count = count.checked_add(1).ok_or(Error::CountOverflow)?;
        } else {
            counts.insert(key.to_vec(), 1);
        }
    }
    Ok(())
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
    fn capacity_hint_preserves_counts_and_allows_growth() {
        for k in [1, 3, 31, 32, 33, 65] {
            for strand in [StrandMode::Canonical, StrandMode::Forward] {
                let mut expected = Counter::new(k, strand).unwrap();
                let mut reserved = Counter::with_capacity(k, strand, 1).unwrap();
                assert!(reserved.is_empty());
                let capacity = match &reserved.counts {
                    Counts::Packed(counts) => counts.capacity(),
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
        for k in [31, 33] {
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
        for k in [1, 32, 33, 65] {
            let mut counter = Counter::new(k, StrandMode::Forward).unwrap();
            counter.add_record(&vec![b'A'; k]).unwrap();
            let mut values = vec![0, u64::MAX];
            for exponent in 0..=19 {
                let power = 10_u64.pow(exponent);
                values.extend([power - 1, power, power + 1]);
            }
            for value in values {
                match &mut counter.counts {
                    Counts::Packed(counts) => *counts.values_mut().next().unwrap() = value,
                    Counts::Bytes(counts) => *counts.values_mut().next().unwrap() = value,
                }
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
        let Counts::Packed(counts) = &mut counter.counts else {
            panic!("expected packed backend");
        };
        *counts.values_mut().next().unwrap() = u64::MAX - 1;
        counter.add_record(b"ACG").unwrap();
        assert_eq!(counter.add_record(b"ACG"), Err(Error::CountOverflow));
        assert_counts(&counter, &[("ACG", u64::MAX)]);
    }

    #[test]
    fn overflow_keeps_earlier_windows_and_stops_before_later_ones() {
        for k in [1, 3, 32, 33, 65] {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let mut counter = Counter::new(k, strand).unwrap();
                counter.add_record(&vec![b'A'; k]).unwrap();
                match &mut counter.counts {
                    Counts::Packed(counts) => *counts.values_mut().next().unwrap() = u64::MAX,
                    Counts::Bytes(counts) => *counts.values_mut().next().unwrap() = u64::MAX,
                }
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
