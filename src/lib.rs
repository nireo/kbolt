use std::collections::HashMap;
use std::fmt;

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
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidK => f.write_str("k must be greater than zero"),
            Self::CountOverflow => f.write_str("k-mer count exceeds u64::MAX"),
        }
    }
}

impl std::error::Error for Error {}

/// An exact, in-memory DNA k-mer counter.
///
/// Keys are uppercase ASCII DNA sequences and counts are cumulative across records.
/// There is no maximum k other than practical memory limits.
pub struct Counter {
    counts: HashMap<Vec<u8>, u64>,
    strand: StrandMode,
    k: usize,
}

impl Counter {
    /// Create an empty counter. Reject k = 0 with `Error::InvalidK`.
    pub fn new(k: usize, strand: StrandMode) -> Result<Self, Error> {
        if k == 0 {
            return Err(Error::InvalidK);
        }

        Ok(Self {
            counts: HashMap::new(),
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

        let sequence = sequence.to_ascii_uppercase();
        let mut reverse_complement = if self.strand == StrandMode::Canonical {
            vec![0; self.k]
        } else {
            Vec::new()
        };

        for kmer in sequence.windows(self.k) {
            if !kmer
                .iter()
                .all(|base| matches!(base, b'A' | b'C' | b'G' | b'T'))
            {
                continue;
            }

            let key = if self.strand == StrandMode::Canonical {
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
            if let Some(count) = self.counts.get_mut(key) {
                *count = count.checked_add(1).ok_or(Error::CountOverflow)?;
            } else {
                self.counts.insert(key.to_vec(), 1);
            }
        }

        Ok(())
    }

    /// Return all distinct keys and their counts. Iteration order is unspecified.
    pub fn counts(&self) -> &HashMap<Vec<u8>, u64> {
        &self.counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_counts(counter: &Counter, expected: &[(&str, u64)]) {
        let expected: HashMap<Vec<u8>, u64> = expected
            .iter()
            .map(|(key, count)| (key.as_bytes().to_vec(), *count))
            .collect();
        assert_eq!(counter.counts(), &expected);
    }

    #[test]
    fn matches_naive_counts_on_mixed_records() {
        let mut state = 42_u64;
        let records: Vec<Vec<u8>> = (0..32)
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

        for k in 1..=8 {
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
                    assert_eq!(counter.counts(), &expected);
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
        counter.counts.insert(b"ACG".to_vec(), u64::MAX - 1);
        counter.add_record(b"ACG").unwrap();
        assert_eq!(counter.add_record(b"ACG"), Err(Error::CountOverflow));
        assert_counts(&counter, &[("ACG", u64::MAX)]);
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
        assert_counts(&counter, &[(&"A".repeat(65), 2)]);
    }
}
