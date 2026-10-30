//! kbolt database format, version 1. All integers are little-endian.
//!
//! The 32-byte header is: magic `KBOLTDB\0` (8 bytes), version (u16, 1),
//! flags (u8: bit 0 = canonical), reserved zeros (5 bytes), k (u64), and the
//! number of distinct entries (u64). Unknown versions, flags, or reserved bytes
//! are rejected. Records follow in unspecified order, with no index or sorting.
//!
//! Each record contains ceil(k/4) key bytes and a u64 count. The key is a
//! little-endian base-4 integer: A=0, C=1, G=2, T=3, with the first DNA base
//! most significant and unused high bits zero. Keys with k <= 64 are written
//! directly from the counter's packed integers. Counts must be positive;
//! keys must be unique and canonical when flagged. No trailing data is allowed.
//! There is no checksum; structural validation does not detect every bit flip.
//!
//! [`DatabaseReader`] streams entries with reusable buffers. It validates the
//! header, records, and end of stream, but does not track duplicate keys.
//! [`Counter::read_binary`] additionally rejects duplicates while loading.
//! Wrap files in buffered readers/writers; writing does not flush the caller's
//! writer. Finish reading the stream to validate its declared length.

use std::io::{self, Read, Write};

use crate::{Counter, Counts, StrandMode, format_count};

const MAGIC: &[u8; 8] = b"KBOLTDB\0";
const VERSION: u16 = 1;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn buffer(size: usize) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(io::Error::other)?;
    bytes.resize(size, 0);
    Ok(bytes)
}

/// Metadata describing a kbolt count database.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatabaseHeader {
    pub k: usize,
    pub strand: StrandMode,
    pub entries: u64,
}

/// A streaming database reader. Memory use is proportional to k, not entry count.
pub struct DatabaseReader<R> {
    reader: R,
    header: DatabaseHeader,
    remaining: u64,
    record: Vec<u8>,
    sequence: Vec<u8>,
    finished: bool,
}

impl<R: Read> DatabaseReader<R> {
    /// Read and validate the header. No count table is allocated.
    pub fn new(mut reader: R) -> io::Result<Self> {
        let mut bytes = [0; 32];
        reader.read_exact(&mut bytes)?;
        if &bytes[..8] != MAGIC {
            return Err(invalid("expected a kbolt binary database"));
        }
        if u16::from_le_bytes(bytes[8..10].try_into().unwrap()) != VERSION {
            return Err(invalid("unsupported kbolt database version"));
        }
        if bytes[10] > 1 || bytes[11..16] != [0; 5] {
            return Err(invalid("unsupported database flags or reserved bytes"));
        }
        let k = usize::try_from(u64::from_le_bytes(bytes[16..24].try_into().unwrap()))
            .map_err(|_| invalid("database k does not fit this platform"))?;
        if k == 0 {
            return Err(invalid("database k must be greater than zero"));
        }
        let entries = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let size = k
            .div_ceil(4)
            .checked_add(8)
            .ok_or_else(|| invalid("database record size overflow"))?;
        // Check the declared payload size without reserving the entry count.
        entries
            .checked_mul(size as u64)
            .ok_or_else(|| invalid("database payload size overflow"))?;
        let header = DatabaseHeader {
            k,
            strand: if bytes[10] == 1 {
                StrandMode::Canonical
            } else {
                StrandMode::Forward
            },
            entries,
        };
        Ok(Self {
            reader,
            header,
            remaining: entries,
            record: if entries == 0 {
                Vec::new()
            } else {
                buffer(size)?
            },
            sequence: if entries == 0 { Vec::new() } else { buffer(k)? },
            finished: false,
        })
    }

    pub fn header(&self) -> DatabaseHeader {
        self.header
    }

    /// Read one uppercase DNA key and its count, borrowing a reusable buffer.
    /// Returns None only after checking for trailing data. Discard on any error.
    /// Duplicate detection is left to consumers that build a count table.
    pub fn next_entry(&mut self) -> io::Result<Option<(&[u8], u64)>> {
        if self.finished {
            return Ok(None);
        }
        if self.remaining == 0 {
            let mut byte = [0];
            match self.reader.read_exact(&mut byte) {
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                    self.finished = true;
                    return Ok(None);
                }
                Err(error) => return Err(error),
                Ok(()) => return Err(invalid("trailing data after database entries")),
            }
        }
        self.reader.read_exact(&mut self.record)?;
        let key_bytes = self.record.len() - 8;
        let used = (self.header.k % 4) * 2;
        if used != 0 && self.record[key_bytes - 1] >> used != 0 {
            return Err(invalid("nonzero padding in database key"));
        }
        let count = u64::from_le_bytes(self.record[key_bytes..].try_into().unwrap());
        if count == 0 {
            return Err(invalid("database count must be greater than zero"));
        }
        for (index, base) in self.sequence.iter_mut().rev().enumerate() {
            *base = b"ACGT"[((self.record[index / 4] >> (2 * (index % 4))) & 3) as usize];
        }
        if self.header.strand == StrandMode::Canonical {
            let reverse = self.sequence.iter().rev().map(|&base| match base {
                b'A' => b'T',
                b'C' => b'G',
                b'G' => b'C',
                b'T' => b'A',
                _ => unreachable!(),
            });
            if self.sequence.iter().copied().cmp(reverse).is_gt() {
                return Err(invalid("noncanonical key in canonical database"));
            }
        }
        self.remaining -= 1;
        Ok(Some((&self.sequence, count)))
    }

    /// Export all entries as TSV without loading a count table. Does not flush.
    /// Invalid input or write failures can leave partial output.
    pub fn write_tsv(mut self, mut writer: impl Write) -> io::Result<()> {
        let mut suffix = [0; 22];
        while let Some((key, count)) = self.next_entry()? {
            writer.write_all(key)?;
            writer.write_all(format_count(count, &mut suffix))?;
        }
        Ok(())
    }
}

impl Counter {
    /// Write a version-1 database with exact counts and no per-entry allocation.
    /// Output is unsorted. Does not flush; write failures may leave partial output.
    pub fn write_binary(&self, mut writer: impl Write) -> io::Result<()> {
        let mut header = [0; 32];
        header[..8].copy_from_slice(MAGIC);
        header[8..10].copy_from_slice(&VERSION.to_le_bytes());
        header[10] = u8::from(self.strand == StrandMode::Canonical);
        header[16..24].copy_from_slice(&(self.k as u64).to_le_bytes());
        header[24..32].copy_from_slice(&(self.len() as u64).to_le_bytes());
        writer.write_all(&header)?;
        if self.is_empty() {
            return Ok(());
        }
        let key_bytes = self.k.div_ceil(4);
        match &self.counts {
            Counts::Packed(counts) => {
                write_packed_binary(
                    counts
                        .iter()
                        .map(|(&key, &count)| (key.to_le_bytes(), count)),
                    key_bytes,
                    &mut writer,
                )?;
            }
            Counts::Packed128(counts) => {
                write_packed_binary(
                    counts.iter().map(|(&(high, low), &count)| {
                        let key = ((high as u128) << 64) | low as u128;
                        (key.to_le_bytes(), count)
                    }),
                    key_bytes,
                    &mut writer,
                )?;
            }
            Counts::Bytes(counts) => {
                let mut record = buffer(key_bytes + 8)?;
                for (key, &count) in counts {
                    record[..key_bytes].fill(0);
                    for (index, base) in key.iter().rev().enumerate() {
                        let value = crate::DNA_BASES[*base as usize];
                        record[index / 4] |= value << (2 * (index % 4));
                    }
                    record[key_bytes..].copy_from_slice(&count.to_le_bytes());
                    writer.write_all(&record)?;
                }
            }
        }
        Ok(())
    }

    /// Load a database into RAM, validating records and rejecting duplicate keys.
    /// The declared entry count is never used for an upfront table allocation.
    pub fn read_binary(reader: impl Read) -> io::Result<Self> {
        let mut reader = DatabaseReader::new(reader)?;
        let header = reader.header();
        let mut counter = Self::new(header.k, header.strand).map_err(io::Error::other)?;
        while let Some((key, count)) = reader.next_entry()? {
            let duplicate = match &mut counter.counts {
                Counts::Packed(counts) => {
                    let mut packed = [0; 8];
                    let size = header.k.div_ceil(4);
                    packed[..size].copy_from_slice(&reader.record[..size]);
                    counts.try_reserve(1).map_err(io::Error::other)?;
                    counts.insert(u64::from_le_bytes(packed), count).is_some()
                }
                Counts::Packed128(counts) => {
                    let mut packed = [0; 16];
                    let size = header.k.div_ceil(4);
                    packed[..size].copy_from_slice(&reader.record[..size]);
                    let key = u128::from_le_bytes(packed);
                    counts.try_reserve(1).map_err(io::Error::other)?;
                    counts
                        .insert(((key >> 64) as u64, key as u64), count)
                        .is_some()
                }
                Counts::Bytes(counts) => {
                    counts.try_reserve(1).map_err(io::Error::other)?;
                    counts.insert(key.to_vec(), count).is_some()
                }
            };
            if duplicate {
                return Err(invalid("duplicate key in database"));
            }
        }
        Ok(counter)
    }
}

fn write_packed_binary<const WIDTH: usize>(
    entries: impl Iterator<Item = ([u8; WIDTH], u64)>,
    key_bytes: usize,
    writer: &mut impl Write,
) -> io::Result<()> {
    let mut batch = [0; 16 * 1024];
    let mut offset = 0;
    for (key, count) in entries {
        if offset + WIDTH + 8 > batch.len() {
            writer.write_all(&batch[..offset])?;
            offset = 0;
        }
        // Constant-width stores avoid a variable-size memcpy for every key.
        // Count bytes replace any unused key padding; only the record is emitted.
        batch[offset..offset + WIDTH].copy_from_slice(&key);
        batch[offset + key_bytes..offset + key_bytes + 8].copy_from_slice(&count.to_le_bytes());
        offset += key_bytes + 8;
    }
    writer.write_all(&batch[..offset])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::{BufReader, Cursor};

    fn fixture(k: u64, entries: u64, canonical: bool) -> Vec<u8> {
        let mut bytes = b"KBOLTDB\0\x01\x00\x00\x00\x00\x00\x00\x00".to_vec();
        bytes[10] = u8::from(canonical);
        bytes.extend_from_slice(&k.to_le_bytes());
        bytes.extend_from_slice(&entries.to_le_bytes());
        bytes
    }

    #[test]
    fn matches_documented_wire_format() {
        for (sequence, packed) in [
            (b"ACG".as_slice(), vec![0x06]),
            (b"TACGT".as_slice(), vec![0x1b, 0x03]),
        ] {
            let mut counter = Counter::new(sequence.len(), StrandMode::Forward).unwrap();
            counter.add_record(sequence).unwrap();
            let mut actual = Vec::new();
            counter.write_binary(&mut actual).unwrap();
            let mut expected = fixture(sequence.len() as u64, 1, false);
            expected.extend(packed);
            expected.extend_from_slice(&1_u64.to_le_bytes());
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn round_trips_both_backends_and_strands_and_streams_tsv() {
        for k in (1..=70).chain([129, 200]) {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let mut counter = Counter::new(k, strand).unwrap();
                for sequence in [
                    b"acgTTacGAGTCAGATTCGAC".repeat(20),
                    vec![b'A'; 205],
                    vec![b'T'; 205],
                    b"ACGTNACGT.--ACGT".repeat(20),
                ] {
                    counter.add_record(&sequence).unwrap();
                }
                let expected = counter.counts().collect::<HashMap<_, _>>();
                let mut bytes = Vec::new();
                counter.write_binary(&mut bytes).unwrap();
                assert_eq!(bytes.len(), 32 + counter.len() * (k.div_ceil(4) + 8));
                let mut reader =
                    DatabaseReader::new(BufReader::with_capacity(1, bytes.as_slice())).unwrap();
                assert_eq!(
                    reader.header(),
                    DatabaseHeader {
                        k,
                        strand,
                        entries: counter.len() as u64
                    }
                );
                let mut streamed = HashMap::new();
                while let Some((key, count)) = reader.next_entry().unwrap() {
                    assert!(streamed.insert(key.to_vec(), count).is_none());
                }
                assert!(reader.next_entry().unwrap().is_none());
                assert_eq!(streamed, expected);

                let loaded = Counter::read_binary(bytes.as_slice()).unwrap();
                assert_eq!(loaded.k, k);
                assert_eq!(loaded.strand, strand);
                assert_eq!(loaded.counts().collect::<HashMap<_, _>>(), expected);
                let mut actual_tsv = Vec::new();
                DatabaseReader::new(bytes.as_slice())
                    .unwrap()
                    .write_tsv(&mut actual_tsv)
                    .unwrap();
                let mut expected_tsv = Vec::new();
                counter.write_tsv(&mut expected_tsv).unwrap();
                assert_eq!(actual_tsv, expected_tsv);
            }
        }
    }

    #[test]
    fn packed_output_preserves_records_across_multiple_output_batches() {
        let mut state = 42_u64;
        let sequence: Vec<u8> = (0..20_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                b"ACGT"[(state & 3) as usize]
            })
            .collect();
        for k in [17, 31, 32, 33, 63, 64] {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let mut counter = Counter::new(k, strand).unwrap();
                counter.add_record(&sequence).unwrap();
                let mut expected = Vec::new();
                for (key, count) in counter.counts() {
                    expected.extend_from_slice(&key);
                    writeln!(expected, "\t{count}").unwrap();
                }
                let mut actual = Vec::new();
                counter.write_tsv(&mut actual).unwrap();
                assert_eq!(actual, expected);
                let mut binary = Vec::new();
                counter.write_binary(&mut binary).unwrap();
                assert!(binary.len() > 2 * 16 * 1024);
                assert_eq!(binary.len(), 32 + counter.len() * (k.div_ceil(4) + 8));
                let mut dumped = Vec::new();
                DatabaseReader::new(binary.as_slice())
                    .unwrap()
                    .write_tsv(&mut dumped)
                    .unwrap();
                assert_eq!(dumped, expected);
                let loaded = Counter::read_binary(binary.as_slice()).unwrap();
                assert_eq!(
                    loaded.counts().collect::<HashMap<_, _>>(),
                    counter.counts().collect::<HashMap<_, _>>()
                );
            }
        }
    }

    #[test]
    fn preserves_full_width_keys_and_maximum_counts() {
        for k in [1, 31, 32, 33, 63, 64, 65] {
            let mut counter = Counter::new(k, StrandMode::Forward).unwrap();
            counter.add_record(&vec![b'T'; k]).unwrap();
            match &mut counter.counts {
                Counts::Packed(counts) => *counts.values_mut().next().unwrap() = u64::MAX,
                Counts::Packed128(counts) => *counts.values_mut().next().unwrap() = u64::MAX,
                Counts::Bytes(counts) => *counts.values_mut().next().unwrap() = u64::MAX,
            }
            let mut bytes = Vec::new();
            counter.write_binary(&mut bytes).unwrap();
            let mut expected = fixture(k as u64, 1, false);
            expected.extend(std::iter::repeat_n(0xff, k / 4));
            if k % 4 != 0 {
                expected.push((1 << (2 * (k % 4))) - 1);
            }
            expected.extend_from_slice(&u64::MAX.to_le_bytes());
            assert_eq!(bytes, expected);
            let loaded = Counter::read_binary(bytes.as_slice()).unwrap();
            assert_eq!(
                loaded.counts().collect::<Vec<_>>(),
                [(vec![b'T'; k], u64::MAX)]
            );
        }
    }

    #[test]
    fn empty_databases_need_no_key_allocation() {
        for k in [1, 33, usize::MAX] {
            let counter = Counter::new(k, StrandMode::Canonical).unwrap();
            let mut bytes = Vec::new();
            counter.write_binary(&mut bytes).unwrap();
            assert_eq!(bytes, fixture(k as u64, 0, true));
            assert!(Counter::read_binary(bytes.as_slice()).unwrap().is_empty());
        }
    }

    #[test]
    fn rejects_truncation_trailing_bytes_and_invalid_records() {
        let mut valid = fixture(3, 1, false);
        valid.extend_from_slice(&[6, 1, 0, 0, 0, 0, 0, 0, 0]);
        for end in 0..valid.len() {
            assert!(Counter::read_binary(&valid[..end]).is_err(), "end={end}");
        }
        let mut corruptions = Vec::new();
        for (offset, value) in [
            (0, b'X'),
            (8, 2),
            (9, 1),
            (10, 2),
            (11, 1),
            (15, 1),
            (16, 0),
            (32, 0xc6),
            (33, 0),
        ] {
            let mut bytes = valid.clone();
            bytes[offset] = value;
            corruptions.push(bytes);
        }
        let mut trailing = valid.clone();
        trailing.push(0);
        corruptions.push(trailing);
        let mut noncanonical = fixture(1, 1, true);
        noncanonical.extend_from_slice(&[3, 1, 0, 0, 0, 0, 0, 0, 0]);
        corruptions.push(noncanonical);
        corruptions.push(fixture(3, u64::MAX, false));
        let mut impossible_allocation = fixture(u64::MAX, 1, false);
        impossible_allocation.extend_from_slice(&[0; 9]);
        corruptions.push(impossible_allocation);
        for bytes in corruptions {
            assert!(Counter::read_binary(bytes.as_slice()).is_err(), "{bytes:?}");
        }
    }

    #[test]
    fn loading_rejects_duplicate_keys() {
        let mut bytes = fixture(3, 2, false);
        bytes.extend_from_slice(&[6, 1, 0, 0, 0, 0, 0, 0, 0]);
        bytes.extend_from_slice(&[6, 2, 0, 0, 0, 0, 0, 0, 0]);
        let error = Counter::read_binary(bytes.as_slice()).err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("duplicate"));
    }

    #[test]
    fn handles_short_reads_interruptions_and_io_failures() {
        struct ShortRead<R>(R);
        impl<R: Read> Read for ShortRead<R> {
            fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
                let len = bytes.len().min(1);
                self.0.read(&mut bytes[..len])
            }
        }
        struct InterruptedOnce<R> {
            inner: R,
            interrupted: bool,
        }
        impl<R: Read> Read for InterruptedOnce<R> {
            fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
                if !self.interrupted {
                    self.interrupted = true;
                    return Err(io::ErrorKind::Interrupted.into());
                }
                self.inner.read(bytes)
            }
        }
        struct FailingRead;
        impl Read for FailingRead {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("injected read failure"))
            }
        }
        let mut bytes = fixture(3, 1, false);
        bytes.extend_from_slice(&[6, 1, 0, 0, 0, 0, 0, 0, 0]);
        let loaded = Counter::read_binary(InterruptedOnce {
            inner: ShortRead(bytes.as_slice()),
            interrupted: false,
        })
        .unwrap();
        assert_eq!(loaded.counts().collect::<Vec<_>>(), [(b"ACG".to_vec(), 1)]);
        // Fail after a complete payload: an I/O error must not look like EOF.
        let error = Counter::read_binary(Cursor::new(&bytes).chain(FailingRead))
            .err()
            .unwrap();
        assert_eq!(error.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn binary_writing_handles_short_writes_and_errors_without_flushing() {
        struct Writer {
            bytes: Vec<u8>,
            remaining: usize,
        }
        impl Write for Writer {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if self.remaining == 0 {
                    return Err(io::Error::other("injected write failure"));
                }
                let len = bytes.len().min(2).min(self.remaining);
                self.bytes.extend_from_slice(&bytes[..len]);
                self.remaining -= len;
                Ok(len)
            }
            fn flush(&mut self) -> io::Result<()> {
                panic!("caller owns flushing")
            }
        }
        for k in [3, 33] {
            let mut counter = Counter::new(k, StrandMode::Forward).unwrap();
            counter.add_record(&vec![b'T'; k]).unwrap();
            let mut expected = Vec::new();
            counter.write_binary(&mut expected).unwrap();
            for limit in [0, 2, 32, expected.len() - 1, expected.len()] {
                let mut writer = Writer {
                    bytes: Vec::new(),
                    remaining: limit,
                };
                let result = counter.write_binary(&mut writer);
                assert_eq!(result.is_ok(), limit == expected.len());
                assert_eq!(writer.bytes, expected[..limit]);
            }
        }
    }
}
