//! kbolt database format, version 2. All integers are little-endian.
//!
//! The 40-byte header is: magic `KBOLTDB\0` (8 bytes), version (u16, 2),
//! flags (u8: bit 0 = canonical, bit 1 = sections ascend), key order (u8, see
//! below), reserved zeros (4 bytes), k (u64), the number of distinct entries
//! (u64), and the seed of the hash (u64, zero unless keys are in hash order).
//! Unknown versions, flags, orders, or reserved bytes are rejected.
//!
//! Records follow in sections, with no index. A section is its number of
//! records (u32, at least one), the number of bytes they take (u32), and the
//! records. A reader can take a section whole, or skip it, without decoding
//! it; a writer closes a section after about 32 KiB.
//!
//! A record is a key and a count. The count is positive and stored as a
//! varint, in one byte when below 128, as nearly all counts of a read set
//! are. How the key is stored depends on the key order:
//!
//! - **0, none**: ceil(k/4) key bytes, in unspecified order. The key is a
//!   little-endian base-4 integer: A=0, C=1, G=2, T=3, with the first DNA
//!   base most significant and unused high bits zero. Used for k > 32.
//! - **1, hash** (k <= 32): the records of a section ascend by the hash of
//!   their key, and each stores, as a varint, its hash less that of the
//!   record before it (the first of a section, its hash). The hash is a
//!   permutation of the 2k-bit keys, so the key is recovered from it and
//!   never stored; consecutive hashes of many keys differ by little, which
//!   takes about five bytes where a key takes eight.
//! - **2, k-mer**: the records of a section ascend by k-mer (A < C < G < T),
//!   which is the order of the keys as integers. For k <= 32 each stores, as
//!   a varint, its key less that of the record before it (the first of a
//!   section, its key); for longer k-mers, its key bytes as under order 0.
//!
//! With `sections ascend` set, every section also continues where the one
//! before it ended, so that the whole database is in order. kbolt writes its
//! in-memory counts in hash order so, split at fixed hash boundaries: the
//! same k-mers then always give the same bytes, whatever the number of
//! threads. A database counted on disk (see `crate::disk`) holds one run of
//! ascending sections per part of its input, without the flag. Databases in
//! k-mer order (see `crate::sorted`) always have it.
//!
//! The hash of a key of `b = 2k` bits, with `s = ceil(b / 2)` and all
//! arithmetic modulo 2^b, is:
//!
//! ```text
//! x = key ^ seed;  x ^= x >> s;  x *= 0xbf58476d1ce4e5b9;
//! x ^= x >> s;     x *= 0x94d049bb133111eb;  hash = x ^ (x >> s)
//! ```
//!
//! A varint of `n` bytes, up to eight, holds a value below 2^(7n): the value
//! is shifted left by `n` bits, bit `n - 1` is set and the bits below it are
//! clear, so that the first byte tells the length. A first byte of zero is
//! followed by the value in eight bytes. Only the shortest encoding of a
//! value is valid. Unlike a varint with a continuation bit in every byte,
//! this one is read and written without a loop.
//!
//! Keys must be unique and canonical when flagged. No trailing data is
//! allowed. There is no checksum; structural validation does not detect
//! every bit flip.
//!
//! [`DatabaseReader`] streams entries with reusable buffers. It validates the
//! header, records, and end of stream, but does not track duplicate keys.
//! [`Counter::read_binary`] additionally rejects duplicates while loading.
//! Wrap files in buffered readers/writers; writing does not flush the caller's
//! writer. Finish reading the stream to validate its declared length.

use std::io::{self, Read, Write};

use crate::table::{self, Mixer, PackedCounts};
use crate::table_wide::{HighKey, PackedCountsWide};
use crate::{Counter, Counts, OutputTable, StrandMode, dispatch_wide, format_count};

const MAGIC: &[u8; 8] = b"KBOLTDB\0";
/// The only version read or written. Version 1, with whole keys and
/// eight-byte counts, is no longer supported.
const VERSION: u16 = 2;
const HEADER_BYTES: usize = 40;
/// Bytes of records after which a writer closes a section.
const SECTION_BYTES: usize = 32 * 1024;
const FLAG_CANONICAL: u8 = 1 << 0;
const FLAG_ASCENDING: u8 = 1 << 1;
const ALL_FLAGS: u8 = FLAG_CANONICAL | FLAG_ASCENDING;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn buffer(size: usize) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(io::Error::other)?;
    bytes.resize(size, 0);
    Ok(bytes)
}

/// Append `value` as a varint (see the module documentation).
#[inline(always)]
fn put_varint(bytes: &mut Vec<u8>, value: u64) {
    let bits = 64 - (value | 1).leading_zeros();
    if bits > 56 {
        bytes.push(0);
        bytes.extend_from_slice(&value.to_le_bytes());
        return;
    }
    let length = bits.div_ceil(7);
    let word = (value << length) | (1 << (length - 1));
    // A whole word is stored and the unused part dropped again, which costs
    // less than a copy of variable length.
    let end = bytes.len() + length as usize;
    bytes.extend_from_slice(&word.to_le_bytes());
    bytes.truncate(end);
}

/// Read the varint at `bytes[*at..]` and advance past it. Returns `None` if
/// it is cut short or is not the shortest encoding of its value.
#[inline(always)]
fn get_varint(bytes: &[u8], at: &mut usize) -> Option<u64> {
    let first = *bytes.get(*at)?;
    if first == 0 {
        let value = u64::from_le_bytes(bytes.get(*at + 1..*at + 9)?.try_into().unwrap());
        *at += 9;
        return (value >> 56 != 0).then_some(value);
    }
    let length = first.trailing_zeros() as usize + 1;
    let word = match bytes.get(*at..*at + 8) {
        Some(word) => u64::from_le_bytes(word.try_into().unwrap()),
        // Near the end of the section: only the varint's own bytes.
        None => {
            let mut word = [0; 8];
            word[..length].copy_from_slice(bytes.get(*at..*at + length)?);
            u64::from_le_bytes(word)
        }
    };
    *at += length;
    let value = (word & (u64::MAX >> (64 - 8 * length))) >> length;
    (length == 1 || value >> (7 * (length - 1)) != 0).then_some(value)
}

/// Writes records as the sections of a version-2 database.
pub(crate) struct SectionWriter<'a, W: Write + ?Sized> {
    writer: &'a mut W,
    /// The open section: room for its two counts, then its records.
    section: Vec<u8>,
    records: u32,
    /// The value of the section's last record, if records hold differences.
    previous: u64,
}

impl<'a, W: Write + ?Sized> SectionWriter<'a, W> {
    pub(crate) fn new(writer: &'a mut W) -> Self {
        Self {
            writer,
            section: vec![0; 8],
            records: 0,
            previous: 0,
        }
    }

    /// Add a record whose key is given by `value`, greater than that of the
    /// record before it, and stored as the difference.
    #[inline(always)]
    pub(crate) fn ascending(&mut self, value: u64, count: u64) -> io::Result<()> {
        debug_assert!(self.records == 0 || value > self.previous);
        put_varint(&mut self.section, value - self.previous);
        self.previous = value;
        put_varint(&mut self.section, count);
        self.added()
    }

    /// Add a record with its key as stored bytes.
    #[inline(always)]
    pub(crate) fn raw(&mut self, key: &[u8], count: u64) -> io::Result<()> {
        self.section.extend_from_slice(key);
        put_varint(&mut self.section, count);
        self.added()
    }

    /// Add a wide packed record directly into the section buffer without an
    /// intermediate key copy.
    #[inline(always)]
    pub(crate) fn raw_wide<H: HighKey>(
        &mut self,
        hi: H,
        lo: u64,
        key_bytes: usize,
        count: u64,
    ) -> io::Result<()> {
        hi.push_le(lo, key_bytes, &mut self.section);
        put_varint(&mut self.section, count);
        self.added()
    }

    #[inline(always)]
    fn added(&mut self) -> io::Result<()> {
        self.records += 1;
        if self.section.len() >= 8 + SECTION_BYTES {
            self.close()?;
        }
        Ok(())
    }

    /// Write the open section, if it holds any record.
    fn close(&mut self) -> io::Result<()> {
        if self.records == 0 {
            return Ok(());
        }
        let bytes = u32::try_from(self.section.len() - 8)
            .map_err(|_| invalid("database record too long"))?;
        self.section[..4].copy_from_slice(&self.records.to_le_bytes());
        self.section[4..8].copy_from_slice(&bytes.to_le_bytes());
        self.writer.write_all(&self.section)?;
        self.section.truncate(8);
        (self.records, self.previous) = (0, 0);
        Ok(())
    }

    /// Write the last section.
    pub(crate) fn finish(mut self) -> io::Result<()> {
        self.close()
    }
}

/// Write `tables` consecutive partitions of packed keys (k <= 32), from
/// `first` on, as sections in hash order.
pub(crate) fn write_hash_sections(
    counts: &PackedCounts,
    first: usize,
    tables: usize,
    writer: &mut (impl Write + ?Sized),
) -> io::Result<()> {
    let mut sections = SectionWriter::new(writer);
    let mut slots = Vec::new();
    for partition in first..first + tables {
        for (mixed, count) in counts.sorted_partition(partition, &mut slots) {
            sections.ascending(mixed, count)?;
        }
    }
    sections.finish()
}

/// How the records of a database are ordered, and with that how their keys
/// are stored; see the [module documentation](self).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyOrder {
    /// No order; keys are stored whole.
    Unordered,
    /// Within a section, by the hash of the key (k <= 32).
    Hash,
    /// Within a section, by k-mer: A < C < G < T.
    Kmer,
}

impl KeyOrder {
    fn from_byte(byte: u8) -> io::Result<Self> {
        match byte {
            0 => Ok(Self::Unordered),
            1 => Ok(Self::Hash),
            2 => Ok(Self::Kmer),
            _ => Err(invalid("unsupported database key order")),
        }
    }

    const fn as_byte(self) -> u8 {
        match self {
            Self::Unordered => 0,
            Self::Hash => 1,
            Self::Kmer => 2,
        }
    }
}

/// Metadata describing a kbolt count database.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatabaseHeader {
    pub k: usize,
    pub strand: StrandMode,
    pub entries: u64,
    /// The order of records within a section.
    pub order: KeyOrder,
    /// Whether the sections follow each other in that order too, so that
    /// the whole database is sorted.
    pub ascending: bool,
}

/// A streaming database reader. Memory use is proportional to k, not entry count.
pub struct DatabaseReader<R> {
    reader: R,
    header: DatabaseHeader,
    remaining: u64,
    /// The current record's key bytes.
    record: Vec<u8>,
    /// The current section's records, and what is left of them.
    section: Vec<u8>,
    position: usize,
    section_records: u32,
    /// For keys stored as differences: the hash, if they are in hash order,
    /// and the last value read (of the section, then of the section before
    /// it).
    mixer: Mixer,
    previous: u64,
    last: Option<u64>,
    /// For whole keys in k-mer order: the k-mer before the current one, if
    /// the current one has to follow it.
    before: Vec<u8>,
    follows: bool,
    sequence: Vec<u8>,
    finished: bool,
}

impl<R: Read> DatabaseReader<R> {
    /// Read and validate the header. No count table is allocated.
    pub fn new(mut reader: R) -> io::Result<Self> {
        let mut bytes = [0; HEADER_BYTES];
        // The magic and version first, so that other files are named as such
        // even when shorter than a header.
        reader.read_exact(&mut bytes[..10])?;
        if &bytes[..8] != MAGIC {
            return Err(invalid("expected a kbolt binary database"));
        }
        if u16::from_le_bytes(bytes[8..10].try_into().unwrap()) != VERSION {
            return Err(invalid("unsupported kbolt database version"));
        }
        reader.read_exact(&mut bytes[10..])?;
        let flags = bytes[10];
        let order = KeyOrder::from_byte(bytes[11])?;
        let seed = u64::from_le_bytes(bytes[32..].try_into().unwrap());
        if flags & !ALL_FLAGS != 0
            || bytes[12..16] != [0; 4]
            || (order != KeyOrder::Hash && seed != 0)
            || (order == KeyOrder::Unordered && flags & FLAG_ASCENDING != 0)
        {
            return Err(invalid("unsupported database flags or reserved bytes"));
        }
        let k = usize::try_from(u64::from_le_bytes(bytes[16..24].try_into().unwrap()))
            .map_err(|_| invalid("database k does not fit this platform"))?;
        if k == 0 {
            return Err(invalid("database k must be greater than zero"));
        }
        if order == KeyOrder::Hash && k > 32 {
            return Err(invalid("database keys in hash order need k <= 32"));
        }
        let entries = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let key_bytes = k.div_ceil(4);
        // Check the declared payload size without reserving the entry count.
        // A count takes at least one byte, as does a key stored as a
        // difference.
        let least = if order != KeyOrder::Unordered && k <= 32 {
            2
        } else {
            key_bytes + 1
        };
        entries
            .checked_mul(least as u64)
            .ok_or_else(|| invalid("database payload size overflow"))?;
        let header = DatabaseHeader {
            k,
            strand: if flags & FLAG_CANONICAL != 0 {
                StrandMode::Canonical
            } else {
                StrandMode::Forward
            },
            entries,
            order,
            ascending: flags & FLAG_ASCENDING != 0,
        };
        Ok(Self {
            reader,
            header,
            remaining: entries,
            record: if entries == 0 {
                Vec::new()
            } else {
                buffer(key_bytes)?
            },
            section: Vec::new(),
            position: 0,
            section_records: 0,
            // (Only used, and only meaningful, for keys in hash order.)
            mixer: Mixer::new(table::key_bits(k.min(32)), seed),
            previous: 0,
            last: None,
            before: if entries == 0 || order != KeyOrder::Kmer || k <= 32 {
                Vec::new()
            } else {
                buffer(k)?
            },
            follows: false,
            sequence: if entries == 0 { Vec::new() } else { buffer(k)? },
            finished: false,
        })
    }

    pub fn header(&self) -> DatabaseHeader {
        self.header
    }

    /// Read the next section of a version-2 database.
    fn next_section(&mut self) -> io::Result<()> {
        if self.position != self.section.len() {
            return Err(invalid("database section longer than its records"));
        }
        let mut counts = [0; 8];
        self.reader.read_exact(&mut counts)?;
        let records = u32::from_le_bytes(counts[..4].try_into().unwrap());
        let bytes = u32::from_le_bytes(counts[4..].try_into().unwrap());
        if records == 0 || u64::from(records) > self.remaining {
            return Err(invalid("invalid database section record count"));
        }
        // Read through `take` so that a corrupt length cannot reserve more
        // than the stream holds.
        self.section.clear();
        let read = (&mut self.reader)
            .take(u64::from(bytes))
            .read_to_end(&mut self.section)?;
        if read as u64 != u64::from(bytes) {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        (self.position, self.section_records) = (0, records);
        // A new section starts from zero; the last value of the section
        // before it is kept to check that sections ascend.
        let later = self.remaining < self.header.entries;
        self.last = (self.header.ascending && later).then_some(self.previous);
        self.previous = 0;
        Ok(())
    }

    /// Read the key of the next record of a section that stores keys as
    /// differences into `record`, as its key bytes.
    fn next_difference(&mut self, first: bool) -> io::Result<()> {
        let difference = get_varint(&self.section, &mut self.position)
            .ok_or_else(|| invalid("invalid key in database section"))?;
        let value = self
            .previous
            .checked_add(difference)
            .filter(|&value| value >> 1 >> (table::key_bits(self.header.k) - 1) == 0)
            .ok_or_else(|| invalid("database key beyond its length"))?;
        // Within a section every record ascends; across sections only if
        // the header says so.
        let ascends = match (first, self.last) {
            (false, _) => difference > 0,
            (true, Some(last)) => value > last,
            (true, None) => true,
        };
        if !ascends {
            return Err(invalid("database keys out of order"));
        }
        self.previous = value;
        let key = match self.header.order {
            KeyOrder::Hash => self.mixer.unmix(value),
            _ => value,
        };
        let key_bytes = self.record.len();
        self.record.copy_from_slice(&key.to_le_bytes()[..key_bytes]);
        Ok(())
    }

    /// Read one uppercase DNA key and its count, borrowing a reusable buffer.
    /// Returns None only after checking for trailing data. Discard on any error.
    /// Duplicate detection is left to consumers that build a count table.
    pub fn next_entry(&mut self) -> io::Result<Option<(&[u8], u64)>> {
        if self.finished {
            return Ok(None);
        }
        if self.remaining == 0 {
            if self.position != self.section.len() {
                return Err(invalid("database section longer than its records"));
            }
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
        let key_bytes = self.record.len();
        let first = self.section_records == 0;
        if first {
            self.next_section()?;
        }
        self.section_records -= 1;
        let whole = self.header.order == KeyOrder::Unordered || self.header.k > 32;
        if !whole {
            self.next_difference(first)?;
        } else {
            let key = self
                .section
                .get(self.position..self.position + key_bytes)
                .ok_or_else(|| invalid("database section shorter than its records"))?;
            self.record.copy_from_slice(key);
            self.position += key_bytes;
        }
        let count = get_varint(&self.section, &mut self.position)
            .ok_or_else(|| invalid("invalid count in database section"))?;
        let used = (self.header.k % 4) * 2;
        if used != 0 && self.record[key_bytes - 1] >> used != 0 {
            return Err(invalid("nonzero padding in database key"));
        }
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
        if !self.before.is_empty() {
            // Whole keys in k-mer order: each after the one before it in its
            // section, and after the last of the section before if sections
            // ascend.
            if (!first || (self.header.ascending && self.follows)) && self.sequence <= self.before {
                return Err(invalid("database keys out of order"));
            }
            self.before.copy_from_slice(&self.sequence);
            self.follows = true;
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
    /// Write a version-2 database with exact counts and no per-entry allocation.
    /// For k <= 32 it is in hash order throughout, and the same k-mers and
    /// counts always give the same bytes; otherwise it is unordered. Does not
    /// flush; write failures may leave partial output.
    pub fn write_binary(&self, mut writer: impl Write) -> io::Result<()> {
        let (order, unit) = match &self.counts {
            Counts::Packed(counts) => (KeyOrder::Hash, self.serial_partition(counts)),
            _ => (KeyOrder::Unordered, 1),
        };
        let ascending = order == KeyOrder::Hash;
        let entries = self.len() as u64;
        writer.write_all(&header(self.k, self.strand, order, ascending, entries))?;
        if self.is_empty() {
            return Ok(());
        }
        let key_bytes = self.k.div_ceil(4);
        // Partitions encode independently; see `Counter::write_tables`.
        self.write_tables(&mut writer, unit, |table, writer| match table {
            OutputTable::Packed(counts, first, tables) => {
                write_hash_sections(counts, first, tables, writer)
            }
            OutputTable::Wide(counts, first, tables) => {
                fn encode_wide<H: HighKey>(
                    counts: &PackedCountsWide<H>,
                    first: usize,
                    tables: usize,
                    key_bytes: usize,
                    writer: &mut dyn Write,
                ) -> io::Result<()> {
                    let mut sections = SectionWriter::new(writer);
                    for partition in first..first + tables {
                        for ((hi, lo), count) in counts.partition_entries(partition) {
                            sections.raw_wide(hi, lo, key_bytes, count)?;
                        }
                    }
                    sections.finish()
                }
                dispatch_wide!(counts, c => encode_wide(c, first, tables, key_bytes, writer))
            }
            OutputTable::Bytes(counts) => {
                let mut sections = SectionWriter::new(writer);
                let mut packed = buffer(key_bytes)?;
                for (key, &count) in counts {
                    pack_bases(key, &mut packed);
                    sections.raw(&packed, count)?;
                }
                sections.finish()
            }
        })
    }

    /// Write a version-2 database containing only k-mers whose counts match
    /// `filter`. Behavior is otherwise identical to [`Counter::write_binary`].
    pub fn write_binary_filtered(
        &self,
        filter: crate::CountFilter,
        mut writer: impl Write,
    ) -> io::Result<()> {
        if filter.is_all() {
            return self.write_binary(writer);
        }
        let (order, unit) = match &self.counts {
            Counts::Packed(counts) => (KeyOrder::Hash, self.serial_partition(counts)),
            _ => (KeyOrder::Unordered, 1),
        };
        let ascending = order == KeyOrder::Hash;
        let entries = self.len_filtered(filter) as u64;
        writer.write_all(&header(self.k, self.strand, order, ascending, entries))?;
        if entries == 0 {
            return Ok(());
        }
        let key_bytes = self.k.div_ceil(4);
        self.write_tables(&mut writer, unit, |table, writer| match table {
            OutputTable::Packed(counts, first, tables) => {
                let mut sections = SectionWriter::new(writer);
                let mut slots = Vec::new();
                for partition in first..first + tables {
                    for (mixed, count) in counts.sorted_partition(partition, &mut slots) {
                        if filter.matches(count) {
                            sections.ascending(mixed, count)?;
                        }
                    }
                }
                sections.finish()
            }
            OutputTable::Wide(counts, first, tables) => {
                fn encode_wide<H: HighKey>(
                    counts: &PackedCountsWide<H>,
                    first: usize,
                    tables: usize,
                    key_bytes: usize,
                    filter: crate::CountFilter,
                    writer: &mut dyn Write,
                ) -> io::Result<()> {
                    let mut sections = SectionWriter::new(writer);
                    for partition in first..first + tables {
                        for (l_rem, h, count) in counts.tables()[partition].iter() {
                            if filter.matches(count) {
                                let (hi, lo) = counts.key(partition, l_rem, h);
                                sections.raw_wide(hi, lo, key_bytes, count)?;
                            }
                        }
                    }
                    sections.finish()
                }
                dispatch_wide!(counts, c => encode_wide(c, first, tables, key_bytes, filter, writer))
            }
            OutputTable::Bytes(counts) => {
                let mut sections = SectionWriter::new(writer);
                let mut packed = buffer(key_bytes)?;
                for (key, &count) in counts {
                    if filter.matches(count) {
                        pack_bases(key, &mut packed);
                        sections.raw(&packed, count)?;
                    }
                }
                sections.finish()
            }
        })
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
                    !counts
                        .insert_new(u64::from_le_bytes(packed), count)
                        .map_err(io::Error::other)?
                }
                Counts::Wide(counts) => {
                    fn insert_wide<H: HighKey>(
                        counts: &mut PackedCountsWide<H>,
                        bytes: &[u8],
                        count: u64,
                    ) -> io::Result<bool> {
                        let (hi, lo) = H::read_le(bytes);
                        Ok(!counts.insert_new(hi, lo, count).map_err(io::Error::other)?)
                    }
                    let size = header.k.div_ceil(4);
                    let bytes = &reader.record[..size];
                    dispatch_wide!(counts, c => insert_wide(c, bytes, count)?)
                }
                Counts::Bytes(counts) => !counts
                    .insert_new(key, count)
                    .map_err(io::Error::other)?,
            };
            if duplicate {
                return Err(invalid("duplicate key in database"));
            }
        }
        Ok(counter)
    }
}

/// Store the uppercase DNA `key` as the key bytes of a record.
pub(crate) fn pack_bases(key: &[u8], packed: &mut [u8]) {
    packed.fill(0);
    for (index, base) in key.iter().rev().enumerate() {
        let value = crate::DNA_BASES[*base as usize];
        packed[index / 4] |= value << (2 * (index % 4));
    }
}

/// The header of a database with `entries` records in `order`, whose
/// sections follow each other in that order too if `ascending`.
pub(crate) fn header(
    k: usize,
    strand: StrandMode,
    order: KeyOrder,
    ascending: bool,
    entries: u64,
) -> [u8; HEADER_BYTES] {
    let mut header = [0; HEADER_BYTES];
    header[..8].copy_from_slice(MAGIC);
    header[8..10].copy_from_slice(&VERSION.to_le_bytes());
    header[10] = if strand == StrandMode::Canonical {
        FLAG_CANONICAL
    } else {
        0
    } | if ascending { FLAG_ASCENDING } else { 0 };
    header[11] = order.as_byte();
    header[16..24].copy_from_slice(&(k as u64).to_le_bytes());
    header[24..32].copy_from_slice(&entries.to_le_bytes());
    if order == KeyOrder::Hash {
        header[32..].copy_from_slice(&table::SEED.to_le_bytes());
    }
    header
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::{BufReader, Cursor};

    /// The header of a database with whole, unordered keys.
    fn header2(k: u64, entries: u64, canonical: bool) -> Vec<u8> {
        let mut bytes = b"KBOLTDB\0\x02\x00\x00\x00\x00\x00\x00\x00".to_vec();
        bytes[10] = u8::from(canonical);
        bytes.extend_from_slice(&k.to_le_bytes());
        bytes.extend_from_slice(&entries.to_le_bytes());
        bytes.extend_from_slice(&[0; 8]);
        bytes
    }

    /// The header kbolt writes for `k`: in hash order throughout for k <= 32.
    fn written_header(k: u64, entries: u64, canonical: bool) -> Vec<u8> {
        let mut bytes = header2(k, entries, canonical);
        if k <= 32 {
            bytes[10] |= 2;
            bytes[11] = 1;
            bytes[32..].copy_from_slice(&table::SEED.to_le_bytes());
        }
        bytes
    }

    /// The hash of a packed key, as a varint.
    fn hashed(k: usize, key: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        put_varint(
            &mut bytes,
            Mixer::new(table::key_bits(k), table::SEED).mix(key),
        );
        bytes
    }

    /// A section of `records` records taking `payload`.
    fn section(records: u32, payload: &[u8]) -> Vec<u8> {
        let mut bytes = records.to_le_bytes().to_vec();
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn matches_documented_wire_format() {
        // The hash of the documentation, written out for six bits.
        let hash = |key: u64| {
            let mut x = key ^ (table::SEED & 63);
            x ^= x >> 3;
            x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9) & 63;
            x ^= x >> 3;
            x = x.wrapping_mul(0x94d0_49bb_1331_11eb) & 63;
            x ^ (x >> 3)
        };
        // ACG = 0b000110 and CGT = 0b011011, seen once and 300 times.
        let mut counter = Counter::new(3, StrandMode::Forward).unwrap();
        counter.add_record(b"ACG").unwrap();
        for _ in 0..300 {
            counter.add_record(b"CGT").unwrap();
        }
        let mut actual = Vec::new();
        counter.write_binary(&mut actual).unwrap();
        let mut records = [
            (hash(0b000110), vec![0b11]),
            (hash(0b011011), vec![0xb2, 0x04]),
        ];
        records.sort();
        let mut expected = written_header(3, 2, false);
        // Each hash is below 64, so a one-byte varint: itself in its own
        // section, as these keys fall in different partitions.
        for (hash, count) in records {
            let mut record = vec![(hash << 1 | 1) as u8];
            record.extend(count);
            expected.extend(section(1, &record));
        }
        assert_eq!(actual, expected);

        // Longer keys are stored whole, the first base most significant.
        let sequence = [b"T".as_slice(), &[b'A'; 32], b"CG"].concat();
        let mut counter = Counter::new(35, StrandMode::Forward).unwrap();
        counter.add_record(&sequence).unwrap();
        let mut actual = Vec::new();
        counter.write_binary(&mut actual).unwrap();
        let mut expected = header2(35, 1, false);
        expected.extend(section(1, &[0x06, 0, 0, 0, 0, 0, 0, 0, 0x30, 0b11]));
        assert_eq!(actual, expected);
    }

    #[test]
    fn packed_databases_are_in_hash_order_and_the_same_for_any_partitioning() {
        let mut state = 7_u64;
        let sequence: Vec<u8> = (0..60_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                b"ACGT"[(state & 3) as usize]
            })
            .collect();
        for k in [1, 5, 16, 31, 32] {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let mut serial = Counter::new(k, strand).unwrap();
                serial.add_record(&sequence).unwrap();
                let mut expected = Vec::new();
                serial.write_binary(&mut expected).unwrap();
                // Hashes ascend from the first record to the last.
                let mixer = Mixer::new(table::key_bits(k), table::SEED);
                let mut reader = DatabaseReader::new(expected.as_slice()).unwrap();
                let header = reader.header();
                assert_eq!((header.order, header.ascending), (KeyOrder::Hash, true));
                let mut previous = None;
                while reader.next_entry().unwrap().is_some() {
                    let size = k.div_ceil(4);
                    let mut key = [0; 8];
                    key[..size].copy_from_slice(&reader.record[..size]);
                    let mixed = mixer.mix(u64::from_le_bytes(key));
                    assert!(previous < Some(mixed), "k={k}");
                    previous = Some(mixed);
                }
                // Counting on several threads partitions the keys more
                // finely, which must not show in the output.
                for threads in [2, 5] {
                    let mut parallel = crate::ParallelCounter::new(k, strand, threads).unwrap();
                    for part in sequence.chunks(7_001) {
                        parallel.add_sequence(part).unwrap();
                    }
                    let parallel = parallel.finish().unwrap();
                    let mut actual = Vec::new();
                    parallel.write_binary(&mut actual).unwrap();
                    assert!(actual == expected, "k={k} threads={threads}");
                    let (mut serial_tsv, mut parallel_tsv) = (Vec::new(), Vec::new());
                    serial.write_tsv(&mut serial_tsv).unwrap();
                    parallel.write_tsv(&mut parallel_tsv).unwrap();
                    assert!(serial_tsv == parallel_tsv, "k={k} threads={threads}");
                }
            }
        }
    }

    #[test]
    fn rejects_hashed_keys_out_of_order_or_beyond_their_length() {
        let read = |bytes: &[u8]| -> io::Result<usize> {
            let mut reader = DatabaseReader::new(bytes)?;
            let mut entries = 0;
            while reader.next_entry()?.is_some() {
                entries += 1;
            }
            Ok(entries)
        };
        // k = 3: hashes below 64. Records are (difference, count) varints.
        let one = |value: u8| value << 1 | 1;
        let with = |ascending: bool, sections: &[Vec<u8>], entries: u64| {
            let mut bytes = written_header(3, entries, false);
            bytes[10] = u8::from(ascending) << 1;
            bytes.extend(sections.concat());
            bytes
        };
        let valid = with(
            true,
            &[
                section(2, &[one(5), one(1), one(4), one(1)]),
                section(1, &[one(10), one(1)]),
            ],
            3,
        );
        assert_eq!(read(&valid).unwrap(), 3);
        for (name, bytes) in [
            (
                "repeated key",
                with(true, &[section(2, &[one(5), one(1), one(0), one(1)])], 2),
            ),
            (
                "key of seven bits",
                with(true, &[section(1, &[one(64), one(1)])], 1),
            ),
            (
                "sum of seven bits",
                with(true, &[section(2, &[one(60), one(1), one(4), one(1)])], 2),
            ),
            (
                "second section starts below the first's end",
                with(
                    true,
                    &[section(1, &[one(9), one(1)]), section(1, &[one(9), one(1)])],
                    2,
                ),
            ),
            ("hash order beyond k = 32", {
                let mut bytes = with(false, &[section(1, &[one(9), one(1)])], 1);
                bytes[16] = 33;
                bytes
            }),
            ("the seed without hash order", {
                let mut bytes = with(false, &[section(1, &[6, one(1)])], 1);
                bytes[11] = 0;
                bytes
            }),
        ] {
            assert!(read(&bytes).is_err(), "{name}");
        }
        // Without the flag, sections need not follow each other in order.
        let unordered = with(
            false,
            &[section(1, &[one(9), one(1)]), section(1, &[one(3), one(1)])],
            2,
        );
        assert_eq!(read(&unordered).unwrap(), 2);
        // A first hash of zero is a key like any other.
        assert_eq!(
            read(&with(
                true,
                &[section(2, &[one(0), one(1), one(1), one(1)])],
                2
            ))
            .unwrap(),
            2
        );
    }

    #[test]
    fn varints_round_trip_at_every_length_and_reject_other_encodings() {
        let mut values = vec![0, 1, 127, 128, u64::MAX, u64::MAX - 1];
        for bits in 1..64 {
            values.extend([(1 << bits) - 1, 1 << bits, (1 << bits) + 1]);
        }
        for value in values {
            let mut bytes = vec![0xaa];
            put_varint(&mut bytes, value);
            let bits = 64 - (value | 1).leading_zeros() as usize;
            let length = if bits > 56 { 9 } else { bits.div_ceil(7) };
            assert_eq!(bytes.len(), 1 + length, "value={value}");
            // Both with a whole word to read and at the end of the bytes.
            for padding in [0, 16] {
                let mut padded = bytes.clone();
                padded.resize(bytes.len() + padding, 0xff);
                let mut at = 1;
                assert_eq!(get_varint(&padded, &mut at), Some(value));
                assert_eq!(at, bytes.len());
            }
            for end in 1..bytes.len() {
                assert_eq!(get_varint(&bytes[..end], &mut 1), None);
            }
            // The same value in one byte more is not accepted.
            if length < 8 {
                let longer = length + 1;
                let word = (u128::from(value) << longer) | (1 << (longer - 1));
                let mut padded = word.to_le_bytes()[..longer].to_vec();
                padded.resize(16, 0);
                assert_eq!(get_varint(&padded, &mut 0), None, "value={value}");
            } else if length == 8 {
                let mut longer = vec![0];
                longer.extend_from_slice(&value.to_le_bytes());
                assert_eq!(get_varint(&longer, &mut 0), None);
            }
        }
    }

    #[test]
    fn reads_version_two_sections_and_rejects_malformed_ones() {
        // ACG twice, then CGT 300 times: a count of two bytes.
        let records = [vec![6, 0b101], vec![0x1b, 0xb2, 0x04]];
        let mut valid = header2(3, 2, false);
        valid.extend(section(1, &records[0]));
        valid.extend(section(1, &records[1]));
        let expected = [(b"ACG".to_vec(), 2), (b"CGT".to_vec(), 300)];
        let read = |bytes: &[u8]| -> io::Result<Vec<(Vec<u8>, u64)>> {
            let mut reader = DatabaseReader::new(bytes)?;
            let mut entries = Vec::new();
            while let Some((key, count)) = reader.next_entry()? {
                entries.push((key.to_vec(), count));
            }
            Ok(entries)
        };
        assert_eq!(read(&valid).unwrap(), expected);
        // The same records in one section.
        let mut joined = header2(3, 2, false);
        joined.extend(section(2, &records.concat()));
        assert_eq!(read(&joined).unwrap(), expected);
        for end in 0..valid.len() {
            assert!(read(&valid[..end]).is_err(), "end={end}");
        }
        let with = |sections: &[Vec<u8>], entries: u64| {
            let mut bytes = header2(3, entries, false);
            bytes.extend(sections.concat());
            bytes
        };
        for (name, bytes) in [
            (
                "no records",
                with(&[section(0, &[]), section(2, &records.concat())], 2),
            ),
            (
                "more records than entries",
                with(&[section(3, &records.concat())], 2),
            ),
            (
                "fewer records than entries",
                with(&[section(1, &records[0])], 2),
            ),
            (
                "records beyond the section",
                with(&[section(2, &records[0])], 2),
            ),
            (
                "bytes after the records",
                with(&[section(1, &records.concat())], 1),
            ),
            ("zero count", with(&[section(1, &[6, 0b1])], 1)),
            ("long count", with(&[section(1, &[6, 0b110, 0])], 1)),
            ("key padding", with(&[section(1, &[0xc6, 0b11])], 1)),
            (
                "trailing byte",
                [with(&[section(1, &records[0])], 1), vec![0]].concat(),
            ),
            ("reserved header bytes", {
                let mut bytes = valid.clone();
                bytes[39] = 1;
                bytes
            }),
            ("unknown order", {
                let mut bytes = valid.clone();
                bytes[11] = 9;
                bytes
            }),
            ("later version", {
                let mut bytes = valid.clone();
                bytes[8] = 3;
                bytes
            }),
        ] {
            assert!(read(&bytes).is_err(), "{name}");
        }
        // A section claiming more bytes than follow fails without reserving them.
        let mut huge = header2(3, 1, false);
        huge.extend_from_slice(&1_u32.to_le_bytes());
        huge.extend_from_slice(&u32::MAX.to_le_bytes());
        huge.extend_from_slice(&[6, 0b11]);
        assert!(read(&huge).is_err());
    }

    #[test]
    fn round_trips_all_backends_and_strands_and_streams_tsv() {
        for k in [
            1, 2, 3, 4, 5, 31, 32, 33, 34, 35, 36, 40, 41, 48, 49, 63, 64, 65, 96, 97, 128, 129,
            200,
        ] {
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
                let mut reader =
                    DatabaseReader::new(BufReader::with_capacity(1, bytes.as_slice())).unwrap();
                assert_eq!(
                    reader.header(),
                    DatabaseHeader {
                        k,
                        strand,
                        entries: counter.len() as u64,
                        order: if k <= 32 {
                            KeyOrder::Hash
                        } else {
                            KeyOrder::Unordered
                        },
                        ascending: k <= 32,
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
                // The same rows; packed keys are written in hash order.
                let rows = |tsv: &[u8]| {
                    let mut rows: Vec<Vec<u8>> =
                        tsv.split(|&b| b == b'\n').map(Vec::from).collect();
                    rows.sort();
                    rows
                };
                assert_eq!(rows(&actual), rows(&expected));
                let mut binary = Vec::new();
                counter.write_binary(&mut binary).unwrap();
                assert!(binary.len() > 2 * SECTION_BYTES);
                let mut dumped = Vec::new();
                DatabaseReader::new(binary.as_slice())
                    .unwrap()
                    .write_tsv(&mut dumped)
                    .unwrap();
                assert_eq!(dumped, actual);
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
        for k in [1, 31, 32, 33, 40, 48, 63, 64, 65, 96, 97, 128, 129] {
            let mut counter = Counter::new(k, StrandMode::Forward).unwrap();
            counter.add_record(&vec![b'T'; k]).unwrap();
            counter.counts.set_counts(u64::MAX);
            let mut bytes = Vec::new();
            counter.write_binary(&mut bytes).unwrap();
            let mut record: Vec<u8> = if k <= 32 {
                hashed(k, u64::MAX >> (64 - 2 * k))
            } else {
                let mut key: Vec<u8> = std::iter::repeat_n(0xff, k / 4).collect();
                if k % 4 != 0 {
                    key.push((1 << (2 * (k % 4))) - 1);
                }
                key
            };
            // The largest count: a zero byte, then all eight of its bytes.
            record.push(0);
            record.extend_from_slice(&u64::MAX.to_le_bytes());
            let mut expected = written_header(k as u64, 1, false);
            expected.extend(section(1, &record));
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
            assert_eq!(bytes, written_header(k as u64, 0, true));
            assert!(Counter::read_binary(bytes.as_slice()).unwrap().is_empty());
        }
    }

    #[test]
    fn rejects_keys_out_of_kmer_order() {
        let read = |bytes: &[u8]| -> io::Result<Vec<Vec<u8>>> {
            let mut reader = DatabaseReader::new(bytes)?;
            let mut keys = Vec::new();
            while let Some((key, _)) = reader.next_entry()? {
                keys.push(key.to_vec());
            }
            Ok(keys)
        };
        let one = |value: u8| value << 1 | 1;
        // k = 3, keys as differences: ACG (6), CGT (27), TTT (63).
        let short = |sections: &[Vec<u8>], entries: u64| {
            let mut bytes = header2(3, entries, false);
            (bytes[10], bytes[11]) = (2, 2);
            bytes.extend(sections.concat());
            bytes
        };
        let valid = short(
            &[
                section(2, &[one(6), one(1), one(21), one(1)]),
                section(1, &[one(63), one(1)]),
            ],
            3,
        );
        assert_eq!(read(&valid).unwrap(), [b"ACG", b"CGT", b"TTT"]);
        for (name, bytes) in [
            (
                "repeated key",
                short(&[section(2, &[one(6), one(1), one(0), one(1)])], 2),
            ),
            (
                "key beyond three bases",
                short(&[section(1, &[one(64), one(1)])], 1),
            ),
            (
                "section before the last one's end",
                short(
                    &[
                        section(1, &[one(27), one(1)]),
                        section(1, &[one(6), one(1)]),
                    ],
                    2,
                ),
            ),
            ("a seed", {
                let mut bytes = valid.clone();
                bytes[32] = 1;
                bytes
            }),
        ] {
            assert!(read(&bytes).is_err(), "{name}");
        }
        // k = 33, whole keys: the last base in the lowest bits.
        let key = |last: u8| {
            let mut bytes = vec![last];
            bytes.extend_from_slice(&[0; 8]);
            bytes.push(one(1));
            bytes
        };
        let long = |ascending: bool, sections: &[Vec<u8>], entries: u64| {
            let mut bytes = header2(33, entries, false);
            (bytes[10], bytes[11]) = (u8::from(ascending) << 1, 2);
            bytes.extend(sections.concat());
            bytes
        };
        let two = |first: u8, second: u8| [key(first), key(second)].concat();
        assert_eq!(
            read(&long(
                true,
                &[section(2, &two(0, 1)), section(1, &key(3))],
                3
            ))
            .unwrap()
            .len(),
            3
        );
        assert!(read(&long(true, &[section(2, &two(1, 1))], 2)).is_err());
        assert!(read(&long(true, &[section(2, &two(2, 1))], 2)).is_err());
        let apart = [section(1, &key(2)), section(1, &key(1))];
        assert!(read(&long(true, &apart, 2)).is_err());
        // Without the flag, only the records of a section are in order.
        assert_eq!(read(&long(false, &apart, 2)).unwrap().len(), 2);
    }

    #[test]
    fn rejects_invalid_headers_noncanonical_and_impossible_allocations() {
        let mut valid = header2(3, 1, false);
        valid.extend(section(1, &[6, 0b11]));
        assert!(Counter::read_binary(valid.as_slice()).is_ok());
        let mut corruptions = Vec::new();
        for (offset, value) in [
            // Magic, version 1 (removed), version high byte, flags, reserved
            // bytes, and k of zero.
            (0, b'X'),
            (8, 1),
            (9, 1),
            (10, 4),
            (12, 1),
            (15, 1),
            (16, 0),
        ] {
            let mut bytes = valid.clone();
            bytes[offset] = value;
            corruptions.push(bytes);
        }
        let mut noncanonical = header2(1, 1, true);
        noncanonical.extend(section(1, &[3, 0b11]));
        corruptions.push(noncanonical);
        corruptions.push(header2(3, u64::MAX, false));
        let mut impossible_allocation = header2(u64::MAX, 1, false);
        impossible_allocation.extend(section(1, &[0; 9]));
        corruptions.push(impossible_allocation);
        for bytes in corruptions {
            assert!(Counter::read_binary(bytes.as_slice()).is_err(), "{bytes:?}");
        }
    }

    #[test]
    fn loading_rejects_duplicate_keys() {
        let mut bytes = header2(3, 2, false);
        bytes.extend(section(2, &[6, 0b11, 6, 0b101]));
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
        let mut bytes = header2(3, 1, false);
        bytes.extend(section(1, &[6, 0b11]));
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
            for limit in [0, 2, 40, expected.len() - 1, expected.len()] {
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
