//! A streaming gzip decoder built for throughput.
//!
//! Decompression is sequential, so on compressed input one thread per file
//! sets the pace of the whole counter. This decoder follows the design of
//! libdeflate's: a 64-bit bit buffer refilled without branches, decode tables
//! whose entries carry everything a symbol needs (so a symbol costs one
//! lookup), several literals per refill, and matches copied a word at a
//! time. Unlike libdeflate it streams: input is read through a bounded
//! buffer and output is produced in chunks.
//!
//! Output goes straight into the caller's chunk. A match may reach back
//! [`WINDOW`] bytes, so each chunk starts with that much room, into which the
//! end of the previous chunk is copied; matches then never leave the chunk.
//! A chunk also ends with [`SLACK`] spare bytes: decoding stops at the first
//! symbol boundary past the chunk's nominal end, and a match is copied in
//! whole words that may run past its own end.
//!
//! Near the end of the buffered input the decoder switches to a careful loop
//! that takes one symbol at a time. The input buffer is followed by zeros,
//! so a symbol can always be decoded; if it turns out to have used bits that
//! have not arrived, it is undone. Everything that can be decoded from the
//! bytes at hand is therefore delivered before the decoder waits for more,
//! which matters when input arrives slowly through a pipe.
//!
//! Concatenated members are decoded as one stream, and each member's CRC-32
//! and length are verified.

use std::io::{self, Read};

/// Bytes a match may reach back, and the room at the start of every chunk.
pub(crate) const WINDOW: usize = 32 * 1024;
/// Spare bytes at the end of every chunk: the longest match, the literals
/// decoded with it, and the overshoot of a word-wise copy.
pub(crate) const SLACK: usize = 320;

/// Compressed bytes buffered at a time.
const INPUT: usize = 256 * 1024;
/// Zero bytes kept after the buffered input. A block header (at most 571
/// bytes) or a symbol is decoded from them before it is known to be whole.
const PADDING: usize = 1024;
/// Buffered bytes the fast loop needs at the start of an iteration, which
/// loads at most 14.
const FAST_INPUT: usize = 64;
/// The largest input buffer, reached only by an oversized gzip header.
const MAX_INPUT: usize = 64 * 1024 * 1024;

const LITLEN_BITS: u32 = 11;
const OFFSET_BITS: u32 = 8;
const PRECODE_BITS: u32 = 7;
/// Table sizes that hold any complete code: the root table and the largest
/// possible set of subtables (zlib's `enough 288 11 15` and `enough 32 8 15`).
const LITLEN_ENOUGH: usize = 2342;
const OFFSET_ENOUGH: usize = 402;
const PRECODE_ENOUGH: usize = 1 << PRECODE_BITS;

// A table entry. The low byte is the number of bits to take from the bit
// buffer. For a literal, bits 16..24 hold the byte. For a length or an
// offset, bits 16..32 hold the base value, bits 8..12 the length of the
// codeword (within this table), and the low byte also counts the extra bits
// that follow the codeword. For a subtable pointer, bits 16..32 hold the
// subtable's first index and bits 8..12 the number of bits that index it.
const LITERAL: u32 = 1 << 31;
const EXCEPTIONAL: u32 = 1 << 15;
const SUBTABLE: u32 = 1 << 14;
const END_OF_BLOCK: u32 = 1 << 13;
const INVALID: u32 = 1 << 12;

/// Per-symbol entries of the literal/length code, without codeword lengths.
const LITLEN_RESULTS: [u32; 288] = {
    const BASES: [u16; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const EXTRA: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    let mut results = [EXCEPTIONAL | INVALID; 288];
    let mut symbol = 0;
    while symbol < 256 {
        results[symbol] = LITERAL | ((symbol as u32) << 16);
        symbol += 1;
    }
    results[256] = EXCEPTIONAL | END_OF_BLOCK;
    let mut index = 0;
    while index < 29 {
        results[257 + index] = ((BASES[index] as u32) << 16) | EXTRA[index] as u32;
        index += 1;
    }
    results
};

/// Per-symbol entries of the offset code.
const OFFSET_RESULTS: [u32; 32] = {
    let mut results = [EXCEPTIONAL | INVALID; 32];
    let mut symbol = 0;
    let mut base = 1_u32;
    while symbol < 30 {
        let extra = if symbol < 4 { 0 } else { symbol as u32 / 2 - 1 };
        results[symbol] = (base << 16) | extra;
        base += 1 << extra;
        symbol += 1;
    }
    results
};

/// Per-symbol entries of the code that encodes codeword lengths.
const PRECODE_RESULTS: [u32; 19] = {
    let mut results = [0; 19];
    let mut symbol = 0;
    while symbol < 19 {
        results[symbol] = (symbol as u32) << 16;
        symbol += 1;
    }
    results
};

/// The order in which a block header lists the precode's codeword lengths.
const PRECODE_ORDER: [u8; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

#[inline(always)]
fn low_bits(count: u32) -> u64 {
    (1_u64 << count) - 1
}

/// Build the decode table of a canonical Huffman code from its codeword
/// lengths (at most `max_len`). The first `2^table_bits` entries are indexed
/// by the next bits of input; longer codewords go through subtables.
///
/// Rejects any code that is not complete, except an empty one and (with
/// `allow_single`) a single one-bit codeword, as zlib does; unused codewords
/// of those decode to an invalid entry.
fn build_table(
    table: &mut [u32],
    lens: &[u8],
    results: &[u32],
    table_bits: u32,
    max_len: usize,
    allow_single: bool,
) -> Result<(), &'static str> {
    let root = 1_usize << table_bits;
    let mut counts = [0_u16; 16];
    for &len in lens {
        counts[usize::from(len)] += 1;
    }
    // Symbols sorted by codeword length, then by value: the order in which
    // a canonical code assigns its codewords.
    let mut starts = [0_u16; 17];
    for len in 1..=max_len {
        starts[len + 1] = starts[len] + counts[len];
    }
    let used = usize::from(starts[max_len + 1]);
    let mut sorted = [0_u16; 288];
    let mut next = starts;
    for (symbol, &len) in lens.iter().enumerate() {
        if len != 0 {
            sorted[usize::from(next[usize::from(len)])] = symbol as u16;
            next[usize::from(len)] += 1;
        }
    }
    let space: u32 = (1..=max_len)
        .map(|len| u32::from(counts[len]) << (max_len - len))
        .sum();
    let full = 1_u32 << max_len;
    if space > full {
        return Err("oversubscribed code");
    }
    if space < full {
        let invalid = EXCEPTIONAL | INVALID | 1;
        if used == 0 {
            table[..root].fill(invalid);
            return Ok(());
        }
        if allow_single && used == 1 && counts[1] == 1 {
            let entry = results[usize::from(sorted[0])] + (1 << 8) + 1;
            for (index, slot) in table[..root].iter_mut().enumerate() {
                *slot = if index & 1 == 0 { entry } else { invalid };
            }
            return Ok(());
        }
        return Err("incomplete code");
    }

    let (mut code, mut index, mut free) = (0_u32, 0, root);
    // The subtable being filled: the root index that leads to it, its first
    // entry, and the number of bits that index it.
    let (mut prefix, mut start, mut bits) = (usize::MAX, 0, 0);
    for len in 1..=max_len {
        for placed in 0..counts[len] {
            let symbol = usize::from(sorted[index]);
            index += 1;
            // Input bits arrive least significant first; codewords are
            // defined most significant bit first.
            let reversed = (code.reverse_bits() >> (32 - len)) as usize;
            code += 1;
            if len as u32 <= table_bits {
                let entry = results[symbol] + ((len as u32) << 8) + len as u32;
                for slot in table[reversed..root].iter_mut().step_by(1 << len) {
                    *slot = entry;
                }
                continue;
            }
            if reversed & (root - 1) != prefix {
                prefix = reversed & (root - 1);
                start = free;
                // A codeword `n` bits longer than the root needs 2^n entries.
                // If fewer codewords of this length remain, longer ones
                // share the subtable, which must then be indexed by more
                // bits; the code is complete, so it fills up eventually.
                bits = len - table_bits as usize;
                let mut filled = u32::from(counts[len] - placed);
                while filled < (1_u32 << bits) {
                    bits += 1;
                    if table_bits as usize + bits > max_len {
                        return Err("invalid code");
                    }
                    filled = (filled << 1) + u32::from(counts[table_bits as usize + bits]);
                }
                free += 1 << bits;
                if free > table.len() {
                    return Err("invalid code");
                }
                table[prefix] = EXCEPTIONAL
                    | SUBTABLE
                    | ((start as u32) << 16)
                    | ((bits as u32) << 8)
                    | table_bits;
            }
            let rest = len as u32 - table_bits;
            let entry = results[symbol] + (rest << 8) + rest;
            for slot in table[start..start + (1 << bits)]
                .iter_mut()
                .skip(reversed >> table_bits)
                .step_by(1_usize << rest)
            {
                *slot = entry;
            }
        }
        code <<= 1;
    }
    Ok(())
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// The length of the gzip member header at the start of `bytes`, or `None`
/// if more bytes are needed to tell.
fn header_length(bytes: &[u8]) -> io::Result<Option<usize>> {
    const BAD: &str = "invalid gzip header";
    for (index, expected) in [0x1f, 0x8b, 8].into_iter().enumerate() {
        match bytes.get(index) {
            Some(&byte) if byte != expected => return Err(invalid(BAD)),
            _ => {}
        }
    }
    if bytes.len() < 10 {
        return Ok(None);
    }
    let flags = bytes[3];
    if flags & 0xe0 != 0 {
        return Err(invalid(BAD));
    }
    let mut at = 10;
    // FEXTRA: a length and that many bytes.
    if flags & 4 != 0 {
        let Some(length) = bytes.get(at..at + 2) else {
            return Ok(None);
        };
        at += 2 + usize::from(u16::from_le_bytes([length[0], length[1]]));
    }
    // FNAME and FCOMMENT: zero-terminated strings.
    for flag in [8, 16] {
        if flags & flag != 0 {
            let Some(rest) = bytes.get(at..) else {
                return Ok(None);
            };
            let Some(end) = rest.iter().position(|&byte| byte == 0) else {
                return Ok(None);
            };
            at += end + 1;
        }
    }
    // FHCRC: a checksum of the header, not verified.
    if flags & 2 != 0 {
        at += 2;
    }
    Ok((at <= bytes.len()).then_some(at))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    /// Before a member's header, or at the end of the input.
    Member,
    /// Before a block's header.
    Block,
    /// In a stored block with this many bytes left.
    Stored(usize),
    /// In a Huffman-coded block whose tables are built.
    Coded,
    /// After a member's last block, before its checksum and length.
    Trailer,
    /// After the last member.
    End,
}

/// What stopped decoding.
enum Step {
    /// The chunk is full.
    Full,
    /// Nothing more can be decoded from the buffered input.
    NeedInput,
    /// The input has ended after a complete member.
    End,
}

/// How a run of the fast loop ended.
enum Fast {
    /// Too little input is buffered, or the chunk is full.
    Pause,
    EndOfBlock,
}

/// A decoder of one or more concatenated gzip members read from `R`.
pub(crate) struct GzDecoder<R> {
    reader: R,
    /// Buffered input: `filled` bytes, then at least `PADDING` zeros.
    input: Vec<u8>,
    /// The next input byte to load into the bit buffer.
    position: usize,
    filled: usize,
    /// The reader has reported its end.
    ended: bool,
    /// Input bits, the next one in bit zero. Bits above `count` are zero or
    /// equal the input bits that follow.
    bits: u64,
    count: u32,
    state: State,
    /// The current block is the member's last.
    last_block: bool,
    litlen: Vec<u32>,
    offsets: Vec<u32>,
    /// The end of the previous chunk: `history` bytes at the end.
    window: Vec<u8>,
    history: usize,
    /// In the current chunk: the earliest byte a match may copy, and the
    /// first byte not yet included in the member's checksum.
    floor: usize,
    checked: usize,
    checksum: crc32fast::Hasher,
    /// The member's decoded length, modulo 2^32.
    length: u32,
}

impl<R: Read> GzDecoder<R> {
    pub(crate) fn new(reader: R) -> Self {
        Self {
            reader,
            input: vec![0; INPUT + PADDING],
            position: 0,
            filled: 0,
            ended: false,
            bits: 0,
            count: 0,
            state: State::Member,
            last_block: false,
            litlen: vec![0; LITLEN_ENOUGH],
            offsets: vec![0; OFFSET_ENOUGH],
            window: vec![0; WINDOW],
            history: 0,
            floor: WINDOW,
            checked: WINDOW,
            checksum: crc32fast::Hasher::new(),
            length: 0,
        }
    }

    /// Decode the next bytes of the stream into `chunk[WINDOW..]`, returning
    /// how many: zero only at the end of the stream, and otherwise at least
    /// one and less than `chunk.len() - WINDOW`. The first `WINDOW` bytes of
    /// the chunk are overwritten too.
    ///
    /// Returns as soon as the chunk is full, or with whatever the input read
    /// so far decodes to: it waits for more input only with nothing to
    /// return. Fails on damaged or truncated input.
    pub(crate) fn read_chunk(&mut self, chunk: &mut [u8]) -> io::Result<usize> {
        assert!(chunk.len() > WINDOW + SLACK);
        let history = self.history;
        chunk[WINDOW - history..WINDOW].copy_from_slice(&self.window[WINDOW - history..]);
        self.floor = WINDOW - history;
        self.checked = WINDOW;
        let limit = chunk.len() - SLACK;
        let mut out = WINDOW;
        loop {
            match self.advance(chunk, &mut out, limit)? {
                Step::Full | Step::End => break,
                Step::NeedInput if out > WINDOW => break,
                Step::NeedInput if self.ended => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "unexpected end of gzip stream",
                    ));
                }
                Step::NeedInput => self.fill()?,
            }
        }
        self.check(chunk, out);
        let kept = (out - self.floor).min(WINDOW);
        self.window[WINDOW - kept..].copy_from_slice(&chunk[out - kept..out]);
        self.history = kept;
        Ok(out - WINDOW)
    }

    /// Include the member's bytes decoded so far in its checksum.
    fn check(&mut self, chunk: &[u8], out: usize) {
        let bytes = &chunk[self.checked..out];
        self.checksum.update(bytes);
        self.length = self.length.wrapping_add(bytes.len() as u32);
        self.checked = out;
    }

    /// Read more input, keeping what has not been consumed.
    fn fill(&mut self) -> io::Result<()> {
        self.unload();
        self.input.copy_within(self.position..self.filled, 0);
        self.filled -= self.position;
        self.position = 0;
        if self.filled + PADDING == self.input.len() {
            // Only a header can fill the buffer without being consumed.
            if self.input.len() >= MAX_INPUT {
                return Err(invalid("gzip header too long"));
            }
            self.input.resize(2 * self.input.len() - PADDING, 0);
        }
        let end = self.input.len() - PADDING;
        let read = loop {
            match self.reader.read(&mut self.input[self.filled..end]) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                result => break result?,
            }
        };
        self.ended = read == 0;
        self.filled += read;
        self.input[self.filled..self.filled + PADDING].fill(0);
        Ok(())
    }

    /// Return the whole bytes in the bit buffer to the input, leaving only
    /// the unused bits of a byte that has been started.
    fn unload(&mut self) {
        self.position -= (self.count / 8) as usize;
        self.count %= 8;
        self.bits &= low_bits(self.count);
    }

    /// Load input bytes until the bit buffer holds at least 56 bits.
    #[inline(always)]
    fn refill(&mut self) {
        let word: [u8; 8] = self.input[self.position..self.position + 8]
            .try_into()
            .unwrap();
        self.bits |= u64::from_le_bytes(word) << self.count;
        self.position += ((63 - self.count) >> 3) as usize;
        self.count |= 56;
    }

    #[inline(always)]
    fn take(&mut self, count: u32) -> u64 {
        let value = self.bits & low_bits(count);
        self.bits >>= count;
        self.count -= count;
        value
    }

    /// Whether more bits have been taken than the input has supplied; the
    /// excess came from the zeros after it.
    #[inline(always)]
    fn overread(&self) -> bool {
        self.position * 8 - self.count as usize > self.filled * 8
    }

    /// Decode as far as the buffered input and the chunk allow.
    fn advance(&mut self, chunk: &mut [u8], out: &mut usize, limit: usize) -> io::Result<Step> {
        loop {
            match self.state {
                State::End => return Ok(Step::End),
                State::Member => {
                    let buffered = &self.input[self.position..self.filled];
                    if buffered.is_empty() && self.ended {
                        self.state = State::End;
                        return Ok(Step::End);
                    }
                    let Some(length) = header_length(buffered)? else {
                        return Ok(Step::NeedInput);
                    };
                    self.position += length;
                    self.checksum = crc32fast::Hasher::new();
                    self.length = 0;
                    // A member cannot refer to the one before it.
                    self.floor = *out;
                    self.checked = *out;
                    self.state = State::Block;
                }
                State::Block => {
                    let saved = (self.bits, self.count, self.position);
                    let result = self.block_header();
                    if self.overread() {
                        (self.bits, self.count, self.position) = saved;
                        self.state = State::Block;
                        return Ok(Step::NeedInput);
                    }
                    result.map_err(invalid)?;
                }
                State::Stored(left) => {
                    if *out >= limit {
                        return Ok(Step::Full);
                    }
                    let length = left.min(self.filled - self.position).min(limit - *out);
                    chunk[*out..*out + length]
                        .copy_from_slice(&self.input[self.position..self.position + length]);
                    self.position += length;
                    *out += length;
                    if length == left {
                        self.state = self.after_block();
                    } else {
                        self.state = State::Stored(left - length);
                        if *out < limit {
                            return Ok(Step::NeedInput);
                        }
                    }
                }
                State::Coded => {
                    if *out >= limit {
                        return Ok(Step::Full);
                    }
                    if self.filled - self.position.min(self.filled) >= FAST_INPUT {
                        if let Fast::EndOfBlock =
                            self.coded_fast(chunk, out, limit).map_err(invalid)?
                        {
                            self.state = self.after_block();
                        }
                    } else if let Some(step) = self.coded_careful(chunk, out, limit)? {
                        return Ok(step);
                    }
                }
                State::Trailer => {
                    // The rest of the last byte is padding.
                    self.position -= (self.count / 8) as usize;
                    (self.bits, self.count) = (0, 0);
                    let Some(trailer) =
                        self.input[..self.filled].get(self.position..self.position + 8)
                    else {
                        return Ok(Step::NeedInput);
                    };
                    let checksum = u32::from_le_bytes(trailer[..4].try_into().unwrap());
                    let length = u32::from_le_bytes(trailer[4..].try_into().unwrap());
                    self.position += 8;
                    self.check(chunk, *out);
                    if self.checksum.clone().finalize() != checksum || self.length != length {
                        return Err(invalid(
                            "corrupt gzip stream does not have a matching checksum",
                        ));
                    }
                    self.floor = *out;
                    self.state = State::Member;
                }
            }
        }
    }

    fn after_block(&self) -> State {
        if self.last_block {
            State::Trailer
        } else {
            State::Block
        }
    }

    /// Read a block header and prepare to decode the block. The caller
    /// checks for an overread, which makes any result meaningless.
    fn block_header(&mut self) -> Result<(), &'static str> {
        self.refill();
        self.last_block = self.take(1) != 0;
        match self.take(2) {
            0 => {
                // Stored: the rest of the byte is skipped, then come the
                // length and its complement.
                self.take(self.count % 8);
                self.refill();
                let length = self.take(16) as u16;
                if length != !(self.take(16) as u16) {
                    return Err("invalid stored block lengths");
                }
                // The bytes themselves are copied from the input buffer.
                self.position -= (self.count / 8) as usize;
                (self.bits, self.count) = (0, 0);
                self.state = State::Stored(usize::from(length));
            }
            1 => {
                let mut lens = [8_u8; 288 + 32];
                lens[144..256].fill(9);
                lens[256..280].fill(7);
                lens[288..].fill(5);
                self.build(&lens, 288, 32)?;
            }
            2 => {
                let literals = self.take(5) as usize + 257;
                let offsets = self.take(5) as usize + 1;
                let precodes = self.take(4) as usize + 4;
                if literals > 286 || offsets > 30 {
                    return Err("too many length or distance symbols");
                }
                let mut lens = [0_u8; 19];
                for &symbol in &PRECODE_ORDER[..precodes] {
                    self.refill();
                    lens[usize::from(symbol)] = self.take(3) as u8;
                }
                let mut precode = [0_u32; PRECODE_ENOUGH];
                build_table(
                    &mut precode,
                    &lens,
                    &PRECODE_RESULTS,
                    PRECODE_BITS,
                    7,
                    false,
                )
                .map_err(|_| "invalid code lengths set")?;
                let mut lens = [0_u8; 288 + 32];
                let mut index = 0;
                while index < literals + offsets {
                    self.refill();
                    let entry = precode[(self.bits & low_bits(PRECODE_BITS)) as usize];
                    if entry & EXCEPTIONAL != 0 {
                        return Err("invalid code lengths set");
                    }
                    self.take(entry & 0xff);
                    let symbol = (entry >> 16) as u8;
                    let (value, repeat) = match symbol {
                        0..=15 => (symbol, 1),
                        16 if index == 0 => return Err("invalid bit length repeat"),
                        16 => (lens[index - 1], 3 + self.take(2) as usize),
                        17 => (0, 3 + self.take(3) as usize),
                        _ => (0, 11 + self.take(7) as usize),
                    };
                    if index + repeat > literals + offsets {
                        return Err("invalid bit length repeat");
                    }
                    lens[index..index + repeat].fill(value);
                    index += repeat;
                }
                if lens[256] == 0 {
                    return Err("invalid code -- missing end-of-block");
                }
                self.build(&lens, literals, offsets)?;
            }
            _ => return Err("invalid block type"),
        }
        Ok(())
    }

    /// Build both decode tables from the codeword lengths of a block.
    fn build(&mut self, lens: &[u8], literals: usize, offsets: usize) -> Result<(), &'static str> {
        build_table(
            &mut self.litlen,
            &lens[..literals],
            &LITLEN_RESULTS,
            LITLEN_BITS,
            15,
            true,
        )
        .map_err(|_| "invalid literal/lengths set")?;
        build_table(
            &mut self.offsets,
            &lens[literals..literals + offsets],
            &OFFSET_RESULTS,
            OFFSET_BITS,
            15,
            true,
        )
        .map_err(|_| "invalid distances set")?;
        self.state = State::Coded;
        Ok(())
    }

    /// Decode symbols one at a time, undoing a symbol that used bits beyond
    /// the buffered input. Returns `None` at the end of the block.
    fn coded_careful(
        &mut self,
        chunk: &mut [u8],
        out: &mut usize,
        limit: usize,
    ) -> io::Result<Option<Step>> {
        while *out < limit {
            let saved = (self.bits, self.count, self.position);
            let symbol = self.symbol();
            if self.overread() {
                (self.bits, self.count, self.position) = saved;
                return Ok(Some(Step::NeedInput));
            }
            match symbol.map_err(invalid)? {
                Symbol::Literal(byte) => {
                    chunk[*out] = byte;
                    *out += 1;
                }
                Symbol::EndOfBlock => {
                    self.state = self.after_block();
                    return Ok(None);
                }
                Symbol::Match { length, distance } => {
                    if distance > *out - self.floor {
                        return Err(invalid("invalid distance too far back"));
                    }
                    for at in *out..*out + length {
                        chunk[at] = chunk[at - distance];
                    }
                    *out += length;
                }
            }
        }
        Ok(Some(Step::Full))
    }

    /// Decode one literal, match or end of block. The caller checks for an
    /// overread, which makes any result meaningless.
    fn symbol(&mut self) -> Result<Symbol, &'static str> {
        self.refill();
        let mut entry = self.litlen[(self.bits & low_bits(LITLEN_BITS)) as usize];
        if entry & (EXCEPTIONAL | SUBTABLE) == EXCEPTIONAL | SUBTABLE {
            self.take(entry & 0xff);
            let index = self.bits & low_bits((entry >> 8) & 0xf);
            entry = self.litlen[(entry >> 16) as usize + index as usize];
        }
        let extra = self.take(entry & 0xff) >> ((entry >> 8) & 0xf);
        if entry & LITERAL != 0 {
            return Ok(Symbol::Literal((entry >> 16) as u8));
        }
        if entry & EXCEPTIONAL != 0 {
            return if entry & END_OF_BLOCK != 0 {
                Ok(Symbol::EndOfBlock)
            } else {
                Err("invalid literal/length code")
            };
        }
        let length = (entry >> 16) as usize + extra as usize;
        self.refill();
        let mut entry = self.offsets[(self.bits & low_bits(OFFSET_BITS)) as usize];
        if entry & (EXCEPTIONAL | SUBTABLE) == EXCEPTIONAL | SUBTABLE {
            self.take(entry & 0xff);
            let index = self.bits & low_bits((entry >> 8) & 0xf);
            entry = self.offsets[(entry >> 16) as usize + index as usize];
        }
        let extra = self.take(entry & 0xff) >> ((entry >> 8) & 0xf);
        if entry & EXCEPTIONAL != 0 {
            return Err("invalid distance code");
        }
        Ok(Symbol::Match {
            length,
            distance: (entry >> 16) as usize + extra as usize,
        })
    }

    /// Decode symbols while at least `FAST_INPUT` bytes are buffered and the
    /// chunk has room, without checking either again within an iteration.
    fn coded_fast(
        &mut self,
        chunk: &mut [u8],
        out: &mut usize,
        limit: usize,
    ) -> Result<Fast, &'static str> {
        debug_assert!(self.filled - self.position >= FAST_INPUT && *out < limit);
        debug_assert!(limit + SLACK <= chunk.len());
        let (litlen, offsets) = (self.litlen.as_ptr(), self.offsets.as_ptr());
        let (input, base) = (self.input.as_ptr(), chunk.as_mut_ptr());
        // The last position at which an iteration may start.
        let last = self.filled - FAST_INPUT;
        let floor = self.floor;
        let (mut bits, mut count, mut position, mut at) =
            (self.bits, self.count, self.position, *out);

        // SAFETY (for all of the below): `position <= last` at the start of
        // an iteration, and an iteration refills at most twice, moving
        // `position` by at most 7 bytes each time; every 8-byte read is
        // therefore inside the buffered input. `at < limit` at the start of
        // an iteration, which writes at most four literals, or three and a
        // match of at most 258 bytes with up to 31 bytes copied beyond it,
        // all within the `SLACK` bytes that follow `limit`. A match is
        // copied only from `floor` or later, which is inside the chunk.
        // Table indices are bounded by the sizes checked when the tables
        // were built: a complete code fills its root table and every
        // subtable it points to.
        macro_rules! refill {
            () => {
                let word = unsafe { input.add(position).cast::<u64>().read_unaligned() };
                bits |= u64::from_le(word) << count;
                position += ((63 - count) >> 3) as usize;
                count |= 56;
            };
        }
        macro_rules! take {
            ($entry:expr) => {
                bits >>= $entry & 0xff;
                count -= $entry & 0xff;
            };
        }
        macro_rules! literal {
            ($entry:expr) => {
                unsafe { *base.add(at) = ($entry >> 16) as u8 };
                at += 1;
            };
        }
        macro_rules! lookup {
            () => {
                unsafe { *litlen.add((bits & low_bits(LITLEN_BITS)) as usize) }
            };
        }
        let result = loop {
            if position > last || at >= limit {
                break Ok(Fast::Pause);
            }
            refill!();
            let mut entry = lookup!();
            if entry & LITERAL != 0 {
                // Up to four literals fit in one refill: a root entry takes
                // at most 11 bits. Each is looked up before the last has
                // been written, so the lookups do not wait for the stores.
                take!(entry);
                literal!(entry);
                entry = lookup!();
                if entry & LITERAL != 0 {
                    take!(entry);
                    literal!(entry);
                    entry = lookup!();
                    if entry & LITERAL != 0 {
                        take!(entry);
                        literal!(entry);
                        entry = lookup!();
                        if entry & LITERAL != 0 {
                            take!(entry);
                            literal!(entry);
                            continue;
                        }
                    }
                }
                // Something else follows, which may need more bits than are
                // left. Refilling keeps the bits that `entry` was read from.
                refill!();
            }
            if entry & EXCEPTIONAL != 0 {
                if entry & SUBTABLE != 0 {
                    take!(entry);
                    let index = bits & low_bits((entry >> 8) & 0xf);
                    entry = unsafe { *litlen.add((entry >> 16) as usize + index as usize) };
                    if entry & LITERAL != 0 {
                        take!(entry);
                        literal!(entry);
                        continue;
                    }
                }
                if entry & EXCEPTIONAL != 0 {
                    take!(entry);
                    break if entry & END_OF_BLOCK != 0 {
                        Ok(Fast::EndOfBlock)
                    } else {
                        Err("invalid literal/length code")
                    };
                }
            }
            // A length: at most 20 bits here, then at most 28 for the
            // offset, of the 56 a refill guarantees.
            let extra = (bits & low_bits(entry & 0xff)) >> ((entry >> 8) & 0xf);
            take!(entry);
            let length = (entry >> 16) as usize + extra as usize;
            let mut entry = unsafe { *offsets.add((bits & low_bits(OFFSET_BITS)) as usize) };
            if entry & EXCEPTIONAL != 0 {
                if entry & SUBTABLE == 0 {
                    take!(entry);
                    break Err("invalid distance code");
                }
                take!(entry);
                let index = bits & low_bits((entry >> 8) & 0xf);
                entry = unsafe { *offsets.add((entry >> 16) as usize + index as usize) };
                if entry & EXCEPTIONAL != 0 {
                    take!(entry);
                    break Err("invalid distance code");
                }
            }
            let extra = (bits & low_bits(entry & 0xff)) >> ((entry >> 8) & 0xf);
            take!(entry);
            let distance = (entry >> 16) as usize + extra as usize;
            if distance > at - floor {
                break Err("invalid distance too far back");
            }
            // Copy whole words, possibly past the end of the match. A word
            // must not overlap its source, so short distances go slower.
            unsafe {
                let target = base.add(at);
                let source = target.sub(distance);
                if distance >= 16 {
                    // Nearly every match is this far back and at most 32
                    // bytes long: two words, copied without asking.
                    std::ptr::copy_nonoverlapping(source, target, 16);
                    std::ptr::copy_nonoverlapping(source.add(16), target.add(16), 16);
                    let mut done = 32;
                    while done < length {
                        std::ptr::copy_nonoverlapping(source.add(done), target.add(done), 16);
                        done += 16;
                    }
                } else if distance >= 8 {
                    let mut done = 0;
                    while done < length {
                        std::ptr::copy_nonoverlapping(source.add(done), target.add(done), 8);
                        done += 8;
                    }
                } else if distance == 1 {
                    let word = u64::from_ne_bytes([*source; 8]);
                    let mut done = 0;
                    while done < length {
                        target.add(done).cast::<u64>().write_unaligned(word);
                        done += 8;
                    }
                } else {
                    for done in 0..length {
                        *target.add(done) = *source.add(done);
                    }
                }
            }
            at += length;
        };
        (self.bits, self.count, self.position, *out) = (bits, count, position, at);
        result
    }
}

enum Symbol {
    Literal(u8),
    EndOfBlock,
    Match { length: usize, distance: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    fn gzip(data: &[u8], level: u32) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level));
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    /// A reader that returns at most `step` bytes per call.
    struct Slow<'a>(&'a [u8], usize);

    impl Read for Slow<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let length = self.0.len().min(self.1).min(buffer.len());
            buffer[..length].copy_from_slice(&self.0[..length]);
            self.0 = &self.0[length..];
            Ok(length)
        }
    }

    /// Decode everything, in chunks of at most `chunk` bytes.
    fn decode(reader: impl Read, chunk: usize) -> io::Result<Vec<u8>> {
        let mut decoder = GzDecoder::new(reader);
        let mut buffer = vec![0xa5; WINDOW + chunk + SLACK];
        let mut decoded = Vec::new();
        loop {
            let length = decoder.read_chunk(&mut buffer)?;
            if length == 0 {
                // The end is reported again, not an error.
                assert_eq!(decoder.read_chunk(&mut buffer)?, 0);
                return Ok(decoded);
            }
            assert!(length < chunk + SLACK);
            decoded.extend_from_slice(&buffer[WINDOW..WINDOW + length]);
        }
    }

    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next() % bound as u64) as usize
        }
    }

    /// Inputs that exercise stored, fixed and dynamic blocks, long and short
    /// codewords, matches at every distance, and runs.
    fn samples() -> Vec<(&'static str, Vec<u8>)> {
        let mut random = Random(0x1234_5678_9abc_def1);
        let mut samples = vec![
            ("empty", Vec::new()),
            ("one byte", vec![b'x']),
            ("short", b"ACGTACGTNNNN".to_vec()),
            ("run", vec![b'A'; 100_000]),
        ];
        // Incompressible bytes: stored blocks.
        samples.push((
            "random",
            (0..300_000).map(|_| random.next() as u8).collect(),
        ));
        // Reads with qualities and headers, like the real input.
        let mut fastq = Vec::new();
        let genome: Vec<u8> = (0..20_000).map(|_| b"ACGT"[random.below(4)]).collect();
        for read in 0..6_000 {
            let start = random.below(genome.len() - 100);
            fastq.extend_from_slice(
                format!("@SRR065390.{read} HWUSI-EAS687:8:1:{start}/1\n").as_bytes(),
            );
            fastq.extend_from_slice(&genome[start..start + 100]);
            fastq.extend_from_slice(b"\n+\n");
            fastq.extend((0..100).map(|_| b'!' + (random.below(40) * random.below(2)) as u8));
            fastq.push(b'\n');
        }
        samples.push(("fastq", fastq));
        // A skewed alphabet of all byte values: codewords of many lengths.
        samples.push((
            "skewed",
            (0..400_000)
                .map(|_| (random.next().trailing_zeros() * 7 + random.below(5) as u32) as u8)
                .collect(),
        ));
        // Copies of random lengths from random distances up to the window.
        let mut copies: Vec<u8> = (0..40_000).map(|_| random.next() as u8).collect();
        while copies.len() < 600_000 {
            let distance = 1 + random.below(WINDOW.min(copies.len()));
            let length = 3 + random.below(300);
            for _ in 0..length {
                copies.push(copies[copies.len() - distance]);
            }
            if random.below(3) == 0 {
                copies.push(random.next() as u8);
            }
        }
        samples.push(("copies", copies));
        // Short periods: overlapping copies at distances below a word.
        let mut periods = Vec::new();
        for period in 1..40 {
            let unit: Vec<u8> = (0..period).map(|_| random.next() as u8).collect();
            for _ in 0..(700 / period + 3) {
                periods.extend_from_slice(&unit);
            }
        }
        samples.push(("periods", periods));
        samples
    }

    #[test]
    fn decodes_what_flate2_encodes_at_every_level_and_chunk_size() {
        for (name, data) in samples() {
            for level in 0..=9 {
                let compressed = gzip(&data, level);
                for (chunk, step) in [(1 << 20, usize::MAX), (4096, 1000), (1, 7), (70_000, 1)] {
                    // A reader of one byte at a time is slow on large input.
                    if step == 1 && data.len() > 150_000 {
                        continue;
                    }
                    let decoded = decode(Slow(&compressed, step), chunk)
                        .unwrap_or_else(|error| panic!("{name} level {level}: {error}"));
                    assert!(
                        decoded == data,
                        "{name} level {level} chunk {chunk} step {step}"
                    );
                }
            }
        }
    }

    #[test]
    fn decodes_concatenated_members_and_optional_header_fields() {
        let mut random = Random(77);
        let parts: Vec<Vec<u8>> = (0..40)
            .map(|index| {
                let length = [0, 1, 5, 300, 40_000, 70_000][index % 6];
                (0..length).map(|_| b"ACGTN\n"[random.below(6)]).collect()
            })
            .collect();
        let mut compressed = Vec::new();
        for (index, part) in parts.iter().enumerate() {
            let mut member = gzip(part, (index % 10) as u32);
            if index % 3 == 1 {
                // Extra field, name, comment and header checksum.
                member[3] = 4 | 8 | 16 | 2;
                let fields = b"\x05\x00ex\x00raname\x00a comment\x00\x12\x34";
                // (The extra field holds a zero byte.)
                member.splice(10..10, fields.iter().copied());
            }
            compressed.extend_from_slice(&member);
        }
        let expected = parts.concat();
        for (chunk, step) in [(1 << 20, usize::MAX), (1000, 333), (65_536, 1)] {
            assert!(decode(Slow(&compressed, step), chunk).unwrap() == expected);
        }
    }

    #[test]
    fn reports_truncated_and_damaged_streams() {
        let data = &samples()[5].1[..3000];
        for level in [0, 1, 6] {
            let compressed = gzip(data, level);
            for length in 1..compressed.len() {
                let error = decode(Slow(&compressed[..length], 100), 500)
                    .expect_err("truncated input must fail");
                assert_eq!(
                    error.kind(),
                    io::ErrorKind::UnexpectedEof,
                    "length {length}"
                );
            }
            // Any damaged bit is either reported or changes nothing (unused
            // header fields); it never panics or returns other data.
            for bit in 0..compressed.len() * 8 {
                let mut damaged = compressed.clone();
                damaged[bit / 8] ^= 1 << (bit % 8);
                match decode(Slow(&damaged, 97), 300) {
                    Ok(decoded) => assert!(decoded == *data, "level {level} bit {bit}"),
                    Err(error) => assert!(matches!(
                        error.kind(),
                        io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof
                    )),
                }
            }
            // Bytes after the last member are not another member.
            let mut trailing = compressed.clone();
            trailing.extend_from_slice(b"more");
            assert_eq!(
                decode(Slow(&trailing, 100), 500).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
        }
    }

    #[test]
    fn random_bytes_never_panic() {
        let mut random = Random(5);
        let header = [0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
        for _ in 0..3000 {
            let mut bytes = header.to_vec();
            let length = random.below(400);
            bytes.extend((0..length).map(|_| random.next() as u8));
            // Favour coded blocks, where most of the decoder is.
            if length > 0 && random.below(4) != 0 {
                bytes[10] = (bytes[10] & !6) | [2, 4][random.below(2)];
            }
            assert!(decode(Slow(&bytes, 64), 1000).is_err());
        }
    }

    #[test]
    fn returns_what_is_decodable_before_waiting_for_more_input() {
        /// Fails the test if asked for more than it was given.
        struct Stalled<'a>(&'a [u8]);
        impl Read for Stalled<'_> {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                assert!(!self.0.is_empty(), "waited for input with output pending");
                let length = self.0.len().min(buffer.len());
                buffer[..length].copy_from_slice(&self.0[..length]);
                self.0 = &self.0[length..];
                Ok(length)
            }
        }
        for data in [b"@x\nAC GT\n+\n!!!!!\n".to_vec(), samples()[5].1.clone()] {
            for level in [1, 6] {
                // A sync flush ends a block without ending the stream.
                let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level));
                encoder.write_all(&data).unwrap();
                // (A second flush completes one that ran out of room.)
                encoder.flush().unwrap();
                encoder.flush().unwrap();
                let mut decoder = GzDecoder::new(Stalled(encoder.get_ref()));
                let mut buffer = vec![0; WINDOW + (1 << 20) + SLACK];
                let mut decoded = Vec::new();
                while decoded.len() < data.len() {
                    let length = decoder.read_chunk(&mut buffer).unwrap();
                    decoded.extend_from_slice(&buffer[WINDOW..WINDOW + length]);
                }
                assert!(decoded == data);
            }
        }
    }

    #[test]
    fn tables_decode_every_codeword_of_extreme_codes() {
        // The most lopsided complete code: lengths 1, 2, ..., 14, 15, 15.
        let mut lens = [0_u8; 288];
        for (symbol, len) in (1..=15).chain([15]).enumerate() {
            lens[symbol * 17] = len;
        }
        let mut table = vec![0; LITLEN_ENOUGH];
        build_table(&mut table, &lens, &LITLEN_RESULTS, LITLEN_BITS, 15, true).unwrap();
        // Walk the canonical code: each codeword must decode to its symbol.
        let mut code = 0_u32;
        for len in 1..=15_u32 {
            for symbol in (0..288).filter(|&symbol| u32::from(lens[symbol]) == len) {
                let bits = u64::from(code.reverse_bits() >> (32 - len));
                let mut entry = table[(bits & low_bits(LITLEN_BITS)) as usize];
                let mut taken = 0;
                if entry & SUBTABLE != 0 && entry & EXCEPTIONAL != 0 {
                    taken = entry & 0xff;
                    let index = (bits >> taken) & low_bits((entry >> 8) & 0xf);
                    entry = table[(entry >> 16) as usize + index as usize];
                }
                assert_eq!(taken + ((entry >> 8) & 0xf), len, "symbol {symbol}");
                assert_eq!(entry & !0xffff, LITLEN_RESULTS[symbol] & !0xffff);
                code += 1;
            }
            code <<= 1;
        }
        // Over- and undersubscribed codes are rejected.
        let mut over = lens;
        over[5] = 1;
        assert!(build_table(&mut table, &over, &LITLEN_RESULTS, LITLEN_BITS, 15, true).is_err());
        let mut under = lens;
        under[0] = 0;
        assert!(build_table(&mut table, &under, &LITLEN_RESULTS, LITLEN_BITS, 15, true).is_err());
    }
}
