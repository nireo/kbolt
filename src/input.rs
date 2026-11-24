//! Streaming FASTA/FASTQ input with automatic gzip detection and concurrent
//! decompression.
//!
//! Headers, sequence lines, and quality lines are consumed in buffer-sized
//! fragments, never accumulated as whole records. The parser accepts LF or
//! CRLF, wrapped FASTA and FASTQ, and an unterminated final line. FASTQ
//! quality lengths are validated without storing qualities.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::{Counter, ParallelInput, gzip};

const BUFFER_SIZE: usize = 64 * 1024;
/// Input read buffer. Larger reads mean fewer syscalls and fewer FASTQ
/// records straddling a refill, which fall back to the line-by-line parser.
const READ_BUFFER_SIZE: usize = 256 * 1024;
/// Decompressed bytes handed from the decoder thread to the parser at a time.
const DECODED_CHUNK_SIZE: usize = 1024 * 1024;
/// A decoded chunk's allocation: room for the decoder's window before the
/// data and its slack after (see `crate::gzip`).
const DECODED_BUFFER_SIZE: usize = gzip::WINDOW + DECODED_CHUNK_SIZE + gzip::SLACK;

/// Record and base counts reported after streaming FASTA/FASTQ input.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InputStats {
    pub records: u64,
    pub bases: u64,
}

/// Periodic progress update emitted while streaming input.
#[derive(Debug, Clone, Copy)]
pub struct Progress<'a> {
    /// File currently being read when multiple files are read concurrently.
    pub source: Option<&'a Path>,
    pub records: u64,
    pub bases: u64,
    /// Distinct k-mers counted so far in single-threaded mode; `None` when
    /// counting asynchronously on worker threads.
    pub distinct: Option<usize>,
    pub elapsed: Duration,
}

/// Callback invoked periodically with [`Progress`] updates while reading input.
pub type ProgressCallback<'p> = &'p (dyn Fn(Progress<'_>) -> io::Result<()> + Sync);

struct SerialBuffer<'a> {
    counter: &'a mut Counter,
    // Batch up to 64 KiB of new sequence bytes from one or more records, then
    // count them together. `ends` closes each finished record in the batch;
    // the current record starts at `record_start`. Each flush keeps the current
    // record's last k-1 bytes so windows continue across flushes.
    overlap: Vec<u8>,
    ends: Vec<usize>,
    record_start: usize,
    pending: usize,
    k: usize,
}

impl SerialBuffer<'_> {
    fn start_record(&mut self) {
        if self.overlap.len() > self.record_start {
            self.ends.push(self.overlap.len());
        }
        self.record_start = self.overlap.len();
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.pending == 0 {
            return Ok(());
        }
        let (bytes, mut start) = (&self.overlap, 0);
        let records = self
            .ends
            .iter()
            .copied()
            .chain(std::iter::once(bytes.len()))
            .map(|end| {
                let record = &bytes[start..end];
                start = end;
                record
            });
        self.counter
            .add_records(records)
            .map_err(io::Error::other)?;
        let keep = (self.k - 1).min(self.overlap.len() - self.record_start);
        let start = self.overlap.len() - keep;
        self.overlap.copy_within(start.., 0);
        self.overlap.truncate(keep);
        self.ends.clear();
        self.record_start = 0;
        self.pending = 0;
        Ok(())
    }

    fn push(&mut self, mut remaining: &[u8]) -> io::Result<()> {
        while !remaining.is_empty() {
            let take = remaining.len().min(BUFFER_SIZE - self.pending);
            self.overlap.extend_from_slice(&remaining[..take]);
            self.pending += take;
            remaining = &remaining[take..];
            if self.pending == BUFFER_SIZE {
                self.flush()?;
            }
        }
        Ok(())
    }
}

enum CountSink<'a> {
    Serial(SerialBuffer<'a>),
    Parallel(ParallelInput<'a>),
}

/// Streams one or more FASTA/FASTQ readers or files into a serial [`Counter`]
/// or a [`ParallelInput`] producer, with optional periodic progress callbacks.
pub struct Stream<'a, 'p> {
    counter: CountSink<'a>,
    records: u64,
    bases: u64,
    on_progress: Option<ProgressCallback<'p>>,
    started: Instant,
    last_report: Instant,
    source: Option<PathBuf>,
}

impl<'a, 'p> Stream<'a, 'p> {
    /// Create a stream that batches and counts into `counter`.
    pub fn serial(counter: &'a mut Counter) -> Self {
        let k = counter.k();
        Self::with_sink(CountSink::Serial(SerialBuffer {
            counter,
            overlap: Vec::new(),
            ends: Vec::new(),
            record_start: 0,
            pending: 0,
            k,
        }))
    }

    /// Create a stream that feeds a [`ParallelInput`] producer.
    pub fn parallel(input: ParallelInput<'a>) -> Self {
        Self::with_sink(CountSink::Parallel(input))
    }

    fn with_sink(counter: CountSink<'a>) -> Self {
        let now = Instant::now();
        Self {
            counter,
            records: 0,
            bases: 0,
            on_progress: None,
            started: now,
            last_report: now,
            source: None,
        }
    }

    /// Call `on_progress` about once per second while reading input.
    pub fn with_progress(mut self, on_progress: ProgressCallback<'p>) -> Self {
        self.on_progress = Some(on_progress);
        self
    }

    /// Attach a source path label reported in [`Progress::source`].
    pub fn with_source(mut self, source: impl Into<PathBuf>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Read FASTA or FASTQ input from `reader`, auto-detecting gzip from the
    /// content rather than a file extension.
    pub fn add_reader<R: Read + Send + 'static>(&mut self, reader: R) -> io::Result<()> {
        count_reader(reader, self)
    }

    /// Open `path` and read FASTA or FASTQ input from it, auto-detecting gzip
    /// from content. Errors are prefixed with `path`.
    pub fn add_file(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        File::open(path)
            .and_then(|file| count_reader(file, self))
            .map_err(|error| io::Error::new(error.kind(), format!("{}: {error}", path.display())))
    }

    /// Flush buffered sequence data, finalize any parallel producer, and return
    /// the total records and bases read.
    pub fn finish(mut self) -> io::Result<InputStats> {
        self.flush()?;
        if let CountSink::Parallel(input) = self.counter {
            input.finish().map_err(io::Error::other)?;
        }
        Ok(InputStats {
            records: self.records,
            bases: self.bases,
        })
    }

    #[cfg(test)]
    fn serial_buffer(&self) -> &SerialBuffer<'a> {
        match &self.counter {
            CountSink::Serial(serial) => serial,
            CountSink::Parallel(_) => panic!("expected serial sink"),
        }
    }

    fn start_record(&mut self) -> io::Result<()> {
        match &mut self.counter {
            CountSink::Parallel(counter) => counter.start_record(),
            CountSink::Serial(serial) => serial.start_record(),
        }
        self.records += 1;
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        match &mut self.counter {
            CountSink::Serial(serial) => serial.flush(),
            CountSink::Parallel(_) => Ok(()),
        }
    }

    fn sequence(&mut self, bytes: &[u8]) -> io::Result<()> {
        match &mut self.counter {
            CountSink::Parallel(counter) => counter.add_sequence(bytes).map_err(io::Error::other)?,
            CountSink::Serial(serial) => serial.push(bytes)?,
        }
        self.bases += bytes.len() as u64;
        if let Some(report) = self.on_progress
            && self.last_report.elapsed() >= Duration::from_secs(1)
        {
            self.flush()?;
            report(Progress {
                source: self.source.as_deref(),
                records: self.records,
                bases: self.bases,
                distinct: match &self.counter {
                    CountSink::Serial(serial) => Some(serial.counter.len()),
                    CountSink::Parallel(_) => None,
                },
                elapsed: self.started.elapsed(),
            })?;
            self.last_report = Instant::now();
        }
        Ok(())
    }
}

fn concurrent_file(path: &Path) -> bool {
    fn descriptor_alias(path: &Path) -> bool {
        path == Path::new("/dev/stdin")
            || path.starts_with("/dev/fd")
            || path.starts_with("/proc/self/fd")
            || path.starts_with("/proc/thread-self/fd")
    }
    if descriptor_alias(path) {
        return false;
    }
    let Ok(resolved) = std::fs::canonicalize(path) else {
        return false;
    };
    // fdescfs can report a regular file while opening it duplicates a shared
    // descriptor offset. Resolve symlinks so these aliases also stay ordered.
    !descriptor_alias(&resolved) && std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file())
}

/// Count FASTA/FASTQ files into parallel producers created by `input`, using
/// up to `readers` concurrent file-reading threads when every path is a regular
/// file.
///
/// Non-regular files (such as FIFOs or descriptor aliases) are read
/// sequentially on the calling thread so an error in one file cannot block on
/// another open pipe.
pub fn count_parallel_files<'a, P: AsRef<Path> + Sync>(
    input: impl Fn() -> ParallelInput<'a> + Sync,
    paths: &[P],
    readers: usize,
    on_progress: Option<ProgressCallback<'_>>,
) -> io::Result<InputStats> {
    if readers == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "reader count must be greater than zero",
        ));
    }
    // Blocking devices/FIFOs (including stdin aliases) cannot be interrupted
    // when another file fails. Only regular files use concurrent readers.
    if readers <= 1
        || paths.len() <= 1
        || !paths.iter().all(|path| concurrent_file(path.as_ref()))
    {
        let mut stream = Stream::parallel(input());
        if let Some(report) = on_progress {
            stream = stream.with_progress(report);
        }
        for path in paths {
            stream.add_file(path)?;
        }
        return stream.finish();
    }
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for index in 0..readers.min(paths.len()) {
            let next = &next;
            let input = &input;
            handles.push(
                std::thread::Builder::new()
                    .name(format!("reader-{index}"))
                    .spawn_scoped(scope, move || {
                        let mut totals = InputStats::default();
                        while let Some(path) = paths.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let path = path.as_ref();
                            let mut stream = Stream::parallel(input()).with_source(path);
                            if let Some(report) = on_progress {
                                stream = stream.with_progress(report);
                            }
                            stream.add_file(path)?;
                            let stats = stream.finish()?;
                            totals.records += stats.records;
                            totals.bases += stats.bases;
                        }
                        Ok::<_, io::Error>(totals)
                    })?,
            );
        }
        let mut totals = InputStats::default();
        for handle in handles {
            let stats = handle
                .join()
                .map_err(|_| io::Error::other("input reader failed"))??;
            totals.records += stats.records;
            totals.bases += stats.bases;
        }
        Ok(totals)
    })
}

fn count_reader<R: Read + Send + 'static>(
    mut reader: R,
    counting: &mut Stream<'_, '_>,
) -> io::Result<()> {
    // Inspect bytes rather than extensions, including for stdin. Loop to
    // handle readers which return only one byte per read. Reading the magic
    // directly (not through a second buffer) avoids copying all input twice.
    let mut magic = [0; 2];
    let mut size = 0;
    while size < magic.len() {
        match reader.read(&mut magic[size..]) {
            Ok(0) => break,
            Ok(n) => size += n,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    let reader = io::Cursor::new(magic).take(size as u64).chain(reader);
    if size == 2 && magic == [0x1f, 0x8b] {
        // The decoder buffers its own input.
        decompress_concurrently(gzip::GzDecoder::new(reader), |decoded| {
            count(decoded, counting)
        })
    } else {
        count(BufReader::with_capacity(READ_BUFFER_SIZE, reader), counting)
    }
}

/// Run `decoder` on its own thread while `parse` consumes its output.
///
/// Inflating and parsing a single gzip stream are both sequential, but they
/// need not share a thread. Bounded, recycled chunks keep memory constant.
/// Decoder errors arrive after all bytes decoded before them, in stream order.
/// Each chunk holds whatever the input read so far decodes to, so input
/// arriving slowly (such as from a pipe) is parsed as soon as it can be
/// decoded.
///
/// The decoder thread owns its input. When parsing fails, this returns
/// without waiting for it: a decoder blocked reading a slow pipe would
/// otherwise delay the error. It exits at its next send (or with the process).
fn decompress_concurrently<R: Read + Send + 'static>(
    mut decoder: gzip::GzDecoder<R>,
    parse: impl FnOnce(&mut DecodedChunks) -> io::Result<()>,
) -> io::Result<()> {
    let (sender, receiver) = crossbeam_channel::bounded::<io::Result<(Vec<u8>, usize)>>(4);
    let (recycle, recycled) = crossbeam_channel::bounded::<Vec<u8>>(8);
    let decoding = std::thread::Builder::new()
        .name("decoder".into())
        .spawn(move || {
            loop {
                // Buffers keep their full length; the decoded bytes follow
                // the first `gzip::WINDOW`.
                let mut chunk = recycled
                    .try_recv()
                    .unwrap_or_else(|_| vec![0; DECODED_BUFFER_SIZE]);
                match decoder.read_chunk(&mut chunk) {
                    Ok(0) => return, // End of stream: dropping the sender signals EOF.
                    Ok(len) => {
                        if sender.send(Ok((chunk, len))).is_err() {
                            return; // The parser stopped early.
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error));
                        return;
                    }
                }
            }
        })?;
    let mut chunks = DecodedChunks {
        receiver,
        recycle,
        chunk: Vec::new(),
        len: 0,
        position: 0,
    };
    let result = parse(&mut chunks);
    // Dropping the receiver releases a decoder blocked on a full queue.
    drop(chunks);
    result?;
    // Parsing succeeded, so it read to the end of the stream and the decoder
    // has finished. A panicking decoder also ends the stream early; report it
    // rather than accepting truncated input.
    decoding
        .join()
        .map_err(|_| io::Error::other("gzip decoder failed"))
}

/// A `BufRead` view over chunks produced by `decompress_concurrently`.
struct DecodedChunks {
    receiver: crossbeam_channel::Receiver<io::Result<(Vec<u8>, usize)>>,
    recycle: crossbeam_channel::Sender<Vec<u8>>,
    // The current chunk; bytes `position..len` are unread.
    chunk: Vec<u8>,
    len: usize,
    position: usize,
}

impl Read for DecodedChunks {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        let len = available.len().min(buffer.len());
        buffer[..len].copy_from_slice(&available[..len]);
        self.consume(len);
        Ok(len)
    }
}

impl BufRead for DecodedChunks {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        while self.position == self.len {
            match self.receiver.recv() {
                Ok(chunk) => {
                    let (chunk, len) = chunk?;
                    let used = std::mem::replace(&mut self.chunk, chunk);
                    (self.len, self.position) = (gzip::WINDOW + len, gzip::WINDOW);
                    // Only full-size buffers are reused for decoding.
                    if used.len() == DECODED_BUFFER_SIZE {
                        let _ = self.recycle.try_send(used);
                    }
                }
                // The decoder finished (or failed after reporting its error).
                Err(_) => return Ok(&[]),
            }
        }
        Ok(&self.chunk[self.position..self.len])
    }

    fn consume(&mut self, amount: usize) {
        self.position = (self.position + amount).min(self.len);
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

struct Lines<R> {
    reader: R,
    number: u64,
}

impl<R: BufRead> Lines<R> {
    fn peek(&mut self) -> io::Result<Option<u8>> {
        Ok(self.reader.fill_buf()?.first().copied())
    }

    fn line(&mut self, mut visit: impl FnMut(&[u8]) -> io::Result<()>) -> io::Result<u64> {
        self.number += 1;
        self.line_inner(&mut visit)
            .map_err(|error| io::Error::new(error.kind(), format!("line {}: {error}", self.number)))
    }

    fn line_inner(&mut self, visit: &mut impl FnMut(&[u8]) -> io::Result<()>) -> io::Result<u64> {
        let mut length = 0_u64;
        // A CR at the end of a buffer might be half of a CRLF. Withhold it
        // until the next fragment, so even a split CRLF is removed correctly.
        let mut pending_cr = false;
        loop {
            let buffer = self.reader.fill_buf()?;
            if buffer.is_empty() {
                return Ok(length);
            }
            let newline = memchr::memchr(b'\n', buffer);
            let end = newline.unwrap_or(buffer.len());
            let mut bytes = &buffer[..end];
            if pending_cr && !bytes.is_empty() {
                visit(b"\r")?;
                length = length
                    .checked_add(1)
                    .ok_or_else(|| invalid("line too long"))?;
            }
            pending_cr = bytes.last() == Some(&b'\r');
            if pending_cr {
                bytes = &bytes[..bytes.len() - 1];
            }
            if !bytes.is_empty() {
                visit(bytes)?;
                length = length
                    .checked_add(bytes.len() as u64)
                    .ok_or_else(|| invalid("line too long"))?;
            }
            self.reader.consume(end + usize::from(newline.is_some()));
            if newline.is_some() {
                return Ok(length);
            }
        }
    }

    fn header(&mut self) -> io::Result<()> {
        let mut first = true;
        let mut has_name = false;
        self.line(|bytes| {
            let bytes = if first {
                first = false;
                &bytes[1..]
            } else {
                bytes
            };
            has_name |= bytes.iter().any(|byte| !byte.is_ascii_whitespace());
            Ok(())
        })?;
        if !has_name {
            return Err(invalid(format!(
                "line {}: missing record name",
                self.number
            )));
        }
        Ok(())
    }

    fn sequence(&mut self, counting: &mut Stream<'_, '_>) -> io::Result<u64> {
        self.line(|bytes| {
            if !valid_sequence(bytes) {
                return Err(invalid(
                    "invalid sequence byte (expected letters, '.' or '-')",
                ));
            }
            counting.sequence(bytes)
        })
    }

    /// Count consecutive complete, single-line FASTQ records held entirely in
    /// the current buffer, stopping before anything else.
    ///
    /// Only records the general parser would accept identically are taken
    /// here: four newline-terminated lines, a named header, no wrapping, valid
    /// bytes, and matching sequence/quality lengths. A record that straddles
    /// the buffer end, wraps, or is invalid is left unconsumed, so the general
    /// parser handles it and reports any error with the same line number.
    fn fastq_records(&mut self, counting: &mut Stream<'_, '_>) -> io::Result<()> {
        let buffer = self.reader.fill_buf()?;
        let mut consumed = 0;
        let mut records = 0;
        let mut result = Ok(());
        while let Some((record, sequence)) = split_fastq_record(&buffer[consumed..]) {
            counting.start_record()?;
            if let Err(error) = counting.sequence(sequence) {
                // Prefix the sequence line's number, as `Lines::line` does.
                let line = self.number + 4 * records + 2;
                result = Err(io::Error::new(
                    error.kind(),
                    format!("line {line}: {error}"),
                ));
                break;
            }
            consumed += record;
            records += 1;
        }
        self.reader.consume(consumed);
        self.number += 4 * records;
        result
    }
}

/// Alphabetic ambiguity codes and alignment gaps break valid runs in Counter;
/// whitespace and structural punctuation are errors.
// Branch-free folds let LLVM validate many bytes per vector instruction.
#[inline]
fn valid_sequence(bytes: &[u8]) -> bool {
    bytes.iter().fold(true, |valid, &byte| {
        let folded = byte | 0x20;
        valid & ((folded.wrapping_sub(b'a') < 26) | (byte == b'.') | (byte == b'-'))
    })
}

/// Quality scores are printable ASCII, '!' through '~'.
#[inline]
fn valid_quality(bytes: &[u8]) -> bool {
    bytes.iter().fold(true, |valid, &byte| {
        valid & (byte.wrapping_sub(b'!') <= b'~' - b'!')
    })
}

/// Remove one line-ending CR, matching `Lines::line`.
#[inline]
fn without_cr(line: &[u8]) -> &[u8] {
    line.strip_suffix(b"\r").unwrap_or(line)
}

/// Split one complete four-line FASTQ record from the start of `bytes`,
/// returning its length including the final newline and its sequence.
/// Returns `None` for anything the fast path does not handle.
#[inline]
fn split_fastq_record(bytes: &[u8]) -> Option<(usize, &[u8])> {
    if bytes.first() != Some(&b'@') {
        return None;
    }
    let header_end = memchr::memchr(b'\n', bytes)?;
    let sequence_start = header_end + 1;
    let sequence_end = sequence_start + memchr::memchr(b'\n', &bytes[sequence_start..])?;
    let separator_start = sequence_end + 1;
    // Anything other than '+' here is a wrapped sequence (or an error).
    if bytes.get(separator_start) != Some(&b'+') {
        return None;
    }
    let quality_start = separator_start + memchr::memchr(b'\n', &bytes[separator_start..])? + 1;
    let quality_end = quality_start + memchr::memchr(b'\n', &bytes[quality_start..])?;
    let header = without_cr(&bytes[1..header_end]);
    let sequence = without_cr(&bytes[sequence_start..sequence_end]);
    let quality = without_cr(&bytes[quality_start..quality_end]);
    // `valid_sequence` rejects a leading '+', which the general parser would
    // read as the separator of an empty record.
    let accepted = sequence.len() == quality.len()
        && header.iter().any(|byte| !byte.is_ascii_whitespace())
        && valid_sequence(sequence)
        && valid_quality(quality);
    accepted.then_some((quality_end + 1, sequence))
}

fn count<R: BufRead>(reader: R, counting: &mut Stream<'_, '_>) -> io::Result<()> {
    let mut lines = Lines { reader, number: 0 };
    let result = match lines.peek()? {
        None => Ok(()),
        Some(b'>') => fasta(&mut lines, counting),
        Some(b'@') => fastq(&mut lines, counting),
        _ => Err(invalid("line 1: expected a FASTA '>' or FASTQ '@' header")),
    };
    result?;
    counting.flush()
}

fn fasta<R: BufRead>(lines: &mut Lines<R>, counting: &mut Stream<'_, '_>) -> io::Result<()> {
    while let Some(first) = lines.peek()? {
        if first == b'>' {
            lines.header()?;
            counting.start_record()?;
        } else {
            lines.sequence(counting)?;
        }
    }
    Ok(())
}

fn fastq<R: BufRead>(lines: &mut Lines<R>, counting: &mut Stream<'_, '_>) -> io::Result<()> {
    loop {
        lines.fastq_records(counting)?;
        let Some(first) = lines.peek()? else { break };
        if first != b'@' {
            return Err(invalid(format!(
                "line {}: expected a FASTQ '@' header",
                lines.number + 1
            )));
        }
        lines.header()?;
        counting.start_record()?;
        let mut sequence_length = 0_u64;
        loop {
            match lines.peek()? {
                Some(b'+') => {
                    // Optional separator text is not stored or interpreted.
                    lines.line(|_| Ok(()))?;
                    break;
                }
                Some(_) => {
                    sequence_length = sequence_length
                        .checked_add(lines.sequence(counting)?)
                        .ok_or_else(|| invalid("sequence too long"))?;
                }
                None => return Err(invalid("unexpected end of FASTQ: missing '+' separator")),
            }
        }
        let mut remaining = sequence_length;
        // Even an empty sequence requires one (empty) quality line.
        loop {
            if lines.peek()?.is_none() {
                return Err(invalid("unexpected end of FASTQ: truncated quality scores"));
            }
            lines.line(|bytes| {
                if !valid_quality(bytes) {
                    return Err(invalid(
                        "quality scores must be printable ASCII ('!' through '~')",
                    ));
                }
                remaining = remaining
                    .checked_sub(bytes.len() as u64)
                    .ok_or_else(|| invalid("more quality scores than sequence bases"))?;
                Ok(())
            })?;
            if remaining == 0 {
                break;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::{BufReader, Cursor, Write};
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::{Counter, DiskCounter, ParallelCounter, StrandMode};

    use super::*;

    fn counts(counter: &Counter) -> HashMap<Vec<u8>, u64> {
        counter.counts().collect()
    }

    fn temp_dir() -> PathBuf {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "kbolt-input-test-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        path
    }

    #[test]
    fn all_chunk_boundaries_match_whole_records_for_all_backends() {
        let records: Vec<Vec<u8>> = vec![
            b"AC".to_vec(),
            b"GT".to_vec(),
            Vec::new(),
            b"acgTNacgtACGT.--acg".to_vec(),
            b"ACGTGTCAGATTCGAC".repeat(9),
        ];
        let mut fasta = Vec::new();
        let mut fastq = Vec::new();
        for (index, record) in records.iter().enumerate() {
            fasta.extend_from_slice(format!(">record {index}\r\n").as_bytes());
            fastq.extend_from_slice(format!("@record {index}\r\n").as_bytes());
            for chunk in record.chunks(7) {
                fasta.extend_from_slice(chunk);
                fasta.extend_from_slice(b"\r\n");
                fastq.extend_from_slice(chunk);
                fastq.extend_from_slice(b"\r\n");
            }
            fastq.extend_from_slice(b"+\r\n");
            // Quality wrapping need not match sequence wrapping; '@' and '+'
            // in qualities must not be mistaken for record markers.
            let qualities: Vec<u8> = b"@+!~".iter().copied().cycle().take(record.len()).collect();
            if qualities.is_empty() {
                fastq.extend_from_slice(b"\r\n");
            }
            for chunk in qualities.chunks(11) {
                fastq.extend_from_slice(chunk);
                fastq.extend_from_slice(b"\r\n");
            }
        }
        // Exercise EOF without a newline.
        fastq.truncate(fastq.len() - 2);
        fasta.truncate(fasta.len() - 2);
        for k in [1, 3, 31, 32, 33, 63, 64, 65, 200] {
            for strand in [StrandMode::Canonical, StrandMode::Forward] {
                let mut expected = Counter::new(k, strand).unwrap();
                for record in &records {
                    expected.add_record(record).unwrap();
                }
                for size in [1, 2, 3, 7, 32, 67, 1024] {
                    for fixture in [&fasta, &fastq] {
                        let mut counter = Counter::new(k, strand).unwrap();
                        let mut counting = Stream::serial(&mut counter);
                        let reader = BufReader::with_capacity(size, Cursor::new(fixture));
                        count(reader, &mut counting).unwrap();
                        assert_eq!(counting.records, records.len() as u64);
                        assert_eq!(counting.bases, records.iter().map(|r| r.len() as u64).sum());
                        assert!(counting.serial_buffer().overlap.len() < k);
                        counting.finish().unwrap();
                        assert_eq!(counts(&counter), counts(&expected), "k={k}, buffer={size}");
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_malformed_inputs_even_with_one_byte_buffers() {
        for fixture in [
            b"ACGT\n".as_slice(),
            b">\nACGT\n",
            b"> \r\nACGT\n",
            b">x\nAC GT\n",
            b">x\nAC\rGT\n",
            b">x\nAC\xffGT\n",
            b"@x\nACGT\n",
            b"@x\nACGT\n+\n!!!\n",
            b"@x\nACGT\n+\n!!!!!\n",
            b"@x\nACGT\n+\n!! !\n",
            b"@x\nACGT\n+\n!!\x7f!\n",
            b"@x\nACGT\n+\n!!!!\nnot a header\n",
            b"@x\n+\n",
            b"@\nACGT\n+\n!!!!\n",
        ] {
            for size in [1, 2, 64] {
                let mut counter = Counter::new(3, StrandMode::Canonical).unwrap();
                let mut counting = Stream::serial(&mut counter);
                let error = count(BufReader::with_capacity(size, fixture), &mut counting)
                    .expect_err(&format!("fixture={fixture:?}"));
                assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            }
        }
    }

    #[test]
    fn skips_large_headers_in_bounded_fragments() {
        let header = vec![b'x'; 200_000];
        let mut fasta = b">".to_vec();
        fasta.extend_from_slice(&header);
        fasta.extend_from_slice(b"\nACG\n");
        let mut counter = Counter::new(3, StrandMode::Forward).unwrap();
        let mut counting = Stream::serial(&mut counter);
        count(
            BufReader::with_capacity(13, fasta.as_slice()),
            &mut counting,
        )
        .unwrap();
        counting.finish().unwrap();
        assert_eq!(counter.counts().collect::<Vec<_>>(), [(b"ACG".to_vec(), 1)]);
    }

    #[test]
    fn accepts_empty_input_and_empty_records() {
        for fixture in [b"".as_slice(), b">empty\n", b"@empty\n\n+\n\n"] {
            let mut counter = Counter::new(3, StrandMode::Canonical).unwrap();
            let mut counting = Stream::serial(&mut counter);
            count(BufReader::new(fixture), &mut counting).unwrap();
            counting.finish().unwrap();
            assert!(counter.is_empty());
        }
    }

    #[test]
    fn batched_wrapped_records_preserve_windows_and_reset_at_records() {
        let mut sequence = b"ACGTTGCAGATTCGAC".repeat(10_000);
        for index in [BUFFER_SIZE - 1, BUFFER_SIZE, 2 * BUFFER_SIZE + 5] {
            sequence[index] = b'N';
        }
        let mut fasta = b">large\r\n".to_vec();
        for part in sequence.chunks(77) {
            fasta.extend_from_slice(part);
            fasta.extend_from_slice(b"\r\n");
        }
        fasta.extend_from_slice(b">short\r\nAC\r\n>short2\r\nGT");
        for k in [1, 31, 32, 33, 63, 64, 65, 128, 129] {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let mut expected = Counter::new(k, strand).unwrap();
                for record in [sequence.as_slice(), b"AC", b"GT"] {
                    expected.add_record(record).unwrap();
                }
                let mut actual = Counter::new(k, strand).unwrap();
                let mut counting = Stream::serial(&mut actual);
                count_reader(Cursor::new(fasta.clone()), &mut counting).unwrap();
                assert_eq!(counting.serial_buffer().pending, 0);
                assert!(counting.serial_buffer().overlap.len() < k);
                assert!(counting.serial_buffer().overlap.capacity() <= 2 * (BUFFER_SIZE + k - 1));
                assert_eq!(counting.records, 3);
                assert_eq!(counting.bases, sequence.len() as u64 + 4);
                counting.finish().unwrap();
                assert_eq!(counts(&actual), counts(&expected), "k={k}, strand={strand:?}");
            }
        }
    }

    #[test]
    fn counts_generated_large_record_with_bounded_input_storage() {
        let bases = 4 * 1024 * 1024;
        let reader = Cursor::new(b">generated\n")
            .chain(io::repeat(b'A').take(bases))
            .chain(Cursor::new(b"\n"));
        let mut counter = Counter::new(31, StrandMode::Canonical).unwrap();
        let mut counting = Stream::serial(&mut counter);
        count_reader(reader, &mut counting).unwrap();
        assert_eq!(counting.records, 1);
        assert_eq!(counting.bases, bases);
        assert_eq!(counting.serial_buffer().overlap.len(), 30);
        assert!(counting.serial_buffer().overlap.capacity() <= 2 * (BUFFER_SIZE + 30));
        counting.finish().unwrap();
        assert_eq!(
            counter.counts().collect::<Vec<_>>(),
            [(vec![b'A'; 31], bases - 30)]
        );
    }

    #[test]
    fn detects_compression_with_short_reads() {
        struct SingleByte<R>(R);
        impl<R: Read> Read for SingleByte<R> {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                let len = buffer.len().min(1);
                self.0.read(&mut buffer[..len])
            }
        }
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(b">x\nACGT\n").unwrap();
        let compressed = encoder.finish().unwrap();
        for bytes in [b">x\nACGT\n".as_slice(), compressed.as_slice()] {
            let mut counter = Counter::new(3, StrandMode::Canonical).unwrap();
            let stats = counter
                .add_reader(SingleByte(Cursor::new(bytes.to_vec())))
                .unwrap();
            assert_eq!(stats, InputStats { records: 1, bases: 4 });
            assert_eq!(counter.counts().collect::<Vec<_>>(), [(b"ACG".to_vec(), 2)]);
        }
    }

    #[test]
    fn propagates_input_io_failures() {
        struct FailingReader;
        impl Read for FailingReader {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("injected read failure"))
            }
        }
        let reader = Cursor::new(b">x\nACGT\n").chain(FailingReader);
        let mut counter = Counter::new(3, StrandMode::Canonical).unwrap();
        assert!(
            counter
                .add_reader(reader)
                .unwrap_err()
                .to_string()
                .contains("injected read failure")
        );
    }

    #[test]
    fn file_helpers_work_across_counter_types_and_compression() {
        let dir = temp_dir();
        let plain = dir.join("a.fa");
        let compressed = dir.join("b.fq.gz");
        std::fs::write(&plain, b">r1\nACGTAC\n").unwrap();
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        gz.write_all(b"@r2\ncgtacg\n+\n!!!!!!\n").unwrap();
        std::fs::write(&compressed, gz.finish().unwrap()).unwrap();

        let files = [&plain, &compressed];

        let mut serial = Counter::new(3, StrandMode::Canonical).unwrap();
        let serial_stats = serial.add_files(files).unwrap();
        assert_eq!(serial_stats, InputStats { records: 2, bases: 12 });

        let parallel = ParallelCounter::new(3, StrandMode::Canonical, 2).unwrap();
        let parallel_stats = parallel.add_files(&files, 2).unwrap();
        assert_eq!(parallel_stats, serial_stats);
        let parallel = parallel.finish().unwrap();
        assert_eq!(counts(&parallel), counts(&serial));

        let disk = DiskCounter::new(3, StrandMode::Canonical, 2, 64 << 20, &dir).unwrap();
        let disk_stats = disk.add_files(&files, 2).unwrap();
        assert_eq!(disk_stats, serial_stats);
        let mut tsv = Vec::new();
        disk.finish().unwrap().write_tsv(&mut tsv).unwrap();
        let mut expected_tsv = Vec::new();
        serial.write_tsv(&mut expected_tsv).unwrap();
        let mut actual_lines: Vec<_> = tsv.split(|&b| b == b'\n').collect();
        let mut expected_lines: Vec<_> = expected_tsv.split(|&b| b == b'\n').collect();
        actual_lines.sort();
        expected_lines.sort();
        assert_eq!(actual_lines, expected_lines);

        // Zero readers is rejected.
        let parallel = ParallelCounter::new(3, StrandMode::Canonical, 2).unwrap();
        assert_eq!(
            parallel.add_files(&files, 0).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
