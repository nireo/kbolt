//! Bounded-memory FASTA/FASTQ parsing. Headers, sequence lines, and quality
//! lines are consumed in buffer-sized fragments, never accumulated as records.
//!
//! The parser accepts LF or CRLF, wrapped FASTA and FASTQ, and an unterminated
//! final line. FASTQ quality lengths are validated without storing qualities.

use std::io::{self, BufRead};

use crate::Counting;

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

    fn sequence(&mut self, counting: &mut Counting<'_>) -> io::Result<u64> {
        self.line(|bytes| {
            // Alphabetic ambiguity codes and alignment gaps break valid runs
            // in Counter. Whitespace and structural punctuation are errors.
            if !bytes
                .iter()
                .all(|byte| byte.is_ascii_alphabetic() || matches!(byte, b'.' | b'-'))
            {
                return Err(invalid(
                    "invalid sequence byte (expected letters, '.' or '-')",
                ));
            }
            counting.sequence(bytes)
        })
    }
}

pub(super) fn count<R: BufRead>(reader: R, counting: &mut Counting<'_>) -> io::Result<()> {
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

fn fasta<R: BufRead>(lines: &mut Lines<R>, counting: &mut Counting<'_>) -> io::Result<()> {
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

fn fastq<R: BufRead>(lines: &mut Lines<R>, counting: &mut Counting<'_>) -> io::Result<()> {
    while let Some(first) = lines.peek()? {
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
                if !bytes.iter().all(|byte| (b'!'..=b'~').contains(byte)) {
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
    use std::io::{BufReader, Cursor};

    use kbolt::{Counter, StrandMode};

    use super::*;

    fn counts(counter: &Counter) -> HashMap<Vec<u8>, u64> {
        counter.counts().collect()
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
                        let mut counting = Counting::new(&mut counter, k, false);
                        let reader = BufReader::with_capacity(size, Cursor::new(fixture));
                        count(reader, &mut counting).unwrap();
                        assert_eq!(counting.records, records.len() as u64);
                        assert_eq!(counting.bases, records.iter().map(|r| r.len() as u64).sum());
                        assert!(counting.overlap.len() < k);
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
                let mut counting = Counting::new(&mut counter, 3, false);
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
        let mut counting = Counting::new(&mut counter, 3, false);
        count(
            BufReader::with_capacity(13, fasta.as_slice()),
            &mut counting,
        )
        .unwrap();
        assert_eq!(counter.counts().collect::<Vec<_>>(), [(b"ACG".to_vec(), 1)]);
    }

    #[test]
    fn accepts_empty_input_and_empty_records() {
        for fixture in [b"".as_slice(), b">empty\n", b"@empty\n\n+\n\n"] {
            let mut counter = Counter::new(3, StrandMode::Canonical).unwrap();
            let mut counting = Counting::new(&mut counter, 3, false);
            count(BufReader::new(fixture), &mut counting).unwrap();
            assert!(counter.is_empty());
        }
    }
}
