mod input;

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use flate2::bufread::MultiGzDecoder;
use kbolt::{Counter, StrandMode};

const BUFFER_SIZE: usize = 64 * 1024;

#[derive(Parser)]
#[command(
    version,
    about = "Exact DNA k-mer counting with streaming FASTA/FASTQ input"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Count k-mers; the distinct-k-mer table remains in memory
    Count {
        /// K-mer length (greater than zero)
        #[arg(short = 'k', long, value_parser = clap::value_parser!(usize))]
        k: usize,
        /// Count forward strands separately instead of canonicalizing
        #[arg(long)]
        forward: bool,
        /// Write tab-separated k-mer/count pairs here (default: stdout)
        #[arg(short, long, default_value = "-")]
        output: PathBuf,
        /// Report periodic progress and a final summary on stderr
        #[arg(long)]
        progress: bool,
        /// FASTA/FASTQ files, optionally gzip-compressed; '-' means stdin
        #[arg(value_name = "INPUT", default_value = "-")]
        inputs: Vec<PathBuf>,
    },
}

struct Counting<'a> {
    counter: &'a mut Counter,
    // Retain only k-1 sequence bytes between chunks. Recounting this overlap
    // produces no duplicate windows, including when a record is shorter than k.
    overlap: Vec<u8>,
    k: usize,
    records: u64,
    bases: u64,
    progress: bool,
    started: Instant,
    last_report: Instant,
}

impl<'a> Counting<'a> {
    fn new(counter: &'a mut Counter, k: usize, progress: bool) -> Self {
        let now = Instant::now();
        Self {
            counter,
            overlap: Vec::new(),
            k,
            records: 0,
            bases: 0,
            progress,
            started: now,
            last_report: now,
        }
    }

    fn start_record(&mut self) {
        self.overlap.clear();
        self.records += 1;
    }

    fn sequence(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.overlap.extend_from_slice(bytes);
        self.counter
            .add_record(&self.overlap)
            .map_err(io::Error::other)?;
        let keep = (self.k - 1).min(self.overlap.len());
        let start = self.overlap.len() - keep;
        self.overlap.copy_within(start.., 0);
        self.overlap.truncate(keep);
        self.bases += bytes.len() as u64;
        if self.progress && self.last_report.elapsed() >= Duration::from_secs(1) {
            self.report()?;
            self.last_report = Instant::now();
        }
        Ok(())
    }

    fn report(&self) -> io::Result<()> {
        writeln!(
            io::stderr().lock(),
            "{} records, {} bases, {} distinct k-mers, {:.1}s elapsed",
            self.records,
            self.bases,
            self.counter.len(),
            self.started.elapsed().as_secs_f64(),
        )
    }
}

fn count_reader<R: Read>(reader: R, counting: &mut Counting<'_>) -> io::Result<()> {
    let mut reader = BufReader::with_capacity(BUFFER_SIZE, reader);
    // Inspect bytes rather than extensions, including for stdin. Loop to
    // handle readers which return only one byte per read.
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
    let reader = io::Cursor::new(magic[..size].to_vec()).chain(reader);
    let reader = BufReader::with_capacity(BUFFER_SIZE, reader);
    if size == 2 && magic == [0x1f, 0x8b] {
        input::count(
            BufReader::with_capacity(BUFFER_SIZE, MultiGzDecoder::new(reader)),
            counting,
        )
    } else {
        input::count(reader, counting)
    }
}

fn run(cli: Cli) -> io::Result<()> {
    let Command::Count {
        k,
        forward,
        output,
        progress,
        inputs,
    } = cli.command;
    let strand = if forward {
        StrandMode::Forward
    } else {
        StrandMode::Canonical
    };
    let mut counter = Counter::new(k, strand).map_err(io::Error::other)?;
    if inputs.iter().filter(|path| path.as_os_str() == "-").count() > 1 {
        return Err(io::Error::other("stdin ('-') may only be specified once"));
    }
    // Refuse the same path (including symlinks) for input and output. Open
    // output only after ALL input has been read and validated.
    let output_path = if output.as_os_str() == "-" {
        None
    } else {
        match std::fs::canonicalize(&output) {
            Ok(path) => Some(path),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error),
        }
    };
    if let Some(output_path) = output_path {
        for path in &inputs {
            if path.as_os_str() != "-" && std::fs::canonicalize(path)? == output_path {
                return Err(io::Error::other("output must not be an input file"));
            }
        }
    }
    let mut counting = Counting::new(&mut counter, k, progress);
    for path in inputs {
        let result = if path.as_os_str() == "-" {
            count_reader(io::stdin().lock(), &mut counting)
        } else {
            File::open(&path).and_then(|file| count_reader(file, &mut counting))
        };
        result.map_err(|error| {
            io::Error::new(error.kind(), format!("{}: {error}", path.display()))
        })?;
    }
    if progress {
        counting.report()?;
    }
    let writer: Box<dyn Write> = if output.as_os_str() == "-" {
        Box::new(io::stdout().lock())
    } else {
        Box::new(File::create(&output)?)
    };
    let mut writer = BufWriter::with_capacity(BUFFER_SIZE, writer);
    for (key, count) in counter.counts() {
        writer.write_all(&key)?;
        writeln!(writer, "\t{count}")?;
    }
    writer.flush()
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("kbolt: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_generated_large_record_with_bounded_input_storage() {
        let bases = 4 * 1024 * 1024;
        let reader = io::Cursor::new(b">generated\n")
            .chain(io::repeat(b'A').take(bases))
            .chain(io::Cursor::new(b"\n"));
        let mut counter = Counter::new(31, StrandMode::Canonical).unwrap();
        let mut counting = Counting::new(&mut counter, 31, false);
        count_reader(reader, &mut counting).unwrap();
        assert_eq!(counting.records, 1);
        assert_eq!(counting.bases, bases);
        assert_eq!(counting.overlap.len(), 30);
        assert!(counting.overlap.capacity() <= 2 * (BUFFER_SIZE + 30));
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
            let mut counting = Counting::new(&mut counter, 3, false);
            count_reader(SingleByte(bytes), &mut counting).unwrap();
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
        let reader = io::Cursor::new(b">x\nACGT\n").chain(FailingReader);
        let mut counter = Counter::new(3, StrandMode::Canonical).unwrap();
        let mut counting = Counting::new(&mut counter, 3, false);
        assert!(
            count_reader(reader, &mut counting)
                .unwrap_err()
                .to_string()
                .contains("injected read failure")
        );
    }
}
