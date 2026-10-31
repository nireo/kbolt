mod input;

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};
use flate2::bufread::MultiGzDecoder;
use kbolt::database::DatabaseReader;
use kbolt::{Counter, ParallelCounter, StrandMode};

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
    Count(CountArgs),
    /// Stream a kbolt binary database as TSV; errors may leave partial output
    Dump {
        /// Binary database file; '-' means stdin
        #[arg(default_value = "-")]
        input: PathBuf,
        /// TSV output file; '-' means stdout
        #[arg(short, long, default_value = "-")]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Tsv,
    Binary,
}

#[derive(Args)]
struct CountArgs {
    /// K-mer length (greater than zero)
    #[arg(short = 'k', long, value_parser = clap::value_parser!(usize))]
    k: usize,
    /// Count forward strands separately instead of canonicalizing
    #[arg(long)]
    forward: bool,
    /// Reserve space for this many distinct k-mers (a hint, not a limit)
    #[arg(long, default_value_t = 0)]
    capacity: usize,
    /// Counting workers; the input parser runs on the calling thread
    #[arg(short = 't', long, default_value_t = 1, value_parser = parse_threads)]
    threads: usize,
    /// Print counting, merge, and output wall times on stderr
    #[arg(long)]
    timings: bool,
    /// Output encoding (binary files can be exported with 'kbolt dump')
    #[arg(long = "format", value_enum, default_value = "tsv")]
    format: OutputFormat,
    /// Output file; '-' means stdout
    #[arg(short, long, default_value = "-")]
    output: PathBuf,
    /// Report periodic progress and a final summary on stderr
    #[arg(long)]
    progress: bool,
    /// FASTA/FASTQ files, optionally gzip-compressed; '-' means stdin
    #[arg(value_name = "INPUT", default_value = "-")]
    inputs: Vec<PathBuf>,
}

fn parse_threads(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|&n| n > 0 && n <= usize::MAX / 2)
        .ok_or_else(|| "threads must be a positive, representable worker count".to_owned())
}

enum CountSink<'a> {
    Serial(&'a mut Counter),
    Parallel(&'a mut ParallelCounter),
}

struct Counting<'a> {
    counter: CountSink<'a>,
    // Batch up to 64 KiB of new sequence bytes, keeping k-1 tail bytes after each
    // flush. Wrapped lines share a batch; record boundaries always flush/reset it.
    overlap: Vec<u8>,
    pending: usize,
    k: usize,
    records: u64,
    bases: u64,
    progress: bool,
    started: Instant,
    last_report: Instant,
}

impl<'a> Counting<'a> {
    fn new(counter: &'a mut Counter, k: usize, progress: bool) -> Self {
        Self::with_sink(CountSink::Serial(counter), k, progress)
    }

    fn with_sink(counter: CountSink<'a>, k: usize, progress: bool) -> Self {
        let now = Instant::now();
        Self {
            counter,
            overlap: Vec::new(),
            pending: 0,
            k,
            records: 0,
            bases: 0,
            progress,
            started: now,
            last_report: now,
        }
    }

    fn start_record(&mut self) -> io::Result<()> {
        self.flush()?;
        self.overlap.clear();
        self.records += 1;
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.pending == 0 {
            return Ok(());
        }
        match &mut self.counter {
            CountSink::Serial(counter) => counter.add_record(&self.overlap),
            CountSink::Parallel(counter) => counter.add_record(&self.overlap),
        }
        .map_err(io::Error::other)?;
        let keep = (self.k - 1).min(self.overlap.len());
        let start = self.overlap.len() - keep;
        self.overlap.copy_within(start.., 0);
        self.overlap.truncate(keep);
        self.pending = 0;
        Ok(())
    }

    fn sequence(&mut self, bytes: &[u8]) -> io::Result<()> {
        let mut remaining = bytes;
        while !remaining.is_empty() {
            let take = remaining.len().min(BUFFER_SIZE - self.pending);
            self.overlap.extend_from_slice(&remaining[..take]);
            self.pending += take;
            remaining = &remaining[take..];
            if self.pending == BUFFER_SIZE {
                self.flush()?;
            }
        }
        self.bases += bytes.len() as u64;
        if self.progress && self.last_report.elapsed() >= Duration::from_secs(1) {
            self.flush()?;
            self.report()?;
            self.last_report = Instant::now();
        }
        Ok(())
    }

    fn report(&self) -> io::Result<()> {
        match &self.counter {
            CountSink::Serial(counter) => writeln!(
                io::stderr().lock(),
                "{} records, {} bases, {} distinct k-mers, {:.1}s elapsed",
                self.records,
                self.bases,
                counter.len(),
                self.started.elapsed().as_secs_f64(),
            ),
            CountSink::Parallel(_) => writeln!(
                io::stderr().lock(),
                "{} records, {} bases submitted, {:.1}s elapsed",
                self.records,
                self.bases,
                self.started.elapsed().as_secs_f64(),
            ),
        }
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
    match cli.command {
        Command::Count(args) => count_inputs(args),
        Command::Dump { input, output } => dump(input, output),
    }
}

fn check_output(inputs: &[PathBuf], output: &Path) -> io::Result<()> {
    if output.as_os_str() == "-" {
        return Ok(());
    }
    let output_path = match std::fs::canonicalize(output) {
        Ok(path) => path,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    for path in inputs {
        if path.as_os_str() != "-" && std::fs::canonicalize(path)? == output_path {
            return Err(io::Error::other("output must not be an input file"));
        }
    }
    Ok(())
}

fn output_writer(output: &Path) -> io::Result<BufWriter<Box<dyn Write>>> {
    let writer: Box<dyn Write> = if output.as_os_str() == "-" {
        Box::new(io::stdout().lock())
    } else {
        Box::new(File::create(output)?)
    };
    Ok(BufWriter::with_capacity(BUFFER_SIZE, writer))
}

fn dump(input: PathBuf, output: PathBuf) -> io::Result<()> {
    check_output(std::slice::from_ref(&input), &output)?;
    let result = (|| {
        let source: Box<dyn Read> = if input.as_os_str() == "-" {
            Box::new(io::stdin().lock())
        } else {
            Box::new(File::open(&input)?)
        };
        let reader = DatabaseReader::new(BufReader::with_capacity(BUFFER_SIZE, source))?;
        let mut writer = output_writer(&output)?;
        reader.write_tsv(&mut writer)?;
        writer.flush()
    })();
    result.map_err(|error: io::Error| {
        io::Error::new(error.kind(), format!("{}: {error}", input.display()))
    })
}

fn count_inputs(args: CountArgs) -> io::Result<()> {
    let CountArgs {
        k,
        forward,
        capacity,
        threads,
        timings,
        format,
        output,
        progress,
        inputs,
    } = args;
    let strand = if forward {
        StrandMode::Forward
    } else {
        StrandMode::Canonical
    };
    if inputs.iter().filter(|path| path.as_os_str() == "-").count() > 1 {
        return Err(io::Error::other("stdin ('-') may only be specified once"));
    }
    // Open output only after all input has been read and validated.
    check_output(&inputs, &output)?;
    let started = Instant::now();
    let mut counter = Counter::with_capacity(k, strand, if threads == 1 { capacity } else { 0 })
        .map_err(io::Error::other)?;
    let mut parallel = if threads > 1 {
        Some(
            ParallelCounter::with_capacity(k, strand, threads, capacity)
                .map_err(io::Error::other)?,
        )
    } else {
        None
    };
    let mut counting = match &mut parallel {
        Some(parallel) => Counting::with_sink(CountSink::Parallel(parallel), k, progress),
        None => Counting::new(&mut counter, k, progress),
    };
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
    let records = counting.records;
    let bases = counting.bases;
    drop(counting);
    let (counting_time, merge_time) = if let Some(parallel) = parallel {
        let (result, stats) = parallel.finish_with_stats().map_err(io::Error::other)?;
        counter = result;
        (stats.counting, stats.merging)
    } else {
        (started.elapsed(), Duration::ZERO)
    };
    if progress {
        writeln!(
            io::stderr().lock(),
            "{records} records, {bases} bases, {} distinct k-mers, {:.1}s elapsed",
            counter.len(),
            started.elapsed().as_secs_f64()
        )?;
    }
    let output_started = Instant::now();
    let mut writer = output_writer(&output)?;
    match format {
        OutputFormat::Tsv => counter.write_tsv(&mut writer)?,
        OutputFormat::Binary => counter.write_binary(&mut writer)?,
    }
    writer.flush()?;
    if timings {
        writeln!(
            io::stderr().lock(),
            "timings: counting={:.6}s merging={:.6}s output={:.6}s total={:.6}s",
            counting_time.as_secs_f64(),
            merge_time.as_secs_f64(),
            output_started.elapsed().as_secs_f64(),
            started.elapsed().as_secs_f64()
        )?;
    }
    Ok(())
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
        for k in [1, 31, 32, 33, 63, 64, 65] {
            for strand in [StrandMode::Forward, StrandMode::Canonical] {
                let mut expected = Counter::new(k, strand).unwrap();
                for record in [sequence.as_slice(), b"AC", b"GT"] {
                    expected.add_record(record).unwrap();
                }
                let mut actual = Counter::new(k, strand).unwrap();
                let mut counting = Counting::new(&mut actual, k, false);
                count_reader(fasta.as_slice(), &mut counting).unwrap();
                assert_eq!(counting.pending, 0);
                assert!(counting.overlap.len() < k);
                assert!(counting.overlap.capacity() <= 2 * (BUFFER_SIZE + k - 1));
                assert_eq!(counting.records, 3);
                assert_eq!(counting.bases, sequence.len() as u64 + 4);
                assert_eq!(
                    actual.counts().collect::<std::collections::HashMap<_, _>>(),
                    expected
                        .counts()
                        .collect::<std::collections::HashMap<_, _>>(),
                    "k={k}, strand={strand:?}",
                );
            }
        }
    }

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
