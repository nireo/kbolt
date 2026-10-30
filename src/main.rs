mod input;

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};
use flate2::bufread::MultiGzDecoder;
use kbolt::database::DatabaseReader;
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
    let mut counter = Counter::with_capacity(k, strand, capacity).map_err(io::Error::other)?;
    if inputs.iter().filter(|path| path.as_os_str() == "-").count() > 1 {
        return Err(io::Error::other("stdin ('-') may only be specified once"));
    }
    // Open output only after all input has been read and validated.
    check_output(&inputs, &output)?;
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
    let mut writer = output_writer(&output)?;
    match format {
        OutputFormat::Tsv => counter.write_tsv(&mut writer)?,
        OutputFormat::Binary => counter.write_binary(&mut writer)?,
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
