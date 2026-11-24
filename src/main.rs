use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};
use kbolt::database::DatabaseReader;
use kbolt::input::{self, InputStats, Progress};
use kbolt::{Counter, DiskCounter, ParallelCounter, ParallelInput, StrandMode};

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
    #[arg(short = 'k', long)]
    k: usize,
    /// Count forward strands separately instead of canonicalizing
    #[arg(long)]
    forward: bool,
    /// Reserve space for this many distinct k-mers (a hint, not a limit)
    #[arg(long, default_value_t = 0)]
    capacity: usize,
    /// Counting workers; input readers run separately
    #[arg(short = 't', long, default_value_t = 1, value_parser = parse_threads)]
    threads: usize,
    /// Concurrent input-file readers when threads > 1 (bounded by input count)
    #[arg(long, default_value_t = 2, value_parser = parse_threads)]
    readers: usize,
    /// Count on disk within about this much memory, such as 4G or 500M
    /// (k <= 32). Without it, all k-mers are counted in memory.
    #[arg(short = 'm', long, value_name = "SIZE", value_parser = parse_size)]
    memory: Option<u64>,
    /// Directory for the temporary file of --memory [default: the system's]
    #[arg(long, value_name = "DIR", requires = "memory")]
    temp_dir: Option<PathBuf>,
    /// Write k-mers in alphabetical order. Sorting takes one more pass over
    /// the distinct k-mers, and with --memory a second temporary file of 8
    /// bytes each
    #[arg(long)]
    sorted: bool,
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

/// Parse a byte count with an optional K, M, G or T suffix (powers of 1024).
fn parse_size(value: &str) -> Result<u64, String> {
    let digits = value.trim_end_matches(|c: char| c.is_ascii_alphabetic());
    let shift = match value[digits.len()..].to_ascii_uppercase().as_str() {
        "" | "B" => 0,
        "K" | "KB" | "KIB" => 10,
        "M" | "MB" | "MIB" => 20,
        "G" | "GB" | "GIB" => 30,
        "T" | "TB" | "TIB" => 40,
        _ => return Err("expected a size such as 4G or 500M".to_owned()),
    };
    digits
        .parse::<u64>()
        .ok()
        .and_then(|size| size.checked_mul(1 << shift))
        .filter(|&size| size >= MIN_MEMORY)
        .ok_or_else(|| "expected a size of at least 64M, such as 4G or 500M".to_owned())
}

/// The smallest --memory accepted: below it, fixed buffers dominate.
const MIN_MEMORY: u64 = 64 << 20;

fn report_progress(progress: Progress<'_>) -> io::Result<()> {
    let mut stderr = io::stderr().lock();
    if let Some(source) = progress.source {
        write!(stderr, "{}: ", source.display())?;
    }
    match progress.distinct {
        Some(distinct) => writeln!(
            stderr,
            "{} records, {} bases, {} distinct k-mers, {:.1}s elapsed",
            progress.records,
            progress.bases,
            distinct,
            progress.elapsed.as_secs_f64(),
        ),
        None => writeln!(
            stderr,
            "{} records, {} bases submitted, {:.1}s elapsed",
            progress.records,
            progress.bases,
            progress.elapsed.as_secs_f64(),
        ),
    }
}

fn count_cli_path(path: &Path, stream: &mut input::Stream<'_, '_>) -> io::Result<()> {
    if path.as_os_str() == "-" {
        stream
            .add_reader(io::stdin())
            .map_err(|error| io::Error::new(error.kind(), format!("-: {error}")))
    } else {
        stream.add_file(path)
    }
}

fn count_parallel_inputs<'a>(
    input: impl Fn() -> ParallelInput<'a> + Sync,
    inputs: &[PathBuf],
    readers: usize,
    progress: bool,
) -> io::Result<InputStats> {
    let on_progress: Option<input::ProgressCallback<'_>> = progress.then_some(&report_progress);
    if inputs.iter().any(|path| path.as_os_str() == "-") {
        let mut stream = input::Stream::parallel(input());
        if let Some(report) = on_progress {
            stream = stream.with_progress(report);
        }
        for path in inputs {
            count_cli_path(path, &mut stream)?;
        }
        return stream.finish();
    }
    input::count_parallel_files(input, inputs, readers, on_progress)
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
        if path.as_os_str() != "-"
            && std::fs::canonicalize(path).map_err(|error| {
                io::Error::new(error.kind(), format!("{}: {error}", path.display()))
            })? == output_path
        {
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
    let with_input = |error: io::Error| {
        io::Error::new(error.kind(), format!("{}: {error}", input.display()))
    };
    let source: Box<dyn Read> = if input.as_os_str() == "-" {
        Box::new(io::stdin().lock())
    } else {
        Box::new(File::open(&input).map_err(with_input)?)
    };
    let reader =
        DatabaseReader::new(BufReader::with_capacity(BUFFER_SIZE, source)).map_err(with_input)?;
    let mut writer = output_writer(&output).map_err(|error| {
        io::Error::new(error.kind(), format!("{}: {error}", output.display()))
    })?;
    reader.write_tsv(&mut writer).map_err(with_input)?;
    writer.flush()
}

fn count_inputs(args: CountArgs) -> io::Result<()> {
    let CountArgs {
        k,
        forward,
        capacity,
        threads,
        readers,
        memory,
        temp_dir,
        sorted,
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
    if let Some(memory) = memory {
        // Bins on disk, counted one at a time; nothing is held as a `Counter`.
        if !(1..=32).contains(&k) {
            return Err(io::Error::other("--memory supports k from 1 to 32"));
        }
        let directory = temp_dir.unwrap_or_else(std::env::temp_dir);
        let counter = DiskCounter::new(k, strand, threads, memory, &directory)?;
        let InputStats { records, bases } =
            count_parallel_inputs(|| counter.input(), &inputs, readers, progress)?;
        // The reader threads are done; give their cores to counting.
        let bins = counter
            .finish()?
            .with_threads(threads + readers.min(inputs.len()))
            .sorted(sorted);
        let splitting = started.elapsed();
        let (kmers, temporary) = (bins.kmers(), bins.temporary_bytes());
        let distinct = match format {
            OutputFormat::Tsv => {
                let mut writer = output_writer(&output)?;
                let distinct = bins.write_tsv(&mut writer)?;
                writer.flush()?;
                distinct
            }
            // The header is completed after the entries, which needs a file.
            OutputFormat::Binary if output.as_os_str() == "-" => {
                return Err(io::Error::other(
                    "--memory with --format binary needs an output file (-o)",
                ));
            }
            OutputFormat::Binary => {
                let mut writer = BufWriter::with_capacity(BUFFER_SIZE, File::create(&output)?);
                let distinct = bins.write_binary(&mut writer)?;
                writer.flush()?;
                distinct
            }
        };
        if progress {
            writeln!(
                io::stderr().lock(),
                "{records} records, {bases} bases, {distinct} distinct k-mers, {:.1}s elapsed",
                started.elapsed().as_secs_f64()
            )?;
        }
        if timings {
            writeln!(
                io::stderr().lock(),
                "timings: splitting={:.6}s counting_and_output={:.6}s total={:.6}s \
                 kmers={kmers} distinct={distinct} temporary_bytes={temporary}",
                splitting.as_secs_f64(),
                (started.elapsed() - splitting).as_secs_f64(),
                started.elapsed().as_secs_f64()
            )?;
        }
        return Ok(());
    }
    let (counter, InputStats { records, bases }, counting_time, merge_time) = if threads == 1 {
        let mut counter = Counter::with_capacity(k, strand, capacity).map_err(io::Error::other)?;
        let mut stream = input::Stream::serial(&mut counter);
        if progress {
            stream = stream.with_progress(&report_progress);
        }
        for path in &inputs {
            count_cli_path(path, &mut stream)?;
        }
        let stats = stream.finish()?;
        (counter, stats, started.elapsed(), Duration::ZERO)
    } else {
        let parallel = ParallelCounter::with_capacity(k, strand, threads, capacity)
            .map_err(io::Error::other)?;
        let stats = count_parallel_inputs(|| parallel.input(), &inputs, readers, progress)?;
        let (counter, parallel_stats) = parallel.finish_with_stats().map_err(io::Error::other)?;
        (counter, stats, parallel_stats.counting, parallel_stats.merging)
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
    match (format, sorted) {
        (OutputFormat::Tsv, false) => counter.write_tsv(&mut writer)?,
        (OutputFormat::Tsv, true) => counter.write_tsv_sorted(&mut writer)?,
        (OutputFormat::Binary, false) => counter.write_binary(&mut writer)?,
        (OutputFormat::Binary, true) => counter.write_binary_sorted(&mut writer)?,
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
