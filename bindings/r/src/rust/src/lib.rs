use std::cell::RefCell;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use extendr_api::prelude::*;
use kbolt::database::{DatabaseHeader, DatabaseReader, KeyOrder};
use kbolt::{CountFilter, Counter, DiskCounter, ParallelCounter, StrandMode};

type Result<T> = std::result::Result<T, String>;

const IO_BUFFER_SIZE: usize = 64 * 1024;
const MIN_DISK_MEMORY: u64 = 64 << 20;

struct RCounter(RefCell<Counter>);

struct RDatabaseReader {
    reader: Option<DatabaseReader<BufReader<File>>>,
    header: DatabaseHeader,
}

struct RDatabase(RefCell<RDatabaseReader>);

fn r_err(msg: impl ToString) -> String {
    msg.to_string()
}

fn strand_mode(canonical: bool) -> StrandMode {
    if canonical {
        StrandMode::Canonical
    } else {
        StrandMode::Forward
    }
}

fn parse_u64_arg(value: f64, name: &str, min: u64) -> Result<u64> {
    if !value.is_finite() || value.fract() != 0.0 || value < min as f64 {
        return Err(r_err(format!(
            "{name} must be a finite integer >= {min}"
        )));
    }
    if value > (1_u64 << 53) as f64 && value > u64::MAX as f64 {
        return Err(r_err(format!("{name} exceeds u64::MAX")));
    }
    Ok(value as u64)
}

fn parse_usize_arg(value: f64, name: &str, min: usize) -> Result<usize> {
    let n = parse_u64_arg(value, name, min as u64)?;
    usize::try_from(n).map_err(|_| r_err(format!("{name} does not fit usize")))
}

fn parse_filter(min_count: f64, max_count: Nullable<f64>) -> Result<CountFilter> {
    let min_count = parse_u64_arg(min_count, "min_count", 0)?;
    let max_count = match max_count {
        Nullable::Null => None,
        Nullable::NotNull(v) => Some(parse_u64_arg(v, "max_count", 0)?),
    };
    Ok(CountFilter {
        min_count,
        max_count,
    })
}

fn parse_paths(paths: Strings) -> Result<Vec<PathBuf>> {
    let mut out = Vec::with_capacity(paths.len());
    for item in paths.iter() {
        let s: &str = item.as_ref();
        if item.is_na() || s.is_empty() {
            return Err(r_err("paths must not contain NA or empty strings"));
        }
        out.push(PathBuf::from(s));
    }
    Ok(out)
}

fn parse_path(path: &str, name: &str) -> Result<PathBuf> {
    if path.is_empty() {
        return Err(r_err(format!("{name} must not be empty")));
    }
    Ok(PathBuf::from(path))
}

fn parse_memory_size(memory: Robj) -> Result<u64> {
    if let Some(s) = memory.as_str() {
        let digits = s.trim_end_matches(|c: char| c.is_ascii_alphabetic());
        let shift = match s[digits.len()..].to_ascii_uppercase().as_str() {
            "" | "B" => 0,
            "K" | "KB" | "KIB" => 10,
            "M" | "MB" | "MIB" => 20,
            "G" | "GB" | "GIB" => 30,
            "T" | "TB" | "TIB" => 40,
            _ => {
                return Err(r_err(
                    "memory must be a byte count or size string such as '4G' or '500M'",
                ));
            }
        };
        let size = digits
            .parse::<u64>()
            .ok()
            .and_then(|n| n.checked_mul(1 << shift))
            .filter(|&n| n >= MIN_DISK_MEMORY)
            .ok_or_else(|| r_err("memory must be at least 64M, such as '4G' or '500M'"))?;
        return Ok(size);
    }
    if let Some(v) = memory.as_real() {
        return parse_u64_arg(v, "memory", MIN_DISK_MEMORY);
    }
    if let Some(v) = memory.as_integer() {
        if v < 0 {
            return Err(r_err("memory must be at least 64 MiB"));
        }
        return parse_u64_arg(v as f64, "memory", MIN_DISK_MEMORY);
    }
    Err(r_err(
        "memory must be a numeric byte count or size string such as '4G' or '500M'",
    ))
}

fn check_output_path(inputs: &[PathBuf], output: &Path) -> Result<()> {
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::metadata(parent).map_err(r_err)?;
    }
    let output_canonical = match std::fs::canonicalize(output) {
        Ok(path) => path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(r_err(error)),
    };
    for input in inputs {
        if let Ok(input_canonical) = std::fs::canonicalize(input)
            && input_canonical == output_canonical
        {
            return Err(r_err("output must not be an input file"));
        }
    }
    Ok(())
}

fn wrap_counter(counter: Counter) -> Robj {
    let mut ptr = ExternalPtr::new(RCounter(RefCell::new(counter)));
    ptr.set_class(["kbolt_counter"])
        .expect("setting S3 class on ExternalPtr");
    ptr.into_robj()
}

fn borrow_counter(obj: &Robj) -> Result<&ExternalPtr<RCounter>> {
    <&ExternalPtr<RCounter>>::try_from(obj)
        .map_err(|_| r_err("expected a kbolt_counter object"))
}

fn wrap_database(reader: DatabaseReader<BufReader<File>>) -> Robj {
    let header = reader.header();
    let mut ptr = ExternalPtr::new(RDatabase(RefCell::new(RDatabaseReader {
        reader: Some(reader),
        header,
    })));
    ptr.set_class(["kbolt_database"])
        .expect("setting S3 class on ExternalPtr");
    ptr.into_robj()
}

fn borrow_database(obj: &Robj) -> Result<&ExternalPtr<RDatabase>> {
    <&ExternalPtr<RDatabase>>::try_from(obj)
        .map_err(|_| r_err("expected a kbolt_database object"))
}

fn key_order_name(order: KeyOrder) -> &'static str {
    match order {
        KeyOrder::Unordered => "unordered",
        KeyOrder::Hash => "hash",
        KeyOrder::Kmer => "kmer",
    }
}

#[extendr]
fn rust_kbolt_counter(k: f64, canonical: bool, capacity: f64) -> Result<Robj> {
    let k = parse_usize_arg(k, "k", 1)?;
    let capacity = parse_usize_arg(capacity, "capacity", 0)?;
    let counter = Counter::with_capacity(k, strand_mode(canonical), capacity).map_err(r_err)?;
    Ok(wrap_counter(counter))
}

#[extendr]
fn rust_kbolt_k(counter: Robj) -> Result<i32> {
    let ptr = borrow_counter(&counter)?;
    let guard = ptr.0.try_borrow().map_err(r_err)?;
    i32::try_from(guard.k()).map_err(|_| r_err("k exceeds R integer range"))
}

#[extendr]
fn rust_kbolt_is_canonical(counter: Robj) -> Result<bool> {
    let ptr = borrow_counter(&counter)?;
    let guard = ptr.0.try_borrow().map_err(r_err)?;
    Ok(guard.strand() == StrandMode::Canonical)
}

#[extendr]
fn rust_kbolt_len(counter: Robj) -> Result<f64> {
    let ptr = borrow_counter(&counter)?;
    let guard = ptr.0.try_borrow().map_err(r_err)?;
    Ok(guard.len() as f64)
}

#[extendr]
fn rust_kbolt_add_records(counter: Robj, records: Strings) -> Result<()> {
    let ptr = borrow_counter(&counter)?;
    let mut guard = ptr.0.try_borrow_mut().map_err(r_err)?;
    let slices = records
        .iter()
        .filter(|s| !s.is_na())
        .map(|s| s.as_bytes());
    guard.add_records(slices).map_err(r_err)
}

#[extendr]
fn rust_kbolt_add_files(counter: Robj, paths: Strings) -> Result<Doubles> {
    let paths = parse_paths(paths)?;
    let ptr = borrow_counter(&counter)?;
    let mut guard = ptr.0.try_borrow_mut().map_err(r_err)?;
    let stats = guard.add_files(&paths).map_err(r_err)?;
    let mut out = Doubles::from_values([stats.records as f64, stats.bases as f64]);
    out.set_names(["records", "bases"]).map_err(r_err)?;
    Ok(out)
}

#[extendr]
fn rust_kbolt_merge(target: Robj, source: Robj) -> Result<()> {
    let target_ptr = borrow_counter(&target)?;
    let source_ptr = borrow_counter(&source)?;
    if std::ptr::eq(
        target_ptr.try_addr().map_err(r_err)?,
        source_ptr.try_addr().map_err(r_err)?,
    ) {
        return Err(r_err("cannot merge a kbolt_counter into itself"));
    }
    let mut target_guard = target_ptr.0.try_borrow_mut().map_err(r_err)?;
    let mut source_guard = source_ptr.0.try_borrow_mut().map_err(r_err)?;
    if target_guard.k() != source_guard.k() || target_guard.strand() != source_guard.strand() {
        return Err(r_err(kbolt::Error::IncompatibleCounters));
    }
    let replacement = Counter::new(source_guard.k(), source_guard.strand()).map_err(r_err)?;
    let mut taken = std::mem::replace(&mut *source_guard, replacement);
    if taken.len() > target_guard.len() {
        std::mem::swap(&mut *target_guard, &mut taken);
    }
    target_guard.merge(taken).map_err(r_err)
}

#[extendr]
fn rust_kbolt_count_files(
    paths: Strings,
    k: f64,
    canonical: bool,
    threads: f64,
    readers: f64,
    capacity: f64,
) -> Result<Robj> {
    let k = parse_usize_arg(k, "k", 1)?;
    let threads = parse_usize_arg(threads, "threads", 1)?;
    let readers = parse_usize_arg(readers, "readers", 1)?;
    let capacity = parse_usize_arg(capacity, "capacity", 0)?;
    let paths = parse_paths(paths)?;
    let strand = strand_mode(canonical);
    let counter = if threads == 1 {
        let mut c = Counter::with_capacity(k, strand, capacity).map_err(r_err)?;
        c.add_files(&paths).map_err(r_err)?;
        c
    } else {
        let parallel = ParallelCounter::with_capacity(k, strand, threads, capacity).map_err(r_err)?;
        parallel.add_files(&paths, readers).map_err(r_err)?;
        parallel.finish().map_err(r_err)?
    };
    Ok(wrap_counter(counter))
}

#[extendr]
fn rust_kbolt_count_records(
    records: Strings,
    k: f64,
    canonical: bool,
    threads: f64,
    capacity: f64,
) -> Result<Robj> {
    let k = parse_usize_arg(k, "k", 1)?;
    let threads = parse_usize_arg(threads, "threads", 1)?;
    let capacity = parse_usize_arg(capacity, "capacity", 0)?;
    let strand = strand_mode(canonical);
    let counter = if threads == 1 {
        let mut c = Counter::with_capacity(k, strand, capacity).map_err(r_err)?;
        let slices = records
            .iter()
            .filter(|s| !s.is_na())
            .map(|s| s.as_bytes());
        c.add_records(slices).map_err(r_err)?;
        c
    } else {
        let mut parallel =
            ParallelCounter::with_capacity(k, strand, threads, capacity).map_err(r_err)?;
        for s in records.iter() {
            if !s.is_na() {
                parallel.add_record(s.as_bytes()).map_err(r_err)?;
            }
        }
        parallel.finish().map_err(r_err)?
    };
    Ok(wrap_counter(counter))
}

#[extendr]
#[allow(clippy::too_many_arguments)]
fn rust_kbolt_count_files_to_disk(
    paths: Strings,
    output: &str,
    k: f64,
    memory: Robj,
    canonical: bool,
    threads: f64,
    readers: f64,
    temp_dir: Nullable<String>,
    format: &str,
    sorted: bool,
) -> Result<f64> {
    let k = parse_usize_arg(k, "k", 1)?;
    if k > 32 {
        return Err(r_err("kbolt_count_files_to_disk supports k from 1 to 32"));
    }
    let threads = parse_usize_arg(threads, "threads", 1)?;
    let readers = parse_usize_arg(readers, "readers", 1)?;
    let binary = match format {
        "binary" => true,
        "tsv" => false,
        _ => return Err(r_err("format must be 'binary' or 'tsv'")),
    };
    let memory_bytes = parse_memory_size(memory)?;
    let paths = parse_paths(paths)?;
    let output = parse_path(output, "output")?;
    check_output_path(&paths, &output)?;
    let directory = match temp_dir {
        Nullable::Null => std::env::temp_dir(),
        Nullable::NotNull(dir) => parse_path(&dir, "temp_dir")?,
    };
    let strand = strand_mode(canonical);

    let counter = DiskCounter::new(k, strand, threads, memory_bytes, &directory).map_err(r_err)?;
    counter.add_files(&paths, readers).map_err(r_err)?;
    let bins = counter
        .finish()
        .map_err(r_err)?
        .with_threads(threads + readers.min(paths.len()))
        .sorted(sorted);
    let mut writer =
        BufWriter::with_capacity(IO_BUFFER_SIZE, File::create(&output).map_err(r_err)?);
    let distinct = if binary {
        bins.write_binary(&mut writer).map_err(r_err)?
    } else {
        bins.write_tsv(&mut writer).map_err(r_err)?
    };
    writer.flush().map_err(r_err)?;
    Ok(distinct as f64)
}

#[extendr]
fn rust_kbolt_get(counter: Robj, kmers: Strings) -> Result<Doubles> {
    let ptr = borrow_counter(&counter)?;
    let guard = ptr.0.try_borrow().map_err(r_err)?;
    let mut out = Doubles::new(kmers.len());
    // Collect non-NA indices so we benefit from batched software prefetching.
    let mut non_na_indices = Vec::with_capacity(kmers.len());
    let mut non_na_slices: Vec<&[u8]> = Vec::with_capacity(kmers.len());
    for (i, s) in kmers.iter().enumerate() {
        if s.is_na() {
            out.set_elt(i, Rfloat::na());
        } else {
            non_na_indices.push(i);
            non_na_slices.push(s.as_bytes());
        }
    }
    let counts = guard.get_many(non_na_slices);
    for (idx, count) in non_na_indices.into_iter().zip(counts) {
        out.set_elt(idx, Rfloat::from(count as f64));
    }
    Ok(out)
}

#[extendr]
fn rust_kbolt_histogram(counter: Robj) -> Result<List> {
    let ptr = borrow_counter(&counter)?;
    let guard = ptr.0.try_borrow().map_err(r_err)?;
    let hist = guard.histogram();
    let mut counts = Doubles::new(hist.len());
    let mut freqs = Doubles::new(hist.len());
    for (i, (c, f)) in hist.into_iter().enumerate() {
        counts.set_elt(i, Rfloat::from(c as f64));
        freqs.set_elt(i, Rfloat::from(f as f64));
    }
    Ok(list!(count = counts, frequency = freqs))
}

#[extendr]
fn rust_kbolt_top_k(counter: Robj, n: f64) -> Result<List> {
    let n = parse_usize_arg(n, "n", 0)?;
    let ptr = borrow_counter(&counter)?;
    let guard = ptr.0.try_borrow().map_err(r_err)?;
    let entries = guard.top_k(n);
    let mut kmers = Strings::new(entries.len());
    let mut counts = Doubles::new(entries.len());
    for (i, (kmer, count)) in entries.into_iter().enumerate() {
        let s = std::str::from_utf8(&kmer).expect("k-mers are ASCII");
        kmers.set_elt(i, Rstr::from(s));
        counts.set_elt(i, Rfloat::from(count as f64));
    }
    Ok(list!(kmer = kmers, count = counts))
}

#[extendr]
fn rust_kbolt_count_filtered(
    counter: Robj,
    min_count: f64,
    max_count: Nullable<f64>,
) -> Result<f64> {
    let filter = parse_filter(min_count, max_count)?;
    let ptr = borrow_counter(&counter)?;
    let guard = ptr.0.try_borrow().map_err(r_err)?;
    Ok(guard.len_filtered(filter) as f64)
}

#[extendr]
fn rust_kbolt_to_list(counter: Robj, min_count: f64, max_count: Nullable<f64>) -> Result<List> {
    let filter = parse_filter(min_count, max_count)?;
    let ptr = borrow_counter(&counter)?;
    let guard = ptr.0.try_borrow().map_err(r_err)?;
    let n = guard.len_filtered(filter);
    let mut kmers = Strings::new(n);
    let mut counts = Doubles::new(n);
    for (i, (kmer, count)) in guard.counts_filtered(filter).enumerate() {
        let s = std::str::from_utf8(&kmer).expect("k-mers are ASCII");
        kmers.set_elt(i, Rstr::from(s));
        counts.set_elt(i, Rfloat::from(count as f64));
    }
    Ok(list!(kmer = kmers, count = counts))
}

#[extendr]
fn rust_kbolt_write_tsv(
    counter: Robj,
    path: &str,
    min_count: f64,
    max_count: Nullable<f64>,
) -> Result<()> {
    let filter = parse_filter(min_count, max_count)?;
    let path = parse_path(path, "path")?;
    let ptr = borrow_counter(&counter)?;
    let guard = ptr.0.try_borrow().map_err(r_err)?;
    let mut writer = BufWriter::with_capacity(IO_BUFFER_SIZE, File::create(&path).map_err(r_err)?);
    guard
        .write_tsv_filtered(filter, &mut writer)
        .map_err(r_err)?;
    writer.flush().map_err(r_err)
}

#[extendr]
fn rust_kbolt_write_binary(
    counter: Robj,
    path: &str,
    min_count: f64,
    max_count: Nullable<f64>,
) -> Result<()> {
    let filter = parse_filter(min_count, max_count)?;
    let path = parse_path(path, "path")?;
    let ptr = borrow_counter(&counter)?;
    let guard = ptr.0.try_borrow().map_err(r_err)?;
    let mut writer = BufWriter::with_capacity(IO_BUFFER_SIZE, File::create(&path).map_err(r_err)?);
    guard
        .write_binary_filtered(filter, &mut writer)
        .map_err(r_err)?;
    writer.flush().map_err(r_err)
}

#[extendr]
fn rust_kbolt_read_binary(path: &str) -> Result<Robj> {
    let path = parse_path(path, "path")?;
    let file = File::open(&path).map_err(r_err)?;
    let counter =
        Counter::read_binary(BufReader::with_capacity(IO_BUFFER_SIZE, file)).map_err(r_err)?;
    Ok(wrap_counter(counter))
}

#[extendr]
fn rust_kbolt_open_database(path: &str) -> Result<Robj> {
    let path = parse_path(path, "path")?;
    let file = File::open(&path).map_err(r_err)?;
    let reader =
        DatabaseReader::new(BufReader::with_capacity(IO_BUFFER_SIZE, file)).map_err(r_err)?;
    Ok(wrap_database(reader))
}

#[extendr]
fn rust_kbolt_database_header(db: Robj) -> Result<List> {
    let ptr = borrow_database(&db)?;
    let guard = ptr.0.try_borrow().map_err(r_err)?;
    let h = guard.header;
    let k = i32::try_from(h.k).map_err(|_| r_err("k exceeds R integer range"))?;
    Ok(list!(
        k = k,
        canonical = h.strand == StrandMode::Canonical,
        entries = h.entries as f64,
        order = key_order_name(h.order),
        ascending = h.ascending,
        open = guard.reader.is_some()
    ))
}

#[extendr]
fn rust_kbolt_database_read(db: Robj, n: f64) -> Result<List> {
    let n = parse_usize_arg(n, "n", 0)?;
    let ptr = borrow_database(&db)?;
    let mut guard = ptr.0.try_borrow_mut().map_err(r_err)?;
    let mut entries: Vec<(Vec<u8>, u64)> = Vec::new();
    if let Some(reader) = guard.reader.as_mut() {
        for _ in 0..n {
            match reader.next_entry() {
                Ok(Some((kmer, count))) => entries.push((kmer.to_vec(), count)),
                Ok(None) => {
                    guard.reader = None;
                    break;
                }
                Err(error) => {
                    guard.reader = None;
                    return Err(r_err(error));
                }
            }
        }
    }
    let mut kmers = Strings::new(entries.len());
    let mut counts = Doubles::new(entries.len());
    for (i, (kmer, count)) in entries.into_iter().enumerate() {
        let s = std::str::from_utf8(&kmer).expect("k-mers are ASCII");
        kmers.set_elt(i, Rstr::from(s));
        counts.set_elt(i, Rfloat::from(count as f64));
    }
    Ok(list!(kmer = kmers, count = counts))
}

#[extendr]
fn rust_kbolt_database_write_tsv(db: Robj, path: &str) -> Result<()> {
    let path = parse_path(path, "path")?;
    let ptr = borrow_database(&db)?;
    let mut guard = ptr.0.try_borrow_mut().map_err(r_err)?;
    let Some(reader) = guard.reader.take() else {
        return Err(r_err("kbolt_database reader is closed"));
    };
    let mut writer = BufWriter::with_capacity(IO_BUFFER_SIZE, File::create(&path).map_err(r_err)?);
    reader.write_tsv(&mut writer).map_err(r_err)?;
    writer.flush().map_err(r_err)
}

#[extendr]
fn rust_kbolt_database_close(db: Robj) -> Result<()> {
    let ptr = borrow_database(&db)?;
    let mut guard = ptr.0.try_borrow_mut().map_err(r_err)?;
    guard.reader = None;
    Ok(())
}

extendr_module! {
    mod kbolt;
    fn rust_kbolt_counter;
    fn rust_kbolt_k;
    fn rust_kbolt_is_canonical;
    fn rust_kbolt_len;
    fn rust_kbolt_add_records;
    fn rust_kbolt_add_files;
    fn rust_kbolt_merge;
    fn rust_kbolt_count_files;
    fn rust_kbolt_count_records;
    fn rust_kbolt_count_files_to_disk;
    fn rust_kbolt_get;
    fn rust_kbolt_histogram;
    fn rust_kbolt_top_k;
    fn rust_kbolt_count_filtered;
    fn rust_kbolt_to_list;
    fn rust_kbolt_write_tsv;
    fn rust_kbolt_write_binary;
    fn rust_kbolt_read_binary;
    fn rust_kbolt_open_database;
    fn rust_kbolt_database_header;
    fn rust_kbolt_database_read;
    fn rust_kbolt_database_write_tsv;
    fn rust_kbolt_database_close;
}
