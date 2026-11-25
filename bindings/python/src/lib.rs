use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use kbolt::database::{DatabaseHeader, DatabaseReader, KeyOrder};
use kbolt::input::{self, Progress};
use kbolt::{CountFilter, Counter, DiskCounter, InputStats, ParallelCounter, StrandMode};
use pyo3::buffer::PyUntypedBuffer;
use pyo3::exceptions::{
    PyMemoryError, PyOSError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyInt, PyList, PyString, PyTuple};

const IO_BUFFER_SIZE: usize = 64 * 1024;
const RECORD_BATCH_BYTES: usize = 1024 * 1024;
const QUERY_BATCH_ITEMS: usize = 4096;
const SIGNAL_CHECK_STRIDE: usize = 65_536;
const MIN_DISK_MEMORY: u64 = 64 << 20;

fn map_error(error: kbolt::Error) -> PyErr {
    match error {
        kbolt::Error::InvalidK
        | kbolt::Error::IncompatibleCounters
        | kbolt::Error::InvalidThreads => PyValueError::new_err(error.to_string()),
        kbolt::Error::CountOverflow => PyOverflowError::new_err(error.to_string()),
        kbolt::Error::ReserveFailed => PyMemoryError::new_err(error.to_string()),
        kbolt::Error::WorkerFailed => PyRuntimeError::new_err(error.to_string()),
        kbolt::Error::TemporaryFile => PyOSError::new_err(error.to_string()),
    }
}

fn map_io_error(error: std::io::Error) -> PyErr {
    if let Some(kbolt_err) = error
        .get_ref()
        .and_then(|inner| inner.downcast_ref::<kbolt::Error>())
    {
        return map_error(*kbolt_err);
    }
    if error.kind() == std::io::ErrorKind::InvalidInput {
        return PyValueError::new_err(error.to_string());
    }
    PyErr::from(error)
}

fn strand_mode(canonical: bool) -> StrandMode {
    if canonical {
        StrandMode::Canonical
    } else {
        StrandMode::Forward
    }
}

fn make_filter(min_count: u64, max_count: Option<u64>) -> CountFilter {
    CountFilter {
        min_count,
        max_count,
    }
}

enum SeqBytes<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

impl SeqBytes<'_> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(s) => s,
            Self::Owned(v) => v.as_slice(),
        }
    }
}

/// Extract sequence bytes from `str`, `bytes`, or a 1-D byte buffer (`bytearray`,
/// `memoryview`, `mmap`, etc.). Immutable `str`/`bytes` are borrowed zero-copy;
/// mutable/generic buffers are copied into an owned `Vec<u8>` so they are safe
/// to read across `py.detach`.
fn extract_seq_bytes<'a>(obj: &'a Bound<'_, PyAny>) -> PyResult<SeqBytes<'a>> {
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(SeqBytes::Borrowed(s.to_str()?.as_bytes()));
    }
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(SeqBytes::Borrowed(b.as_bytes()));
    }
    if let Ok(buf) = PyUntypedBuffer::get(obj) {
        if buf.dimensions() != 1 || buf.item_size() != 1 || buf.suboffsets().is_some() {
            return Err(PyTypeError::new_err(
                "expected str, bytes, or a 1-D byte buffer",
            ));
        }
        let n = buf.shape()[0];
        let mut out = Vec::with_capacity(n);
        if n > 0 {
            let base = buf.buf_ptr() as *const u8;
            if buf.is_c_contiguous() {
                // SAFETY: `buf` is a valid C-contiguous 1-D byte buffer of length `n`.
                let slice = unsafe { std::slice::from_raw_parts(base, n) };
                out.extend_from_slice(slice);
            } else {
                let stride = buf.strides()[0];
                for i in 0..n {
                    // SAFETY: `i < shape[0]` with no suboffsets; offset is in bounds of `buf`.
                    let byte = unsafe { *base.offset(i as isize * stride) };
                    out.push(byte);
                }
            }
        }
        return Ok(SeqBytes::Owned(out));
    }
    Err(PyTypeError::new_err(
        "expected str, bytes, or a 1-D byte buffer",
    ))
}

/// Reject passing a bare sequence (`str`, `bytes`, `bytearray`, `memoryview`)
/// where an iterable of sequences is expected.
fn reject_single_sequence(obj: &Bound<'_, PyAny>, fn_name: &str) -> PyResult<()> {
    if obj.is_instance_of::<PyString>()
        || obj.is_instance_of::<PyBytes>()
        || obj.is_instance_of::<PyByteArray>()
        || PyUntypedBuffer::get(obj).is_ok()
    {
        return Err(PyTypeError::new_err(format!(
            "{fn_name} expects an iterable of sequences, not a single str/bytes/buffer"
        )));
    }
    Ok(())
}

fn extract_paths(paths: &Bound<'_, PyAny>) -> PyResult<Vec<PathBuf>> {
    if paths.is_instance_of::<PyString>() || paths.hasattr("__fspath__")? {
        return Ok(vec![paths.extract::<PathBuf>()?]);
    }
    if paths.is_instance_of::<PyBytes>() || paths.is_instance_of::<PyByteArray>() {
        return Err(PyTypeError::new_err(
            "paths must be str, os.PathLike, or an iterable of them",
        ));
    }
    let mut out = Vec::new();
    for item in paths.try_iter()? {
        out.push(item?.extract::<PathBuf>()?);
    }
    Ok(out)
}

#[derive(Clone, Copy)]
enum PackedBufKind {
    U64 { big_endian: Option<bool> },
    I64 { big_endian: Option<bool> },
    Bytes,
}

fn classify_packed_format(fmt: &str, item_size: usize) -> Option<PackedBufKind> {
    let (big_endian, code) = match fmt.as_bytes() {
        [c] => (None, *c),
        [b'@' | b'=', c] => (None, *c),
        [b'<', c] => (Some(false), *c),
        [b'>' | b'!', c] => (Some(true), *c),
        _ => return None,
    };
    match code {
        b'Q' | b'L' if item_size == 8 => Some(PackedBufKind::U64 { big_endian }),
        b'q' | b'l' if item_size == 8 => Some(PackedBufKind::I64 { big_endian }),
        b'B' if item_size == 1 => Some(PackedBufKind::Bytes),
        _ => None,
    }
}

/// Extract 2-bit packed `u64` keys from either a 1-D buffer (`uint64`, `int64`,
/// or native-endian `B` byte buffer whose length is a multiple of 8, handling
/// explicit `<`/`>`/`!` endianness and arbitrary strides) or a non-buffer
/// iterable of integers.
fn extract_packed_keys(keys: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    if let Ok(buf) = PyUntypedBuffer::get(keys) {
        if buf.dimensions() != 1 {
            return Err(PyValueError::new_err(format!(
                "expected a 1-D buffer for packed keys, got {}-D",
                buf.dimensions()
            )));
        }
        if buf.suboffsets().is_some() {
            return Err(PyTypeError::new_err(
                "buffers with suboffsets are not supported",
            ));
        }
        let fmt = buf.format().to_str().unwrap_or("");
        let kind = classify_packed_format(fmt, buf.item_size()).ok_or_else(|| {
            PyTypeError::new_err(format!(
                "unsupported buffer format {fmt:?} (itemsize={}) for packed u64 keys",
                buf.item_size()
            ))
        })?;
        let n = buf.shape()[0];
        let base = buf.buf_ptr() as *const u8;
        return match kind {
            PackedBufKind::Bytes => {
                if !buf.is_c_contiguous() {
                    return Err(PyValueError::new_err(
                        "byte buffers for packed keys must be C-contiguous",
                    ));
                }
                if !n.is_multiple_of(8) {
                    return Err(PyValueError::new_err(
                        "byte buffer length must be a multiple of 8 bytes",
                    ));
                }
                let count = n / 8;
                let mut out = Vec::with_capacity(count);
                for i in 0..count {
                    // SAFETY: `buf` is C-contiguous with `n == count * 8` bytes.
                    let raw = unsafe { std::ptr::read_unaligned(base.add(i * 8).cast::<[u8; 8]>()) };
                    out.push(u64::from_ne_bytes(raw));
                }
                Ok(out)
            }
            PackedBufKind::U64 { big_endian } => {
                let stride = buf.strides()[0];
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    // SAFETY: `i < shape[0]` with `item_size == 8` and no suboffsets.
                    let raw = unsafe {
                        std::ptr::read_unaligned(base.offset(i as isize * stride).cast::<[u8; 8]>())
                    };
                    let val = match big_endian {
                        None => u64::from_ne_bytes(raw),
                        Some(false) => u64::from_le_bytes(raw),
                        Some(true) => u64::from_be_bytes(raw),
                    };
                    out.push(val);
                }
                Ok(out)
            }
            PackedBufKind::I64 { big_endian } => {
                let stride = buf.strides()[0];
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    // SAFETY: `i < shape[0]` with `item_size == 8` and no suboffsets.
                    let raw = unsafe {
                        std::ptr::read_unaligned(base.offset(i as isize * stride).cast::<[u8; 8]>())
                    };
                    let val = match big_endian {
                        None => i64::from_ne_bytes(raw),
                        Some(false) => i64::from_le_bytes(raw),
                        Some(true) => i64::from_be_bytes(raw),
                    };
                    if val < 0 {
                        return Err(PyValueError::new_err("packed keys must be non-negative"));
                    }
                    out.push(val as u64);
                }
                Ok(out)
            }
        };
    }

    if keys.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "expected a uint64 buffer or iterable of ints, not str",
        ));
    }
    let mut vec = Vec::new();
    for item in keys.try_iter()? {
        vec.push(item?.extract::<u64>()?);
    }
    Ok(vec)
}

fn check_output_path(inputs: &[PathBuf], output: &Path) -> PyResult<()> {
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::metadata(parent).map_err(map_io_error)?;
    }
    let output_canonical = match std::fs::canonicalize(output) {
        Ok(path) => path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(map_io_error(error)),
    };
    for input in inputs {
        if let Ok(input_canonical) = std::fs::canonicalize(input)
            && input_canonical == output_canonical
        {
            return Err(PyValueError::new_err("output must not be an input file"));
        }
    }
    Ok(())
}

fn parse_memory_size(value: &Bound<'_, PyAny>) -> PyResult<u64> {
    if let Ok(int_val) = value.cast::<PyInt>() {
        let size: u64 = int_val.extract()?;
        if size < MIN_DISK_MEMORY {
            return Err(PyValueError::new_err(
                "memory must be at least 64 MiB (67108864 bytes)",
            ));
        }
        return Ok(size);
    }
    if let Ok(str_val) = value.cast::<PyString>() {
        let s = str_val.to_str()?;
        let digits = s.trim_end_matches(|c: char| c.is_ascii_alphabetic());
        let shift = match s[digits.len()..].to_ascii_uppercase().as_str() {
            "" | "B" => 0,
            "K" | "KB" | "KIB" => 10,
            "M" | "MB" | "MIB" => 20,
            "G" | "GB" | "GIB" => 30,
            "T" | "TB" | "TIB" => 40,
            _ => {
                return Err(PyValueError::new_err(
                    "expected a byte count or size string such as '4G' or '500M'",
                ));
            }
        };
        let size = digits
            .parse::<u64>()
            .ok()
            .and_then(|n| n.checked_mul(1 << shift))
            .filter(|&n| n >= MIN_DISK_MEMORY)
            .ok_or_else(|| {
                PyValueError::new_err("expected a memory size of at least 64M, such as '4G'")
            })?;
        return Ok(size);
    }
    Err(PyTypeError::new_err(
        "memory must be an int (bytes) or str (such as '4G' or '500M')",
    ))
}

fn key_order_name(order: KeyOrder) -> &'static str {
    match order {
        KeyOrder::Unordered => "unordered",
        KeyOrder::Hash => "hash",
        KeyOrder::Kmer => "kmer",
    }
}

fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

/// Progress hook that periodically re-attaches to Python to check pending
/// signals (such as Ctrl-C / `KeyboardInterrupt`) during long streaming I/O.
fn signal_progress_callback(
    interrupted: &Mutex<Option<PyErr>>,
) -> impl Fn(Progress<'_>) -> std::io::Result<()> + Sync + '_ {
    move |_| {
        let err = Python::attach(|py| py.check_signals().err());
        if let Some(py_err) = err {
            let mut slot = interrupted.lock().unwrap_or_else(|p| p.into_inner());
            if slot.is_none() {
                *slot = Some(py_err);
            }
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "interrupted by signal",
            ));
        }
        Ok(())
    }
}

fn finish_with_signal<T>(
    result: std::io::Result<T>,
    interrupted: Mutex<Option<PyErr>>,
) -> PyResult<T> {
    if let Some(py_err) = interrupted.into_inner().unwrap_or_else(|p| p.into_inner()) {
        return Err(py_err);
    }
    result.map_err(map_io_error)
}

/// Sentinel allowing `Counter.get(kmer, default=0)` to distinguish an omitted
/// default (returning `0`) from an explicit `default=None` (returning `None`).
enum GetDefault {
    Zero,
    Value(Py<PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for GetDefault {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self::Value(obj.to_owned().unbind()))
    }
}

/// Record and base counts reported after reading FASTA/FASTQ files.
#[pyclass(name = "InputStats", module = "kbolt", frozen, eq, hash, get_all)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PyInputStats {
    records: u64,
    bases: u64,
}

impl From<InputStats> for PyInputStats {
    fn from(stats: InputStats) -> Self {
        Self {
            records: stats.records,
            bases: stats.bases,
        }
    }
}

#[pymethods]
impl PyInputStats {
    fn __repr__(&self) -> String {
        format!("InputStats(records={}, bases={})", self.records, self.bases)
    }
}

/// Metadata header of a `.kbolt` binary count database.
#[pyclass(name = "DatabaseHeader", module = "kbolt", frozen, eq, hash, get_all)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PyDatabaseHeader {
    k: usize,
    canonical: bool,
    entries: u64,
    order: String,
    ascending: bool,
}

impl From<DatabaseHeader> for PyDatabaseHeader {
    fn from(header: DatabaseHeader) -> Self {
        Self {
            k: header.k,
            canonical: header.strand == StrandMode::Canonical,
            entries: header.entries,
            order: key_order_name(header.order).to_owned(),
            ascending: header.ascending,
        }
    }
}

#[pymethods]
impl PyDatabaseHeader {
    fn __repr__(&self) -> String {
        format!(
            "DatabaseHeader(k={}, canonical={}, entries={}, order='{}', ascending={})",
            self.k,
            py_bool(self.canonical),
            self.entries,
            self.order,
            py_bool(self.ascending)
        )
    }
}

/// Streaming reader for a `.kbolt` binary database (memory is O(k)).
#[pyclass(name = "DatabaseReader", module = "kbolt")]
struct PyDatabaseReader {
    reader: Option<DatabaseReader<BufReader<File>>>,
    header: PyDatabaseHeader,
}

#[pymethods]
impl PyDatabaseReader {
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let reader = py
            .detach(|| -> std::io::Result<_> {
                let file = File::open(&path)?;
                DatabaseReader::new(BufReader::with_capacity(IO_BUFFER_SIZE, file))
            })
            .map_err(map_io_error)?;
        let header = PyDatabaseHeader::from(reader.header());
        Ok(Self {
            reader: Some(reader),
            header,
        })
    }

    #[getter]
    fn header(&self) -> PyDatabaseHeader {
        self.header.clone()
    }

    #[getter]
    fn k(&self) -> usize {
        self.header.k
    }

    #[getter]
    fn canonical(&self) -> bool {
        self.header.canonical
    }

    #[getter]
    fn entries(&self) -> u64 {
        self.header.entries
    }

    /// Stream remaining entries directly to a TSV file without loading into RAM.
    fn write_tsv(&mut self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let Some(reader) = self.reader.take() else {
            return Err(PyValueError::new_err("database reader is closed"));
        };
        py.detach(|| -> std::io::Result<()> {
            let mut writer = BufWriter::with_capacity(IO_BUFFER_SIZE, File::create(&path)?);
            reader.write_tsv(&mut writer)?;
            writer.flush()
        })
        .map_err(map_io_error)?;
        Ok(())
    }

    fn close(&mut self) {
        self.reader = None;
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<(Py<PyString>, u64)>> {
        let Some(reader) = self.reader.as_mut() else {
            return Ok(None);
        };
        match reader.next_entry() {
            Ok(Some((kmer, count))) => {
                let s = std::str::from_utf8(kmer).expect("k-mers are ASCII");
                Ok(Some((PyString::new(py, s).unbind(), count)))
            }
            Ok(None) => {
                self.reader = None;
                Ok(None)
            }
            Err(error) => {
                self.reader = None;
                Err(map_io_error(error))
            }
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        self.close();
        false
    }

    fn __repr__(&self) -> String {
        let state = if self.reader.is_some() {
            "open"
        } else {
            "closed"
        };
        format!(
            "DatabaseReader(k={}, canonical={}, entries={}, state='{}')",
            self.header.k,
            py_bool(self.header.canonical),
            self.header.entries,
            state
        )
    }
}

/// Exact in-memory DNA k-mer counter backed by Rust.
#[pyclass(name = "Counter", module = "kbolt")]
struct PyCounter {
    inner: Counter,
}

#[pymethods]
impl PyCounter {
    #[new]
    #[pyo3(signature = (k, *, canonical = true, capacity = 0))]
    fn new(k: usize, canonical: bool, capacity: usize) -> PyResult<Self> {
        let inner = Counter::with_capacity(k, strand_mode(canonical), capacity).map_err(map_error)?;
        Ok(Self { inner })
    }

    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    #[getter]
    fn canonical(&self) -> bool {
        self.inner.strand() == StrandMode::Canonical
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __bool__(&self) -> bool {
        !self.inner.is_empty()
    }

    fn __contains__(&self, kmer: &Bound<'_, PyAny>) -> PyResult<bool> {
        let bytes = extract_seq_bytes(kmer)?;
        Ok(self.inner.get(bytes.as_slice()) > 0)
    }

    fn __getitem__(&self, kmer: &Bound<'_, PyAny>) -> PyResult<u64> {
        let bytes = extract_seq_bytes(kmer)?;
        Ok(self.inner.get(bytes.as_slice()))
    }

    /// Iterate over all k-mer strings. Note: materializes the key list in
    /// memory; for very large counters prefer `to_numpy()` or `DatabaseReader`.
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let keys = self.keys(py, 1, None)?;
        Ok(keys.try_iter()?.into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "Counter(k={}, canonical={}, distinct={})",
            self.inner.k(),
            py_bool(self.inner.strand() == StrandMode::Canonical),
            self.inner.len()
        )
    }

    /// Pickle support using kbolt's compact binary format.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let bytes = py
            .detach(|| -> std::io::Result<Vec<u8>> {
                let mut buf = Vec::new();
                self.inner.write_binary(&mut buf)?;
                Ok(buf)
            })
            .map_err(map_io_error)?;
        let cls = py.get_type::<PyCounter>();
        let constructor = cls.getattr("_from_binary_bytes")?;
        let args = PyTuple::new(py, [PyBytes::new(py, &bytes)])?;
        PyTuple::new(py, [constructor.into_any(), args.into_any()])
    }

    #[staticmethod]
    fn _from_binary_bytes(py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let slice = data.as_bytes();
        let inner = py
            .detach(|| Counter::read_binary(Cursor::new(slice)))
            .map_err(map_io_error)?;
        Ok(Self { inner })
    }

    /// Count all valid k-mers in one DNA record (`str`, `bytes`, or 1-D byte buffer).
    fn add_record(&mut self, py: Python<'_>, sequence: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = extract_seq_bytes(sequence)?;
        py.detach(|| self.inner.add_record(bytes.as_slice()))
            .map_err(map_error)
    }

    /// Count all valid k-mers across an iterable of DNA records.
    fn add_records(&mut self, py: Python<'_>, records: &Bound<'_, PyAny>) -> PyResult<()> {
        reject_single_sequence(records, "add_records()")?;
        let mut buf = Vec::with_capacity(RECORD_BATCH_BYTES);
        let mut ends = Vec::new();
        let flush =
            |counter: &mut Counter, buf: &mut Vec<u8>, ends: &mut Vec<usize>| -> PyResult<()> {
                if ends.is_empty() {
                    return Ok(());
                }
                let result = py.detach(|| {
                    let mut start = 0;
                    let slices = ends.iter().map(|&end| {
                        let slice = &buf[start..end];
                        start = end;
                        slice
                    });
                    counter.add_records(slices)
                });
                buf.clear();
                ends.clear();
                result.map_err(map_error)
            };

        for (i, item) in records.try_iter()?.enumerate() {
            if i.is_multiple_of(SIGNAL_CHECK_STRIDE) {
                py.check_signals()?;
            }
            let item = item?;
            let bytes = extract_seq_bytes(&item)?;
            let slice = bytes.as_slice();
            if slice.len() >= RECORD_BATCH_BYTES {
                flush(&mut self.inner, &mut buf, &mut ends)?;
                py.detach(|| self.inner.add_record(slice))
                    .map_err(map_error)?;
                continue;
            }
            buf.extend_from_slice(slice);
            ends.push(buf.len());
            if buf.len() >= RECORD_BATCH_BYTES {
                flush(&mut self.inner, &mut buf, &mut ends)?;
            }
        }
        flush(&mut self.inner, &mut buf, &mut ends)
    }

    /// Stream FASTA/FASTQ records (plain or gzipped) from `path` into this counter.
    fn add_file(&mut self, py: Python<'_>, path: PathBuf) -> PyResult<PyInputStats> {
        let interrupted = Mutex::new(None);
        let result = py.detach(|| {
            let cb = signal_progress_callback(&interrupted);
            let mut stream = input::Stream::serial(&mut self.inner).with_progress(&cb);
            stream.add_file(&path)?;
            stream.finish()
        });
        Ok(finish_with_signal(result, interrupted)?.into())
    }

    /// Stream FASTA/FASTQ records (plain or gzipped) from `paths` into this counter.
    fn add_files(&mut self, py: Python<'_>, paths: &Bound<'_, PyAny>) -> PyResult<PyInputStats> {
        let paths = extract_paths(paths)?;
        let interrupted = Mutex::new(None);
        let result = py.detach(|| {
            let cb = signal_progress_callback(&interrupted);
            let mut stream = input::Stream::serial(&mut self.inner).with_progress(&cb);
            for path in &paths {
                stream.add_file(path)?;
            }
            stream.finish()
        });
        Ok(finish_with_signal(result, interrupted)?.into())
    }

    /// Merge all counts from `other` into this counter, leaving `other` empty.
    ///
    /// Note: on `OverflowError`, `other` has already been consumed and `self`
    /// may contain partially merged counts.
    fn merge(
        slf: &Bound<'_, PyCounter>,
        py: Python<'_>,
        other: &Bound<'_, PyCounter>,
    ) -> PyResult<()> {
        if slf.as_ptr() == other.as_ptr() {
            return Err(PyValueError::new_err("cannot merge a Counter into itself"));
        }
        let mut this = slf.try_borrow_mut()?;
        let mut other_ref = other.try_borrow_mut()?;
        if this.inner.k() != other_ref.inner.k() || this.inner.strand() != other_ref.inner.strand()
        {
            return Err(map_error(kbolt::Error::IncompatibleCounters));
        }
        let replacement =
            Counter::new(other_ref.inner.k(), other_ref.inner.strand()).map_err(map_error)?;
        let mut taken = std::mem::replace(&mut other_ref.inner, replacement);
        drop(other_ref);
        let target = &mut this.inner;
        if taken.len() > target.len() {
            std::mem::swap(target, &mut taken);
        }
        py.detach(|| target.merge(taken)).map_err(map_error)
    }

    /// Look up the count of `kmer`, returning `default` (0) if absent or invalid.
    #[pyo3(signature = (kmer, default = GetDefault::Zero))]
    fn get(
        &self,
        py: Python<'_>,
        kmer: &Bound<'_, PyAny>,
        default: GetDefault,
    ) -> PyResult<Py<PyAny>> {
        let bytes = extract_seq_bytes(kmer)?;
        let count = self.inner.get(bytes.as_slice());
        if count > 0 {
            return Ok(count.into_pyobject(py)?.into_any().unbind());
        }
        match default {
            GetDefault::Zero => Ok(0_u64.into_pyobject(py)?.into_any().unbind()),
            GetDefault::Value(value) => Ok(value),
        }
    }

    /// Look up many k-mers at once, returning a `list[int]` in input order.
    fn get_many(&self, py: Python<'_>, kmers: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
        reject_single_sequence(kmers, "get_many()")?;
        let mut results = Vec::new();
        let mut buf = Vec::new();
        let mut ends = Vec::with_capacity(QUERY_BATCH_ITEMS);
        let flush = |buf: &mut Vec<u8>, ends: &mut Vec<usize>, results: &mut Vec<u64>| {
            if ends.is_empty() {
                return;
            }
            let chunk = py.detach(|| {
                let mut start = 0;
                let slices = ends.iter().map(|&end| {
                    let slice = &buf[start..end];
                    start = end;
                    slice
                });
                self.inner.get_many(slices)
            });
            results.extend(chunk);
            buf.clear();
            ends.clear();
        };

        for (i, item) in kmers.try_iter()?.enumerate() {
            if i.is_multiple_of(SIGNAL_CHECK_STRIDE) {
                py.check_signals()?;
            }
            let item = item?;
            let bytes = extract_seq_bytes(&item)?;
            buf.extend_from_slice(bytes.as_slice());
            ends.push(buf.len());
            if ends.len() >= QUERY_BATCH_ITEMS {
                flush(&mut buf, &mut ends, &mut results);
            }
        }
        flush(&mut buf, &mut ends, &mut results);
        Ok(results)
    }

    /// Look up a 2-bit packed k-mer (`k <= 32`). Raises `ValueError` if
    /// `k > 32` or if `key` has nonzero bits above `2 * k`.
    fn get_packed(&self, key: u64) -> PyResult<u64> {
        if self.inner.k() > 32 {
            return Err(PyValueError::new_err("packed queries require k <= 32"));
        }
        self.inner.get_packed(key).ok_or_else(|| {
            PyValueError::new_err(format!(
                "packed key out of range for k={}",
                self.inner.k()
            ))
        })
    }

    /// Look up many 2-bit packed k-mers (`k <= 32`), returning a `list[int]`.
    ///
    /// Accepts a 1-D `uint64`/`int64` buffer (native or explicit `<`/`>` endian),
    /// a contiguous native-endian byte buffer whose length is a multiple of 8
    /// (such as from `to_packed_buffers()`), or an iterable of ints. Out-of-range
    /// keys yield `0`.
    fn get_packed_many(&self, py: Python<'_>, keys: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
        if self.inner.k() > 32 {
            return Err(PyValueError::new_err("packed queries require k <= 32"));
        }
        let key_vec = extract_packed_keys(keys)?;
        Ok(py
            .detach(|| self.inner.get_packed_many(&key_vec))
            .expect("k <= 32 verified"))
    }

    /// Look up many 2-bit packed k-mers (`k <= 32`), returning a raw
    /// native-endian `uint64` `bytearray` (wrap with `numpy.frombuffer(..., dtype="uint64")`).
    fn get_packed_many_buffer<'py>(
        &self,
        py: Python<'py>,
        keys: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyByteArray>> {
        if self.inner.k() > 32 {
            return Err(PyValueError::new_err("packed queries require k <= 32"));
        }
        let key_vec = extract_packed_keys(keys)?;
        let byte_len = key_vec
            .len()
            .checked_mul(8)
            .ok_or_else(|| PyMemoryError::new_err("query result buffer size overflow"))?;
        let out_ba = PyByteArray::new_with(py, byte_len, |_| Ok(()))?;
        // SAFETY: `out_ba` is freshly created, not GC-tracked, exclusively owned
        // by this stack frame, and kept alive across `detach`.
        let out_buf = unsafe { out_ba.as_bytes_mut() };
        py.detach(|| {
            let counts = self
                .inner
                .get_packed_many(&key_vec)
                .expect("k <= 32 verified");
            for (i, count) in counts.into_iter().enumerate() {
                let offset = i * 8;
                out_buf[offset..offset + 8].copy_from_slice(&count.to_ne_bytes());
            }
        });
        Ok(out_ba)
    }

    /// Return the k-mer count spectrum as sorted `(count, frequency)` pairs.
    fn histogram(&self, py: Python<'_>) -> Vec<(u64, u64)> {
        py.detach(|| self.inner.histogram())
    }

    /// Return up to `n` most abundant `(kmer, count)` pairs (count descending,
    /// ties broken alphabetically).
    fn top_k<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyList>> {
        let entries = py.detach(|| self.inner.top_k(n));
        let mut items = Vec::with_capacity(entries.len());
        for (kmer, count) in entries {
            let s = std::str::from_utf8(&kmer).expect("k-mers are ASCII");
            items.push((PyString::new(py, s), count));
        }
        PyList::new(py, items)
    }

    /// Count distinct k-mers whose count falls within `[min_count, max_count]`.
    #[pyo3(signature = (*, min_count = 1, max_count = None))]
    fn count_filtered(&self, py: Python<'_>, min_count: u64, max_count: Option<u64>) -> usize {
        let filter = make_filter(min_count, max_count);
        py.detach(|| self.inner.len_filtered(filter))
    }

    /// Return `(kmer, count)` pairs matching the optional count bounds.
    #[pyo3(signature = (*, min_count = 1, max_count = None))]
    fn items<'py>(
        &self,
        py: Python<'py>,
        min_count: u64,
        max_count: Option<u64>,
    ) -> PyResult<Bound<'py, PyList>> {
        let filter = make_filter(min_count, max_count);
        let list = PyList::empty(py);
        for (i, (kmer, count)) in self.inner.counts_filtered(filter).enumerate() {
            if i.is_multiple_of(SIGNAL_CHECK_STRIDE) {
                py.check_signals()?;
            }
            let s = std::str::from_utf8(&kmer).expect("k-mers are ASCII");
            list.append((PyString::new(py, s), count))?;
        }
        Ok(list)
    }

    /// Return k-mer strings matching the optional count bounds.
    #[pyo3(signature = (*, min_count = 1, max_count = None))]
    fn keys<'py>(
        &self,
        py: Python<'py>,
        min_count: u64,
        max_count: Option<u64>,
    ) -> PyResult<Bound<'py, PyList>> {
        let filter = make_filter(min_count, max_count);
        let list = PyList::empty(py);
        for (i, (kmer, _)) in self.inner.counts_filtered(filter).enumerate() {
            if i.is_multiple_of(SIGNAL_CHECK_STRIDE) {
                py.check_signals()?;
            }
            let s = std::str::from_utf8(&kmer).expect("k-mers are ASCII");
            list.append(PyString::new(py, s))?;
        }
        Ok(list)
    }

    /// Return k-mer counts matching the optional count bounds.
    #[pyo3(signature = (*, min_count = 1, max_count = None))]
    fn values(&self, py: Python<'_>, min_count: u64, max_count: Option<u64>) -> Vec<u64> {
        let filter = make_filter(min_count, max_count);
        py.detach(|| {
            self.inner
                .counts_filtered(filter)
                .map(|(_, count)| count)
                .collect()
        })
    }

    /// Materialize matching entries as a Python `dict[str, int]`.
    #[pyo3(signature = (*, min_count = 1, max_count = None))]
    fn to_dict<'py>(
        &self,
        py: Python<'py>,
        min_count: u64,
        max_count: Option<u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let filter = make_filter(min_count, max_count);
        let dict = PyDict::new(py);
        for (i, (kmer, count)) in self.inner.counts_filtered(filter).enumerate() {
            if i.is_multiple_of(SIGNAL_CHECK_STRIDE) {
                py.check_signals()?;
            }
            let s = std::str::from_utf8(&kmer).expect("k-mers are ASCII");
            dict.set_item(PyString::new(py, s), count)?;
        }
        Ok(dict)
    }

    /// Export matching entries (`k <= 32`) as a `(keys, counts)` pair of raw
    /// native-endian `uint64` bytearrays (wrap with `numpy.frombuffer`).
    #[pyo3(signature = (*, min_count = 1, max_count = None))]
    fn to_packed_buffers<'py>(
        &self,
        py: Python<'py>,
        min_count: u64,
        max_count: Option<u64>,
    ) -> PyResult<(Bound<'py, PyByteArray>, Bound<'py, PyByteArray>)> {
        if self.inner.k() > 32 {
            return Err(PyValueError::new_err("packed export requires k <= 32"));
        }
        let filter = make_filter(min_count, max_count);
        let n = py.detach(|| self.inner.len_filtered(filter));
        let byte_len = n
            .checked_mul(8)
            .ok_or_else(|| PyMemoryError::new_err("export buffer size overflow"))?;
        let keys_ba = PyByteArray::new_with(py, byte_len, |_| Ok(()))?;
        let counts_ba = PyByteArray::new_with(py, byte_len, |_| Ok(()))?;
        // SAFETY: Both bytearrays are freshly created, not GC-tracked,
        // exclusively owned by this stack frame (unreachable from Python),
        // and kept alive across `detach`.
        let (keys_buf, counts_buf) =
            unsafe { (keys_ba.as_bytes_mut(), counts_ba.as_bytes_mut()) };
        py.detach(|| {
            let iter = self
                .inner
                .packed_counts_filtered(filter)
                .expect("k <= 32 verified");
            for (i, (key, count)) in iter.enumerate() {
                let offset = i * 8;
                keys_buf[offset..offset + 8].copy_from_slice(&key.to_ne_bytes());
                counts_buf[offset..offset + 8].copy_from_slice(&count.to_ne_bytes());
            }
        });
        Ok((keys_ba, counts_ba))
    }

    /// Export matching entries (any `k`) as a `(keys, counts)` pair of raw
    /// bytearrays: `keys` has fixed-width `k`-byte ASCII entries and `counts`
    /// has native-endian `uint64` values.
    #[pyo3(signature = (*, min_count = 1, max_count = None))]
    fn to_ascii_buffers<'py>(
        &self,
        py: Python<'py>,
        min_count: u64,
        max_count: Option<u64>,
    ) -> PyResult<(Bound<'py, PyByteArray>, Bound<'py, PyByteArray>)> {
        let k = self.inner.k();
        let filter = make_filter(min_count, max_count);
        let n = py.detach(|| self.inner.len_filtered(filter));
        let keys_len = n
            .checked_mul(k)
            .ok_or_else(|| PyMemoryError::new_err("export buffer size overflow"))?;
        let counts_len = n
            .checked_mul(8)
            .ok_or_else(|| PyMemoryError::new_err("export buffer size overflow"))?;
        let keys_ba = PyByteArray::new_with(py, keys_len, |_| Ok(()))?;
        let counts_ba = PyByteArray::new_with(py, counts_len, |_| Ok(()))?;
        // SAFETY: Freshly created, non-GC-tracked, unshared bytearrays kept alive across `detach`.
        let (keys_buf, counts_buf) =
            unsafe { (keys_ba.as_bytes_mut(), counts_ba.as_bytes_mut()) };
        py.detach(|| {
            for (i, (kmer, count)) in self.inner.counts_filtered(filter).enumerate() {
                let k_off = i * k;
                let c_off = i * 8;
                keys_buf[k_off..k_off + k].copy_from_slice(&kmer);
                counts_buf[c_off..c_off + 8].copy_from_slice(&count.to_ne_bytes());
            }
        });
        Ok((keys_ba, counts_ba))
    }

    /// Export matching entries as NumPy arrays `(keys, counts)` backed directly
    /// by Rust-filled bytearrays (zero-copy between the bytearray and NumPy).
    ///
    /// With `packed=False` (default), `keys` has dtype `S{k}` (fixed-width
    /// ASCII bytes) and works for any `k`. With `packed=True` (`k <= 32`),
    /// `keys` has dtype `uint64`. `counts` always has dtype `uint64`.
    #[pyo3(signature = (*, min_count = 1, max_count = None, packed = false))]
    fn to_numpy<'py>(
        &self,
        py: Python<'py>,
        min_count: u64,
        max_count: Option<u64>,
        packed: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let np = py.import("numpy")?;
        let (keys_ba, counts_ba, key_dtype) = if packed {
            let (k, c) = self.to_packed_buffers(py, min_count, max_count)?;
            (k, c, "uint64".to_owned())
        } else {
            let (k, c) = self.to_ascii_buffers(py, min_count, max_count)?;
            (k, c, format!("S{}", self.inner.k()))
        };
        let keys_arr = np.call_method1("frombuffer", (keys_ba, key_dtype))?;
        let counts_arr = np.call_method1("frombuffer", (counts_ba, "uint64"))?;
        PyTuple::new(py, [keys_arr, counts_arr])
    }

    /// Write headerless `KMER<TAB>COUNT` rows to `path`.
    #[pyo3(signature = (path, *, min_count = 1, max_count = None))]
    fn write_tsv(
        &self,
        py: Python<'_>,
        path: PathBuf,
        min_count: u64,
        max_count: Option<u64>,
    ) -> PyResult<()> {
        let filter = make_filter(min_count, max_count);
        py.detach(|| -> std::io::Result<()> {
            let mut writer = BufWriter::with_capacity(IO_BUFFER_SIZE, File::create(&path)?);
            self.inner.write_tsv_filtered(filter, &mut writer)?;
            writer.flush()
        })
        .map_err(map_io_error)?;
        Ok(())
    }

    /// Write a `.kbolt` binary database to `path`.
    #[pyo3(signature = (path, *, min_count = 1, max_count = None))]
    fn write_binary(
        &self,
        py: Python<'_>,
        path: PathBuf,
        min_count: u64,
        max_count: Option<u64>,
    ) -> PyResult<()> {
        let filter = make_filter(min_count, max_count);
        py.detach(|| -> std::io::Result<()> {
            let mut writer = BufWriter::with_capacity(IO_BUFFER_SIZE, File::create(&path)?);
            self.inner.write_binary_filtered(filter, &mut writer)?;
            writer.flush()
        })
        .map_err(map_io_error)?;
        Ok(())
    }

    /// Load a `.kbolt` binary database from `path` into memory.
    #[staticmethod]
    fn read_binary(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py
            .detach(|| -> std::io::Result<Counter> {
                let file = File::open(&path)?;
                Counter::read_binary(BufReader::with_capacity(IO_BUFFER_SIZE, file))
            })
            .map_err(map_io_error)?;
        Ok(Self { inner })
    }
}

/// Count k-mers directly from one or more FASTA/FASTQ files (plain or gzipped).
#[pyfunction]
#[pyo3(signature = (paths, k, *, canonical = true, threads = 1, readers = 2, capacity = 0))]
fn count_files(
    py: Python<'_>,
    paths: &Bound<'_, PyAny>,
    k: usize,
    canonical: bool,
    threads: usize,
    readers: usize,
    capacity: usize,
) -> PyResult<PyCounter> {
    if threads == 0 {
        return Err(map_error(kbolt::Error::InvalidThreads));
    }
    if readers == 0 {
        return Err(PyValueError::new_err("readers must be greater than zero"));
    }
    let paths = extract_paths(paths)?;
    let strand = strand_mode(canonical);
    let interrupted = Mutex::new(None);
    let inner = py.detach(|| -> PyResult<Counter> {
        let cb = signal_progress_callback(&interrupted);
        if threads == 1 {
            let mut counter = Counter::with_capacity(k, strand, capacity).map_err(map_error)?;
            let mut stream = input::Stream::serial(&mut counter).with_progress(&cb);
            for path in &paths {
                stream.add_file(path).map_err(map_io_error)?;
            }
            stream.finish().map_err(map_io_error)?;
            Ok(counter)
        } else {
            let parallel =
                ParallelCounter::with_capacity(k, strand, threads, capacity).map_err(map_error)?;
            input::count_parallel_files(|| parallel.input(), &paths, readers, Some(&cb))
                .map_err(map_io_error)?;
            parallel.finish().map_err(map_error)
        }
    });
    if let Some(py_err) = interrupted.into_inner().unwrap_or_else(|p| p.into_inner()) {
        return Err(py_err);
    }
    Ok(PyCounter { inner: inner? })
}

/// Count k-mers from FASTA/FASTQ files within a memory budget (`k <= 32`) and
/// write the result straight to `output`, returning the distinct k-mer count.
#[pyfunction]
#[pyo3(signature = (paths, output, k, *, memory, canonical = true, threads = 1, readers = 2, temp_dir = None, format = "binary", sorted = false))]
#[allow(clippy::too_many_arguments)]
fn count_files_to_disk(
    py: Python<'_>,
    paths: &Bound<'_, PyAny>,
    output: PathBuf,
    k: usize,
    memory: &Bound<'_, PyAny>,
    canonical: bool,
    threads: usize,
    readers: usize,
    temp_dir: Option<PathBuf>,
    format: &str,
    sorted: bool,
) -> PyResult<u64> {
    if threads == 0 {
        return Err(map_error(kbolt::Error::InvalidThreads));
    }
    if readers == 0 {
        return Err(PyValueError::new_err("readers must be greater than zero"));
    }
    if !(1..=32).contains(&k) {
        return Err(PyValueError::new_err(
            "count_files_to_disk supports k from 1 to 32",
        ));
    }
    let binary = match format {
        "binary" => true,
        "tsv" => false,
        _ => {
            return Err(PyValueError::new_err(
                "format must be 'binary' or 'tsv'",
            ));
        }
    };
    let memory_bytes = parse_memory_size(memory)?;
    let paths = extract_paths(paths)?;
    check_output_path(&paths, &output)?;
    let strand = strand_mode(canonical);
    let directory = temp_dir.unwrap_or_else(std::env::temp_dir);
    let interrupted = Mutex::new(None);

    let result = py.detach(|| -> std::io::Result<u64> {
        let cb = signal_progress_callback(&interrupted);
        let counter = DiskCounter::new(k, strand, threads, memory_bytes, &directory)?;
        input::count_parallel_files(|| counter.input(), &paths, readers, Some(&cb))?;
        let bins = counter
            .finish()?
            .with_threads(threads + readers.min(paths.len()))
            .sorted(sorted);
        let mut writer = BufWriter::with_capacity(IO_BUFFER_SIZE, File::create(&output)?);
        let distinct = if binary {
            bins.write_binary(&mut writer)?
        } else {
            bins.write_tsv(&mut writer)?
        };
        writer.flush()?;
        Ok(distinct)
    });
    finish_with_signal(result, interrupted)
}

/// Count k-mers across an iterable of DNA records (`str`, `bytes`, or 1-D byte buffers).
#[pyfunction]
#[pyo3(signature = (records, k, *, canonical = true, threads = 1, capacity = 0))]
fn count_records(
    py: Python<'_>,
    records: &Bound<'_, PyAny>,
    k: usize,
    canonical: bool,
    threads: usize,
    capacity: usize,
) -> PyResult<PyCounter> {
    if threads == 0 {
        return Err(map_error(kbolt::Error::InvalidThreads));
    }
    reject_single_sequence(records, "count_records()")?;
    let strand = strand_mode(canonical);
    if threads == 1 {
        let mut counter = PyCounter::new(k, canonical, capacity)?;
        counter.add_records(py, records)?;
        return Ok(counter);
    }
    let mut parallel =
        ParallelCounter::with_capacity(k, strand, threads, capacity).map_err(map_error)?;
    let mut buf = Vec::with_capacity(RECORD_BATCH_BYTES);
    let mut ends = Vec::new();
    let flush = |parallel: &mut ParallelCounter,
                 buf: &mut Vec<u8>,
                 ends: &mut Vec<usize>|
     -> PyResult<()> {
        if ends.is_empty() {
            return Ok(());
        }
        let result = py.detach(|| {
            let mut start = 0;
            for &end in ends.iter() {
                parallel.add_record(&buf[start..end])?;
                start = end;
            }
            Ok::<_, kbolt::Error>(())
        });
        buf.clear();
        ends.clear();
        result.map_err(map_error)
    };

    for (i, item) in records.try_iter()?.enumerate() {
        if i.is_multiple_of(SIGNAL_CHECK_STRIDE) {
            py.check_signals()?;
        }
        let item = item?;
        let bytes = extract_seq_bytes(&item)?;
        let slice = bytes.as_slice();
        if slice.len() >= RECORD_BATCH_BYTES {
            flush(&mut parallel, &mut buf, &mut ends)?;
            py.detach(|| parallel.add_record(slice))
                .map_err(map_error)?;
            continue;
        }
        buf.extend_from_slice(slice);
        ends.push(buf.len());
        if buf.len() >= RECORD_BATCH_BYTES {
            flush(&mut parallel, &mut buf, &mut ends)?;
        }
    }
    flush(&mut parallel, &mut buf, &mut ends)?;
    let inner = py.detach(|| parallel.finish()).map_err(map_error)?;
    Ok(PyCounter { inner })
}

/// Load a `.kbolt` binary database from `path` into memory.
#[pyfunction]
fn read_binary(py: Python<'_>, path: PathBuf) -> PyResult<PyCounter> {
    PyCounter::read_binary(py, path)
}

/// Open a `.kbolt` binary database for O(k)-memory streaming iteration.
#[pyfunction]
fn open_database(py: Python<'_>, path: PathBuf) -> PyResult<PyDatabaseReader> {
    PyDatabaseReader::new(py, path)
}

#[pymodule]
fn _kbolt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyCounter>()?;
    m.add_class::<PyInputStats>()?;
    m.add_class::<PyDatabaseHeader>()?;
    m.add_class::<PyDatabaseReader>()?;
    m.add_function(wrap_pyfunction!(count_files, m)?)?;
    m.add_function(wrap_pyfunction!(count_files_to_disk, m)?)?;
    m.add_function(wrap_pyfunction!(count_records, m)?)?;
    m.add_function(wrap_pyfunction!(read_binary, m)?)?;
    m.add_function(wrap_pyfunction!(open_database, m)?)?;
    Ok(())
}
