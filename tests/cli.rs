use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use flate2::Compression;
use flate2::write::GzEncoder;

fn run(args: &[&str], input: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_kbolt"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Input and output can both exceed pipe capacity; feed input concurrently.
    let mut stdin = child.stdin.take().unwrap();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let _ = stdin.write_all(input);
        });
        child.wait_with_output().unwrap()
    })
}

fn table(output: &Output) -> BTreeMap<String, u64> {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout.clone())
        .unwrap()
        .lines()
        .map(|line| {
            let (key, count) = line.split_once('\t').unwrap();
            (key.to_owned(), count.parse().unwrap())
        })
        .collect()
}

fn gzip(input: &[u8]) -> Vec<u8> {
    let mut writer = GzEncoder::new(Vec::new(), Compression::fast());
    writer.write_all(input).unwrap();
    writer.finish().unwrap()
}

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "kbolt-test-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn file(&self, name: &str, contents: &[u8]) -> String {
        let path = self.0.join(name);
        fs::write(&path, contents).unwrap();
        path.to_str().unwrap().to_owned()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn stdin_defaults_to_canonical_and_forward_is_explicit() {
    let input = b">a\nACG\n>b\ncgt\n";
    let canonical = run(&["count", "-k", "3"], input);
    assert_eq!(table(&canonical), BTreeMap::from([("ACG".to_owned(), 2)]));
    assert!(canonical.stderr.is_empty());
    let forward = run(&["count", "-k", "3", "--forward", "-"], input);
    assert_eq!(
        table(&forward),
        BTreeMap::from([("ACG".to_owned(), 1), ("CGT".to_owned(), 1)])
    );
}

#[test]
fn capacity_hint_preserves_output_and_is_not_a_limit() {
    let mut input = b">reads\n".to_vec();
    input.extend_from_slice(&b"acGTTACGAGTCAGATTCGAC".repeat(8));
    input.extend_from_slice(b"\n>ambiguous\nACGTNNACGT\n");
    for k in [1, 3, 31, 32, 33, 63, 64, 65] {
        let k = k.to_string();
        for forward in [false, true] {
            let mut args = vec!["count", "-k", &k];
            if forward {
                args.push("--forward");
            }
            let expected = table(&run(&args, &input));
            for capacity in ["0", "1", "1000"] {
                let mut args = args.clone();
                args.extend(["--capacity", capacity]);
                assert_eq!(table(&run(&args, &input)), expected);
            }
        }
    }
}

#[test]
fn rejects_invalid_and_unrepresentable_capacity() {
    for capacity in ["-1", "abc", &usize::MAX.to_string()] {
        let output = run(&["count", "-k", "31", "--capacity", capacity], b">x\nACG\n");
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!error.is_empty());
        assert!(!error.contains("panicked"), "{error}");
    }
}

#[test]
fn accepts_wrapped_fastq_crlf_and_no_final_newline() {
    let output = run(&["count", "-k", "3"], b"@x\r\nAC\r\nGT\r\n+\r\n@+\r\n!!");
    assert_eq!(table(&output), BTreeMap::from([("ACG".to_owned(), 2)]));
}

#[test]
fn detects_gzip_stdin_and_concatenated_members() {
    let mut compressed = gzip(b">a\nAC");
    // A gzip member boundary need not be a record or line boundary.
    compressed.extend(gzip(b"GT\n"));
    let output = run(&["count", "-k", "3"], &compressed);
    assert_eq!(table(&output), BTreeMap::from([("ACG".to_owned(), 2)]));
}

#[test]
fn combines_files_and_stdin_without_joining_records() {
    let temp = TempDir::new();
    let first = temp.file("no-extension", b">x\nAC\n");
    let second = temp.file("compressed.data", &gzip(b"@x\nGT\n+\n!!\n"));
    let output = run(&["count", "-k", "3", &first, "-", &second], b">x\nACG\n");
    assert_eq!(table(&output), BTreeMap::from([("ACG".to_owned(), 1)]));
}

#[test]
fn huge_single_line_and_wrapped_fasta_preserve_chunk_windows() {
    let sequence = b"ACGTGTCAGATTCGAC".repeat(20_000);
    let mut input = b">large\n".to_vec();
    input.extend_from_slice(&sequence);
    input.extend_from_slice(b"\n>short\nAC\n>short2\nGT\n");
    for k in [1, 3, 32, 33, 63, 64, 65] {
        let output = run(&["count", "-k", &k.to_string(), "--forward"], &input);
        let actual = table(&output);
        let mut expected = kbolt::Counter::new(k, kbolt::StrandMode::Forward).unwrap();
        expected.add_record(&sequence).unwrap();
        expected.add_record(b"AC").unwrap();
        expected.add_record(b"GT").unwrap();
        let expected = expected
            .counts()
            .map(|(key, count)| (String::from_utf8(key).unwrap(), count))
            .collect();
        assert_eq!(actual, expected, "k={k}");
    }
    let mut wrapped = b">large\n".to_vec();
    for chunk in sequence.chunks(71) {
        wrapped.extend_from_slice(chunk);
        wrapped.push(b'\n');
    }
    let output = run(&["count", "-k", "31"], &wrapped);
    let total: u64 = table(&output).values().sum();
    assert_eq!(total, (sequence.len() - 30) as u64);
}

#[test]
fn writes_output_file_and_progress_only_to_stderr() {
    let temp = TempDir::new();
    let path = temp.0.join("counts.tsv");
    let output = run(
        &[
            "count",
            "-k",
            "3",
            "--output",
            path.to_str().unwrap(),
            "--progress",
        ],
        b">x\nACGT\n",
    );
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(fs::read_to_string(path).unwrap(), "ACG\t2\n");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("1 records, 4 bases, 1 distinct k-mers"));
}

#[test]
fn input_errors_do_not_truncate_output_and_identify_file_and_line() {
    let temp = TempDir::new();
    let input = temp.file("bad.fa", b">x\nAC GT\n");
    let output_path = temp.file("keep.tsv", b"keep me\n");
    let output = run(&["count", "-k", "3", "-o", &output_path, &input], b"");
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("bad.fa: line 2:"), "{stderr}");
    assert_eq!(fs::read(output_path).unwrap(), b"keep me\n");
}

#[test]
fn rejects_malformed_fastq_and_damaged_gzip() {
    let mut truncated = gzip(b">x\nACGT\n");
    truncated.truncate(truncated.len() - 4);
    let mut corrupt = gzip(b">x\nACGT\n");
    let crc = corrupt.len() - 8;
    corrupt[crc] ^= 1;
    for input in [
        b"@x\nACGT\n+\n!!!\n".to_vec(),
        b"@x\nACGT\n+\n!!!!!\n".to_vec(),
        b"@x\nACGT\n".to_vec(),
        truncated,
        corrupt,
    ] {
        let output = run(&["count", "-k", "3"], &input);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
    }
}

#[test]
fn handles_empty_and_short_inputs() {
    for input in [b"".as_slice(), b">x\nAC\n", b"@x\n\n+\n\n", b">x\nNNNN\n"] {
        let output = run(&["count", "-k", "3"], input);
        assert!(table(&output).is_empty());
    }
    assert!(table(&run(&["count", "-k", "3"], &gzip(b""))).is_empty());
}

#[test]
fn validates_arguments_and_missing_files() {
    for args in [
        vec!["count", "-k", "0"],
        vec!["count", "-k", "-1"],
        vec!["count", "-k", "abc"],
        vec!["count"],
        vec!["count", "-k", "3", "-", "-"],
        vec!["count", "-k", "3", "/nonexistent/kbolt-input.fa"],
    ] {
        let output = run(&args, b"");
        assert!(!output.status.success(), "{args:?}");
        assert!(!output.stderr.is_empty());
    }
    assert!(run(&["--help"], b"").status.success());
    assert!(run(&["count", "--help"], b"").status.success());
    assert!(run(&["--version"], b"").status.success());
}

#[test]
fn refuses_overwriting_an_input_path() {
    let temp = TempDir::new();
    let input = temp.file("input.fa", b">x\nACGT\n");
    let output = run(&["count", "-k", "3", "-o", &input, &input], b"");
    assert!(!output.status.success());
    assert_eq!(fs::read(input).unwrap(), b">x\nACGT\n");
}

#[test]
fn binary_output_and_streaming_dump_match_tsv_for_all_modes() {
    let mut input = b">mixed\n".to_vec();
    input.extend_from_slice(&b"acGTTACGAGTCAGATTCGAC".repeat(8));
    input.extend_from_slice(b"\n>ambiguous\nACGTNNACGT\n");
    for k in [1, 3, 31, 32, 33, 63, 64, 65] {
        let k = k.to_string();
        for forward in [false, true] {
            let mut args = vec!["count", "-k", &k];
            if forward {
                args.push("--forward");
            }
            let expected = table(&run(&args, &input));
            args.extend(["--format", "binary", "--capacity", "1000"]);
            let binary = run(&args, &input);
            assert!(
                binary.status.success(),
                "{}",
                String::from_utf8_lossy(&binary.stderr)
            );
            assert!(binary.stdout.starts_with(b"KBOLTDB\0"));
            assert!(binary.stderr.is_empty());
            let loaded = kbolt::Counter::read_binary(binary.stdout.as_slice()).unwrap();
            assert_eq!(loaded.len(), expected.len());
            assert_eq!(table(&run(&["dump"], &binary.stdout)), expected);
        }
    }
    for input in [b"".as_slice(), b">empty\n", b">short\nAC\n", b">n\nNNNN\n"] {
        let binary = run(&["count", "-k", "31", "--format", "binary"], input);
        assert!(binary.status.success());
        assert_eq!(binary.stdout.len(), 32);
        assert!(table(&run(&["dump", "-"], &binary.stdout)).is_empty());
    }
}

#[test]
fn writes_binary_file_and_exports_tsv_file() {
    let temp = TempDir::new();
    let database = temp.0.join("counts.kbolt");
    let tsv = temp.0.join("counts.tsv");
    let output = run(
        &[
            "count",
            "-k",
            "3",
            "--format",
            "binary",
            "--output",
            database.to_str().unwrap(),
            "--progress",
        ],
        &gzip(b">x\nACGT\n"),
    );
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("1 records, 4 bases, 1 distinct k-mers")
    );
    let output = run(
        &[
            "dump",
            database.to_str().unwrap(),
            "-o",
            tsv.to_str().unwrap(),
        ],
        b"",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    assert_eq!(fs::read_to_string(tsv).unwrap(), "ACG\t2\n");
}

#[test]
fn dump_rejects_invalid_databases_and_identifies_input() {
    let binary = run(&["count", "-k", "3", "--format", "binary"], b">x\nACG\n").stdout;
    let mut trailing = binary.clone();
    trailing.push(0);
    let mut version = binary.clone();
    version[8] = 2;
    let mut zero_count = binary.clone();
    zero_count[33..].fill(0);
    for input in [
        b"".to_vec(),
        b"not a database".to_vec(),
        binary[..binary.len() - 1].to_vec(),
        trailing,
        version,
        zero_count,
    ] {
        let output = run(&["dump"], &input);
        assert!(!output.status.success());
        assert!(!output.stderr.is_empty());
    }
    let temp = TempDir::new();
    let bad = temp.file("bad.kbolt", b"bad");
    let keep = temp.file("keep.tsv", b"keep me\n");
    let output = run(&["dump", &bad, "-o", &keep], b"");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("bad.kbolt"));
    assert_eq!(fs::read(keep).unwrap(), b"keep me\n");
    assert!(
        !run(&["dump", "/nonexistent/kbolt.db"], b"")
            .status
            .success()
    );
    assert!(
        !run(&["count", "-k", "3", "--format", "unknown"], b"")
            .status
            .success()
    );
}

#[test]
fn binary_count_and_dump_refuse_overwriting_inputs() {
    let temp = TempDir::new();
    let input = temp.file("input.fa", b">x\nACGT\n");
    let output = run(
        &[
            "count", "-k", "3", "--format", "binary", "-o", &input, &input,
        ],
        b"",
    );
    assert!(!output.status.success());
    assert_eq!(fs::read(&input).unwrap(), b">x\nACGT\n");
    let binary = run(&["count", "-k", "3", "--format", "binary"], b">x\nACGT\n").stdout;
    let database = temp.file("counts.kbolt", &binary);
    assert!(
        !run(&["dump", &database, "-o", &database], b"")
            .status
            .success()
    );
    assert_eq!(fs::read(database).unwrap(), binary);
}

#[test]
fn parallel_cli_matches_serial_for_formats_compression_and_record_boundaries() {
    let mut sequence = b"acGTTGCAGATTCGAC".repeat(40_000);
    for index in [65_535, 65_536, 262_143, 262_144] {
        sequence[index] = b'N';
    }
    let mut fasta = b">long\r\n".to_vec();
    for part in sequence.chunks(77) {
        fasta.extend_from_slice(part);
        fasta.extend_from_slice(b"\r\n");
    }
    fasta.extend_from_slice(b">short\nAC\n>short2\nGT\n");
    for k in [1, 31, 32, 33, 63, 64, 65] {
        let k = k.to_string();
        for forward in [false, true] {
            let mut args = vec!["count", "-k", &k];
            if forward {
                args.push("--forward");
            }
            let expected = table(&run(&args, &fasta));
            args.extend(["--threads", "3", "--progress", "--timings"]);
            let parallel = run(&args, &gzip(&fasta));
            assert_eq!(table(&parallel), expected);
            let stderr = String::from_utf8_lossy(&parallel.stderr);
            assert!(stderr.contains("timings: counting="), "{stderr}");
            assert!(stderr.contains("distinct k-mers"), "{stderr}");
            args.extend(["--format", "binary"]);
            let binary = run(&args, &fasta);
            assert!(binary.status.success());
            assert_eq!(table(&run(&["dump"], &binary.stdout)), expected);
        }
    }
    // FASTQ qualities, mixed files, stdin, and invalid input retain their semantics.
    let temp = TempDir::new();
    let path = temp.file("reads.fastq", b"@a\nACG\n+\n!!!\n@b\nCGT\n+\n@@@\n");
    assert_eq!(
        table(&run(
            &["count", "-k", "3", "-t", "2", &path, "-"],
            b">c\nACG\n"
        )),
        BTreeMap::from([("ACG".to_owned(), 3)])
    );
    let output = temp.file("keep.tsv", b"keep me\n");
    let bad = run(
        &["count", "-k", "3", "-t", "2", "-o", &output],
        b"@x\nACGT\n+\n!!!\n",
    );
    assert!(!bad.status.success());
    assert!(bad.stdout.is_empty());
    assert_eq!(fs::read(output).unwrap(), b"keep me\n");
    assert!(table(&run(&["count", "-k", "31", "-t", "2",], b"")).is_empty());
    for threads in ["0", "-1", "18446744073709551615"] {
        assert!(
            !run(&["count", "-k", "3", "--threads", threads], b"")
                .status
                .success()
        );
    }
}
