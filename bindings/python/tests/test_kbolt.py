import array
import gzip
import pickle
import sys
import threading
import time
from collections import Counter as PyCounterRef
from pathlib import Path
from unittest.mock import patch

import pytest

import kbolt


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


def reference_counts(records: list[str], k: int, canonical: bool = True) -> dict[str, int]:
    counts: PyCounterRef[str] = PyCounterRef()
    for rec in records:
        upper = rec.upper()
        for i in range(len(upper) - k + 1):
            kmer = upper[i : i + k]
            if any(c not in "ACGT" for c in kmer):
                continue
            if canonical:
                kmer = min(kmer, revcomp(kmer))
            counts[kmer] += 1
    return dict(counts)


def encode_2bit(kmer: str) -> int:
    table = {"A": 0, "C": 1, "G": 2, "T": 3}
    val = 0
    for ch in kmer.upper():
        val = (val << 2) | table[ch]
    return val


@pytest.mark.parametrize("k", [1, 3, 31, 32, 33, 48, 64, 65, 97, 128, 129])
@pytest.mark.parametrize("canonical", [True, False])
def test_basic_counting_and_queries_all_backends(k: int, canonical: bool) -> None:
    records = [
        "acgtACGTtgcaTGCA" * 12,
        "A" * 160,
        "T" * 160,
        "ACGTTN" * 30,
        "AC",
        "",
    ]
    expected = reference_counts(records, k, canonical)

    c = kbolt.Counter(k, canonical=canonical)
    assert c.k == k
    assert c.canonical is canonical
    assert len(c) == 0
    assert not bool(c)
    assert c["A" * k] == 0
    assert ("A" * k) not in c

    c.add_record(records[0])
    c.add_records(r.encode() if i % 2 else r for i, r in enumerate(records[1:]))

    assert len(c) == len(expected)
    assert bool(c) == (len(expected) > 0)
    assert c.to_dict() == expected
    assert dict(c.items()) == expected
    assert set(c.keys()) == set(expected.keys())
    assert set(iter(c)) == set(expected.keys())
    assert sorted(c.values()) == sorted(expected.values())

    sample_keys = list(expected.keys())[:25]
    queries: list[str | bytes] = []
    expected_q: list[int] = []
    for kmer in sample_keys:
        cnt = expected[kmer]
        assert kmer in c
        assert c[kmer] == cnt
        assert c[kmer.lower()] == cnt
        assert c[kmer.encode()] == cnt
        assert c[bytearray(kmer.encode())] == cnt
        assert c[memoryview(kmer.encode())] == cnt
        assert c.get(kmer) == cnt

        queries.extend([kmer, kmer.lower(), kmer.encode()])
        expected_q.extend([cnt, cnt, cnt])

        rc = revcomp(kmer)
        rc_cnt = cnt if canonical else expected.get(rc, 0)
        assert c[rc] == rc_cnt
        queries.append(rc)
        expected_q.append(rc_cnt)

    # Absent / invalid / wrong length queries.
    bad_kmer = "N" * k
    assert bad_kmer not in c
    assert c[bad_kmer] == 0
    assert c.get(bad_kmer) == 0
    assert c.get(bad_kmer, -1) == -1
    assert c.get(bad_kmer, None) is None
    assert c[""] == 0
    assert c["A" * (k + 1)] == 0
    queries.extend([bad_kmer, "", "A" * (k + 1)])
    expected_q.extend([0, 0, 0])

    assert c.get_many(queries) == expected_q

    # Packed lookups for k <= 32, ValueError for k > 32.
    if k <= 32:
        packed_keys = [encode_2bit(kmer) for kmer in sample_keys]
        packed_expected = [expected[kmer] for kmer in sample_keys]
        for pk, pe in zip(packed_keys, packed_expected):
            assert c.get_packed(pk) == pe
        assert c.get_packed_many(packed_keys) == packed_expected
        assert c.get_packed_many(array.array("Q", packed_keys)) == packed_expected

        # Packed bytearray round-trip (from to_packed_buffers back into get_packed_many).
        pkeys_buf, pcounts_buf = c.to_packed_buffers()
        assert c.get_packed_many(pkeys_buf) == list(memoryview(pcounts_buf).cast("Q"))
        assert c.get_packed_many_buffer(pkeys_buf) == pcounts_buf

        if k < 32:
            with pytest.raises(ValueError):
                c.get_packed(1 << (2 * k))
    else:
        with pytest.raises(ValueError):
            c.get_packed(0)
        with pytest.raises(ValueError):
            c.get_packed_many([0])


def test_packed_buffer_endianness_dtypes_and_ndim() -> None:
    np = pytest.importorskip("numpy")

    c = kbolt.count_records(["ACGTACGTACGT", "TTTTGGGGCCCC"], 5, canonical=True)
    pkeys, pcounts = c.to_numpy(packed=True)
    expected = pcounts.tolist()

    # Native, explicit little-endian, explicit big-endian, strided, and reversed uint64.
    assert c.get_packed_many(pkeys) == expected
    assert c.get_packed_many(pkeys.astype("<u8")) == expected
    assert c.get_packed_many(pkeys.astype(">u8")) == expected
    assert c.get_packed_many(pkeys[::-1]) == expected[::-1]
    assert c.get_packed_many(pkeys[::2]) == expected[::2]

    # Signed int64 (native & big-endian) works for non-negative keys, rejects negative.
    assert c.get_packed_many(pkeys.astype("<i8")) == expected
    assert c.get_packed_many(pkeys.astype(">i8")) == expected
    with pytest.raises(ValueError):
        c.get_packed_many(np.array([-1], dtype=np.int64))

    # Vectorized buffer output matches.
    out_buf = c.get_packed_many_buffer(pkeys.astype(">u8"))
    assert np.frombuffer(out_buf, dtype=np.uint64).tolist() == expected

    # Byte buffer with non-multiple-of-8 length is rejected (not iterated byte-by-byte).
    with pytest.raises(ValueError):
        c.get_packed_many(bytearray(5))

    # 2-D buffer is rejected.
    with pytest.raises(ValueError):
        c.get_packed_many(np.zeros((2, 2), dtype=np.uint64))

    # Unsupported dtypes (float64, uint32) raise TypeError instead of falling back.
    with pytest.raises(TypeError):
        c.get_packed_many(np.zeros(4, dtype=np.float64))
    with pytest.raises(TypeError):
        c.get_packed_many(np.zeros(4, dtype=np.uint32))


def test_sequence_buffers_and_large_records_release_gil() -> None:
    c = kbolt.Counter(4, canonical=True)
    c.add_record(bytearray(b"ACGTACGT"))
    c.add_record(memoryview(b"ACGTACGT"))
    assert c["ACGT"] == 4

    # Large record (> 1 MiB batch threshold) counted without copying and with GIL released.
    big = "ACGT" * 400_000  # 1.6 Mbp
    ticks = 0
    stop = False

    def ticker() -> None:
        nonlocal ticks
        while not stop:
            ticks += 1
            time.sleep(0.001)

    t = threading.Thread(target=ticker)
    t.start()
    c_big = kbolt.Counter(31, canonical=True)
    c_big.add_record(big)
    c_big.add_records([big])
    c_par = kbolt.count_records([big], 31, canonical=True, threads=2)
    stop = True
    t.join()

    assert c_big.len if hasattr(c_big, "len") else len(c_big) == len(c_par)
    assert c_big[big[:31]] == 2 * c_par[big[:31]]
    assert ticks > 0


def test_count_records_parallel_merge_and_pickle() -> None:
    records = ["ACGTACGTACGT" * 20, "TGCATGCATGCA" * 15, "acgtnacgt" * 10] * 50
    expected = reference_counts(records, 31, canonical=True)

    c1 = kbolt.count_records(records, 31, canonical=True, threads=1)
    c4 = kbolt.count_records(records, 31, canonical=True, threads=4)
    assert c1.to_dict() == expected
    assert c4.to_dict() == expected

    # Merge small into large AND large into small (tests swap optimization).
    left = kbolt.count_records(records[:10], 31, canonical=True)
    right = kbolt.count_records(records[10:], 31, canonical=True)
    left.merge(right)
    assert left.to_dict() == expected
    assert len(right) == 0

    with pytest.raises(ValueError, match="itself"):
        left.merge(left)

    incompatible = kbolt.Counter(31, canonical=False)
    incompatible.add_record("A" * 31)
    with pytest.raises(ValueError):
        left.merge(incompatible)
    assert len(incompatible) == 1  # untouched on error

    # Pickle round-trip.
    restored = pickle.loads(pickle.dumps(left))
    assert restored.k == left.k
    assert restored.canonical == left.canonical
    assert restored.to_dict() == expected


def test_histogram_top_k_and_filters() -> None:
    records = (
        ["ACGT" * 20] * 10
        + ["TGCA" * 20] * 4
        + ["AAAA" * 20] * 2
        + ["CCCC" * 20]
    )
    for k in [3, 31, 45, 130]:
        c = kbolt.count_records(records, k, canonical=True, threads=2)
        ref = reference_counts(records, k, canonical=True)

        ref_hist: PyCounterRef[int] = PyCounterRef(ref.values())
        assert c.histogram() == sorted(ref_hist.items())

        sorted_ref = sorted(ref.items(), key=lambda kv: (-kv[1], kv[0]))
        assert c.top_k(0) == []
        assert c.top_k(5) == sorted_ref[:5]
        assert c.top_k(len(ref) + 10) == sorted_ref

        ref_ge5 = {kmer: cnt for kmer, cnt in ref.items() if cnt >= 5}
        assert c.count_filtered(min_count=5) == len(ref_ge5)
        assert c.to_dict(min_count=5) == ref_ge5
        assert dict(c.items(min_count=5)) == ref_ge5
        assert set(c.keys(min_count=5)) == set(ref_ge5.keys())
        assert sorted(c.values(min_count=5)) == sorted(ref_ge5.values())

        ref_mid = {kmer: cnt for kmer, cnt in ref.items() if 2 <= cnt <= 10}
        assert c.count_filtered(min_count=2, max_count=10) == len(ref_mid)
        assert c.to_dict(min_count=2, max_count=10) == ref_mid

        assert c.count_filtered(min_count=10, max_count=2) == 0
        assert c.items(min_count=10, max_count=2) == []


@pytest.mark.parametrize("k", [3, 31, 32, 45, 130])
def test_numpy_and_buffer_export(k: int) -> None:
    np = pytest.importorskip("numpy")

    records = ["ACGTACGTACGT" * 15, "GGGAAACCC" * 20] * 5
    c = kbolt.count_records(records, k, canonical=True)
    ref = reference_counts(records, k, canonical=True)

    keys_arr, counts_arr = c.to_numpy(packed=False)
    assert keys_arr.dtype == np.dtype(f"S{k}")
    assert counts_arr.dtype == np.uint64
    assert len(keys_arr) == len(ref)
    assert {k_b.decode("ascii"): int(cnt) for k_b, cnt in zip(keys_arr, counts_arr)} == ref

    keys_f, counts_f = c.to_numpy(min_count=6, packed=False)
    ref_f = {kmer: cnt for kmer, cnt in ref.items() if cnt >= 6}
    assert {k_b.decode("ascii"): int(cnt) for k_b, cnt in zip(keys_f, counts_f)} == ref_f

    keys_e, counts_e = c.to_numpy(min_count=10**9, packed=False)
    assert len(keys_e) == 0 and len(counts_e) == 0

    if k <= 32:
        pkeys, pcounts = c.to_numpy(packed=True)
        assert pkeys.dtype == np.uint64
        assert pcounts.dtype == np.uint64
        assert len(pkeys) == len(ref)
        assert c.get_packed_many(pkeys) == pcounts.tolist()
    else:
        with pytest.raises(ValueError):
            c.to_numpy(packed=True)
        with pytest.raises(ValueError):
            c.to_packed_buffers()


def test_to_numpy_without_numpy_installed() -> None:
    c = kbolt.count_records(["ACGT"], 3)
    with patch.dict(sys.modules, {"numpy": None}):
        with pytest.raises(ModuleNotFoundError):
            c.to_numpy()


def test_file_counting_binary_tsv_and_database_reader(tmp_path: Path) -> None:
    fa = tmp_path / "reads1.fa"
    fq_gz = tmp_path / "reads2.fq.gz"

    fa.write_text(">r1\nACGTACGTACGT\n>r2\nTTTTGGGGCCCC\n")
    with gzip.open(fq_gz, "wt") as fh:
        fh.write("@q1\nacgtacgtacgt\n+\n!!!!!!!!!!!!\n")
        fh.write("@q2\nACGTNNNNACGT\n+\n############\n")

    expected = reference_counts(
        ["ACGTACGTACGT", "TTTTGGGGCCCC", "acgtacgtacgt", "ACGTNNNNACGT"],
        k=5,
        canonical=True,
    )

    c_single = kbolt.count_files(fa, 5)
    assert c_single.to_dict() == reference_counts(["ACGTACGTACGT", "TTTTGGGGCCCC"], 5)

    c_serial = kbolt.count_files([str(fa), fq_gz], 5, threads=1)
    c_parallel = kbolt.count_files([fa, fq_gz], 5, threads=4, readers=2)
    assert c_serial.to_dict() == expected
    assert c_parallel.to_dict() == expected

    c_inc = kbolt.Counter(5)
    s1 = c_inc.add_file(fa)
    assert s1.records == 2 and s1.bases == 24
    s2 = c_inc.add_files([fq_gz])
    assert s2.records == 2 and s2.bases == 24
    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert (s1 == "not stats") is False
    assert c_inc.to_dict() == expected

    tsv_path = tmp_path / "counts.tsv"
    bin_path = tmp_path / "counts.kbolt"
    bin_filt_path = tmp_path / "counts_ge2.kbolt"
    dumped_tsv = tmp_path / "dumped.tsv"

    c_parallel.write_tsv(tsv_path)
    tsv_parsed = {
        line.split("\t")[0]: int(line.split("\t")[1])
        for line in tsv_path.read_text().splitlines()
    }
    assert tsv_parsed == expected

    c_parallel.write_binary(bin_path)
    loaded = kbolt.read_binary(bin_path)
    assert loaded.to_dict() == expected
    assert kbolt.Counter.read_binary(bin_path).to_dict() == expected

    with kbolt.open_database(bin_path) as db:
        assert db.k == 5
        assert db.canonical is True
        assert db.entries == len(expected)
        assert db.header.order == "hash"
        assert db.header.ascending is True
        assert hash(db.header) == hash(db.header)
        assert (db.header == "other") is False
        assert "canonical=True" in repr(db)
        assert "canonical=True" in repr(db.header)
        streamed = dict(db)
        assert streamed == expected

    # Closed DatabaseReader behavior.
    db_closed = kbolt.DatabaseReader(bin_path)
    db_closed.close()
    assert list(db_closed) == []
    with pytest.raises(ValueError, match="closed"):
        db_closed.write_tsv(dumped_tsv)

    # Truncated database closes cleanly after raising on first error.
    trunc_path = tmp_path / "truncated.kbolt"
    raw = bin_path.read_bytes()
    trunc_path.write_bytes(raw[:-2])
    db_trunc = kbolt.open_database(trunc_path)
    with pytest.raises(OSError):
        list(db_trunc)
    assert list(db_trunc) == []
    assert "state='closed'" in repr(db_trunc)

    with kbolt.DatabaseReader(bin_path) as db:
        db.write_tsv(dumped_tsv)
    assert {
        line.split("\t")[0]: int(line.split("\t")[1])
        for line in dumped_tsv.read_text().splitlines()
    } == expected

    c_parallel.write_binary(bin_filt_path, min_count=2)
    loaded_filt = kbolt.read_binary(bin_filt_path)
    assert loaded_filt.to_dict() == {k: v for k, v in expected.items() if v >= 2}

    # Disk-backed counting (binary & sorted TSV, string & int memory).
    disk_bin = tmp_path / "disk.kbolt"
    disk_tsv = tmp_path / "disk_sorted.tsv"
    n_bin = kbolt.count_files_to_disk(
        [fa, fq_gz],
        disk_bin,
        k=5,
        memory="64M",
        threads=2,
        temp_dir=tmp_path,
        format="binary",
        sorted=True,
    )
    assert n_bin == len(expected)
    with kbolt.open_database(disk_bin) as db:
        assert db.header.order == "kmer"
        assert db.header.ascending is True
        assert list(db) == sorted(expected.items())

    n_tsv = kbolt.count_files_to_disk(
        [fa, fq_gz],
        disk_tsv,
        k=5,
        memory=64 * 1024 * 1024,
        threads=2,
        temp_dir=tmp_path,
        format="tsv",
        sorted=True,
    )
    assert n_tsv == len(expected)
    tsv_rows = [
        (line.split("\t")[0], int(line.split("\t")[1]))
        for line in disk_tsv.read_text().splitlines()
    ]
    assert tsv_rows == sorted(expected.items())

    # Guard against overwriting an input file.
    orig_fa = fa.read_text()
    with pytest.raises(ValueError, match="input"):
        kbolt.count_files_to_disk(fa, fa, 5, memory="64M", format="tsv")
    assert fa.read_text() == orig_fa

    # Missing output parent directory fails upfront.
    with pytest.raises(FileNotFoundError):
        kbolt.count_files_to_disk(fa, tmp_path / "nonexistent_dir" / "out.kbolt", 5, memory="64M")


def test_validation_and_error_handling(tmp_path: Path) -> None:
    assert isinstance(kbolt.__version__, str) and len(kbolt.__version__) > 0
    assert "canonical=True" in repr(kbolt.Counter(3))

    with pytest.raises(ValueError):
        kbolt.Counter(0)
    with pytest.raises(ValueError):
        kbolt.count_files([], 31, threads=0)
    with pytest.raises(ValueError):
        kbolt.count_files([], 31, threads=1, readers=0)
    with pytest.raises(ValueError):
        kbolt.count_files([], 31, threads=2, readers=0)
    with pytest.raises(ValueError):
        kbolt.count_records([], 31, threads=0)
    with pytest.raises(FileNotFoundError):
        kbolt.count_files(tmp_path / "missing.fa", 31)
    with pytest.raises(TypeError):
        kbolt.Counter(3).add_record(12345)  # type: ignore[arg-type]

    # Passing bare str/bytes/bytearray to iterable APIs raises TypeError instead of silently iterating chars.
    c = kbolt.Counter(3)
    for bad in ("ACGTACGT", b"ACGTACGT", bytearray(b"ACGTACGT")):
        with pytest.raises(TypeError):
            c.add_records(bad)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            kbolt.count_records(bad, 3)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            c.get_many(bad)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        kbolt.count_files(b"reads.fa", 31)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        kbolt.count_files_to_disk([], tmp_path / "out.kbolt", 33, memory="64M")
    with pytest.raises(ValueError):
        kbolt.count_files_to_disk([], tmp_path / "out.kbolt", 5, memory="1M")
    with pytest.raises(ValueError):
        kbolt.count_files_to_disk([], tmp_path / "out.kbolt", 5, memory="64M", readers=0)
    with pytest.raises(ValueError):
        kbolt.count_files_to_disk([], tmp_path / "out.kbolt", 5, memory="64M", format="csv")  # type: ignore[arg-type]
