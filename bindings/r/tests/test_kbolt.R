library(kbolt)

assert_true <- function(cond, msg = "assertion failed") {
  if (!isTRUE(cond)) stop(msg, call. = FALSE)
}

assert_equal <- function(actual, expected, msg = NULL) {
  if (!identical(actual, expected)) {
    stop(
      paste0(
        if (!is.null(msg)) paste0(msg, ": ") else "",
        "expected ", paste(deparse(expected), collapse = " "),
        ", got ", paste(deparse(actual), collapse = " ")
      ),
      call. = FALSE
    )
  }
}

assert_error <- function(expr, pattern = NULL) {
  res <- tryCatch(
    {
      force(expr)
      NULL
    },
    error = function(e) conditionMessage(e)
  )
  if (is.null(res)) {
    stop("expected an error, but expression succeeded", call. = FALSE)
  }
  if (!is.null(pattern) && !grepl(pattern, res, ignore.case = TRUE)) {
    stop(sprintf("error %s did not match %s", sQuote(res), sQuote(pattern)), call. = FALSE)
  }
}

revcomp <- function(seq) {
  comp <- chartr("ACGTacgt", "TGCAtgca", seq)
  paste(rev(strsplit(comp, "", fixed = TRUE)[[1L]]), collapse = "")
}

ref_counts <- function(records, k, canonical = TRUE) {
  env <- new.env(hash = TRUE, parent = emptyenv())
  for (rec in records) {
    if (is.na(rec) || nchar(rec) < k) next
    up <- toupper(rec)
    chars <- strsplit(up, "", fixed = TRUE)[[1L]]
    for (i in seq_len(length(chars) - k + 1L)) {
      sub <- chars[i:(i + k - 1L)]
      if (any(!(sub %in% c("A", "C", "G", "T")))) next
      kmer <- paste(sub, collapse = "")
      if (canonical) {
        rc <- revcomp(kmer)
        if (rc < kmer) kmer <- rc
      }
      cur <- env[[kmer]]
      env[[kmer]] <- if (is.null(cur)) 1 else cur + 1
    }
  }
  ks <- sort(ls(env))
  if (length(ks) == 0L) {
    return(data.frame(kmer = character(0), count = numeric(0), stringsAsFactors = FALSE))
  }
  vals <- vapply(ks, function(x) env[[x]], numeric(1), USE.NAMES = FALSE)
  data.frame(kmer = ks, count = vals, stringsAsFactors = FALSE)
}

sort_df <- function(df) {
  if (nrow(df) == 0L) return(data.frame(kmer = character(0), count = numeric(0)))
  o <- order(df$kmer)
  out <- df[o, , drop = FALSE]
  rownames(out) <- NULL
  out
}

# 1. All backends (packed, wide, bytes) & both strand modes
records <- c(
  paste(rep("acgtACGTtgcaTGCA", 12), collapse = ""),
  paste(rep("A", 160), collapse = ""),
  paste(rep("T", 160), collapse = ""),
  paste(rep("ACGTTN", 30), collapse = ""),
  "AC",
  "",
  NA_character_
)

for (k in c(1, 3, 31, 32, 33, 48, 64, 65, 97, 128, 129)) {
  for (canonical in c(TRUE, FALSE)) {
    expected <- ref_counts(records, k, canonical = canonical)

    c <- kbolt_counter(k, canonical = canonical)
    assert_equal(kbolt_k(c), as.integer(k))
    assert_equal(kbolt_is_canonical(c), canonical)
    assert_equal(length(c), 0L)
    assert_equal(kbolt_len(c), 0)

    kbolt_add_records(c, records)
    assert_equal(length(c), nrow(expected))
    assert_equal(kbolt_len(c), as.double(nrow(expected)))

    df <- sort_df(as.data.frame(c))
    assert_equal(df, expected, sprintf("k=%d canonical=%s", k, canonical))

    # Vectorized lookups: present, lowercase, revcomp, absent, invalid, wrong length, NA
    sample_kmers <- head(expected$kmer, 20)
    sample_exp <- head(expected$count, 20)
    assert_equal(c[sample_kmers], sample_exp)
    assert_equal(kbolt_get(c, tolower(sample_kmers)), sample_exp)

    rc_kmers <- vapply(sample_kmers, revcomp, character(1), USE.NAMES = FALSE)
    if (canonical) {
      assert_equal(c[rc_kmers], sample_exp)
    }

    bad <- c(paste(rep("N", k), collapse = ""), "", paste(rep("A", k + 1), collapse = ""), NA_character_)
    got_bad <- c[bad]
    assert_equal(got_bad[1:3], c(0, 0, 0))
    assert_true(is.na(got_bad[4]))
  }
}

# 2. Parallel count_records and merge (including swap & self-merge error)
recs_par <- rep(c(
  paste(rep("ACGTACGTACGT", 20), collapse = ""),
  paste(rep("TGCATGCATGCA", 15), collapse = ""),
  paste(rep("acgtnacgt", 10), collapse = "")
), 20)
exp_par <- ref_counts(recs_par, 31, canonical = TRUE)

c1 <- kbolt_count_records(recs_par, k = 31, canonical = TRUE, threads = 1)
c4 <- kbolt_count_records(recs_par, k = 31, canonical = TRUE, threads = 4)
assert_equal(sort_df(as.data.frame(c1)), exp_par)
assert_equal(sort_df(as.data.frame(c4)), exp_par)

left <- kbolt_count_records(recs_par[1:5], k = 31)
right <- kbolt_count_records(recs_par[6:length(recs_par)], k = 31)
kbolt_merge(left, right)
assert_equal(sort_df(as.data.frame(left)), exp_par)
assert_equal(length(right), 0L)
assert_error(kbolt_merge(left, left), "itself")

incompat <- kbolt_counter(31, canonical = FALSE)
kbolt_add_records(incompat, paste(rep("A", 31), collapse = ""))
assert_error(kbolt_merge(left, incompat))
assert_equal(length(incompat), 1L)

# 3. Histogram, top_k, and filtered export
recs_h <- c(
  rep(paste(rep("ACGT", 20), collapse = ""), 10),
  rep(paste(rep("TGCA", 20), collapse = ""), 4),
  rep(paste(rep("AAAA", 20), collapse = ""), 2),
  paste(rep("CCCC", 20), collapse = "")
)

for (k in c(3, 31, 45, 130)) {
  c <- kbolt_count_records(recs_h, k = k, threads = 2)
  ref <- ref_counts(recs_h, k)

  tab <- table(ref$count)
  exp_hist <- data.frame(
    count = as.numeric(names(tab)),
    frequency = as.numeric(tab),
    stringsAsFactors = FALSE
  )
  assert_equal(kbolt_histogram(c), exp_hist, sprintf("hist k=%d", k))

  ord <- order(-ref$count, ref$kmer)
  exp_top <- ref[ord, , drop = FALSE]
  rownames(exp_top) <- NULL
  assert_equal(nrow(kbolt_top_k(c, 0)), 0L)
  assert_equal(kbolt_top_k(c, 5), head(exp_top, 5))
  assert_equal(kbolt_top_k(c, nrow(ref) + 10), exp_top)

  ref_ge5 <- ref[ref$count >= 5, , drop = FALSE]
  rownames(ref_ge5) <- NULL
  assert_equal(kbolt_count_filtered(c, min_count = 5), as.double(nrow(ref_ge5)))
  assert_equal(sort_df(as.data.frame(c, min_count = 5)), ref_ge5)

  ref_mid <- ref[ref$count >= 2 & ref$count <= 10, , drop = FALSE]
  rownames(ref_mid) <- NULL
  assert_equal(kbolt_count_filtered(c, min_count = 2, max_count = 10), as.double(nrow(ref_mid)))
  assert_equal(sort_df(kbolt_to_data_frame(c, min_count = 2, max_count = 10)), ref_mid)

  assert_equal(kbolt_count_filtered(c, min_count = 10, max_count = 2), 0)
  assert_equal(nrow(as.data.frame(c, min_count = 10, max_count = 2)), 0L)
}

# 4. File counting (FASTA + gzipped FASTQ), binary/TSV I/O, DatabaseReader, disk counting
tmp <- tempfile("kbolt_r_test_")
dir.create(tmp)
on.exit(unlink(tmp, recursive = TRUE), add = TRUE)

fa <- file.path(tmp, "reads1.fa")
fq_gz <- file.path(tmp, "reads2.fq.gz")
writeLines(c(">r1", "ACGTACGTACGT", ">r2", "TTTTGGGGCCCC"), fa)
gz_con <- gzfile(fq_gz, "wb")
writeLines(c(
  "@q1", "acgtacgtacgt", "+", "!!!!!!!!!!!!",
  "@q2", "ACGTNNNNACGT", "+", "############"
), gz_con)
close(gz_con)

exp_files <- ref_counts(c("ACGTACGTACGT", "TTTTGGGGCCCC", "acgtacgtacgt", "ACGTNNNNACGT"), 5)

cf1 <- kbolt_count_files(c(fa, fq_gz), k = 5, threads = 1)
cf4 <- kbolt_count_files(c(fa, fq_gz), k = 5, threads = 4, readers = 2)
assert_equal(sort_df(as.data.frame(cf1)), exp_files)
assert_equal(sort_df(as.data.frame(cf4)), exp_files)

c_inc <- kbolt_counter(5)
stats <- kbolt_add_files(c_inc, c(fa, fq_gz))
assert_equal(stats, c(records = 4, bases = 48))
assert_equal(sort_df(as.data.frame(c_inc)), exp_files)

# Print methods
out_c <- capture.output(print(cf4))
assert_true(grepl("k=5, canonical=TRUE", out_c))

# TSV & binary I/O + filtering
tsv_path <- file.path(tmp, "counts.tsv")
bin_path <- file.path(tmp, "counts.kbolt")
bin_filt <- file.path(tmp, "counts_ge2.kbolt")
dumped_tsv <- file.path(tmp, "dumped.tsv")

kbolt_write_tsv(cf4, tsv_path)
tsv_df <- read.table(tsv_path, sep = "\t", col.names = c("kmer", "count"), stringsAsFactors = FALSE)
tsv_df$count <- as.numeric(tsv_df$count)
assert_equal(sort_df(tsv_df), exp_files)

kbolt_write_binary(cf4, bin_path)
loaded <- kbolt_read_binary(bin_path)
assert_equal(sort_df(as.data.frame(loaded)), exp_files)

kbolt_write_binary(cf4, bin_filt, min_count = 2)
loaded_filt <- kbolt_read_binary(bin_filt)
exp_ge2 <- exp_files[exp_files$count >= 2, , drop = FALSE]
rownames(exp_ge2) <- NULL
assert_equal(sort_df(as.data.frame(loaded_filt)), exp_ge2)

# Streaming DatabaseReader (chunked read + header + close + write_tsv + truncated error)
db <- kbolt_open_database(bin_path)
h <- kbolt_database_header(db)
assert_equal(h$k, 5L)
assert_true(h$canonical)
assert_equal(h$entries, as.double(nrow(exp_files)))
assert_equal(h$order, "hash")
assert_true(h$ascending)
assert_true(h$open)
assert_true(grepl("state='open'", capture.output(print(db))))

chunks <- list()
repeat {
  chunk <- kbolt_database_read(db, n = 3)
  if (nrow(chunk) == 0L) break
  chunks[[length(chunks) + 1L]] <- chunk
}
streamed <- do.call(rbind, chunks)
assert_equal(sort_df(streamed), exp_files)
assert_true(!kbolt_database_header(db)$open)

db2 <- kbolt_open_database(bin_path)
kbolt_database_write_tsv(db2, dumped_tsv)
assert_error(kbolt_database_write_tsv(db2, dumped_tsv), "closed")
dumped_df <- read.table(dumped_tsv, sep = "\t", col.names = c("kmer", "count"), stringsAsFactors = FALSE)
dumped_df$count <- as.numeric(dumped_df$count)
assert_equal(sort_df(dumped_df), exp_files)

# Truncated database closes after raising error
trunc_path <- file.path(tmp, "trunc.kbolt")
raw_bytes <- readBin(bin_path, "raw", n = file.info(bin_path)$size)
writeBin(raw_bytes[seq_len(length(raw_bytes) - 2L)], trunc_path)
db_trunc <- kbolt_open_database(trunc_path)
assert_error(kbolt_database_read(db_trunc, n = 100))
assert_true(!kbolt_database_header(db_trunc)$open)
assert_equal(nrow(kbolt_database_read(db_trunc, n = 100)), 0L)

# Disk-backed counting (binary & sorted TSV, string & numeric memory)
disk_bin <- file.path(tmp, "disk.kbolt")
disk_tsv <- file.path(tmp, "disk_sorted.tsv")
n_bin <- kbolt_count_files_to_disk(
  c(fa, fq_gz),
  output = disk_bin,
  k = 5,
  memory = "64M",
  threads = 2,
  temp_dir = tmp,
  format = "binary",
  sorted = TRUE
)
assert_equal(n_bin, as.double(nrow(exp_files)))
db_disk <- kbolt_open_database(disk_bin)
assert_equal(kbolt_database_header(db_disk)$order, "kmer")
assert_true(kbolt_database_header(db_disk)$ascending)
assert_equal(kbolt_database_read(db_disk, n = 100), exp_files)
kbolt_database_close(db_disk)

n_tsv <- kbolt_count_files_to_disk(
  c(fa, fq_gz),
  output = disk_tsv,
  k = 5,
  memory = 64 * 1024 * 1024,
  threads = 2,
  temp_dir = tmp,
  format = "tsv",
  sorted = TRUE
)
assert_equal(n_tsv, as.double(nrow(exp_files)))
disk_tsv_df <- read.table(disk_tsv, sep = "\t", col.names = c("kmer", "count"), stringsAsFactors = FALSE)
disk_tsv_df$count <- as.numeric(disk_tsv_df$count)
assert_equal(disk_tsv_df, exp_files)

# Guard against overwriting input file and missing parent directory
orig_fa <- readLines(fa)
assert_error(
  kbolt_count_files_to_disk(fa, output = fa, k = 5, memory = "64M", format = "tsv"),
  "input"
)
assert_equal(readLines(fa), orig_fa)
assert_error(
  kbolt_count_files_to_disk(fa, output = file.path(tmp, "missing_dir", "out.kbolt"), k = 5, memory = "64M")
)

# 5. Argument validation & error handling
assert_error(kbolt_counter(0))
assert_error(kbolt_counter(-1))
assert_error(kbolt_counter(3.5))
assert_error(kbolt_count_files(fa, k = 5, threads = 0))
assert_error(kbolt_count_files(fa, k = 5, threads = 1, readers = 0))
assert_error(kbolt_count_files(fa, k = 5, threads = 2, readers = 0))
assert_error(kbolt_count_files(c(fa, NA_character_), k = 5))
assert_error(kbolt_count_files(file.path(tmp, "missing.fa"), k = 5))
assert_error(kbolt_count_files_to_disk(fa, output = disk_bin, k = 33, memory = "64M"))
assert_error(kbolt_count_files_to_disk(fa, output = disk_bin, k = 5, memory = "1M"))
assert_error(kbolt_count_files_to_disk(fa, output = disk_bin, k = 5, memory = "64M", readers = 0))
assert_error(kbolt_get("not_a_counter", "ACG"))

cat("All R kbolt tests passed!\n")
