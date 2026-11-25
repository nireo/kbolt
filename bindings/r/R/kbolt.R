# Internal helper: unwrap an extendr `result_list` into a value or clean R stop().
.unwrap <- function(res) {
  if (!is.null(res$err)) {
    stop(res$err, call. = FALSE)
  }
  res$ok
}

# Internal helper: attach data.frame attributes in O(1) without copying columns.
.as_df <- function(cols) {
  n <- if (length(cols) > 0L) length(cols[[1L]]) else 0L
  structure(cols, class = "data.frame", row.names = .set_row_names(n))
}

.check_scalar_logical <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("`%s` must be a single non-NA logical value", name), call. = FALSE)
  }
  x
}

.check_scalar_numeric <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("`%s` must be a single non-NA numeric value", name), call. = FALSE)
  }
  as.double(x)
}

.check_optional_numeric <- function(x, name) {
  if (is.null(x)) {
    return(NULL)
  }
  .check_scalar_numeric(x, name)
}

.check_scalar_string <- function(x, name) {
  if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
    stop(sprintf("`%s` must be a single non-empty character string", name), call. = FALSE)
  }
  x
}

.check_character_vec <- function(x, name) {
  if (!is.character(x)) {
    stop(sprintf("`%s` must be a character vector", name), call. = FALSE)
  }
  x
}

#' Create an empty kbolt k-mer counter
#' @export
kbolt_counter <- function(k, canonical = TRUE, capacity = 0) {
  k <- .check_scalar_numeric(k, "k")
  canonical <- .check_scalar_logical(canonical, "canonical")
  capacity <- .check_scalar_numeric(capacity, "capacity")
  .unwrap(.Call(wrap__rust_kbolt_counter, k, canonical, capacity))
}

#' Get k-mer length of a counter
#' @export
kbolt_k <- function(counter) {
  .unwrap(.Call(wrap__rust_kbolt_k, counter))
}

#' Check whether a counter combines reverse complements
#' @export
kbolt_is_canonical <- function(counter) {
  .unwrap(.Call(wrap__rust_kbolt_is_canonical, counter))
}

#' Number of distinct k-mers in a counter (returned as numeric/double)
#' @export
kbolt_len <- function(counter) {
  .unwrap(.Call(wrap__rust_kbolt_len, counter))
}

#' @export
length.kbolt_counter <- function(x) {
  kbolt_len(x)
}

#' @export
print.kbolt_counter <- function(x, ...) {
  cat(sprintf(
    "<kbolt_counter: k=%d, canonical=%s, distinct=%.0f>\n",
    kbolt_k(x),
    if (kbolt_is_canonical(x)) "TRUE" else "FALSE",
    kbolt_len(x)
  ))
  invisible(x)
}

#' Add DNA sequence records (character vector) into a counter
#' @export
kbolt_add_records <- function(counter, records) {
  records <- .check_character_vec(records, "records")
  .unwrap(.Call(wrap__rust_kbolt_add_records, counter, records))
  invisible(counter)
}

#' Stream FASTA/FASTQ files (plain or gzipped) into a counter
#' @export
kbolt_add_files <- function(counter, paths) {
  paths <- path.expand(.check_character_vec(paths, "paths"))
  .unwrap(.Call(wrap__rust_kbolt_add_files, counter, paths))
}

#' Merge counts from source into target (emptying source)
#' @export
kbolt_merge <- function(target, source) {
  .unwrap(.Call(wrap__rust_kbolt_merge, target, source))
  invisible(target)
}

#' Count k-mers directly from FASTA/FASTQ files (plain or gzipped)
#' @export
kbolt_count_files <- function(
  paths,
  k,
  canonical = TRUE,
  threads = 1,
  readers = 2,
  capacity = 0
) {
  paths <- path.expand(.check_character_vec(paths, "paths"))
  k <- .check_scalar_numeric(k, "k")
  canonical <- .check_scalar_logical(canonical, "canonical")
  threads <- .check_scalar_numeric(threads, "threads")
  readers <- .check_scalar_numeric(readers, "readers")
  capacity <- .check_scalar_numeric(capacity, "capacity")
  .unwrap(.Call(wrap__rust_kbolt_count_files, paths, k, canonical, threads, readers, capacity))
}

#' Count k-mers from a character vector of DNA sequences
#' @export
kbolt_count_records <- function(
  records,
  k,
  canonical = TRUE,
  threads = 1,
  capacity = 0
) {
  records <- .check_character_vec(records, "records")
  k <- .check_scalar_numeric(k, "k")
  canonical <- .check_scalar_logical(canonical, "canonical")
  threads <- .check_scalar_numeric(threads, "threads")
  capacity <- .check_scalar_numeric(capacity, "capacity")
  .unwrap(.Call(wrap__rust_kbolt_count_records, records, k, canonical, threads, capacity))
}

#' Count k-mers from files within a memory budget on disk (k <= 32)
#' @export
kbolt_count_files_to_disk <- function(
  paths,
  output,
  k,
  memory,
  canonical = TRUE,
  threads = 1,
  readers = 2,
  temp_dir = NULL,
  format = c("binary", "tsv"),
  sorted = FALSE
) {
  paths <- path.expand(.check_character_vec(paths, "paths"))
  output <- path.expand(.check_scalar_string(output, "output"))
  k <- .check_scalar_numeric(k, "k")
  if (length(memory) != 1L || is.na(memory) || !(is.character(memory) || is.numeric(memory))) {
    stop("`memory` must be a single numeric byte count or size string such as '2G'", call. = FALSE)
  }
  canonical <- .check_scalar_logical(canonical, "canonical")
  threads <- .check_scalar_numeric(threads, "threads")
  readers <- .check_scalar_numeric(readers, "readers")
  if (!is.null(temp_dir)) {
    temp_dir <- path.expand(.check_scalar_string(temp_dir, "temp_dir"))
  }
  format <- match.arg(format)
  sorted <- .check_scalar_logical(sorted, "sorted")
  .unwrap(.Call(
    wrap__rust_kbolt_count_files_to_disk,
    paths,
    output,
    k,
    memory,
    canonical,
    threads,
    readers,
    temp_dir,
    format,
    sorted
  ))
}

#' Vectorized lookup of k-mer counts
#' @export
kbolt_get <- function(counter, kmers) {
  kmers <- .check_character_vec(kmers, "kmers")
  .unwrap(.Call(wrap__rust_kbolt_get, counter, kmers))
}

#' @export
`[.kbolt_counter` <- function(x, i) {
  kbolt_get(x, i)
}

#' Compute the k-mer count spectrum (histogram) as a data.frame
#' @export
kbolt_histogram <- function(counter) {
  .as_df(.unwrap(.Call(wrap__rust_kbolt_histogram, counter)))
}

#' Return up to n most abundant k-mers as a data.frame
#' @export
kbolt_top_k <- function(counter, n = 20) {
  n <- .check_scalar_numeric(n, "n")
  .as_df(.unwrap(.Call(wrap__rust_kbolt_top_k, counter, n)))
}

#' Count distinct k-mers falling within [min_count, max_count]
#' @export
kbolt_count_filtered <- function(counter, min_count = 1, max_count = NULL) {
  min_count <- .check_scalar_numeric(min_count, "min_count")
  max_count <- .check_optional_numeric(max_count, "max_count")
  .unwrap(.Call(wrap__rust_kbolt_count_filtered, counter, min_count, max_count))
}

#' Export k-mers and counts matching optional count bounds as a data.frame
#' @export
kbolt_to_data_frame <- function(counter, min_count = 1, max_count = NULL) {
  min_count <- .check_scalar_numeric(min_count, "min_count")
  max_count <- .check_optional_numeric(max_count, "max_count")
  .as_df(.unwrap(.Call(wrap__rust_kbolt_to_list, counter, min_count, max_count)))
}

#' @export
as.data.frame.kbolt_counter <- function(
  x,
  row.names = NULL,
  optional = FALSE,
  ...,
  min_count = 1,
  max_count = NULL
) {
  df <- kbolt_to_data_frame(x, min_count = min_count, max_count = max_count)
  if (!is.null(row.names)) {
    attr(df, "row.names") <- row.names
  }
  df
}

#' Write headerless KMER<TAB>COUNT TSV file
#' @export
kbolt_write_tsv <- function(counter, path, min_count = 1, max_count = NULL) {
  path <- path.expand(.check_scalar_string(path, "path"))
  min_count <- .check_scalar_numeric(min_count, "min_count")
  max_count <- .check_optional_numeric(max_count, "max_count")
  .unwrap(.Call(wrap__rust_kbolt_write_tsv, counter, path, min_count, max_count))
  invisible(path)
}

#' Write a .kbolt binary count database
#' @export
kbolt_write_binary <- function(counter, path, min_count = 1, max_count = NULL) {
  path <- path.expand(.check_scalar_string(path, "path"))
  min_count <- .check_scalar_numeric(min_count, "min_count")
  max_count <- .check_optional_numeric(max_count, "max_count")
  .unwrap(.Call(wrap__rust_kbolt_write_binary, counter, path, min_count, max_count))
  invisible(path)
}

#' Load a .kbolt binary database into memory
#' @export
kbolt_read_binary <- function(path) {
  path <- path.expand(.check_scalar_string(path, "path"))
  .unwrap(.Call(wrap__rust_kbolt_read_binary, path))
}

#' Open a .kbolt binary database for O(k)-memory streaming reads
#' @export
kbolt_open_database <- function(path) {
  path <- path.expand(.check_scalar_string(path, "path"))
  .unwrap(.Call(wrap__rust_kbolt_open_database, path))
}

#' Read metadata header from an open kbolt_database
#' @export
kbolt_database_header <- function(db) {
  .unwrap(.Call(wrap__rust_kbolt_database_header, db))
}

#' Read up to n entries from a kbolt_database as a data.frame
#' @export
kbolt_database_read <- function(db, n = 100000) {
  n <- .check_scalar_numeric(n, "n")
  .as_df(.unwrap(.Call(wrap__rust_kbolt_database_read, db, n)))
}

#' Stream remaining entries of a kbolt_database directly to a TSV file
#' @export
kbolt_database_write_tsv <- function(db, path) {
  path <- path.expand(.check_scalar_string(path, "path"))
  .unwrap(.Call(wrap__rust_kbolt_database_write_tsv, db, path))
  invisible(path)
}

#' Close a kbolt_database reader
#' @export
kbolt_database_close <- function(db) {
  .unwrap(.Call(wrap__rust_kbolt_database_close, db))
  invisible(NULL)
}

#' @export
print.kbolt_database <- function(x, ...) {
  h <- kbolt_database_header(x)
  cat(sprintf(
    "<kbolt_database: k=%d, canonical=%s, entries=%.0f, order='%s', state='%s'>\n",
    h$k,
    if (h$canonical) "TRUE" else "FALSE",
    h$entries,
    h$order,
    if (h$open) "open" else "closed"
  ))
  invisible(x)
}
