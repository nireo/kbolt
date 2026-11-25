"""Fast exact DNA k-mer counting, backed by Rust."""

from kbolt._kbolt import (
    Counter,
    DatabaseHeader,
    DatabaseReader,
    InputStats,
    __version__,
    count_files,
    count_files_to_disk,
    count_records,
    open_database,
    read_binary,
)

__all__ = [
    "Counter",
    "DatabaseHeader",
    "DatabaseReader",
    "InputStats",
    "__version__",
    "count_files",
    "count_files_to_disk",
    "count_records",
    "open_database",
    "read_binary",
]
