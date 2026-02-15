#!/usr/bin/env python3
"""Minimal stats for DATA/background_edges.csv."""

from __future__ import annotations

import argparse
from pathlib import Path


def _read_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        header_line = handle.readline().strip("\n")
    if not header_line:
        return []
    return header_line.split(",")


def _count_newlines(csv_path: Path, chunk_size: int = 1024 * 1024) -> int:
    """Count rows by counting newlines (fast). Assumes no embedded newlines."""
    newline_count = 0
    with csv_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            newline_count += chunk.count(b"\n")
    return max(newline_count - 1, 0)


def get_stats(csv_path: Path, count_rows: bool) -> tuple[int | None, list[str]]:
    header = _read_header(csv_path)
    row_count = _count_newlines(csv_path) if count_rows else None
    return row_count, header


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal stats for background_edges.csv",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="CSV path (default: DATA/background_edges.csv)",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Count rows by scanning the file (can be slow on huge CSVs)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = args.path or (repo_root / "DATA" / "background_edges.csv")

    row_count, columns = get_stats(csv_path, args.count)

    print(f"File: {csv_path}")
    if row_count is None:
        print("Rows: skipped (use --count to enable)")
    else:
        print("Rows (fast newline count; assumes no embedded newlines):")
        print(row_count)
    print("Columns:")
    for column in columns:
        print(f"- {column}")


if __name__ == "__main__":
    main()
