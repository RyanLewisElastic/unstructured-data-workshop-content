"""Database content extraction for SQLite and Microsoft Access files.

Extracts schema and row data from embedded databases, producing one
logical document per table with full schema context.
"""

from __future__ import annotations

import csv
import io
import logging
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_ROWS_PER_TABLE = 5000
MAX_TEXT_PER_TABLE = 50000


@dataclass
class TableExtract:
    """Content extracted from a single database table."""

    table_name: str
    column_names: list[str]
    row_count: int
    text_content: str
    schema_ddl: str = ""


@dataclass
class DatabaseResult:
    """Result of extracting an entire database file."""

    tables: list[TableExtract] = field(default_factory=list)
    db_type: str = "unknown"
    errors: list[str] = field(default_factory=list)


def extract_sqlite(path: Path) -> DatabaseResult:
    """Extract all tables from a SQLite database."""
    result = DatabaseResult(db_type="sqlite")

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except Exception as exc:
        result.errors.append(f"Failed to open SQLite database: {exc}")
        return result

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = cursor.fetchall()

        for table_name, ddl in tables:
            try:
                table_extract = _extract_sqlite_table(conn, table_name, ddl or "")
                result.tables.append(table_extract)
            except Exception as exc:
                result.errors.append(f"Failed to extract table {table_name}: {exc}")
    except Exception as exc:
        result.errors.append(f"Failed to read schema: {exc}")
    finally:
        conn.close()

    return result


def _extract_sqlite_table(conn: sqlite3.Connection, table_name: str, ddl: str) -> TableExtract:
    cursor = conn.cursor()

    cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
    total_rows = cursor.fetchone()[0]

    cursor.execute(f"SELECT * FROM [{table_name}] LIMIT {MAX_ROWS_PER_TABLE}")
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description] if cursor.description else []

    text_parts: list[str] = []
    text_parts.append(f"Table: {table_name}")
    text_parts.append(f"Columns: {', '.join(col_names)}")
    text_parts.append(f"Total rows: {total_rows}")
    if ddl:
        text_parts.append(f"Schema: {ddl}")
    text_parts.append("")

    for row in rows:
        row_str = " | ".join(str(v) if v is not None else "NULL" for v in row)
        text_parts.append(row_str)
        if sum(len(p) for p in text_parts) > MAX_TEXT_PER_TABLE:
            text_parts.append(f"... ({total_rows - len(rows)} more rows truncated)")
            break

    return TableExtract(
        table_name=table_name,
        column_names=col_names,
        row_count=total_rows,
        text_content="\n".join(text_parts)[:MAX_TEXT_PER_TABLE],
        schema_ddl=ddl,
    )


def extract_access(path: Path) -> DatabaseResult:
    """Extract tables from a Microsoft Access database using mdbtools."""
    result = DatabaseResult(db_type="access")

    mdb_tables = shutil.which("mdb-tables")
    mdb_export = shutil.which("mdb-export")

    if not mdb_tables or not mdb_export:
        result.errors.append(
            "mdbtools not installed (mdb-tables / mdb-export not found). "
            "Install with: brew install mdbtools (macOS) or apt-get install mdbtools (Debian)"
        )
        return result

    try:
        tables_result = subprocess.run(
            [mdb_tables, "-1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if tables_result.returncode != 0:
            result.errors.append(f"mdb-tables failed: {tables_result.stderr[:200]}")
            return result

        table_names = [t.strip() for t in tables_result.stdout.strip().split("\n") if t.strip()]
    except Exception as exc:
        result.errors.append(f"Failed to list Access tables: {exc}")
        return result

    for table_name in table_names:
        try:
            export_result = subprocess.run(
                [mdb_export, str(path), table_name],
                capture_output=True, text=True, timeout=60,
            )
            if export_result.returncode != 0:
                result.errors.append(f"mdb-export failed for {table_name}: {export_result.stderr[:200]}")
                continue

            csv_text = export_result.stdout
            reader = csv.reader(io.StringIO(csv_text))
            rows = list(reader)

            if not rows:
                continue

            col_names = rows[0] if rows else []
            data_rows = rows[1:]

            text_parts = [
                f"Table: {table_name}",
                f"Columns: {', '.join(col_names)}",
                f"Total rows: {len(data_rows)}",
                "",
            ]
            for row in data_rows[:MAX_ROWS_PER_TABLE]:
                text_parts.append(" | ".join(row))
                if sum(len(p) for p in text_parts) > MAX_TEXT_PER_TABLE:
                    break

            result.tables.append(TableExtract(
                table_name=table_name,
                column_names=col_names,
                row_count=len(data_rows),
                text_content="\n".join(text_parts)[:MAX_TEXT_PER_TABLE],
            ))
        except subprocess.TimeoutExpired:
            result.errors.append(f"mdb-export timed out for {table_name}")
        except Exception as exc:
            result.errors.append(f"Failed to export Access table {table_name}: {exc}")

    return result


def extract_database(path: Path) -> DatabaseResult:
    """Auto-detect database type and extract content."""
    ext = path.suffix.lower()

    if ext in {".sqlite", ".sqlite3", ".db", ".sdb"}:
        return extract_sqlite(path)

    if ext in {".mdb", ".accdb"}:
        return extract_access(path)

    # Try SQLite first as a heuristic (many .db files are SQLite)
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        if header.startswith(b"SQLite format 3"):
            return extract_sqlite(path)
    except Exception:
        pass

    return DatabaseResult(errors=[f"Unrecognized database format: {ext}"])
