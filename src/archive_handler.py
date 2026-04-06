"""Archive and container recursive extraction with provenance tracking.

Handles zip, tar, gz, bz2, xz, 7z, rar, iso, mbox, and pst containers.
Extracts contents to a temporary directory and returns paths with provenance
metadata linking each extracted file back to its source container.
"""

from __future__ import annotations

import logging
import mailbox
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_ARCHIVE_DEPTH = 5
MAX_EXTRACTED_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB safety limit


@dataclass
class ArchiveEntry:
    """A file extracted from a container, with provenance chain."""

    extracted_path: Path
    original_name: str
    archive_source: str
    path_within_archive: str
    provenance_chain: list[str] = field(default_factory=list)


def _safe_extract_path(base: Path, member_name: str) -> Path | None:
    """Prevent path traversal attacks in archive members."""
    target = (base / member_name).resolve()
    if not str(target).startswith(str(base.resolve())):
        logger.warning("Skipping path-traversal member: %s", member_name)
        return None
    return target


def extract_zip(path: Path, dest: Path, password: bytes | None = None) -> list[ArchiveEntry]:
    entries: list[ArchiveEntry] = []
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                safe = _safe_extract_path(dest, info.filename)
                if safe is None:
                    continue
                try:
                    safe.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info, pwd=password) as src, open(safe, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    entries.append(ArchiveEntry(
                        extracted_path=safe,
                        original_name=Path(info.filename).name,
                        archive_source=str(path),
                        path_within_archive=info.filename,
                    ))
                except Exception as exc:
                    logger.warning("Failed to extract %s from %s: %s", info.filename, path, exc)
    except Exception as exc:
        logger.warning("Failed to open zip %s: %s", path, exc)
    return entries


def extract_tar(path: Path, dest: Path) -> list[ArchiveEntry]:
    entries: list[ArchiveEntry] = []
    try:
        with tarfile.open(path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                safe = _safe_extract_path(dest, member.name)
                if safe is None:
                    continue
                try:
                    safe.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src:  # type: ignore[arg-type]
                        if src is None:
                            continue
                        with open(safe, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    entries.append(ArchiveEntry(
                        extracted_path=safe,
                        original_name=Path(member.name).name,
                        archive_source=str(path),
                        path_within_archive=member.name,
                    ))
                except Exception as exc:
                    logger.warning("Failed to extract %s from %s: %s", member.name, path, exc)
    except Exception as exc:
        logger.warning("Failed to open tar %s: %s", path, exc)
    return entries


def extract_7z(path: Path, dest: Path) -> list[ArchiveEntry]:
    entries: list[ArchiveEntry] = []
    try:
        import py7zr

        with py7zr.SevenZipFile(path, "r") as z:
            if z.needs_password():
                logger.warning("7z file %s is password-protected, skipping", path)
                return entries
            z.extractall(path=dest)
            for name in z.getnames():
                extracted = dest / name
                if extracted.is_file():
                    entries.append(ArchiveEntry(
                        extracted_path=extracted,
                        original_name=Path(name).name,
                        archive_source=str(path),
                        path_within_archive=name,
                    ))
    except ImportError:
        logger.warning("py7zr not installed; cannot extract %s", path)
    except Exception as exc:
        logger.warning("Failed to extract 7z %s: %s", path, exc)
    return entries


def extract_rar(path: Path, dest: Path) -> list[ArchiveEntry]:
    entries: list[ArchiveEntry] = []
    try:
        import rarfile

        with rarfile.RarFile(str(path), "r") as rf:
            for info in rf.infolist():
                if info.is_dir():
                    continue
                safe = _safe_extract_path(dest, info.filename)
                if safe is None:
                    continue
                try:
                    safe.parent.mkdir(parents=True, exist_ok=True)
                    rf.extract(info, dest)
                    entries.append(ArchiveEntry(
                        extracted_path=safe,
                        original_name=Path(info.filename).name,
                        archive_source=str(path),
                        path_within_archive=info.filename,
                    ))
                except Exception as exc:
                    logger.warning("Failed to extract %s from %s: %s", info.filename, path, exc)
    except ImportError:
        logger.warning("rarfile not installed; cannot extract %s", path)
    except Exception as exc:
        logger.warning("Failed to extract rar %s: %s", path, exc)
    return entries


def extract_iso(path: Path, dest: Path) -> list[ArchiveEntry]:
    entries: list[ArchiveEntry] = []
    try:
        import pycdlib

        iso = pycdlib.PyCdlib()
        iso.open(str(path))
        try:
            for dirpath, _, filenames in iso.walk(iso_path="/"):
                for fname in filenames:
                    iso_path = f"{dirpath}/{fname}" if dirpath != "/" else f"/{fname}"
                    clean_name = fname.rstrip(";1").rstrip(";2")
                    out_path = dest / clean_name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        iso.get_file_from_iso(str(out_path), iso_path=iso_path)
                        entries.append(ArchiveEntry(
                            extracted_path=out_path,
                            original_name=clean_name,
                            archive_source=str(path),
                            path_within_archive=iso_path,
                        ))
                    except Exception as exc:
                        logger.warning("Failed to extract %s from ISO %s: %s", iso_path, path, exc)
        finally:
            iso.close()
    except ImportError:
        logger.warning("pycdlib not installed; cannot extract %s", path)
    except Exception as exc:
        logger.warning("Failed to open ISO %s: %s", path, exc)
    return entries


def extract_mbox(path: Path, dest: Path) -> list[ArchiveEntry]:
    """Extract individual messages from an mbox file as .eml files."""
    entries: list[ArchiveEntry] = []
    try:
        mbox = mailbox.mbox(str(path))
        for i, message in enumerate(mbox):
            eml_name = f"message_{i:04d}.eml"
            out_path = dest / eml_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                out_path.write_text(str(message))
                entries.append(ArchiveEntry(
                    extracted_path=out_path,
                    original_name=eml_name,
                    archive_source=str(path),
                    path_within_archive=eml_name,
                ))
            except Exception as exc:
                logger.warning("Failed to extract message %d from mbox %s: %s", i, path, exc)
    except Exception as exc:
        logger.warning("Failed to open mbox %s: %s", path, exc)
    return entries


def extract_pst(path: Path, dest: Path) -> list[ArchiveEntry]:
    """Extract messages from a PST file using readpst subprocess."""
    import subprocess

    entries: list[ArchiveEntry] = []

    readpst = shutil.which("readpst")
    if not readpst:
        logger.warning("readpst not found; cannot extract PST %s", path)
        return entries

    try:
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [readpst, "-e", "-o", str(dest), str(path)],
            capture_output=True, text=True, timeout=120,
        )
        for extracted in dest.rglob("*"):
            if extracted.is_file():
                entries.append(ArchiveEntry(
                    extracted_path=extracted,
                    original_name=extracted.name,
                    archive_source=str(path),
                    path_within_archive=str(extracted.relative_to(dest)),
                ))
    except subprocess.TimeoutExpired:
        logger.warning("readpst timed out for %s", path)
    except Exception as exc:
        logger.warning("Failed to extract PST %s: %s", path, exc)
    return entries


def extract_archive(path: Path, dest: Path) -> list[ArchiveEntry]:
    """Dispatch to the correct extractor based on file type."""
    ext = path.suffix.lower()
    name_lower = path.name.lower()

    if ext == ".zip":
        return extract_zip(path, dest)
    if ext in {".tar", ".tgz", ".tbz2", ".txz"} or name_lower.endswith(
        (".tar.gz", ".tar.bz2", ".tar.xz")
    ):
        return extract_tar(path, dest)
    if ext == ".gz" and not name_lower.endswith(".tar.gz"):
        return _extract_single_gz(path, dest)
    if ext == ".bz2" and not name_lower.endswith(".tar.bz2"):
        return _extract_single_bz2(path, dest)
    if ext == ".xz" and not name_lower.endswith(".tar.xz"):
        return _extract_single_xz(path, dest)
    if ext == ".7z":
        return extract_7z(path, dest)
    if ext == ".rar":
        return extract_rar(path, dest)
    if ext == ".iso":
        return extract_iso(path, dest)
    if ext == ".mbox":
        return extract_mbox(path, dest)
    if ext in {".pst", ".ost"}:
        return extract_pst(path, dest)
    if ext == ".dmg":
        return extract_7z(path, dest)

    logger.warning("No archive handler for %s", path)
    return []


def _extract_single_gz(path: Path, dest: Path) -> list[ArchiveEntry]:
    import gzip

    out_name = path.stem
    out_path = dest / out_name
    try:
        with gzip.open(path, "rb") as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return [ArchiveEntry(
            extracted_path=out_path,
            original_name=out_name,
            archive_source=str(path),
            path_within_archive=out_name,
        )]
    except Exception as exc:
        logger.warning("Failed to decompress gz %s: %s", path, exc)
        return []


def _extract_single_bz2(path: Path, dest: Path) -> list[ArchiveEntry]:
    import bz2

    out_name = path.stem
    out_path = dest / out_name
    try:
        with bz2.open(path, "rb") as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return [ArchiveEntry(
            extracted_path=out_path,
            original_name=out_name,
            archive_source=str(path),
            path_within_archive=out_name,
        )]
    except Exception as exc:
        logger.warning("Failed to decompress bz2 %s: %s", path, exc)
        return []


def _extract_single_xz(path: Path, dest: Path) -> list[ArchiveEntry]:
    import lzma

    out_name = path.stem
    out_path = dest / out_name
    try:
        with lzma.open(path, "rb") as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return [ArchiveEntry(
            extracted_path=out_path,
            original_name=out_name,
            archive_source=str(path),
            path_within_archive=out_name,
        )]
    except Exception as exc:
        logger.warning("Failed to decompress xz %s: %s", path, exc)
        return []


def extract_archive_recursive(
    path: Path,
    temp_root: Path,
    provenance_chain: list[str] | None = None,
    depth: int = 0,
) -> list[ArchiveEntry]:
    """Recursively extract archives, tracking the full provenance chain.

    Returns flat list of leaf files (non-archive) with their full provenance.
    """
    if depth >= MAX_ARCHIVE_DEPTH:
        logger.warning("Max archive nesting depth reached for %s", path)
        return [ArchiveEntry(
            extracted_path=path,
            original_name=path.name,
            archive_source=str(provenance_chain[-1]) if provenance_chain else str(path),
            path_within_archive=path.name,
            provenance_chain=list(provenance_chain or []),
        )]

    chain = list(provenance_chain or [])
    chain.append(str(path))

    dest = temp_root / f"depth_{depth}" / path.stem
    dest.mkdir(parents=True, exist_ok=True)

    entries = extract_archive(path, dest)

    from .classifier import classify_file, Route

    final_entries: list[ArchiveEntry] = []
    for entry in entries:
        entry.provenance_chain = list(chain)
        classified = classify_file(entry.extracted_path)
        if classified and classified.route == Route.ARCHIVE:
            nested = extract_archive_recursive(
                entry.extracted_path, temp_root, chain, depth + 1,
            )
            final_entries.extend(nested)
        else:
            final_entries.append(entry)

    return final_entries
