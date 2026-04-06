"""Robust file extraction pipeline with multi-handler routing.

Classifies every file, dispatches to the correct handler (Unstructured,
image OCR, audio/video transcription, database extraction, archive
unpacking), and produces standardized ExtractedFile records with rich
provenance metadata.  Files that defeat all handlers are quarantined.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Generator

from PIL import Image
from rich.progress import Progress

from .classifier import ClassifiedFile, Route, classify_directory, classify_file

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFile:
    """Standardized record produced by extraction."""

    file_path: str
    file_name: str
    file_size: int
    mime_type: str
    file_extension: str
    text_content: str
    element_types: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    is_image: bool = False
    image_base64: str | None = None
    source_archive: str | None = None
    source_archive_path: str | None = None
    extraction_method: str = "unstructured"
    extraction_status: str = "success"
    extraction_errors: list[str] = field(default_factory=list)
    detected_mime: str | None = None
    language_detected: str | None = None
    content_hash: str | None = None
    page_count: int | None = None
    duration_seconds: float | None = None
    is_encrypted: bool = False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _extract_image_base64(path: Path, max_size: int = 512) -> str | None:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        logger.warning("Failed to process image %s: %s", path, exc)
        return None


def _ocr_image_subprocess(path: Path, timeout: int = 30) -> str:
    import subprocess
    import sys

    script = (
        "import sys\n"
        "from unstructured.partition.image import partition_image\n"
        "elements = partition_image(filename=sys.argv[1], strategy='ocr_only')\n"
        "print('\\n\\n'.join(str(el) for el in elements))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.warning("OCR timed out for %s", path)
    except Exception as exc:
        logger.warning("OCR subprocess failed for %s: %s", path, exc)
    return ""


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

def _handle_image(path: Path, rel_path: str, classified: ClassifiedFile) -> ExtractedFile:
    ext = path.suffix.lower()
    image_b64 = _extract_image_base64(path)
    text_content = _ocr_image_subprocess(path)
    element_types = ["Image"] if not text_content else ["Image", "OCRText"]

    return ExtractedFile(
        file_path=rel_path,
        file_name=path.name,
        file_size=classified.file_size,
        mime_type=classified.mime_type,
        file_extension=ext,
        text_content=text_content[:100000],
        element_types=element_types,
        is_image=True,
        image_base64=image_b64,
        extraction_method="ocr",
        detected_mime=classified.mime_type,
        content_hash=_sha256(path),
    )


def _handle_unstructured(path: Path, rel_path: str, classified: ClassifiedFile) -> ExtractedFile:
    ext = path.suffix.lower()
    text_content = ""
    element_types: list[str] = []
    metadata: dict = {}
    mime_type = classified.mime_type
    errors: list[str] = []
    page_count = None
    status = "success"

    try:
        from unstructured.partition.auto import partition

        elements = partition(filename=str(path), strategy="auto")

        if elements:
            mime_type = getattr(elements[0].metadata, "filetype", mime_type) or mime_type
            text_parts = []
            pages_seen: set[int] = set()
            for el in elements:
                el_type = type(el).__name__
                element_types.append(el_type)
                text_parts.append(str(el))

                el_meta = el.metadata
                if hasattr(el_meta, "page_number") and el_meta.page_number:
                    pages_seen.add(el_meta.page_number)
                if hasattr(el_meta, "subject") and el_meta.subject:
                    metadata["email_subject"] = el_meta.subject
                if hasattr(el_meta, "sent_from") and el_meta.sent_from:
                    metadata["email_from"] = el_meta.sent_from
                if hasattr(el_meta, "sent_to") and el_meta.sent_to:
                    metadata["email_to"] = el_meta.sent_to

            text_content = "\n\n".join(text_parts)
            if pages_seen:
                page_count = max(pages_seen)

    except Exception as exc:
        errors.append(f"Unstructured failed: {exc}")
        logger.warning("Unstructured failed for %s: %s — falling back to raw read", rel_path, exc)
        try:
            raw = path.read_text(errors="replace")
            text_content = raw[:50000]
            element_types = ["RawText"]
            mime_type = "text/plain"
            status = "partial"
        except Exception:
            text_content = ""
            element_types = ["Binary"]
            status = "failed"

    lang = _detect_language(text_content)

    return ExtractedFile(
        file_path=rel_path,
        file_name=path.name,
        file_size=classified.file_size,
        mime_type=mime_type,
        file_extension=ext,
        text_content=text_content[:100000],
        element_types=list(set(element_types)),
        metadata=metadata,
        extraction_method="unstructured",
        extraction_status=status,
        extraction_errors=errors,
        detected_mime=classified.mime_type,
        content_hash=_sha256(path),
        language_detected=lang,
        page_count=page_count,
    )


def _handle_audio_video(
    path: Path, rel_path: str, classified: ClassifiedFile, whisper_model: str = "base",
) -> ExtractedFile:
    from .av_handler import process_av_file

    result = process_av_file(path, whisper_model=whisper_model)
    meta_dict: dict = {}
    if result.metadata.tags:
        meta_dict["av_tags"] = result.metadata.tags
    if result.metadata.codec:
        meta_dict["codec"] = result.metadata.codec
    if result.metadata.width and result.metadata.height:
        meta_dict["resolution"] = f"{result.metadata.width}x{result.metadata.height}"

    return ExtractedFile(
        file_path=rel_path,
        file_name=path.name,
        file_size=classified.file_size,
        mime_type=classified.mime_type,
        file_extension=path.suffix.lower(),
        text_content=result.transcript[:100000],
        element_types=["Transcript"] if result.transcript else ["AudioVideo"],
        metadata=meta_dict,
        extraction_method="whisper" if result.transcript else "ffprobe",
        extraction_status="success" if result.transcript else "partial",
        extraction_errors=result.errors,
        detected_mime=classified.mime_type,
        content_hash=_sha256(path),
        language_detected=result.language_detected,
        duration_seconds=result.metadata.duration_seconds,
    )


def _handle_database(path: Path, rel_path: str, classified: ClassifiedFile) -> list[ExtractedFile]:
    """Returns multiple ExtractedFile records, one per table."""
    from .db_handler import extract_database

    result = extract_database(path)
    content_hash = _sha256(path)
    files: list[ExtractedFile] = []

    if not result.tables:
        return [ExtractedFile(
            file_path=rel_path,
            file_name=path.name,
            file_size=classified.file_size,
            mime_type=classified.mime_type,
            file_extension=path.suffix.lower(),
            text_content="",
            element_types=["Database"],
            extraction_method="database",
            extraction_status="failed",
            extraction_errors=result.errors,
            detected_mime=classified.mime_type,
            content_hash=content_hash,
        )]

    for table in result.tables:
        table_rel = f"{rel_path}::{table.table_name}"
        files.append(ExtractedFile(
            file_path=table_rel,
            file_name=f"{path.name}::{table.table_name}",
            file_size=classified.file_size,
            mime_type=classified.mime_type,
            file_extension=path.suffix.lower(),
            text_content=table.text_content[:100000],
            element_types=["DatabaseTable"],
            metadata={
                "db_type": result.db_type,
                "table_name": table.table_name,
                "columns": table.column_names,
                "row_count": table.row_count,
                "schema_ddl": table.schema_ddl,
            },
            extraction_method="database",
            extraction_status="success",
            extraction_errors=result.errors,
            detected_mime=classified.mime_type,
            content_hash=content_hash,
        ))

    return files


def _detect_language(text: str) -> str | None:
    if not text or len(text.strip()) < 20:
        return None
    try:
        from langdetect import detect

        return detect(text[:2000])
    except ImportError:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def _extract_single_classified(
    classified: ClassifiedFile,
    root: Path,
    whisper_model: str = "base",
    source_archive: str | None = None,
    source_archive_path: str | None = None,
) -> list[ExtractedFile]:
    """Extract content from a single classified file. Returns a list because
    databases produce one record per table."""
    try:
        rel_path = str(classified.path.relative_to(root))
    except ValueError:
        rel_path = classified.path.name

    if source_archive:
        # Override rel_path for archive-sourced files
        rel_path = f"{source_archive}/{source_archive_path or classified.path.name}"

    route = classified.route

    results: list[ExtractedFile] = []

    if route == Route.IMAGE:
        results = [_handle_image(classified.path, rel_path, classified)]
    elif route == Route.AUDIO_VIDEO:
        results = [_handle_audio_video(classified.path, rel_path, classified, whisper_model)]
    elif route == Route.DATABASE:
        results = _handle_database(classified.path, rel_path, classified)
    elif route == Route.UNSTRUCTURED:
        results = [_handle_unstructured(classified.path, rel_path, classified)]
    elif route in (Route.ENCRYPTED, Route.UNKNOWN):
        results = [ExtractedFile(
            file_path=rel_path,
            file_name=classified.path.name,
            file_size=classified.file_size,
            mime_type=classified.mime_type,
            file_extension=classified.path.suffix.lower(),
            text_content="",
            element_types=["Encrypted"] if route == Route.ENCRYPTED else ["Unknown"],
            extraction_method="none",
            extraction_status="failed",
            extraction_errors=[
                f"File classified as {route.value}: {classified.magic_description}"
            ],
            detected_mime=classified.mime_type,
            content_hash=_sha256(classified.path),
            is_encrypted=classified.is_encrypted,
        )]

    for r in results:
        if source_archive:
            r.source_archive = source_archive
            r.source_archive_path = source_archive_path
        r.is_encrypted = classified.is_encrypted

    return results


def extract_directory(
    root: Path,
    max_workers: int = 4,
    whisper_model: str = "base",
    quarantine_dir: Path | None = None,
) -> Generator[ExtractedFile, None, None]:
    """Extract all files from a directory tree.

    Classifies files, unpacks archives recursively, dispatches to handlers,
    deduplicates by SHA-256, and quarantines failures.
    """
    from .archive_handler import extract_archive_recursive
    from .problem_child import quarantine_file

    if quarantine_dir is None:
        quarantine_dir = root.parent / "problem_children"

    classified_files = classify_directory(root)
    logger.info("Classified %d files in %s", len(classified_files), root)

    # Separate archives from non-archives
    archives = [c for c in classified_files if c.route == Route.ARCHIVE]
    non_archives = [c for c in classified_files if c.route != Route.ARCHIVE]

    # Phase 1: Unpack archives recursively into a temp dir
    archive_entries: list[tuple[ClassifiedFile, str, str]] = []
    if archives:
        temp_root = Path(tempfile.mkdtemp(prefix="triage_archives_"))
        logger.info("Unpacking %d archives...", len(archives))
        for arch in archives:
            try:
                rel_archive = str(arch.path.relative_to(root))
            except ValueError:
                rel_archive = arch.path.name

            entries = extract_archive_recursive(arch.path, temp_root)
            for entry in entries:
                inner_classified = classify_file(entry.extracted_path)
                if inner_classified:
                    archive_entries.append((
                        inner_classified,
                        rel_archive,
                        entry.path_within_archive,
                    ))
        logger.info("Archives yielded %d inner files", len(archive_entries))

    # Phase 2: Process all files in parallel
    seen_hashes: set[str] = set()

    all_work: list[tuple[ClassifiedFile, str | None, str | None]] = []
    for c in non_archives:
        all_work.append((c, None, None))
    for c, arch_src, arch_path in archive_entries:
        all_work.append((c, arch_src, arch_path))

    total = len(all_work)
    logger.info("Processing %d files total (%d direct + %d from archives)",
                total, len(non_archives), len(archive_entries))

    with Progress() as progress:
        task = progress.add_task("Extracting files...", total=total)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _extract_single_classified, c, root, whisper_model, arch_src, arch_path,
                ): c
                for c, arch_src, arch_path in all_work
            }

            for future in as_completed(futures):
                progress.advance(task)
                classified = futures[future]
                try:
                    results = future.result()
                    for result in results:
                        # Dedup by content hash
                        if result.content_hash and result.content_hash in seen_hashes:
                            logger.debug("Skipping duplicate: %s", result.file_path)
                            continue
                        if result.content_hash:
                            seen_hashes.add(result.content_hash)

                        # Quarantine failures (encrypted, unknown, no text)
                        if (result.extraction_status == "failed"
                                or (not result.text_content.strip()
                                    and result.element_types != ["Image"])):
                            quarantine_file(
                                source_path=classified.path,
                                quarantine_dir=quarantine_dir,
                                root=root,
                                mime_type=classified.mime_type,
                                magic_description=classified.magic_description,
                                entropy=classified.entropy,
                                attempted_methods=[result.extraction_method],
                                error_messages=result.extraction_errors,
                            )
                            # Still yield the record so indexer can record it
                            yield result
                        else:
                            yield result

                except Exception as exc:
                    logger.error("Extraction error for %s: %s", classified.path, exc)
                    quarantine_file(
                        source_path=classified.path,
                        quarantine_dir=quarantine_dir,
                        root=root,
                        mime_type=classified.mime_type,
                        magic_description=classified.magic_description,
                        entropy=classified.entropy,
                        attempted_methods=["exception"],
                        error_messages=[str(exc)],
                    )
