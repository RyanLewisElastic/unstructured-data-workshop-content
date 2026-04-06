"""File classification using libmagic, entropy analysis, and encryption detection.

Classifies every file into a processing route before extraction begins,
enabling the pipeline to dispatch to the correct handler and detect
encrypted or proprietary formats early.
"""

from __future__ import annotations

import logging
import math
import os
import zipfile
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class Route(str, Enum):
    UNSTRUCTURED = "unstructured"
    IMAGE = "image"
    AUDIO_VIDEO = "audio_video"
    ARCHIVE = "archive"
    DATABASE = "database"
    ENCRYPTED = "encrypted"
    UNKNOWN = "unknown"


_IMAGE_MIMES = frozenset({
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "image/bmp", "image/tiff", "image/heic", "image/heif",
})

_AV_MIMES = frozenset({
    "audio/mpeg", "audio/mp4", "audio/x-wav", "audio/wav", "audio/flac",
    "audio/ogg", "audio/x-m4a", "audio/aac", "audio/x-aiff",
    "video/mp4", "video/x-msvideo", "video/x-matroska", "video/quicktime",
    "video/webm", "video/mpeg", "video/3gpp",
})

_ARCHIVE_MIMES = frozenset({
    "application/zip", "application/x-tar", "application/gzip",
    "application/x-bzip2", "application/x-7z-compressed",
    "application/x-rar-compressed", "application/x-rar",
    "application/x-iso9660-image", "application/x-apple-diskimage",
})

_ARCHIVE_EXTENSIONS = frozenset({
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".tbz2", ".xz", ".txz",
    ".7z", ".rar", ".iso", ".dmg", ".pst", ".ost", ".mbox",
})

_DATABASE_MIMES = frozenset({
    "application/x-sqlite3", "application/vnd.sqlite3",
})

_DATABASE_EXTENSIONS = frozenset({
    ".sqlite", ".sqlite3", ".db", ".sdb",
    ".mdb", ".accdb",
})

_AV_EXTENSIONS = frozenset({
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".aiff", ".wma",
    ".mp4", ".avi", ".mkv", ".mov", ".webm", ".mpeg", ".mpg", ".3gp", ".wmv",
})

ENTROPY_ENCRYPTED_THRESHOLD = 7.5
ENTROPY_SAMPLE_SIZE = 65536


@dataclass
class ClassifiedFile:
    """Result of classifying a single file."""

    path: Path
    mime_type: str
    magic_description: str
    route: Route
    is_encrypted: bool
    entropy: float
    file_size: int


def _detect_mime(path: Path) -> tuple[str, str]:
    """Return (mime_type, description) via python-magic, with fallback."""
    try:
        import magic

        mime = magic.from_file(str(path), mime=True) or "application/octet-stream"
        desc = magic.from_file(str(path)) or ""
        return mime, desc
    except ImportError:
        logger.debug("python-magic not installed; falling back to extension-based detection")
    except Exception as exc:
        logger.debug("libmagic failed for %s: %s", path, exc)

    ext = path.suffix.lower()
    _ext_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".doc": "application/msword",
        ".xls": "application/vnd.ms-excel",
        ".ppt": "application/vnd.ms-powerpoint",
        ".html": "text/html", ".htm": "text/html",
        ".csv": "text/csv", ".tsv": "text/tab-separated-values",
        ".txt": "text/plain", ".log": "text/plain", ".md": "text/markdown",
        ".eml": "message/rfc822", ".msg": "application/vnd.ms-outlook",
        ".json": "application/json", ".xml": "application/xml",
        ".zip": "application/zip", ".tar": "application/x-tar",
        ".gz": "application/gzip", ".7z": "application/x-7z-compressed",
        ".rar": "application/x-rar-compressed",
        ".mp3": "audio/mpeg", ".wav": "audio/x-wav",
        ".mp4": "video/mp4", ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska", ".mov": "video/quicktime",
        ".sqlite": "application/x-sqlite3", ".db": "application/x-sqlite3",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    return _ext_map.get(ext, "application/octet-stream"), f"file extension: {ext}"


def _shannon_entropy(path: Path, sample_size: int = ENTROPY_SAMPLE_SIZE) -> float:
    """Compute Shannon entropy of the first `sample_size` bytes."""
    try:
        with open(path, "rb") as f:
            data = f.read(sample_size)
    except OSError:
        return 0.0

    if not data:
        return 0.0

    counts = Counter(data)
    length = len(data)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _is_encrypted_pdf(path: Path) -> bool:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return reader.is_encrypted
    except Exception:
        return False


def _is_encrypted_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.flag_bits & 0x1:
                    return True
        return False
    except Exception:
        return False


def _is_encrypted_ole(path: Path) -> bool:
    """Detect Microsoft OLE2 encrypted documents (.doc, .xls, .ppt)."""
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        return header[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" and _check_ole_encryption(path)
    except Exception:
        return False


def _check_ole_encryption(path: Path) -> bool:
    """Heuristic: check if OLE2 file has EncryptedPackage stream name."""
    try:
        with open(path, "rb") as f:
            content = f.read(4096)
        return b"E\x00n\x00c\x00r\x00y\x00p\x00t\x00e\x00d" in content
    except Exception:
        return False


def _is_encrypted_7z(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            sig = f.read(6)
        if sig != b"7z\xbc\xaf\x27\x1c":
            return False
        import py7zr

        with py7zr.SevenZipFile(path, "r") as z:
            return z.needs_password()
    except ImportError:
        return False
    except Exception:
        return False


def _detect_encryption(path: Path, mime: str, entropy: float) -> bool:
    ext = path.suffix.lower()

    if mime == "application/pdf" or ext == ".pdf":
        if _is_encrypted_pdf(path):
            return True

    if mime == "application/zip" or ext == ".zip":
        if _is_encrypted_zip(path):
            return True

    if ext == ".7z" or mime == "application/x-7z-compressed":
        if _is_encrypted_7z(path):
            return True

    if ext in {".doc", ".xls", ".ppt"} or mime in {
        "application/msword", "application/vnd.ms-excel", "application/vnd.ms-powerpoint",
    }:
        if _is_encrypted_ole(path):
            return True

    if entropy >= ENTROPY_ENCRYPTED_THRESHOLD:
        if ext not in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
                       ".mp3", ".mp4", ".avi", ".mkv", ".mov", ".flac",
                       ".ogg", ".wav", ".zip", ".gz", ".bz2", ".xz",
                       ".7z", ".rar", ".tar"}:
            return True

    return False


def _determine_route(mime: str, ext: str, is_encrypted: bool) -> Route:
    if is_encrypted:
        return Route.ENCRYPTED

    if mime in _IMAGE_MIMES or ext in {".jpg", ".jpeg", ".png", ".gif",
                                         ".webp", ".bmp", ".tiff", ".tif",
                                         ".heic", ".heif"}:
        return Route.IMAGE

    if mime in _AV_MIMES or ext in _AV_EXTENSIONS:
        return Route.AUDIO_VIDEO

    if mime in _ARCHIVE_MIMES or ext in _ARCHIVE_EXTENSIONS:
        return Route.ARCHIVE

    if mime in _DATABASE_MIMES or ext in _DATABASE_EXTENSIONS:
        return Route.DATABASE

    # Anything else goes to Unstructured (which handles its own fallback)
    return Route.UNSTRUCTURED


def classify_file(path: Path) -> ClassifiedFile | None:
    """Classify a single file and determine its processing route."""
    try:
        stat = path.stat()
    except OSError:
        return None

    if stat.st_size == 0:
        return None

    mime, desc = _detect_mime(path)
    entropy = _shannon_entropy(path)
    ext = path.suffix.lower()
    is_encrypted = _detect_encryption(path, mime, entropy)
    route = _determine_route(mime, ext, is_encrypted)

    return ClassifiedFile(
        path=path,
        mime_type=mime,
        magic_description=desc,
        route=route,
        is_encrypted=is_encrypted,
        entropy=entropy,
        file_size=stat.st_size,
    )


def classify_directory(root: Path) -> list[ClassifiedFile]:
    """Walk a directory tree and classify every file."""
    from .config import SKIP_EXTENSIONS

    classified: list[ClassifiedFile] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.startswith("."):
                continue
            fp = Path(dirpath) / fname
            if fp.suffix.lower() in SKIP_EXTENSIONS:
                continue
            result = classify_file(fp)
            if result is not None:
                classified.append(result)

    return sorted(classified, key=lambda c: str(c.path))
