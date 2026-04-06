"""Audio/video metadata extraction and speech-to-text transcription.

Uses ffprobe for media metadata and OpenAI Whisper for transcription.
Both are optional: ffprobe requires ffmpeg installed, Whisper requires
the openai-whisper package (or can be skipped entirely).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AVMetadata:
    """Metadata extracted from an audio or video file."""

    duration_seconds: float | None = None
    codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bit_rate: int | None = None
    width: int | None = None
    height: int | None = None
    format_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class AVResult:
    """Result of processing an audio/video file."""

    transcript: str
    metadata: AVMetadata
    language_detected: str | None = None
    errors: list[str] = field(default_factory=list)


def probe_media(path: Path) -> AVMetadata:
    """Extract metadata from a media file using ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        logger.debug("ffprobe not found; skipping media probe for %s", path)
        return AVMetadata()

    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning("ffprobe failed for %s: %s", path, result.stderr[:200])
            return AVMetadata()

        data = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe timed out for %s", path)
        return AVMetadata()
    except Exception as exc:
        logger.warning("ffprobe error for %s: %s", path, exc)
        return AVMetadata()

    meta = AVMetadata()

    fmt = data.get("format", {})
    meta.duration_seconds = _safe_float(fmt.get("duration"))
    meta.bit_rate = _safe_int(fmt.get("bit_rate"))
    meta.format_name = fmt.get("format_name")
    meta.tags = {k.lower(): v for k, v in fmt.get("tags", {}).items()}

    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type", "")
        if codec_type == "audio" and meta.codec is None:
            meta.codec = stream.get("codec_name")
            meta.sample_rate = _safe_int(stream.get("sample_rate"))
            meta.channels = _safe_int(stream.get("channels"))
        elif codec_type == "video" and meta.width is None:
            meta.codec = stream.get("codec_name")
            meta.width = _safe_int(stream.get("width"))
            meta.height = _safe_int(stream.get("height"))

    return meta


def transcribe_whisper(path: Path, model_name: str = "base") -> tuple[str, str | None]:
    """Transcribe audio/video using OpenAI Whisper.

    Returns (transcript_text, detected_language).
    """
    try:
        import whisper
    except ImportError:
        logger.info("openai-whisper not installed; skipping transcription for %s", path)
        return "", None

    try:
        model = _get_whisper_model(model_name)
        result = model.transcribe(str(path), verbose=False)
        text = result.get("text", "").strip()
        lang = result.get("language")
        return text, lang
    except Exception as exc:
        logger.warning("Whisper transcription failed for %s: %s", path, exc)
        return "", None


_whisper_model_cache: dict[str, object] = {}


def _get_whisper_model(name: str) -> object:
    """Cache loaded Whisper models to avoid reloading per file."""
    if name not in _whisper_model_cache:
        import whisper

        logger.info("Loading Whisper model '%s'...", name)
        _whisper_model_cache[name] = whisper.load_model(name)
    return _whisper_model_cache[name]


def process_av_file(
    path: Path,
    whisper_model: str = "base",
    skip_transcription: bool = False,
) -> AVResult:
    """Full processing pipeline for an audio/video file."""
    errors: list[str] = []

    metadata = probe_media(path)

    transcript = ""
    lang = None

    if not skip_transcription:
        try:
            transcript, lang = transcribe_whisper(path, whisper_model)
        except Exception as exc:
            errors.append(f"Transcription failed: {exc}")

    return AVResult(
        transcript=transcript,
        metadata=metadata,
        language_detected=lang,
        errors=errors,
    )


def _safe_float(val: str | None) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: str | int | None) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None
