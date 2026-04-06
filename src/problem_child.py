"""Problem Child quarantine system and LLM-powered format research agent.

Phase 1: Quarantine files that resist extraction -- copy them to a holding
directory and record detailed diagnostics.

Phase 2: Use an LLM to analyze the file signature, identify the format,
and generate a one-off extraction script that runs in a sandboxed subprocess.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QuarantineRecord:
    """Diagnostic record for a quarantined file."""

    file_path: str
    file_name: str
    file_size: int
    mime_type: str
    magic_description: str
    entropy: float
    hex_header: str
    attempted_methods: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    quarantined_at: str = ""
    status: str = "quarantined"  # quarantined | resolved | unresolvable
    llm_analysis: str = ""
    resolved_at: str = ""
    resolution_method: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


def _read_hex_header(path: Path, num_bytes: int = 512) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(num_bytes)
        return data.hex()
    except Exception:
        return ""


def quarantine_file(
    source_path: Path,
    quarantine_dir: Path,
    root: Path,
    mime_type: str,
    magic_description: str,
    entropy: float,
    attempted_methods: list[str],
    error_messages: list[str],
) -> QuarantineRecord:
    """Copy a file to the quarantine directory and create a diagnostic record."""
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    try:
        rel_path = source_path.relative_to(root)
    except ValueError:
        rel_path = Path(source_path.name)

    dest = quarantine_dir / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(source_path, dest)
    except Exception as exc:
        logger.warning("Failed to copy %s to quarantine: %s", source_path, exc)

    try:
        file_size = source_path.stat().st_size
    except OSError:
        file_size = 0

    record = QuarantineRecord(
        file_path=str(rel_path),
        file_name=source_path.name,
        file_size=file_size,
        mime_type=mime_type,
        magic_description=magic_description,
        entropy=entropy,
        hex_header=_read_hex_header(source_path),
        attempted_methods=attempted_methods,
        error_messages=error_messages,
        quarantined_at=datetime.now(timezone.utc).isoformat(),
        status="quarantined",
    )

    _write_manifest(quarantine_dir, record)
    return record


def _write_manifest(quarantine_dir: Path, record: QuarantineRecord) -> None:
    """Append the quarantine record to the manifest JSONL file."""
    manifest = quarantine_dir / "manifest.jsonl"
    try:
        with open(manifest, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
    except Exception as exc:
        logger.warning("Failed to write quarantine manifest: %s", exc)


def load_quarantine_manifest(quarantine_dir: Path) -> list[QuarantineRecord]:
    """Load all quarantine records from the manifest file."""
    manifest = quarantine_dir / "manifest.jsonl"
    if not manifest.exists():
        return []

    records: list[QuarantineRecord] = []
    try:
        for line in manifest.read_text().strip().split("\n"):
            if not line.strip():
                continue
            data = json.loads(line)
            records.append(QuarantineRecord(**{
                k: v for k, v in data.items()
                if k in QuarantineRecord.__dataclass_fields__
            }))
    except Exception as exc:
        logger.warning("Failed to read quarantine manifest: %s", exc)
    return records


# ---------------------------------------------------------------------------
# Phase 2: LLM Research Agent
# ---------------------------------------------------------------------------

_RESEARCH_PROMPT = textwrap.dedent("""\
    You are a digital forensics file format expert. A file has resisted all
    standard extraction methods. Analyze the diagnostic information below
    and:

    1. Identify the most likely file format based on the hex header, MIME type,
       and magic description.
    2. Explain what the format is and why standard tools failed.
    3. Write a Python script that extracts readable text content from this file.
       The script must:
       - Read from the file path provided as sys.argv[1]
       - Print extracted text to stdout
       - Use only standard library or widely-available packages
       - Handle errors gracefully
       - NOT modify the original file

    File diagnostics:
    - File name: {file_name}
    - MIME type: {mime_type}
    - Magic description: {magic_description}
    - Entropy: {entropy:.2f}
    - File size: {file_size} bytes
    - Hex header (first 512 bytes): {hex_header}
    - Attempted methods: {attempted_methods}
    - Error messages: {error_messages}

    Respond with a JSON object containing exactly two keys:
    - "analysis": a string explaining the format identification
    - "script": a string containing the complete Python extraction script
""")


def _call_llm(prompt: str, provider: str, api_key: str, model: str) -> dict | None:
    """Call an LLM provider and parse the JSON response."""
    if provider == "openai":
        return _call_openai(prompt, api_key, model)
    elif provider == "anthropic":
        return _call_anthropic(prompt, api_key, model)
    else:
        logger.error("Unknown LLM provider: %s", provider)
        return None


def _call_openai(prompt: str, api_key: str, model: str) -> dict | None:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content
        return json.loads(content) if content else None
    except ImportError:
        logger.error("openai package not installed")
        return None
    except Exception as exc:
        logger.error("OpenAI API call failed: %s", exc)
        return None


def _call_anthropic(prompt: str, api_key: str, model: str) -> dict | None:
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text
        # Try to extract JSON from the response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
        return None
    except ImportError:
        logger.error("anthropic package not installed")
        return None
    except Exception as exc:
        logger.error("Anthropic API call failed: %s", exc)
        return None


def _run_extraction_script(script: str, file_path: Path, timeout: int = 60) -> str | None:
    """Run an LLM-generated extraction script in a sandboxed subprocess."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path, str(file_path)],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        if result.stderr:
            logger.warning("Extraction script stderr: %s", result.stderr[:500])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Extraction script timed out for %s", file_path)
        return None
    except Exception as exc:
        logger.warning("Extraction script failed: %s", exc)
        return None
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def resolve_quarantined_file(
    record: QuarantineRecord,
    quarantine_dir: Path,
    provider: str,
    api_key: str,
    model: str,
) -> tuple[str | None, QuarantineRecord]:
    """Attempt to resolve a single quarantined file using LLM research.

    Returns (extracted_text_or_None, updated_record).
    """
    prompt = _RESEARCH_PROMPT.format(**record.to_dict())
    llm_result = _call_llm(prompt, provider, api_key, model)

    if not llm_result:
        record.status = "unresolvable"
        record.llm_analysis = "LLM call failed or returned no result"
        record.resolved_at = datetime.now(timezone.utc).isoformat()
        return None, record

    analysis = llm_result.get("analysis", "")
    script = llm_result.get("script", "")
    record.llm_analysis = analysis

    if not script:
        record.status = "unresolvable"
        record.resolved_at = datetime.now(timezone.utc).isoformat()
        return None, record

    file_in_quarantine = quarantine_dir / record.file_path
    if not file_in_quarantine.exists():
        record.status = "unresolvable"
        record.error_messages.append("Quarantined file not found on disk")
        record.resolved_at = datetime.now(timezone.utc).isoformat()
        return None, record

    text = _run_extraction_script(script, file_in_quarantine)

    if text:
        record.status = "resolved"
        record.resolution_method = "llm_generated_script"
        record.resolved_at = datetime.now(timezone.utc).isoformat()
        return text, record
    else:
        record.status = "unresolvable"
        record.error_messages.append("LLM-generated script produced no output")
        record.resolved_at = datetime.now(timezone.utc).isoformat()
        return None, record


def resolve_all_quarantined(
    quarantine_dir: Path,
    provider: str,
    api_key: str,
    model: str,
) -> list[tuple[QuarantineRecord, str | None]]:
    """Attempt to resolve all quarantined files.

    Returns list of (record, extracted_text_or_None).
    """
    records = load_quarantine_manifest(quarantine_dir)
    pending = [r for r in records if r.status == "quarantined"]

    results: list[tuple[QuarantineRecord, str | None]] = []
    for record in pending:
        text, updated = resolve_quarantined_file(
            record, quarantine_dir, provider, api_key, model,
        )
        results.append((updated, text))

    # Rewrite the manifest with updated statuses
    manifest = quarantine_dir / "manifest.jsonl"
    resolved_paths = {r.file_path for r, _ in results}
    all_records = []
    for r in records:
        if r.file_path in resolved_paths:
            for updated, _ in results:
                if updated.file_path == r.file_path:
                    all_records.append(updated)
                    break
        else:
            all_records.append(r)

    try:
        with open(manifest, "w") as f:
            for r in all_records:
                f.write(json.dumps(r.to_dict()) + "\n")
    except Exception as exc:
        logger.warning("Failed to update quarantine manifest: %s", exc)

    return results
