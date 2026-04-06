"""Shared configuration, environment loading, and Elasticsearch client setup.

Uses .multilingual-e5-small for retrieval embeddings (deployed locally in the cluster
via semantic_text fields). Format-based entities (emails, financial refs) are regex-extracted
at index time; person/company identification is delegated to the LLM at query time.
"""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DRIVE_DIR = PROJECT_ROOT / "sample_drive"

# --- Elasticsearch indices ---
INDEX_FILE_METADATA = "triage-file-metadata"
INDEX_RETRIEVAL = "triage-retrieval"
INDEX_SAR_REPORTS = "triage-sar-reports"
INDEX_QUARANTINE = "triage-quarantine"

# --- Embedding model ---
# Uses the built-in .multilingual-e5-small model deployed locally in the cluster.
# On Elastic Cloud, you can switch to a cloud-hosted model (e.g. jina-embeddings-v5-text-nano)
# by changing these constants.
ELASTIC_INFERENCE_ENDPOINT_ID = "e5-small-retrieval"
ELASTIC_INFERENCE_MODEL = ".multilingual-e5-small_linux-x86_64"

# --- Regex-extracted entity fields (format-based, no predefined lists) ---
ENTITY_FIELDS = (
    "entities_financial_ref",
    "entities_email",
)

# --- Agent Builder ---
AGENT_NAME = "forensic-triage-assistant"

SKIP_EXTENSIONS = frozenset({
    ".dll", ".exe", ".sys", ".msi", ".cab", ".inf", ".drv",
    ".so", ".dylib", ".o", ".a", ".pyc", ".pyo", ".class",
    ".DS_Store", ".Thumbs.db",
})

SUPPORTED_IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif",
})


@dataclass
class Config:
    es_url: str = field(default_factory=lambda: os.getenv("ELASTICSEARCH_URL", "https://localhost:9200"))
    es_api_key: str = field(default_factory=lambda: os.getenv("ELASTICSEARCH_API_KEY", ""))
    es_cloud_id: str = field(default_factory=lambda: os.getenv("ELASTICSEARCH_CLOUD_ID", ""))
    es_username: str = field(default_factory=lambda: os.getenv("ELASTICSEARCH_USERNAME", ""))
    es_password: str = field(default_factory=lambda: os.getenv("ELASTICSEARCH_PASSWORD", ""))
    es_insecure: bool = field(default_factory=lambda: os.getenv("ELASTICSEARCH_INSECURE", "false").lower() == "true")
    kibana_url: str = field(default_factory=lambda: os.getenv("KIBANA_URL", "https://localhost:5601"))

    # --- Audio/Video transcription ---
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))

    # --- Problem Child LLM resolution ---
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))

    def get_es_client(self) -> Elasticsearch:
        kwargs: dict = {}

        if self.es_cloud_id:
            kwargs["cloud_id"] = self.es_cloud_id
        else:
            kwargs["hosts"] = [self.es_url]

        if self.es_api_key:
            kwargs["api_key"] = self.es_api_key
        elif self.es_username and self.es_password:
            kwargs["basic_auth"] = (self.es_username, self.es_password)

        if self.es_insecure and self.es_url.startswith("https"):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            kwargs["ssl_context"] = ctx
            kwargs["verify_certs"] = False

        return Elasticsearch(**kwargs)


def get_config() -> Config:
    return Config()
