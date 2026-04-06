# Unstructured data workshop content

Public, **standalone** package used by the [Instruqt](https://instruqt.com) lab
**Unstructured Data Triage with Elasticsearch**. It provides the `triage` CLI:

- `triage generate-sample --scenario workshop`
- `triage ingest sample_drive/ --clean`
- `triage setup-agent`
- `triage sar "<entity name>"`
- `triage status`

This repo is a **subset** of a larger internal project: only what the workshop
needs is published here so sandboxes can `git clone` without credentials.

## Disclaimer

This lab uses **fictional, computer-generated sample data** for training. Names
and entities are **not real**; any resemblance to actual people or organizations
is **coincidental**.

## Requirements

- Python 3.11+
- Node.js 18+ (`node` on `PATH`) for `triage setup-agent` (Kibana Agent Builder API)
- System packages commonly preinstalled on workshop VMs: `poppler-utils`,
  `tesseract-ocr`, `libmagic1`, `git`, `curl`, `jq`

## Install

```bash
git clone https://github.com/RyanLewisElastic/unstructured-data-workshop-content.git
cd unstructured-data-workshop-content
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configure

Create a `.env` file (or export variables). Example for Instruqt-style sandboxes:

```bash
ELASTICSEARCH_URL=http://kubernetes-vm:30920
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=...
# Or API key (same key works for ES and Kibana in typical setups):
ELASTICSEARCH_API_KEY=...
KIBANA_API_KEY=...   # optional mirror of ES API key for Node agent-builder scripts
KIBANA_URL=http://kubernetes-vm:30002
ELASTICSEARCH_INSECURE=true
INVESTIGATION_CONTEXT=workshop
```

If you only set `ELASTICSEARCH_API_KEY`, ensure `KIBANA_API_KEY` is set to the
same value when running `triage setup-agent` (the bundled Node scripts read
`KIBANA_API_KEY`).

## Instruqt (Elastic managed VM)

- **Install location:** copy `instruqt/setup-host-1.example` to your track as
  `01-ingest-the-data/setup-host-1`. By default it clones this repo into
  **`/workspace/workshop`** and runs `triage generate-sample --scenario workshop`
  there (same path participants land in on many Elastic presets).
- **Check scripts:** use `instruqt/check-host-1.example` paths (`WORKSHOP_ROOT`).
- **Agent Builder LLM:** replacing workshop challenges does **not** configure
  Kibana’s default AI connector. See **`instruqt/LLM_CONNECTOR.md`** and challenge
  2 in the lab guide (GenAI Settings + OpenAI-compatible connector for LiteLLM).

Override clone target with env `WORKSHOP_REPO_DIR` or repo URL with `WORKSHOP_REPO_URL`.

## Layout

| Path | Purpose |
|------|---------|
| `src/` | Python package (`triage` CLI) |
| `workflows/sar-report.yaml` | Elastic Workflow deployed by `setup-agent` |
| `.agents/skills/kibana-agent-builder/scripts/` | Node helpers for Agent Builder API |
| `instruqt/` | `setup-host-1.example`, check/solve examples, `LLM_CONNECTOR.md` |

## License

Add a `LICENSE` file per your organization’s open-source policy.
