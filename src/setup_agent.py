"""Set up the Elastic Agent Builder agent for forensic triage.

Creates custom tools (index_search + ES|QL) and a forensic-triage-assistant agent
using the kibana-agent-builder skill scripts. Delegates all Kibana API interaction
to the existing Node.js scripts. Also deploys the SAR workflow.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import httpx
from rich.console import Console

from .config import (
    AGENT_NAME,
    INDEX_FILE_METADATA,
    INDEX_RETRIEVAL,
    INDEX_SAR_REPORTS,
    Config,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)
console = Console()

# Resolve the skill script -- prefer project-local, fall back to user-level
_PROJECT_SKILL = PROJECT_ROOT / ".agents" / "skills" / "kibana-agent-builder" / "scripts" / "agent-builder.js"
_USER_SKILL = Path.home() / ".agents" / "skills" / "kibana-agent-builder" / "scripts" / "agent-builder.js"
SKILL_SCRIPT = _PROJECT_SKILL if _PROJECT_SKILL.exists() else _USER_SKILL

RETRIEVAL_SEARCH_TOOL_ID = "triage-content-search"
METADATA_ESQL_TOOL_ID = "triage-file-query"
FILE_LIST_ESQL_TOOL_ID = "triage-file-list"
SAR_WORKFLOW_FILE = PROJECT_ROOT / "workflows" / "sar-report.yaml"
_SAR_WORKFLOW_ID_FILE = PROJECT_ROOT / ".sar_workflow_id"


def _get_sar_workflow_id() -> str | None:
    """Read the deployed SAR workflow ID from the state file."""
    if _SAR_WORKFLOW_ID_FILE.exists():
        return _SAR_WORKFLOW_ID_FILE.read_text().strip()
    return None

_INVESTIGATION_CONTEXTS = {
    "default": (
        "This data comes from a seized device in a potential money laundering investigation. "
        "The data source contains financial records, communications in multiple languages, "
        "business documents, media files, and digital artifacts mixed with mundane personal "
        "and work content."
    ),
    "workshop": (
        "This data comes from a seized device in a potential sanctions evasion and "
        "trade fraud investigation. The data source contains shipping documents, financial "
        "records, communications in multiple languages, business contracts, media files, "
        "and digital artifacts mixed with mundane personal and work content."
    ),
}


def _get_system_prompt() -> str:
    """Build the agent system prompt, adapting context based on INVESTIGATION_CONTEXT env var."""
    context_key = os.getenv("INVESTIGATION_CONTEXT", "default")
    context = _INVESTIGATION_CONTEXTS.get(context_key, _INVESTIGATION_CONTEXTS["default"])

    return (
        "You are a forensic data analyst assistant helping investigate a seized data source "
        "(e.g. computer hard drive). Your role is to help law enforcement triage the contents "
        "of this data source.\n\n"
        "RULES:\n"
        "- Always use tools to retrieve data. Never answer data questions from memory.\n"
        "- Cite specific file paths when referencing evidence.\n"
        "- Flag potentially suspicious content (unusual financial patterns, coded language, "
        "offshore references).\n"
        "- When asked about file types or counts, use triage-file-query or triage-file-list.\n"
        "- When asked about file content or searching for specific topics, use triage-content-search.\n"
        "- If a question is ambiguous, ask for clarification before querying.\n"
        "- Present findings in a structured format suitable for investigative reports.\n\n"
        "ENTITY RESOLUTION:\n"
        "Documents contain fragmented identity information -- first names, last names, email "
        "handles, and localized names in multiple languages. No person, company, or operation "
        "names have been pre-extracted. You must search broadly, cross-reference across "
        "documents, and synthesize your own understanding of who's who. Email addresses and "
        "financial reference IDs are the only structured entity fields available.\n\n"
        "PRE-EXTRACTED FIELDS (keyword, exact-queryable):\n"
        "- entities_email: email addresses found in document text\n"
        "- entities_financial_ref: transaction IDs, IBANs, Bitcoin addresses\n"
        "- email_from, email_to: sender/recipient from .eml metadata\n"
        "- email_subject: email subject lines\n\n"
        "SAR WORKFLOW:\n"
        "When asked to generate a Suspicious Activity Report (SAR) for a person or company, "
        "invoke the triage-sar-report workflow with the entity name. This workflow uses full-text "
        "search across all indexed documents to find mentions of the entity, then formats a "
        "FinCEN-aligned SAR and indexes it to triage-sar-reports. No AI is involved in the SAR "
        "generation; every field comes from indexed evidence.\n\n"
        "AVAILABLE DATA:\n"
        "- triage-retrieval: Full-text content with semantic embeddings (use for meaning-based search)\n"
        "- triage-file-metadata: File metadata, extracted text, email metadata, and the "
        "pre-extracted keyword fields listed above\n"
        "- triage-sar-reports: SAR reports generated by the workflow\n\n"
        "CONTEXT:\n"
        f"{context}"
    )


def _run_script(args: list[str], env_vars: dict[str, str], timeout: int = 60) -> tuple[bool, str]:
    """Run the agent-builder.js script with given arguments."""
    if not SKILL_SCRIPT.exists():
        return False, (
            f"Skill script not found. Install the kibana-agent-builder skill:\n"
            f"  npx skills add elastic/agent-skills\n"
            f"Checked: {_PROJECT_SKILL}\n"
            f"         {_USER_SKILL}"
        )

    cmd = ["node", str(SKILL_SCRIPT)] + args

    try:
        merged_env = {**os.environ, **env_vars}
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=merged_env,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as exc:
        return False, str(exc)


def _get_env_vars(config: Config) -> dict[str, str]:
    """Build environment variables for the Kibana skill scripts."""
    env = {"KIBANA_URL": config.kibana_url}
    if config.es_api_key:
        env["KIBANA_API_KEY"] = config.es_api_key
    if config.es_username:
        env["KIBANA_USERNAME"] = config.es_username
    if config.es_password:
        env["KIBANA_PASSWORD"] = config.es_password
    if config.es_insecure:
        env["KIBANA_INSECURE"] = "true"
    return env


def _delete_tool_if_exists(tool_id: str, env: dict[str, str]) -> None:
    """Silently delete a tool if it already exists (for idempotent re-creation)."""
    _run_script(["delete-tool", "--id", tool_id], env, timeout=30)


def create_tools(config: Config) -> list[str]:
    """Create custom Agent Builder tools for forensic triage.

    Returns the list of tool IDs that were successfully created.
    """
    env = _get_env_vars(config)
    created_tools: list[str] = []

    # 1. Index search tool for semantic retrieval -- primary tool for entity discovery
    console.print(f"Creating index search tool: [cyan]{RETRIEVAL_SEARCH_TOOL_ID}[/cyan]")
    _delete_tool_if_exists(RETRIEVAL_SEARCH_TOOL_ID, env)
    ok, output = _run_script([
        "create-tool",
        "--id", RETRIEVAL_SEARCH_TOOL_ID,
        "--type", "index_search",
        "--description",
        "Searches the full-text content of all files extracted from the seized data source "
        "using semantic (meaning-based) search. This is the PRIMARY tool for discovering "
        "person names, company names, and relationships -- these are NOT pre-extracted as "
        "keyword fields. Search broadly for name fragments, aliases, and related concepts.",
        "--pattern", INDEX_RETRIEVAL,
    ], env)
    if ok:
        console.print(f"  [green]Created {RETRIEVAL_SEARCH_TOOL_ID}[/green]")
        created_tools.append(RETRIEVAL_SEARCH_TOOL_ID)
    else:
        console.print(f"  [red]Failed: {output[:300]}[/red]")

    # 2. ES|QL tool for structured metadata queries (parameterized)
    esql_query = (
        f"FROM {INDEX_FILE_METADATA} "
        "| WHERE file_extension == ?extension "
        "| STATS count = COUNT(*), total_size = SUM(file_size) BY file_extension, mime_type "
        "| SORT count DESC "
        "| KEEP file_extension, mime_type, count, total_size "
        "| LIMIT 50"
    )
    params_json = json.dumps({
        "extension": {
            "type": "keyword",
            "description": "File extension to filter by, e.g. .csv, .pdf, .eml",
        },
    })
    console.print(f"Creating ES|QL tool: [cyan]{METADATA_ESQL_TOOL_ID}[/cyan]")
    _delete_tool_if_exists(METADATA_ESQL_TOOL_ID, env)
    ok, output = _run_script([
        "create-tool",
        "--id", METADATA_ESQL_TOOL_ID,
        "--type", "esql",
        "--description",
        "Queries file metadata from the seized data source by file extension. Use to find "
        "files by type, count files, calculate total sizes. Pass a file extension like "
        ".csv, .pdf, .eml, .jpg etc. Note: person and company names are NOT available as "
        "keyword fields -- use triage-content-search or platform.core.execute_esql with "
        "MATCH(text_content, ...) to search for names in document text.",
        "--query", esql_query,
        "--params", params_json,
    ], env)
    if ok:
        console.print(f"  [green]Created {METADATA_ESQL_TOOL_ID}[/green]")
        created_tools.append(METADATA_ESQL_TOOL_ID)
    else:
        console.print(f"  [red]Failed: {output[:300]}[/red]")

    # 3. ES|QL tool for listing all file types (no parameters)
    list_query = (
        f"FROM {INDEX_FILE_METADATA} "
        "| STATS count = COUNT(*), total_bytes = SUM(file_size) BY file_extension "
        "| EVAL total_mb = ROUND(total_bytes / 1048576.0, 2) "
        "| SORT count DESC "
        "| KEEP file_extension, count, total_mb "
        "| LIMIT 50"
    )
    console.print(f"Creating ES|QL tool: [cyan]{FILE_LIST_ESQL_TOOL_ID}[/cyan]")
    _delete_tool_if_exists(FILE_LIST_ESQL_TOOL_ID, env)
    ok, output = _run_script([
        "create-tool",
        "--id", FILE_LIST_ESQL_TOOL_ID,
        "--type", "esql",
        "--description",
        "Lists all file types found on the seized data source with counts and total sizes "
        "in MB. Use when the user asks what types of files are on this drive or wants a "
        "summary of the data source contents. For searching by person/company name, use "
        "triage-content-search instead. No parameters needed.",
        "--query", list_query,
        "--params", "{}",
    ], env)
    if ok:
        console.print(f"  [green]Created {FILE_LIST_ESQL_TOOL_ID}[/green]")
        created_tools.append(FILE_LIST_ESQL_TOOL_ID)
    else:
        console.print(f"  [red]Failed: {output[:300]}[/red]")

    return created_tools


def create_agent(config: Config, custom_tool_ids: list[str]) -> bool:
    """Create the forensic triage assistant agent."""
    env = _get_env_vars(config)

    # List available tools first
    console.print("\nListing available tools...")
    ok, output = _run_script(["list-tools"], env)
    if ok:
        console.print("  [green]Available tools listed successfully[/green]")
        logger.debug("Available tools:\n%s", output)
    else:
        console.print(f"  [yellow]Could not list tools: {output[:200]}[/yellow]")

    # Build tool list from successfully created custom tools + built-in platform tools
    tool_ids = [
        *custom_tool_ids,
        "platform.core.search",
        "platform.core.execute_esql",
        "platform.core.list_indices",
        "platform.core.get_index_mapping",
    ]

    # Delete existing agent if present (for idempotent re-creation)
    agent_id = AGENT_NAME.lower().replace(" ", "-")
    _run_script(["delete-agent", "--id", agent_id], env, timeout=30)

    console.print(f"\nCreating agent: [cyan]{AGENT_NAME}[/cyan]")
    console.print(f"  Tools: {', '.join(tool_ids)}")

    ok, output = _run_script([
        "create-agent",
        "--name", AGENT_NAME,
        "--description", "Forensic data analyst for triaging seized data sources in investigations",
        "--instructions", _get_system_prompt(),
        "--tool-ids", ",".join(tool_ids),
    ], env, timeout=120)

    if ok:
        console.print(f"  [green]Agent '{AGENT_NAME}' created successfully![/green]")
    else:
        console.print(f"  [red]Agent creation failed: {output[:500]}[/red]")

    # Verify
    console.print("\nVerifying agent creation...")
    ok, output = _run_script(["list-agents"], env)
    if ok:
        if agent_id in output.lower():
            console.print(f"  [green]Agent verified in agent list[/green]")
        else:
            console.print(f"  [yellow]Agent not found in list -- may need manual verification[/yellow]")
    console.print(output[:500] if output else "")

    return ok


def chat_with_agent(config: Config, message: str, conversation_id: str | None = None) -> str:
    """Send a message to the forensic triage agent and return the response."""
    env = _get_env_vars(config)

    agent_id = AGENT_NAME.lower().replace(" ", "-")
    args = ["chat", "--id", agent_id, "--message", message]
    if conversation_id:
        args.extend(["--conversation-id", conversation_id])

    ok, output = _run_script(args, env, timeout=180)
    return output


def _kibana_headers(config: Config) -> dict[str, str]:
    """Build standard Kibana API headers."""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "kbn-xsrf": "true",
        "x-elastic-internal-origin": "Kibana",
    }
    if config.es_api_key:
        headers["Authorization"] = f"ApiKey {config.es_api_key}"
    return headers


def _kibana_auth(config: Config) -> tuple[str, str] | None:
    """Build basic-auth tuple when API key is not set."""
    if config.es_username and config.es_password and not config.es_api_key:
        return (config.es_username, config.es_password)
    return None


def deploy_workflow(config: Config) -> bool:
    """Deploy the SAR workflow YAML to Kibana.

    Reads workflows/sar-report.yaml and POSTs it to the Kibana Workflows API
    as JSON-wrapped YAML per the Elastic Workflows spec.
    Returns True if deployment succeeded.
    """
    if not SAR_WORKFLOW_FILE.exists():
        console.print(f"  [red]Workflow file not found: {SAR_WORKFLOW_FILE}[/red]")
        return False

    yaml_content = SAR_WORKFLOW_FILE.read_text(encoding="utf-8")
    headers = _kibana_headers(config)
    auth = _kibana_auth(config)
    base = config.kibana_url.rstrip("/")

    try:
        with httpx.Client(verify=not config.es_insecure, timeout=30) as client:
            resp = client.post(
                f"{base}/api/workflows",
                json={"yaml": yaml_content},
                headers=headers,
                auth=auth,
            )

            if resp.status_code in (200, 201):
                data = resp.json()
                wf_id = data.get("id", "")
                if wf_id:
                    _SAR_WORKFLOW_ID_FILE.write_text(wf_id)
                console.print(f"  [green]Deployed workflow '{wf_id}'[/green]")
                return True
            else:
                console.print(
                    f"  [yellow]Workflow deployment returned {resp.status_code}: "
                    f"{resp.text[:300]}[/yellow]"
                )
                console.print(
                    "  [dim]The SAR workflow YAML is saved at workflows/sar-report.yaml. "
                    "You can import it manually via Kibana if the Workflows API is not "
                    "available on your deployment.[/dim]"
                )
                return False
    except Exception as exc:
        console.print(f"  [yellow]Workflow deployment failed: {exc}[/yellow]")
        console.print(
            "  [dim]The SAR workflow YAML is saved at workflows/sar-report.yaml. "
            "You can import it manually via Kibana.[/dim]"
        )
        return False


def run_workflow(config: Config, entity_name: str) -> str | None:
    """Invoke the SAR workflow and retrieve the indexed report.

    1. Triggers the async workflow via POST /api/workflows/{id}/run
    2. Polls triage-sar-reports for a new document matching the entity
    3. Renders the structured report from the indexed data

    Returns the rendered report text or None on failure.
    """
    import time

    wf_id = _get_sar_workflow_id()
    if not wf_id:
        console.print(
            "[red]No SAR workflow ID found. Run 'triage setup-agent' first to deploy the workflow.[/red]"
        )
        return None

    headers = _kibana_headers(config)
    auth = _kibana_auth(config)
    base = config.kibana_url.rstrip("/")
    payload = {"inputs": {"entity_name": entity_name}}

    # Capture the timestamp before triggering so we can find the new report
    trigger_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        with httpx.Client(verify=not config.es_insecure, timeout=30) as client:
            resp = client.post(
                f"{base}/api/workflows/{wf_id}/run",
                json=payload,
                headers=headers,
                auth=auth,
            )

            if resp.status_code not in (200, 201):
                console.print(
                    f"[red]Workflow trigger failed ({resp.status_code}): "
                    f"{resp.text[:500]}[/red]"
                )
                return None

            data = resp.json()
            exec_id = data.get("workflowExecutionId", "unknown")
            console.print(f"[dim]Workflow execution started: {exec_id}[/dim]")
    except Exception as exc:
        console.print(f"[red]Workflow trigger error: {exc}[/red]")
        return None

    # Poll triage-sar-reports for the new document
    es = config.get_es_client()
    console.print("[dim]Waiting for workflow to complete and index report...[/dim]")

    for attempt in range(30):
        time.sleep(2)
        try:
            result = es.search(
                index=INDEX_SAR_REPORTS,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"entity": entity_name}},
                                {"range": {"generated_at": {"gte": trigger_time}}},
                            ]
                        }
                    },
                    "sort": [{"generated_at": "desc"}],
                    "size": 1,
                },
            )
            hits = result.get("hits", {}).get("hits", [])
            if hits:
                return _render_report(entity_name, hits[0]["_source"])
        except Exception:
            pass

    console.print(
        "[yellow]Timed out waiting for report. The workflow may still be running.\n"
        "Check Kibana Workflows UI for execution status, or search the "
        "triage-sar-reports index.[/yellow]"
    )
    return None


def _render_report(entity_name: str, doc: dict) -> str:
    """Render a human-readable SAR from the indexed document.

    Column indices match the KEEP clauses in sar-report.yaml:
      search_financial_refs: file_path[0], file_name[1], entities_financial_ref[2]
      search_transactions:   file_path[0], file_name[1], text_preview[2], entities_financial_ref[3]
      search_communications: file_path[0], file_name[1], email_from[2], email_to[3],
                             email_subject[4], text_preview[5], entities_email[6]
      search_all_docs:       file_path[0], file_name[1], file_extension[2], text_preview[3],
                             email_from[4], email_to[5], email_subject[6],
                             entities_email[7], entities_financial_ref[8]
    """
    lines = [
        "=" * 69,
        "SUSPICIOUS ACTIVITY REPORT (SAR)",
        "FinCEN Form 111 -- Evidence Summary",
        "=" * 69,
        f"Generated: {doc.get('generated_at', 'N/A')}",
        "",
        "=" * 69,
        "PART I -- SUBJECT INFORMATION",
        "=" * 69,
        f"Subject Name:            {entity_name}",
        f"Total Documents:         {doc.get('total_documents', 0)}",
        f"Financial Documents:     {doc.get('transaction_count', 0)}",
        f"Communications:          {doc.get('communication_count', 0)}",
        f"With Financial Refs:     {doc.get('financial_ref_count', 0)}",
    ]

    def _safe(row, idx, default=""):
        return row[idx] if len(row) > idx and row[idx] else default

    # --- Financial references ---
    lines += ["", "=" * 69, "PART II -- FINANCIAL REFERENCES & TRANSACTIONS", "=" * 69]
    fin_refs_raw = doc.get("financial_refs", "[]")
    try:
        fin_refs = json.loads(fin_refs_raw) if isinstance(fin_refs_raw, str) else fin_refs_raw
    except (json.JSONDecodeError, TypeError):
        fin_refs = []

    if fin_refs:
        lines.append(f"\n{len(fin_refs)} document(s) with financial identifiers:\n")
        for row in fin_refs:
            lines.append(f"  * {_safe(row, 1, '?')}")
            lines.append(f"    Path: {_safe(row, 0, '?')}")
            refs = _safe(row, 2)
            if refs:
                lines.append(f"    Refs: {refs}")
    else:
        lines.append("No financial reference identifiers found.")

    # --- Financial documents ---
    fin_docs_raw = doc.get("financial_documents", "[]")
    try:
        fin_docs = json.loads(fin_docs_raw) if isinstance(fin_docs_raw, str) else fin_docs_raw
    except (json.JSONDecodeError, TypeError):
        fin_docs = []

    if fin_docs:
        lines += [f"\n{len(fin_docs)} financial document(s):\n"]
        for row in fin_docs:
            lines.append(f"  * {_safe(row, 1, '?')}")
            refs = _safe(row, 3)
            if refs:
                lines.append(f"    Refs: {refs}")
            preview = _safe(row, 2)
            if preview:
                lines.append(f"    Preview: {str(preview).strip()[:150]}...")

    # --- Communications ---
    lines += ["", "=" * 69, "PART III -- COMMUNICATIONS", "=" * 69]
    comms_raw = doc.get("communications", "[]")
    try:
        comms = json.loads(comms_raw) if isinstance(comms_raw, str) else comms_raw
    except (json.JSONDecodeError, TypeError):
        comms = []

    if comms:
        lines.append(f"\n{len(comms)} communication(s):\n")
        for row in comms:
            lines.append(f"  * {_safe(row, 1, '?')}")
            subj = _safe(row, 4)
            if subj:
                lines.append(f"    Subject: {subj}")
            frm = _safe(row, 2)
            if frm:
                lines.append(f"    From: {frm}")
            to = _safe(row, 3)
            if to:
                lines.append(f"    To: {to}")
            preview = _safe(row, 5)
            if preview:
                lines.append(f"    Preview: {str(preview).strip()[:150]}...")
    else:
        lines.append("No communications found.")

    # --- Evidence index ---
    lines += ["", "=" * 69, "PART IV -- SUPPORTING DOCUMENTATION INDEX", "=" * 69]
    evidence_raw = doc.get("evidence_files", "[]")
    try:
        evidence = json.loads(evidence_raw) if isinstance(evidence_raw, str) else evidence_raw
    except (json.JSONDecodeError, TypeError):
        evidence = []

    lines.append(
        f"\n{len(evidence)} file(s) referencing subject (preserve as evidence):\n"
    )
    for row in evidence:
        ext = _safe(row, 2, "?")
        path = _safe(row, 0, "?")
        emails = _safe(row, 7)
        fin_refs_val = _safe(row, 8)
        detail_parts = []
        if emails:
            detail_parts.append(f"emails={emails}")
        if fin_refs_val:
            detail_parts.append(f"fin_refs={fin_refs_val}")
        detail = " | ".join(detail_parts)
        lines.append(f"  [{ext}] {path}")
        if detail:
            lines.append(f"        {detail}")

    lines += [
        "",
        "=" * 69,
        "END OF REPORT",
        "=" * 69,
        "This report was generated from indexed evidence.",
        "All fields are populated from Elasticsearch query results.",
        "=" * 69,
    ]

    return "\n".join(lines)


def setup_all(config: Config) -> None:
    """Run the complete Agent Builder setup: tools + agent + workflow."""
    console.print("\n[bold]Setting up Elastic Agent Builder[/bold]\n")

    console.print("[bold]Step 1: Creating custom tools[/bold]")
    created_tool_ids = create_tools(config)

    if not created_tool_ids:
        console.print(
            "\n[yellow]No custom tools were created. The agent will use "
            "built-in platform tools only.[/yellow]"
        )

    console.print("\n[bold]Step 2: Creating agent[/bold]")
    create_agent(config, created_tool_ids)

    console.print("\n[bold]Step 3: Deploying SAR workflow[/bold]")
    deploy_workflow(config)

    console.print("\n[bold green]Agent Builder setup complete![/bold green]")
    console.print(f'Chat with the agent in Kibana or via: triage chat "<your question>"')
    console.print(f'Generate a SAR report via: triage sar "<entity name>"')
