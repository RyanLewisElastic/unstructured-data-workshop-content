"""CLI orchestrator for the Multimodal Forensic Triage Tool.

Subcommands:
  generate-sample      -- Generate synthetic forensic test data
  ingest <path>        -- Extract, embed, and index all files from a directory
  setup-agent          -- Configure the Elastic Agent Builder agent
  chat <message>       -- Send a message to the forensic triage agent
  sar <entity>         -- Generate a SAR report for an entity
  resolve-quarantine   -- LLM-assisted resolution of quarantined files
  status               -- Show ingest pipeline summary
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .config import SAMPLE_DRIVE_DIR, get_config

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Multimodal Forensic Triage Tool -- analyze seized data sources."""
    _setup_logging(verbose)


@cli.command("generate-sample")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output directory (default: sample_drive/)")
@click.option("--skip-noise", is_flag=True, default=False,
              help="Skip noise file generation (only create investigation-relevant files)")
@click.option("--scenario", type=click.Choice(["default", "workshop"]), default="default",
              help="Scenario: 'default' (3-thread AML demo) or 'workshop' (single-thread Instruqt lab)")
def generate_sample(output: str | None, skip_noise: bool, scenario: str) -> None:
    """Generate synthetic forensic test data."""
    from .generate_sample_data import generate_sample_drive

    output_path = Path(output) if output else SAMPLE_DRIVE_DIR
    console.print(f"[bold]Generating sample data in {output_path}[/bold]")
    if scenario == "workshop":
        console.print("[dim]Scenario: workshop (sanctions evasion via trade fraud)[/dim]")
    if skip_noise:
        console.print("[dim]Noise generation skipped (--skip-noise)[/dim]")
    generate_sample_drive(output_path, skip_noise=skip_noise, scenario=scenario)
    console.print("[bold green]Sample data generation complete![/bold green]")


@cli.command("ingest")
@click.argument("drive_path", type=click.Path(exists=True))
@click.option("--workers", "-w", default=4, help="Parallel extraction workers")
@click.option("--clean", is_flag=True, help="Delete existing indices before ingesting")
@click.option("--quarantine-dir", "-q", type=click.Path(), default=None,
              help="Directory for quarantined files (default: <drive_path>/../problem_children)")
@click.option("--skip-quarantine-resolve", is_flag=True, default=False,
              help="Skip automatic LLM resolution of quarantined files")
def ingest(
    drive_path: str,
    workers: int,
    clean: bool,
    quarantine_dir: str | None,
    skip_quarantine_resolve: bool,
) -> None:
    """Extract, embed, and index all files from a directory."""
    from .extractor import extract_directory
    from .indexer import (
        create_indices,
        create_inference_endpoint,
        delete_indices,
        index_metadata,
        index_quarantine_records,
        index_retrieval_content,
    )
    from .problem_child import load_quarantine_manifest

    config = get_config()
    es = config.get_es_client()

    # Verify connection
    console.print("[bold]Connecting to Elasticsearch...[/bold]")
    info = es.info()
    console.print(f"  Connected to {info['name']} v{info['version']['number']}")

    if clean:
        console.print("[yellow]Deleting existing indices...[/yellow]")
        delete_indices(es)

    # Step 1: Create inference endpoint and indices
    console.print("\n[bold]Step 1: Setting up Elastic Inference endpoint and indices[/bold]")
    create_inference_endpoint(es, config)
    create_indices(es)

    # Step 2: Extract files (with classification, archive unpacking, and quarantine)
    root = Path(drive_path).resolve()
    q_dir = Path(quarantine_dir) if quarantine_dir else root.parent / "problem_children"

    console.print(f"\n[bold]Step 2: Classifying and extracting files from {drive_path}[/bold]")
    console.print(f"  Quarantine directory: {q_dir}")

    extracted = list(extract_directory(
        root,
        max_workers=workers,
        whisper_model=config.whisper_model,
        quarantine_dir=q_dir,
    ))
    console.print(f"  Extracted {len(extracted)} file records")

    if not extracted:
        console.print("[red]No files extracted. Check the directory path.[/red]")
        return

    # Report extraction methods
    methods: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for f in extracted:
        methods[f.extraction_method] = methods.get(f.extraction_method, 0) + 1
        statuses[f.extraction_status] = statuses.get(f.extraction_status, 0) + 1

    console.print("\n  [dim]Extraction methods:[/dim]")
    for method, count in sorted(methods.items(), key=lambda x: -x[1]):
        console.print(f"    {method}: {count}")
    console.print("  [dim]Extraction statuses:[/dim]")
    for status, count in sorted(statuses.items(), key=lambda x: -x[1]):
        color = "green" if status == "success" else "yellow" if status == "partial" else "red"
        console.print(f"    [{color}]{status}: {count}[/{color}]")

    # Step 3: Index file metadata with entity extraction
    console.print("\n[bold]Step 3: Indexing file metadata with entity extraction[/bold]")
    index_metadata(es, extracted)

    # Step 4: Index retrieval content (Elasticsearch auto-embeds via semantic_text)
    console.print("\n[bold]Step 4: Indexing retrieval content (Elastic Inference auto-embeds)[/bold]")
    text_files = [f for f in extracted if f.text_content.strip()]
    index_retrieval_content(es, text_files)

    # Step 5: Index quarantine records
    quarantine_records = load_quarantine_manifest(q_dir)
    if quarantine_records:
        console.print(f"\n[bold]Step 5: Indexing {len(quarantine_records)} quarantine records[/bold]")
        index_quarantine_records(es, quarantine_records)
    else:
        console.print("\n[bold]Step 5: No quarantined files[/bold]")

    # Summary
    console.print("\n[bold green]Ingestion complete![/bold green]")
    for idx_name in ["triage-file-metadata", "triage-retrieval", "triage-quarantine"]:
        try:
            count = es.count(index=idx_name)["count"]
            console.print(f"  {idx_name}: {count} documents")
        except Exception:
            console.print(f"  {idx_name}: [yellow]not available[/yellow]")

    if quarantine_records and not skip_quarantine_resolve:
        console.print(
            f"\n[yellow]  {len(quarantine_records)} files quarantined. "
            f"Run 'triage resolve-quarantine {q_dir}' to attempt LLM-assisted resolution.[/yellow]"
        )


@cli.command("resolve-quarantine")
@click.argument("quarantine_path", type=click.Path(exists=True))
@click.option("--file", "-f", "single_file", default=None,
              help="Resolve a single file by its relative path in the quarantine")
def resolve_quarantine(quarantine_path: str, single_file: str | None) -> None:
    """Attempt LLM-assisted resolution of quarantined files."""
    from .problem_child import (
        load_quarantine_manifest,
        resolve_all_quarantined,
        resolve_quarantined_file,
    )

    config = get_config()

    if not config.llm_api_key:
        console.print(
            "[red]LLM_API_KEY not set. Configure LLM_PROVIDER, LLM_API_KEY, "
            "and LLM_MODEL in .env to enable quarantine resolution.[/red]"
        )
        return

    q_dir = Path(quarantine_path)
    records = load_quarantine_manifest(q_dir)
    pending = [r for r in records if r.status == "quarantined"]

    if not pending:
        console.print("[green]No pending quarantined files to resolve.[/green]")
        return

    console.print(f"[bold]Resolving {len(pending)} quarantined files using {config.llm_provider}...[/bold]\n")

    if single_file:
        target = [r for r in pending if r.file_path == single_file]
        if not target:
            console.print(f"[red]File '{single_file}' not found in quarantine manifest.[/red]")
            return
        text, updated = resolve_quarantined_file(
            target[0], q_dir, config.llm_provider, config.llm_api_key, config.llm_model,
        )
        _print_resolution_result(updated, text)
    else:
        results = resolve_all_quarantined(
            q_dir, config.llm_provider, config.llm_api_key, config.llm_model,
        )
        resolved = sum(1 for r, t in results if r.status == "resolved")
        unresolvable = sum(1 for r, t in results if r.status == "unresolvable")

        for record, text in results:
            _print_resolution_result(record, text)

        console.print(f"\n[bold]Results: {resolved} resolved, {unresolvable} unresolvable[/bold]")

        if resolved > 0:
            console.print(
                "[dim]Re-run 'triage ingest' with --clean to re-index resolved files, "
                "or manually index the extracted text.[/dim]"
            )


def _print_resolution_result(record, text: str | None) -> None:
    status_color = "green" if record.status == "resolved" else "red"
    console.print(
        f"  [{status_color}]{record.status}[/{status_color}] {record.file_path}"
    )
    if record.llm_analysis:
        console.print(f"    [dim]{record.llm_analysis[:200]}[/dim]")
    if text:
        console.print(f"    [green]Extracted {len(text)} chars of text[/green]")


@cli.command("status")
@click.option("--quarantine-dir", "-q", type=click.Path(), default=None,
              help="Quarantine directory to check")
def status(quarantine_dir: str | None) -> None:
    """Show ingest pipeline summary from Elasticsearch."""
    from .config import INDEX_FILE_METADATA, INDEX_QUARANTINE, INDEX_RETRIEVAL

    config = get_config()
    es = config.get_es_client()

    console.print("[bold]Pipeline Status[/bold]\n")

    # Index counts
    index_table = Table(title="Index Document Counts")
    index_table.add_column("Index", style="cyan")
    index_table.add_column("Documents", justify="right")

    for idx_name in [INDEX_FILE_METADATA, INDEX_RETRIEVAL, INDEX_QUARANTINE, "triage-sar-reports"]:
        try:
            count = es.count(index=idx_name)["count"]
            index_table.add_row(idx_name, str(count))
        except Exception:
            index_table.add_row(idx_name, "[yellow]not available[/yellow]")

    console.print(index_table)

    # Extraction method breakdown
    try:
        result = es.search(
            index=INDEX_FILE_METADATA,
            body={
                "size": 0,
                "aggs": {
                    "methods": {"terms": {"field": "extraction_method", "size": 20}},
                    "statuses": {"terms": {"field": "extraction_status", "size": 10}},
                    "extensions": {"terms": {"field": "file_extension", "size": 30}},
                    "encrypted": {"filter": {"term": {"is_encrypted": True}}},
                },
            },
        )
        aggs = result.get("aggregations", {})

        if aggs.get("methods", {}).get("buckets"):
            console.print("\n")
            method_table = Table(title="Extraction Methods")
            method_table.add_column("Method", style="cyan")
            method_table.add_column("Count", justify="right")
            for bucket in aggs["methods"]["buckets"]:
                method_table.add_row(bucket["key"], str(bucket["doc_count"]))
            console.print(method_table)

        if aggs.get("statuses", {}).get("buckets"):
            console.print("\n")
            status_table = Table(title="Extraction Statuses")
            status_table.add_column("Status", style="cyan")
            status_table.add_column("Count", justify="right")
            for bucket in aggs["statuses"]["buckets"]:
                color = "green" if bucket["key"] == "success" else "yellow" if bucket["key"] == "partial" else "red"
                status_table.add_row(f"[{color}]{bucket['key']}[/{color}]", str(bucket["doc_count"]))
            console.print(status_table)

        if aggs.get("extensions", {}).get("buckets"):
            console.print("\n")
            ext_table = Table(title="File Types")
            ext_table.add_column("Extension", style="cyan")
            ext_table.add_column("Count", justify="right")
            for bucket in aggs["extensions"]["buckets"]:
                ext_table.add_row(bucket["key"], str(bucket["doc_count"]))
            console.print(ext_table)

        encrypted_count = aggs.get("encrypted", {}).get("doc_count", 0)
        if encrypted_count:
            console.print(f"\n[yellow]  Encrypted files detected: {encrypted_count}[/yellow]")

    except Exception as exc:
        console.print(f"\n[yellow]Could not fetch aggregations: {exc}[/yellow]")

    # Quarantine summary
    if quarantine_dir:
        from .problem_child import load_quarantine_manifest

        q_dir = Path(quarantine_dir)
        records = load_quarantine_manifest(q_dir)
        if records:
            console.print("\n")
            q_table = Table(title="Quarantine Summary")
            q_table.add_column("Status", style="cyan")
            q_table.add_column("Count", justify="right")
            status_counts: dict[str, int] = {}
            for r in records:
                status_counts[r.status] = status_counts.get(r.status, 0) + 1
            for s, c in sorted(status_counts.items()):
                q_table.add_row(s, str(c))
            console.print(q_table)


@cli.command("setup-agent")
def setup_agent() -> None:
    """Configure the Elastic Agent Builder agent."""
    from .setup_agent import setup_all

    config = get_config()
    setup_all(config)


@cli.command("chat")
@click.argument("message")
@click.option("--conversation-id", "-c", default=None,
              help="Continue an existing conversation")
def chat(message: str, conversation_id: str | None) -> None:
    """Send a message to the forensic triage agent."""
    from .setup_agent import chat_with_agent

    config = get_config()
    console.print(f"[dim]Sending to forensic-triage-assistant...[/dim]\n")
    output = chat_with_agent(config, message, conversation_id)
    console.print(output)


@cli.command("sar")
@click.argument("entity_name")
def sar(entity_name: str) -> None:
    """Generate a SAR report for an entity.

    Invokes the triage-sar-report Elastic Workflow which queries all
    financial documents, communications, and associations for the given
    entity and produces a FinCEN-aligned Suspicious Activity Report.
    The report is also indexed to triage-sar-reports for audit trail.
    """
    from .setup_agent import run_workflow

    config = get_config()
    console.print(
        f"[bold]Generating SAR for entity: [cyan]{entity_name}[/cyan][/bold]\n"
    )
    console.print("[dim]Running SAR workflow (query-based, no AI involved)...[/dim]\n")

    report = run_workflow(config, entity_name)
    if report:
        console.print(report)
        console.print(
            f"\n[bold green]SAR report generated and indexed to "
            f"triage-sar-reports[/bold green]"
        )
    else:
        console.print("[red]SAR generation failed. Check the logs above.[/red]")


if __name__ == "__main__":
    cli()
