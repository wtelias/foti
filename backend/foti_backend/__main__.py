"""CLI entry point for the Foti backend.

Subcommands:
    foti-backend serve              start the HTTP daemon
    foti-backend scan PATH          one-shot folder scan
    foti-backend search TEXT        text search
    foti-backend similar PHOTO_ID   image similarity
    foti-backend info               print catalog summary
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from .backfill import backfill_embeddings
from .config import get_settings
from .db import connect
from .importer import import_from_photos_db
from .scanner import scan_root
from .search import search_similar, search_text

cli = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)


@cli.command()
def serve(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the FastAPI daemon."""
    s = get_settings()
    uvicorn.run(
        "foti_backend.api:app",
        host=host or s.host,
        port=port or s.port,
        reload=reload,
    )


@cli.command()
def scan(path: Path) -> None:
    """Index a folder of photos (recursive)."""
    summary = scan_root(path)
    console.print(summary)


@cli.command()
def search(query: str, limit: int = typer.Option(20, "--limit", "-n")) -> None:
    """Text-search the catalog."""
    results = search_text(query, limit=limit)
    table = Table(title=f"top {len(results)} for {query!r}")
    table.add_column("score", justify="right", style="cyan")
    table.add_column("path")
    for r in results:
        table.add_row(f"{r['score']:.3f}", r["path"])
    console.print(table)


@cli.command()
def similar(photo_id: int, limit: int = typer.Option(20, "--limit", "-n")) -> None:
    """Find photos similar to PHOTO_ID."""
    results = search_similar(photo_id, limit=limit)
    table = Table(title=f"similar to photo #{photo_id}")
    table.add_column("score", justify="right", style="cyan")
    table.add_column("path")
    for r in results:
        table.add_row(f"{r['score']:.3f}", r["path"])
    console.print(table)


@cli.command("import-photosdb")
def import_photosdb(
    src: Path,
    limit: int | None = typer.Option(None, "--limit", "-n",
                                     help="Stop after importing N new photos."),
    check_files: bool = typer.Option(False, "--check-files",
                                     help="stat() every source path; slow on NFS."),
) -> None:
    """Import a sibling photos.db (qwen-tagged corpus) into the Foti catalog."""
    summary = import_from_photos_db(src, limit=limit, check_files_exist=check_files)
    console.print(summary)


@cli.command("backfill-embeddings")
def backfill_embeddings_cli(
    limit: int | None = typer.Option(None, "--limit", "-n"),
    batch_size: int | None = typer.Option(None, "--batch", "-b"),
    no_thumbs: bool = typer.Option(False, "--no-thumbs"),
    no_colors: bool = typer.Option(False, "--no-colors"),
) -> None:
    """CLIP-encode photos that arrived without an embedding (e.g. imported)."""
    summary = backfill_embeddings(
        limit=limit, batch_size=batch_size,
        with_thumbs=not no_thumbs, with_colors=not no_colors,
    )
    console.print(summary)


@cli.command()
def info() -> None:
    """Print catalog summary."""
    s = get_settings()
    conn = connect()
    n_photos = conn.execute("SELECT COUNT(*) FROM photo").fetchone()[0]
    n_embed = conn.execute("SELECT COUNT(*) FROM photo_embedding").fetchone()[0]
    n_roots = conn.execute("SELECT COUNT(*) FROM scan_root").fetchone()[0]
    console.print(f"[bold]catalog:[/bold] {s.catalog_path}")
    console.print(f"[bold]photos:[/bold]  {n_photos}")
    console.print(f"[bold]embed:[/bold]   {n_embed}")
    console.print(f"[bold]roots:[/bold]   {n_roots}")


if __name__ == "__main__":
    cli()
