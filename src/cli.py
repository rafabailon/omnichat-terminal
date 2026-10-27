from __future__ import annotations

import argparse
import sqlite3
from dataclasses import replace
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from _version import __version__
from app import OmniChatApp
from config import Settings
from database import ChatStore
from registry import PROVIDER_LABELS, PROVIDER_NAMES, configured_providers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnichat",
        description="OmniChat Terminal: chat with multiple AI providers from your terminal.",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDER_NAMES,
        help="provider to use (otherwise an interactive selection is shown)",
    )
    parser.add_argument("--session", help="resume a saved session by its ID")
    parser.add_argument("--db", help="SQLite database path (overrides OMNICHAT_DB_PATH)")
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="show configured providers and exit",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"OmniChat Terminal {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        settings = Settings.from_environment()
    except ValueError as error:
        console.print(f"[red]Invalid configuration:[/red] {escape(str(error))}")
        return 2
    if args.db is not None:
        if not args.db.strip():
            console.print("[red]The --db option requires a non-empty path.[/red]")
            return 2
        settings = replace(settings, database_path=Path(args.db).expanduser())
    providers = configured_providers(settings)

    if args.list_providers:
        if not providers:
            console.print("No providers configured. Add API keys to .env.")
            return 0
        table = Table(title="Configured AI providers")
        table.add_column("Provider")
        table.add_column("Model")
        for name, factory in providers.items():
            label = PROVIDER_LABELS.get(name, name)
            provider = None
            try:
                provider = factory()
                table.add_row(escape(label), escape(provider.model))
            except (RuntimeError, ValueError) as error:
                table.add_row(escape(label), f"[red]unavailable: {escape(str(error))}[/red]")
            finally:
                if provider is not None:
                    provider.close()
        console.print(table)
        return 0

    try:
        store = ChatStore(settings.database_path)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        console.print(f"[red]Could not open the chat database:[/red] {escape(str(error))}")
        return 2
    app = OmniChatApp(settings, store, console=console, providers=providers)
    try:
        return app.run(provider_name=args.provider, session_id=args.session)
    except KeyboardInterrupt:
        console.print("\n[dim]Conversation ended.[/dim]")
        return 130
