from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

from chat import ChatService
from config import Settings
from database import ChatStore
from export import default_export_filename, export_session
from models import ChatMessage, ChatSession, MessageMatch
from providers import Provider, ProviderError
from registry import PROVIDER_LABELS, ProviderFactory, configured_providers


class OmniChatApp:
    """Interactive terminal application for OmniChat Terminal."""

    def __init__(
        self,
        settings: Settings,
        store: ChatStore,
        *,
        console: Console | None = None,
        input_fn: Callable[[str], str] | None = None,
        providers: dict[str, ProviderFactory] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.service = ChatService(store)
        self.console = console or Console()
        self.input_fn = input_fn or self.console.input
        self.providers = providers if providers is not None else configured_providers(settings)
        self.provider: Provider | None = None
        self.session: ChatSession | None = None
        self._history: list[ChatMessage] = []

    def run(self, *, provider_name: str | None = None, session_id: str | None = None) -> int:
        if not self.providers:
            self.console.print(
                "[red]No provider is configured.[/red] Add an API key or compatible "
                "endpoint to your .env file. See .env.example."
            )
            return 2
        try:
            if session_id:
                if not self._load_session(session_id):
                    return 2
            else:
                self.provider = self._choose_provider(provider_name)
                if self.provider is None:
                    return 0
                self._new_session()
            provider_label = PROVIDER_LABELS.get(self.provider.name, self.provider.name)
            self.console.print(
                f"[bold cyan]OmniChat Terminal[/bold cyan] · {provider_label}"
                f" · [dim]{self.provider.model}[/dim]"
            )
            self.console.print(
                "Type [bold]/help[/bold] for commands or [bold]/quit[/bold] to exit."
            )
            while True:
                try:
                    prompt = self.input_fn("[bold green]you > [/bold green]").strip()
                except (EOFError, KeyboardInterrupt):
                    self.console.print("\n[dim]Conversation ended.[/dim]")
                    return 0
                if not prompt:
                    continue
                if prompt.startswith("/"):
                    try:
                        if not self._handle_command(prompt):
                            return 0
                    except KeyboardInterrupt:
                        self.console.print("\n[dim]Command cancelled.[/dim]")
                    continue
                try:
                    self._ask(prompt)
                except KeyboardInterrupt:
                    self.console.print("\n[dim]Request cancelled.[/dim]")
        finally:
            self._close_provider()

    def _create_provider(self, name: str, factory: ProviderFactory) -> Provider | None:
        try:
            return factory()
        except (RuntimeError, ValueError) as error:
            label = PROVIDER_LABELS.get(name, name)
            self.console.print(
                f"[red]Could not initialize {escape(label)}:[/red] {escape(str(error))}"
            )
            return None

    def _choose_provider(self, requested: str | None) -> Provider | None:
        if requested:
            requested = requested.lower()
            if requested not in self.providers:
                self.console.print(f"[red]Provider {requested!r} is not configured.[/red]")
                return None
            return self._create_provider(requested, self.providers[requested])
        if len(self.providers) == 1:
            name = next(iter(self.providers))
            return self._create_provider(name, self.providers[name])

        table = Table(title="Choose an AI provider")
        table.add_column("#", style="cyan")
        table.add_column("Provider")
        table.add_column("Model")
        names = list(self.providers)
        for index, name in enumerate(names, start=1):
            label = PROVIDER_LABELS.get(name, name)
            provider = self._create_provider(name, self.providers[name])
            if provider is None:
                table.add_row(str(index), escape(label), "[red]unavailable[/red]")
                continue
            try:
                table.add_row(str(index), escape(label), escape(provider.model))
            finally:
                provider.close()
        self.console.print(table)
        try:
            choice = self.input_fn("Provider number (or q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if choice.lower() in {"q", "quit", "exit"}:
            return None
        try:
            number = int(choice)
        except ValueError:
            self.console.print("[red]Invalid provider selection.[/red]")
            return None
        if not 1 <= number <= len(names):
            self.console.print("[red]Invalid provider selection.[/red]")
            return None
        name = names[number - 1]
        return self._create_provider(name, self.providers[name])

    def _new_session(self) -> None:
        assert self.provider is not None
        self.session = self.store.create_session(self.provider.name, self.provider.model)
        self._history = []
        self.console.print(f"[dim]New session {escape(self.session.id)}[/dim]")

    def _load_session(self, session_id: str) -> bool:
        session = self.store.get_session(session_id)
        if session is None:
            self.console.print(f"[red]No chat session found with ID {escape(session_id)}.[/red]")
            return False
        factory = self.providers.get(session.provider)
        if factory is None:
            self.console.print(
                f"[red]This session uses {escape(session.provider)}, which is not configured. "
                "Add its credentials to .env first.[/red]"
            )
            return False
        provider = self._create_provider(session.provider, factory)
        if provider is None:
            return False
        self._close_provider()
        self.provider = provider
        self.session = session
        self._history = [
            ChatMessage(role=message.role, content=message.content)
            for message in self.store.get_messages(session.id)
        ]
        self.console.print(f"[dim]Loaded: {escape(session.title)} ({escape(session.id)})[/dim]")
        return True

    def _ask(self, prompt: str) -> None:
        assert self.provider is not None and self.session is not None
        try:
            answer = self.service.ask(self.provider, prompt, self.session, self._history)
        except ProviderError as error:
            self.console.print(f"[red]Provider error:[/red] {escape(str(error))}")
            return
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
            self.console.print(
                f"[red]Could not complete this conversation:[/red] {escape(str(error))}"
            )
            return
        self._history.extend(
            [
                ChatMessage(role="user", content=prompt),
                ChatMessage(role="assistant", content=answer),
            ]
        )
        provider_label = PROVIDER_LABELS.get(self.provider.name, self.provider.name)
        self.console.print(f"\n[bold magenta]{escape(provider_label)}[/bold magenta]")
        self.console.print(Markdown(answer))
        self.console.print()

    def _handle_command(self, command: str) -> bool:
        name, _, argument = command.partition(" ")
        name = name.lower()
        argument = argument.strip()
        if name in {"/quit", "/exit"}:
            return False
        if name == "/help":
            self._show_help()
        elif name == "/new":
            self._new_session()
        elif name == "/switch":
            self._switch_provider()
        elif name == "/history":
            self._show_history()
        elif name == "/search":
            self._search(argument)
        elif name == "/export":
            self._export(argument)
        elif name == "/load":
            if not argument:
                self.console.print("[yellow]Usage: /load <session-id>[/yellow]")
            else:
                self._load_session(argument)
        elif name in {"/delete", "/clear"}:
            session_id = argument or (self.session.id if self.session else "")
            self._delete_session(session_id)
        else:
            self.console.print(f"[yellow]Unknown command {escape(name)}. Type /help.[/yellow]")
        return True

    def _switch_provider(self) -> None:
        selected = self._choose_provider(None)
        if selected is None:
            return
        self._close_provider()
        self.provider = selected
        self._new_session()
        self.console.print("[dim]Provider switched; started a new session.[/dim]")

    def _show_history(self) -> None:
        sessions = self.store.list_sessions()
        if not sessions:
            self.console.print("[dim]No saved chats yet.[/dim]")
            return
        table = Table(title="Recent chats")
        table.add_column("Session ID", style="cyan")
        table.add_column("Updated")
        table.add_column("Provider")
        table.add_column("Title")
        for session in sessions:
            table.add_row(
                escape(session.id),
                session.updated_at.astimezone().strftime("%Y-%m-%d %H:%M"),
                escape(session.provider),
                escape(session.title),
            )
        self.console.print(table)

    def _search(self, query: str) -> None:
        if not query:
            self.console.print("[yellow]Usage: /search <text>[/yellow]")
            return
        try:
            matches = self.store.search_messages(query, limit=20)
        except (OSError, ValueError, sqlite3.Error) as error:
            self.console.print(f"[red]Could not search history:[/red] {escape(str(error))}")
            return
        if not matches:
            self.console.print(f"[dim]No saved messages match {escape(query)!r}.[/dim]")
            return
        table = Table(title=f"Search results for {escape(query)!r}")
        table.add_column("Session ID", style="cyan")
        table.add_column("Updated")
        table.add_column("Role")
        table.add_column("Excerpt")
        for match in matches:
            table.add_row(
                escape(match.session_id),
                match.created_at.astimezone().strftime("%Y-%m-%d %H:%M"),
                escape(match.role),
                escape(_excerpt(match)),
            )
        self.console.print(table)
        self.console.print("[dim]Use /load <session-id> to resume a conversation.[/dim]")

    def _export(self, destination: str) -> None:
        assert self.session is not None
        path = (
            Path(destination).expanduser()
            if destination
            else Path.cwd() / default_export_filename(self.session)
        )
        try:
            exported = export_session(self.store, self.session.id, path)
        except (KeyError, OSError) as error:
            self.console.print(
                f"[red]Could not export this conversation:[/red] {escape(str(error))}"
            )
            return
        self.console.print(f"[dim]Conversation exported to {escape(str(exported))}[/dim]")

    def _delete_session(self, session_id: str) -> None:
        if not session_id:
            self.console.print(
                "[yellow]Usage: /delete <session-id> (or /clear for this chat)[/yellow]"
            )
            return
        if not self.store.delete_session(session_id):
            self.console.print(
                f"[yellow]No chat session found with ID {escape(session_id)}.[/yellow]"
            )
            return
        if self.session and self.session.id == session_id:
            self._new_session()
        self.console.print("[dim]Chat history deleted.[/dim]")

    def _show_help(self) -> None:
        self.console.print(
            Markdown(
                """
                - `/help` — show commands
                - `/switch` — choose another configured provider
                - `/new` — start a fresh conversation
                - `/history` — list saved conversations
                - `/search <text>` — search messages across saved conversations
                - `/export [path]` — export this conversation to Markdown
                - `/load <id>` — resume a saved conversation
                - `/delete <id>` — permanently delete a conversation
                - `/clear` — delete this conversation and start another
                - `/quit` — exit
                """
            )
        )

    def _close_provider(self) -> None:
        if self.provider is not None:
            self.provider.close()
            self.provider = None


def _excerpt(match: MessageMatch, length: int = 90) -> str:
    collapsed = " ".join(match.content.split())
    if len(collapsed) <= length:
        return collapsed
    return f"{collapsed[: length - 1]}…"
