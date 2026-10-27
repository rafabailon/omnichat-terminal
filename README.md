# OmniChat Terminal

OmniChat Terminal is an interactive Python CLI for chatting with OpenAI, Google Gemini, Anthropic Claude, and OpenAI-compatible endpoints from one terminal. It keeps conversations and ordered message history in a local SQLite database, supports resumable sessions, and can search or export saved conversations.

## Features

- Interactive selection of configured providers and their model names.
- One provider interface for OpenAI Chat Completions, Gemini `generateContent`, Anthropic Messages, and OpenAI-compatible endpoints.
- Optional Azure OpenAI v1-compatible configuration using a custom API-key header.
- Markdown-formatted model replies using Rich.
- Persistent sessions with titles, history listing, session loading, deletion, and new chats.
- Cross-session message search and Markdown conversation export.
- Configurable network timeout and bounded retries for transient provider failures.
- Clear handling of network failures, timeouts, invalid credentials, quotas, and unexpected provider responses.
- Automated provider, persistence, command-loop, export, configuration, and CLI tests.

## Requirements

- Python 3.10 or newer.
- An API key for at least one supported provider, or an OpenAI-compatible endpoint.
- Network access to the provider you select.

## Install

From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The editable install provides both the `omnichat` and `omnichat-terminal` commands.

## Configure provider credentials

Edit `.env` and set at least one API key:

```dotenv
OPENAI_API_KEY=...
# GEMINI_API_KEY=...
# ANTHROPIC_API_KEY=...
```

Only configure providers you intend to use. Model and base URL variables can be changed to match your account and API access. OmniChat Terminal loads `.env` from the current working directory and does not print or store API keys in chat history.

### OpenAI-compatible and Azure OpenAI endpoints

Set the compatible endpoint URL and model/deployment name:

```dotenv
COMPATIBLE_BASE_URL=https://your-resource.openai.azure.com/openai/v1
COMPATIBLE_MODEL=your-deployment-name
COMPATIBLE_API_KEY=...
COMPATIBLE_API_KEY_HEADER=api-key
COMPATIBLE_API_KEY_PREFIX=
```

The endpoint must implement the OpenAI Chat Completions API at `{base URL}/chat/completions`. Other compatible services commonly use `Authorization: Bearer ...`; those are the defaults for `COMPATIBLE_API_KEY_HEADER` and `COMPATIBLE_API_KEY_PREFIX`. The adapter does not use undocumented GitHub Copilot APIs or credentials. For any endpoint, follow its provider terms and verify the expected route and authentication scheme.

### Request tuning

| Variable | Default | Description |
| --- | --- | --- |
| `OMNICHAT_REQUEST_TIMEOUT` | `60` | HTTP timeout in seconds. |
| `OMNICHAT_REQUEST_RETRIES` | `1` | Retries for timeouts, network errors, HTTP 408/429, and 5xx responses (`0`-`5`). |
| `ANTHROPIC_MAX_TOKENS` | `2048` | Anthropic response limit (`1`-`100000`). |

Retries use exponential backoff and honour a numeric `Retry-After` header for 429 responses. A retry can repeat a request that the provider already processed, which may duplicate usage or charges; set `OMNICHAT_REQUEST_RETRIES=0` when strict single-attempt billing is required.

## Run

```bash
omnichat
# or the explicit project command
omnichat-terminal
# or, from a source checkout
python src
```

Choose a configured provider when prompted. When only one provider is configured, OmniChat Terminal selects it automatically. Then type a message and press Enter.

Useful startup options:

```bash
omnichat --list-providers
omnichat --provider openai
omnichat --provider gemini --db ./local-chats.sqlite3
omnichat --session <session-id>
omnichat --version
```

`python main.py` also works from a source checkout without installing the project.

### Chat commands

| Command | Action |
| --- | --- |
| `/help` | Show command help. |
| `/switch` | Select another configured provider and start a fresh conversation. |
| `/new` | Start a new conversation with the current provider. |
| `/history` | List up to 100 recent saved chats and their IDs. |
| `/search <text>` | Search message text across saved conversations (literal text, up to 20 matches). |
| `/export [path]` | Export the current conversation to Markdown. |
| `/load <id>` | Resume a session, including its provider and message context. |
| `/delete <id>` | Permanently delete the selected session and its messages. |
| `/clear` | Delete the current session and start another. |
| `/quit` or `/exit` | Exit the CLI. |

`/export` without a path writes `omnichat-<session-prefix>.md` in the current directory. A path without a suffix receives `.md`, and an existing directory receives the default filename. On POSIX systems exported files are created with owner-only permissions (`0600`).

## Data and privacy

By default, chat history is stored in the platform data directory:

- Linux: `$XDG_DATA_HOME/omnichat-terminal/history.sqlite3`, falling back to `~/.local/share/omnichat-terminal/history.sqlite3`.
- macOS: `~/Library/Application Support/omnichat-terminal/history.sqlite3`.
- Windows: `%LOCALAPPDATA%\omnichat-terminal\history.sqlite3`.

Set `OMNICHAT_DB_PATH` or use `--db` to select another file. On POSIX systems the database file is restricted to the current user (`0600`); SQLite messages are still readable text and are not encrypted by this application. Deleting a session removes its messages from SQLite but does not guarantee secure erasure from filesystem backups or storage.

When you send a message, OmniChat Terminal sends that prompt and the current conversation history to the selected model provider so it can answer. Review the provider's privacy, retention, and billing terms before use. Do not paste secrets or sensitive material unless you have permission and understand the provider's handling. API credentials remain in environment configuration and are not written to the chat database.

Successful user/assistant turns are stored together after a response arrives. If the request fails, that prompt is not added to local history.

## Tests and quality checks

Install the development extra, then run:

```bash
pytest
ruff check .
ruff format --check .
python -m compileall -q src tests scripts main.py
```

Tests use HTTPX's in-process transport and temporary SQLite databases; the normal test suite makes no external network calls. The offline suite covers provider parsing and error mapping, retry/backoff behaviour, configuration validation, SQLite persistence and search, Markdown export, the interactive command loop, and the CLI.

Run a local chat-service/SQLite latency sample (the stub provider makes no network request):

```bash
PYTHONPATH=src python scripts/benchmark_local.py
```

### Live generic-prompt test

The opt-in integration test sends a real, generic arithmetic question to a configured model and checks the provider's actual answer. It is skipped by default because live requests require credentials, network access, and may incur provider charges:

```bash
OMNICHAT_LIVE_TEST=1 OMNICHAT_TEST_PROVIDER=openai pytest -m live -s
```

Choose `openai`, `gemini`, `anthropic`, or `compatible` for `OMNICHAT_TEST_PROVIDER`, and configure that provider in `.env`. The test asks “What is 6 multiplied by 7?” and expects the real response to contain `42`. Do not enable this in routine CI unless its costs and credentials are managed intentionally.

## Project structure

| Path | Contents |
| --- | --- |
| `src/_version.py` | Project version metadata. |
| `src/__main__.py` | Module entry point for `python src`. |
| `src/app.py` | Interactive command loop and terminal UI. |
| `src/chat.py` | Provider invocation and successful-turn persistence. |
| `src/cli.py` | Argument parsing and executable entry point. |
| `src/config.py` | Environment, provider, and request-policy settings. |
| `src/database.py` | SQLite schema, history, and cross-session search. |
| `src/export.py` | Markdown transcript rendering and file export. |
| `src/models.py` | Typed chat/session/search data objects. |
| `src/registry.py` | Configured provider factories and labels. |
| `src/providers/base.py` | Shared provider API, retries, and safe error mapping. |
| `src/providers/openai.py` | OpenAI Chat Completions adapter. |
| `src/providers/gemini.py` | Gemini generateContent adapter. |
| `src/providers/anthropic.py` | Anthropic Messages adapter. |
| `src/providers/compatible.py` | OpenAI-compatible endpoint adapter. |
| `.env.example` | Provider configuration template with no credentials. |
| `tests/conftest.py` | Shared test fixtures. |
| `tests/test_app.py` | Interactive commands, search, export, and failures. |
| `tests/test_chat.py` | Service behaviour and turn persistence. |
| `tests/test_cli.py` | CLI options and configuration errors. |
| `tests/test_config.py` | Dotenv, environment precedence, and validation. |
| `tests/test_database.py` | SQLite schema, persistence, permissions, and search. |
| `tests/test_export.py` | Markdown rendering and private file export. |
| `tests/test_live_provider.py` | Opt-in real-provider generic question. |
| `tests/test_providers.py` | Provider requests, responses, errors, and retries. |
| `tests/test_registry.py` | Configured provider factories. |
| `scripts/benchmark_local.py` | Local SQLite/service overhead benchmark. |
| `main.py` | Source-checkout convenience entry point. |

Runtime modules live directly in `src/`, with provider adapters grouped in `src/providers/`. The source root is added to the import path by `main.py` and by the editable install, so `python main.py`, `python src`, and the installed `omnichat` commands all use the same modules.

## License

MIT. See [LICENSE](LICENSE).
