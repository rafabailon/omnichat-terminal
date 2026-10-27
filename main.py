"""Convenience entry point for running OmniChat Terminal from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    source = Path(__file__).resolve().parent / "src"
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
