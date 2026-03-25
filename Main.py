#!/usr/bin/env python3
"""Entry point that starts the controller loop."""

from __future__ import annotations

import os
import sys


def _bootstrap_parsingfiles_path() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    parsingfiles_dir = os.path.join(repo_root, "CSV", "train", "ParsingFiles")
    if parsingfiles_dir not in sys.path:
        sys.path.insert(0, parsingfiles_dir)


_bootstrap_parsingfiles_path()

from Core import main  # noqa: E402  # type: ignore[reportMissingImports]


if __name__ == "__main__":
    main()

