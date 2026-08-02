#!/usr/bin/env python3
"""Backward-compatible entry point for running NEUJOH.img from a checkout."""

from __future__ import annotations

import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from neujoh import *  # noqa: E402,F403
from neujoh.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
