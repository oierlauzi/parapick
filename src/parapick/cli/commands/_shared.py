"""Argument helpers shared by several sub-commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from parapick.cli.base import format_argument
from parapick.formats import FormatRegistry, Role

__all__ = ["add_particles_input"]


def add_particles_input(
    parser: argparse.ArgumentParser, registry: FormatRegistry, dest: str, help_text: str
) -> None:
    """Positional particles path plus its ``--<dest>-format`` option."""
    parser.add_argument(dest, type=Path, metavar=dest.upper(), help=help_text)
    format_argument(parser, f"--{dest}-format", Role.PARTICLES_READER, registry)
