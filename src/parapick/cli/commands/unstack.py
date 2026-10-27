"""``parapick unstack``: one 2D MRC file per tilt image, for single-particle pickers."""

from __future__ import annotations

import argparse
from pathlib import Path

from parapick.cli.base import add_tomograms_arguments, format_argument, read_options_from
from parapick.formats import FormatRegistry, Role
from parapick.workflows import run_unstack

__all__ = ["UnstackCommand"]


class UnstackCommand:
    """Split tilt-series stacks into individual micrographs."""

    name = "unstack"
    help = "Write each tilt image as its own 2D MRC file (for SPA pickers) plus matching geometry."

    def add_arguments(self, parser: argparse.ArgumentParser, registry: FormatRegistry) -> None:
        """Declare arguments."""
        add_tomograms_arguments(parser, registry)
        parser.add_argument(
            "-o",
            "--output",
            type=Path,
            required=True,
            metavar="DIR",
            help="Output directory: micrographs/<TS>_<NNN>.mrc plus the new geometry.",
        )
        format_argument(
            parser, "--output-format", Role.TOMOGRAMS_WRITER, registry, default="relion5",
            allow_auto=False,
        )  # fmt: skip
        parser.add_argument(
            "--link",
            action="store_true",
            help="Symlink tilts that already are single 2D .mrc files instead of copying them.",
        )
        parser.add_argument(
            "--overwrite", action="store_true", help="Replace existing output images."
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            metavar="N",
            help="Images read and written per batch; bounds memory use (default: 8).",
        )

    def run(self, args: argparse.Namespace, registry: FormatRegistry) -> int:
        """Run."""
        run_unstack(
            args.tomograms,
            args.output,
            tomograms_format=args.tomograms_format,
            output_format=args.output_format,
            read_options=read_options_from(args),
            link=args.link,
            overwrite=args.overwrite,
            batch_size=args.batch_size,
            registry=registry,
        )
        return 0
