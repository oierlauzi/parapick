"""``parapick convert-picks``: Convert 2D picks between formats."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from parapick.cli.base import (
    add_tomograms_arguments,
    format_argument,
    read_options_from,
)
from parapick.formats import FormatRegistry, Role
from parapick.workflows import (
    convert_picks,
)

__all__ = ["ConvertPicksCommand"]

logger = logging.getLogger(__name__)


class ConvertPicksCommand:
    """Convert 2D picks between formats."""

    name = "convert-picks"
    help = "Convert 2D picks between formats (e.g. Topaz or per-micrograph STARs to one STAR)."

    def add_arguments(self, parser: argparse.ArgumentParser, registry: FormatRegistry) -> None:
        """Declare arguments."""
        parser.add_argument("input", type=Path, metavar="INPUT", help="Picks file or directory.")
        format_argument(parser, "--input-format", Role.PICKS_READER, registry)
        parser.add_argument("-o", "--output", type=Path, required=True, metavar="PATH")
        format_argument(
            parser, "--output-format", Role.PICKS_WRITER, registry, default="relion5",
            allow_auto=False,
        )  # fmt: skip
        add_tomograms_arguments(parser, registry, required=False)
        parser.add_argument(
            "--pixel-size",
            type=float,
            metavar="ANGST",
            help="Record the pixel size of the picked images in the output.",
        )
        parser.add_argument(
            "--flip-y",
            action="store_true",
            help="Mirror the picks' y axis on the image height (pickers or micrographs with the "
            "y axis pointing the other way).",
        )

    def run(self, args: argparse.Namespace, registry: FormatRegistry) -> int:
        """Run."""
        picks, written = convert_picks(
            args.input,
            args.output,
            input_format=args.input_format,
            output_format=args.output_format,
            tomograms_path=args.tomograms,
            tomograms_format=args.tomograms_format,
            read_options=read_options_from(args),
            pixel_size=args.pixel_size,
            flip_y=args.flip_y,
            registry=registry,
        )
        logger.info("Wrote %d picks to %d file(s)", len(picks), len(written))
        return 0
