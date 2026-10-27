"""``parapick project``: Project 3D particles onto tilt images."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from parapick.cli.base import (
    add_tomograms_arguments,
    format_argument,
    read_options_from,
)
from parapick.cli.commands._shared import add_particles_input
from parapick.formats import FormatRegistry, Role
from parapick.workflows import (
    project_particles,
)

__all__ = ["ProjectCommand"]

logger = logging.getLogger(__name__)


class ProjectCommand:
    """Project 3D particles onto every tilt image."""

    name = "project"
    help = "Project 3D particles onto the tilt images (e.g. ground truth to train 2D pickers)."

    def add_arguments(self, parser: argparse.ArgumentParser, registry: FormatRegistry) -> None:
        """Declare arguments."""
        add_particles_input(parser, registry, "particles", "3D particles to project.")
        add_tomograms_arguments(parser, registry)
        parser.add_argument("-o", "--output", type=Path, required=True, metavar="PATH")
        format_argument(
            parser, "--output-format", Role.PICKS_WRITER, registry, default="relion5",
            allow_auto=False,
        )  # fmt: skip
        parser.add_argument(
            "--pixel-size",
            type=float,
            metavar="ANGST",
            help="Pixel size of the output coordinates (default: native tilt-series pixel size).",
        )
        parser.add_argument(
            "--keep-outside", action="store_true", help="Keep projections outside the images."
        )

    def run(self, args: argparse.Namespace, registry: FormatRegistry) -> int:
        """Run."""
        tomograms = registry.tomograms_reader(args.tomograms_format, args.tomograms).read_tomograms(
            args.tomograms, read_options_from(args)
        )
        particles = registry.particles_reader(args.particles_format, args.particles).read_particles(
            args.particles, tomograms
        )
        picks = project_particles(
            particles, tomograms, pixel_size=args.pixel_size, keep_outside=args.keep_outside
        )
        registry.picks_writer(args.output_format).write_picks(picks, args.output)
        logger.info("Wrote %d projected picks to %s", len(picks), args.output)
        return 0
