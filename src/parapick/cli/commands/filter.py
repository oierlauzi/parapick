"""``parapick filter``: Select localised particles."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from parapick.cli.base import (
    format_argument,
)
from parapick.cli.commands._shared import add_particles_input
from parapick.cli.commands.localize import add_selection_arguments, selection_from
from parapick.formats import FormatRegistry, Role, WriteContext
from parapick.workflows import (
    select_particles,
)

__all__ = ["FilterCommand"]

logger = logging.getLogger(__name__)


class FilterCommand:
    """Select particles by score, support or rank."""

    name = "filter"
    help = "Select localised particles by score, support or rank."

    def add_arguments(self, parser: argparse.ArgumentParser, registry: FormatRegistry) -> None:
        """Declare arguments."""
        add_particles_input(parser, registry, "particles", "Particles to filter.")
        parser.add_argument("-o", "--output", type=Path, required=True, metavar="PATH")
        format_argument(
            parser, "--output-format", Role.PARTICLES_WRITER, registry, default="relion5",
            allow_auto=False,
        )  # fmt: skip
        parser.add_argument(
            "--tomograms",
            type=Path,
            metavar="PATH",
            help="Geometry to reference from the output (e.g. in optimisation_set.star).",
        )
        add_selection_arguments(parser)

    def run(self, args: argparse.Namespace, registry: FormatRegistry) -> int:
        """Run."""
        particles = registry.particles_reader(args.particles_format, args.particles).read_particles(
            args.particles
        )
        selected = select_particles(particles, selection_from(args))
        registry.particles_writer(args.output_format).write_particles(
            selected, args.output, WriteContext(tomograms_path=args.tomograms)
        )
        logger.info("Kept %d of %d particles", len(selected), len(particles))
        return 0
