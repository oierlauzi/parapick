"""``parapick simulate``: Generate synthetic datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from parapick.cli.base import (
    format_argument,
)
from parapick.formats import FormatRegistry, Role
from parapick.workflows import (
    SimulationConfig,
    simulate,
    write_dataset,
)

__all__ = ["SimulateCommand"]

logger = logging.getLogger(__name__)


class SimulateCommand:
    """Generate a synthetic dataset."""

    name = "simulate"
    help = "Generate a synthetic dataset (geometry, ground truth and noisy picks)."

    def add_arguments(self, parser: argparse.ArgumentParser, registry: FormatRegistry) -> None:
        """Declare arguments."""
        d = SimulationConfig()
        parser.add_argument("-o", "--output", type=Path, required=True, metavar="DIR")
        format_argument(
            parser, "--output-format", Role.TOMOGRAMS_WRITER, registry, default="relion5",
            allow_auto=False,
        )  # fmt: skip
        format_argument(
            parser, "--picks-format", Role.PICKS_WRITER, registry, default="relion5",
            allow_auto=False,
        )  # fmt: skip
        parser.add_argument("--n-tilt-series", type=int, default=d.n_tilt_series, metavar="N")
        parser.add_argument("--n-particles", type=int, default=d.n_particles, metavar="N")
        parser.add_argument(
            "--tilt-range", type=float, nargs=2, default=(d.tilt_min, d.tilt_max),
            metavar=("MIN", "MAX"),
        )  # fmt: skip
        parser.add_argument("--tilt-step", type=float, default=d.tilt_step, metavar="DEG")
        parser.add_argument("--tilt-axis", type=float, default=d.tilt_axis, metavar="DEG")
        parser.add_argument("--pixel-size", type=float, default=d.pixel_size, metavar="ANGST")
        parser.add_argument(
            "--image-size", type=int, nargs=2, default=d.image_size, metavar=("W", "H")
        )
        parser.add_argument("--thickness", type=float, default=d.thickness, metavar="ANGST")
        parser.add_argument("--pick-noise", type=float, default=d.pick_noise, metavar="ANGST")
        parser.add_argument("--recall", type=float, default=d.recall, metavar="P")
        parser.add_argument(
            "--false-positives", type=int, default=d.false_positives, metavar="N",
            help="Uniform random false picks per tilt.",
        )  # fmt: skip
        parser.add_argument("--seed", type=int, default=d.seed)

    def run(self, args: argparse.Namespace, registry: FormatRegistry) -> int:
        """Run."""
        config = SimulationConfig(
            n_tilt_series=args.n_tilt_series,
            n_particles=args.n_particles,
            tilt_min=args.tilt_range[0],
            tilt_max=args.tilt_range[1],
            tilt_step=args.tilt_step,
            tilt_axis=args.tilt_axis,
            pixel_size=args.pixel_size,
            image_size=tuple(args.image_size),
            thickness=args.thickness,
            pick_noise=args.pick_noise,
            recall=args.recall,
            false_positives=args.false_positives,
            seed=args.seed,
        )
        dataset = simulate(config)
        written = write_dataset(
            dataset,
            args.output,
            output_format=args.output_format,
            picks_format=args.picks_format,
            registry=registry,
        )
        logger.info(
            "Simulated %d tilt-series, %d particles, %d picks -> %s",
            len(dataset.tomograms),
            len(dataset.ground_truth),
            len(dataset.picks),
            ", ".join(str(p) for p in written[:3]) + ("..." if len(written) > 3 else ""),
        )
        return 0
