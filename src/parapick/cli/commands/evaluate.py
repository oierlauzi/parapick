"""``parapick evaluate``: Evaluate localised particles against a reference."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from parapick.cli.commands._shared import add_particles_input
from parapick.cli.commands.localize import add_selection_arguments, selection_from
from parapick.formats import FormatRegistry, TomogramsReadOptions
from parapick.workflows import (
    evaluate,
    select_particles,
)

__all__ = ["EvaluateCommand"]

logger = logging.getLogger(__name__)


class EvaluateCommand:
    """Compare particles with a reference set."""

    name = "evaluate"
    help = "Compare localised particles with a reference (precision, recall, F1, RMSE)."

    def add_arguments(self, parser: argparse.ArgumentParser, registry: FormatRegistry) -> None:
        """Declare arguments."""
        add_particles_input(parser, registry, "predicted", "Particles to evaluate.")
        add_particles_input(parser, registry, "reference", "Reference (e.g. ground truth).")
        parser.add_argument(
            "--cutoff", type=float, required=True, metavar="ANGST", help="Matching distance."
        )
        parser.add_argument("--json", type=Path, metavar="PATH", help="Also write a JSON report.")
        parser.add_argument(
            "--tomograms",
            type=Path,
            metavar="PATH",
            help="Geometry, needed only to read uncentred (legacy) coordinates.",
        )
        add_selection_arguments(parser)

    def run(self, args: argparse.Namespace, registry: FormatRegistry) -> int:
        """Run."""
        tomograms = None
        if args.tomograms is not None:
            tomograms = registry.tomograms_reader(path=args.tomograms).read_tomograms(
                args.tomograms, TomogramsReadOptions()
            )
        predicted = registry.particles_reader(args.predicted_format, args.predicted)
        reference = registry.particles_reader(args.reference_format, args.reference)
        result = evaluate(
            select_particles(
                predicted.read_particles(args.predicted, tomograms), selection_from(args)
            ),
            reference.read_particles(args.reference, tomograms),
            cutoff=args.cutoff,
        )
        print(
            f"{'tilt-series':<20} {'pred':>6} {'ref':>6} {'TP':>6} "
            f"{'prec':>6} {'recall':>6} {'F1':>6} {'RMSE':>7}"
        )
        for m in (*result.per_tilt_series, result.overall):
            print(
                f"{m.tilt_series:<20} {m.n_predicted:>6} {m.n_reference:>6} "
                f"{m.true_positives:>6} {m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f} "
                f"{m.rmse:>7.2f}"
            )
        if args.json is not None:
            args.json.write_text(json.dumps(result.as_dict(), indent=2))
        return 0
