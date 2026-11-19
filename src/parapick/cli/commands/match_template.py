"""``parapick match-template``: 2D template matching of every tilt image with Leopard-EM.

Registered only when Leopard-EM is installed (``pip install parapick[leopard]``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from parapick.cli.base import add_tomograms_arguments, read_options_from
from parapick.formats import FormatRegistry
from parapick.matching.schedule import SAMPLINGS

__all__ = ["MatchTemplateCommand"]


class MatchTemplateCommand:
    """Search the tilt images for a template, with parameters inferred for each tilt."""

    name = "match-template"
    help = (
        "Find particles on every tilt image by 2D template matching (Leopard-EM), with "
        "parameters inferred from the acquisition of each tilt. Writes picks for localize."
    )
    uses_jax = False  # Leopard-EM runs on PyTorch: leave the GPU memory to it

    def add_arguments(self, parser: argparse.ArgumentParser, registry: FormatRegistry) -> None:
        """Declare arguments."""
        from parapick.matching.leopard import CITATION

        parser.epilog = f"The search is done by Leopard-EM. Please cite: {CITATION}"
        io = parser.add_argument_group("input/output")
        add_tomograms_arguments(io, registry)
        io.add_argument(
            "--template",
            type=Path,
            required=True,
            metavar="MRC",
            help="Template volume, at the pixel size of the tilt-series.",
        )
        io.add_argument(
            "-o",
            "--output",
            type=Path,
            required=True,
            metavar="DIR",
            help="Output directory: picks.star and tomograms.star for localize, schedule.csv, "
            "and the Leopard-EM configurations, results and peaks of each tilt.",
        )
        io.add_argument(
            "--tilt-series",
            nargs="+",
            metavar="NAME",
            help="Only process these tilt-series (default: all).",
        )

        sample = parser.add_argument_group("sample")
        sample.add_argument(
            "--particle-diameter",
            type=float,
            required=True,
            metavar="Å",
            help="Diameter of the target. Sets the high-pass, the angular steps and the "
            "minimum distance between peaks.",
        )
        sample.add_argument(
            "--thickness",
            type=float,
            metavar="Å",
            help="Thickness of the sample: widens the defocus search (default: the Z size of "
            "the tomogram).",
        )
        sample.add_argument(
            "--dose-per-tilt",
            type=float,
            metavar="E/Å²",
            help="Dose of each tilt image (default: from rlnMicrographPreExposure). Needed if "
            "the pre-exposure is unknown; a dose-symmetric acquisition is then assumed.",
        )
        sample.add_argument(
            "--no-energy-filter",
            dest="energy_filter",
            action="store_false",
            help="The images were not zero-loss filtered (only changes the SNR ranking).",
        )
        sample.add_argument(
            "--symmetry",
            default="C1",
            metavar="SYM",
            help="Symmetry of the template, to restrict the search (default: C1).",
        )

        search = parser.add_argument_group("search")
        search.add_argument(
            "--binning",
            type=int,
            default=1,
            metavar="N",
            help="Search the tilt images binned by N (Fourier cropping): about N² faster per "
            "orientation, and coarser steps follow from the coarser pixel. The template may be "
            "at the native or the binned pixel size. Picks stay in native pixels (default: 1).",
        )
        search.add_argument(
            "--sampling",
            choices=tuple(SAMPLINGS),
            default="default",
            help="'default' caps the angular steps at 4°/3° and the defocus step at 500 Å. "
            "'fast' keeps them proportional to the resolution of each tilt (the relative "
            "accuracy of Leopard-EM's defaults): much faster on binned or late tilts, for "
            "locations rather than poses (default: default).",
        )
        search.add_argument(
            "--max-tilts",
            type=int,
            metavar="N",
            help="Search only the N tilts of each tilt-series with the highest predicted SNR "
            "(default: all).",
        )
        search.add_argument(
            "--false-positives",
            type=float,
            default=1.0,
            metavar="N",
            help="Expected false positives per tilt image, sets the threshold. localize "
            "tolerates false positives on single tilts, so a value well above 1 keeps more "
            "true particles (default: 1).",
        )
        search.add_argument(
            "--orientation-batch-size",
            type=int,
            default=16,
            metavar="N",
            help="Orientations correlated at once; lower it if the GPU runs out of memory "
            "(default: 16).",
        )
        search.add_argument(
            "--schedule-only",
            action="store_true",
            help="Only write the schedule and the Leopard-EM configurations.",
        )
        search.add_argument(
            "--overwrite",
            action="store_true",
            help="Search again the tilts already searched (by default, runs resume).",
        )

    def run(self, args: argparse.Namespace, registry: FormatRegistry) -> int:
        """Run."""
        from parapick.workflows.match_template import run_match_template

        run_match_template(
            args.tomograms,
            args.template,
            args.output,
            particle_diameter=args.particle_diameter,
            thickness=args.thickness,
            dose_per_tilt=args.dose_per_tilt,
            energy_filter=args.energy_filter,
            max_tilts=args.max_tilts,
            tilt_series=args.tilt_series,
            false_positives=args.false_positives,
            symmetry=args.symmetry,
            device=args.device,
            orientation_batch_size=args.orientation_batch_size,
            binning=args.binning,
            sampling=args.sampling,
            schedule_only=args.schedule_only,
            overwrite=args.overwrite,
            tomograms_format=args.tomograms_format,
            read_options=read_options_from(args),
            registry=registry,
        )
        return 0
