"""``parapick localize``: 3D particle coordinates from 2D picks on tilt images."""

from __future__ import annotations

import argparse
from pathlib import Path

from parapick.cli.base import add_tomograms_arguments, format_argument, read_options_from
from parapick.formats import FormatRegistry, Role
from parapick.gmm import INITIALIZERS, EMFitter, LocalizeConfig
from parapick.workflows import Selection, run_localize

_DEFAULTS = LocalizeConfig()


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """``--min-score``, ``--min-support`` and ``--top-n``."""
    group = parser.add_argument_group("selection")
    group.add_argument(
        "--min-score",
        type=float,
        metavar="S",
        help="Keep particles with score s_k = n_k/N >= S (fraction of tilts supporting them).",
    )
    group.add_argument(
        "--min-support",
        type=float,
        metavar="N",
        help="Keep particles supported by at least N effective detections.",
    )
    group.add_argument(
        "--top-n", type=int, metavar="N", help="Keep the N best particles per tilt-series."
    )


def selection_from(args: argparse.Namespace) -> Selection:
    """Build a :class:`Selection` from parsed arguments."""
    return Selection(min_score=args.min_score, min_support=args.min_support, top_n=args.top_n)


class LocalizeCommand:
    """Fit a projected 3D Gaussian mixture to the picks of every tilt-series."""

    name = "localize"
    help = "Localise particles in 3D from 2D picks on tilt images (main task)."

    def add_arguments(self, parser: argparse.ArgumentParser, registry: FormatRegistry) -> None:
        """Declare arguments."""
        io = parser.add_argument_group("input/output")
        add_tomograms_arguments(io, registry)
        io.add_argument("--picks", type=Path, required=True, metavar="PATH", help="2D picks.")
        format_argument(io, "--picks-format", Role.PICKS_READER, registry)
        io.add_argument(
            "--picks-pixel-size",
            type=float,
            metavar="ANGST",
            help="Pixel size of the picked images (if binned and not recorded in the picks).",
        )
        io.add_argument(
            "--min-pick-score",
            type=float,
            metavar="FOM",
            help="Ignore 2D picks with a figure of merit below FOM.",
        )
        io.add_argument(
            "--flip-y",
            action="store_true",
            help="Mirror the picks' y axis on the image height (pickers or micrographs with the "
            "y axis pointing the other way).",
        )
        io.add_argument(
            "-o", "--output", type=Path, required=True, metavar="DIR", help="Output directory."
        )
        format_argument(
            io,
            "--output-format",
            Role.PARTICLES_WRITER,
            registry,
            default="relion5",
            allow_auto=False,
        )
        io.add_argument(
            "--tilt-series",
            nargs="+",
            metavar="NAME",
            help="Only process these tilt-series (default: all).",
        )

        model = parser.add_argument_group("model")
        model.add_argument(
            "--n-components",
            type=int,
            metavar="K",
            help="Initial number of components "
            "(default: overcomplete factor x max picks per tilt).",
        )
        model.add_argument(
            "--overcomplete-factor",
            type=float,
            default=_DEFAULTS.overcomplete_factor,
            metavar="F",
            help="Initial components per pick of the most crowded tilt "
            f"(default: {_DEFAULTS.overcomplete_factor}).",
        )
        model.add_argument(
            "--alpha",
            type=float,
            default=_DEFAULTS.alpha,
            help=f"Dirichlet concentration (<1 prunes; default: {_DEFAULTS.alpha}).",
        )
        model.add_argument(
            "--sigma",
            type=float,
            metavar="ANGST",
            help="Initial pick standard deviation (default: sigma fraction x box size).",
        )
        model.add_argument(
            "--sigma-fraction",
            type=float,
            default=_DEFAULTS.init_sigma_fraction,
            metavar="F",
            help="Initial sigma as a fraction of the box "
            f"(default: {_DEFAULTS.init_sigma_fraction}).",
        )
        model.add_argument(
            "--min-sigma",
            type=float,
            default=_DEFAULTS.min_sigma,
            metavar="ANGST",
            help="Lower bound of the per-tilt sigma.",
        )
        background = model.add_mutually_exclusive_group()
        background.add_argument(
            "--background-fraction",
            type=float,
            default=_DEFAULTS.background_fraction,
            metavar="P",
            help="Initial weight of the uniform background that absorbs false-positive picks "
            f"(default: {_DEFAULTS.background_fraction}).",
        )
        background.add_argument(
            "--no-background",
            action="store_true",
            help="Disable the background term (for ablations).",
        )
        model.add_argument(
            "--merge-distance",
            type=float,
            metavar="ANGST",
            help="Merge fitted components closer than this (default: 2x median sigma; 0: off).",
        )
        model.add_argument(
            "--box",
            type=float,
            nargs=3,
            metavar=("X", "Y", "Z"),
            help="Initialisation box in Å (default: tomogram size).",
        )
        model.add_argument(
            "--init",
            choices=sorted(INITIALIZERS),
            default="lifted",
            help="Initialisation: 'lifted' back-projects the picks of the least tilted image at "
            "random depths; 'uniform' draws them uniformly in the box (default: lifted).",
        )

        optimisation = parser.add_argument_group("optimisation")
        optimisation.add_argument(
            "--max-iterations", type=int, default=_DEFAULTS.max_iterations, metavar="N"
        )
        optimisation.add_argument(
            "--tolerance",
            type=float,
            default=_DEFAULTS.tolerance,
            metavar="ANGST",
            help="Convergence threshold on the mean displacement.",
        )
        optimisation.add_argument(
            "--annealing",
            type=float,
            default=_DEFAULTS.annealing,
            metavar="RATE",
            help="Per-iteration decay of the sigma floor (deterministic annealing; 0 disables; "
            f"default: {_DEFAULTS.annealing}).",
        )
        optimisation.add_argument(
            "--sinkhorn-iterations", type=int, default=_DEFAULTS.sinkhorn_iterations, metavar="N"
        )
        optimisation.add_argument(
            "--regularization", type=float, default=_DEFAULTS.regularization, metavar="EPS"
        )
        optimisation.add_argument("--seed", type=int, default=0, help="Random seed.")
        add_selection_arguments(parser)

    def run(self, args: argparse.Namespace, registry: FormatRegistry) -> int:
        """Run the localisation."""
        config = LocalizeConfig(
            n_components=args.n_components,
            overcomplete_factor=args.overcomplete_factor,
            alpha=args.alpha,
            init_sigma=args.sigma,
            init_sigma_fraction=args.sigma_fraction,
            min_sigma=args.min_sigma,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            sinkhorn_iterations=args.sinkhorn_iterations,
            regularization=args.regularization,
            box=None if args.box is None else tuple(args.box),
            background_fraction=None if args.no_background else args.background_fraction,
            merge_distance=args.merge_distance,
            annealing=args.annealing,
        )
        fitter = EMFitter(config, initializer=INITIALIZERS[args.init]())
        run_localize(
            args.tomograms,
            args.picks,
            args.output,
            config=config,
            fitter=fitter,
            tomograms_format=args.tomograms_format,
            picks_format=args.picks_format,
            output_format=args.output_format,
            read_options=read_options_from(args),
            pixel_size=args.picks_pixel_size,
            min_pick_score=args.min_pick_score,
            flip_y=args.flip_y,
            tilt_series=args.tilt_series,
            selection=selection_from(args),
            seed=args.seed,
            registry=registry,
        )
        return 0
