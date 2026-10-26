"""Building blocks shared by the sub-commands."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from parapick.formats import AUTO, FormatRegistry, Role, TomogramsReadOptions

__all__ = [
    "ArgumentContainer",
    "Command",
    "CommandError",
    "add_tomograms_arguments",
    "configure_jax",
    "configure_logging",
    "format_argument",
    "read_options_from",
]


class CommandError(RuntimeError):
    """A user-facing error: printed without a traceback."""


@runtime_checkable
class Command(Protocol):
    """A ``parapick`` sub-command."""

    name: str
    help: str

    def add_arguments(self, parser: argparse.ArgumentParser, registry: FormatRegistry) -> None:
        """Declare the sub-command's arguments."""
        ...

    def run(self, args: argparse.Namespace, registry: FormatRegistry) -> int:
        """Execute and return a process exit code."""
        ...


ArgumentContainer = argparse._ActionsContainer  # parsers and argument groups


def format_argument(
    parser: ArgumentContainer,
    flag: str,
    role: Role,
    registry: FormatRegistry,
    *,
    default: str = AUTO,
    allow_auto: bool = True,
) -> None:
    """Add a ``--*-format`` option whose choices are the registered formats for ``role``."""
    choices = ([AUTO] if allow_auto else []) + registry.names(role)
    parser.add_argument(
        flag,
        default=default,
        choices=choices,
        metavar="FORMAT",
        help=f"Format to {role.label} ({', '.join(choices)}; default: {default}).",
    )


def add_tomograms_arguments(
    parser: ArgumentContainer, registry: FormatRegistry, *, required: bool = True
) -> None:
    """``--tomograms``, ``--tomograms-format`` and ``--tilt-image-size``."""
    parser.add_argument(
        "--tomograms",
        type=Path,
        required=required,
        metavar="PATH",
        help="Tilt-series geometry (e.g. RELION 5 tomograms.star).",
    )
    format_argument(parser, "--tomograms-format", Role.TOMOGRAMS_READER, registry)
    parser.add_argument(
        "--tilt-image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Tilt-image size in native pixels, if it cannot be read from the image headers.",
    )


def read_options_from(args: argparse.Namespace) -> TomogramsReadOptions:
    """Build :class:`TomogramsReadOptions` from parsed arguments."""
    size = getattr(args, "tilt_image_size", None)
    return TomogramsReadOptions(tilt_image_size=None if size is None else (size[0], size[1]))


def configure_logging(verbosity: int) -> None:
    """Map ``-v``/``-q`` counts to logging levels (default: INFO)."""
    level = {-2: logging.ERROR, -1: logging.WARNING, 0: logging.INFO}.get(
        max(min(verbosity, 1), -2), logging.DEBUG
    )
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    if level > logging.DEBUG:
        for noisy in ("jax", "absl"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def configure_jax(device: str, x64: bool) -> None:
    """Select the JAX backend and precision. Must run before any JAX computation."""
    import jax

    if x64:
        jax.config.update("jax_enable_x64", True)
    if device == "cpu":
        jax.config.update("jax_platforms", "cpu")
    elif device == "gpu":
        try:
            jax.devices("gpu")
        except RuntimeError as error:
            raise CommandError(
                "No GPU is visible to JAX. Install a CUDA build (pip install 'parapick[cuda12]') "
                "or use --device cpu."
            ) from error
    logging.getLogger(__name__).debug("JAX devices: %s", jax.devices())
