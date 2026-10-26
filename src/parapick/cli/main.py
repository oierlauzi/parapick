"""``parapick`` command-line entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence

from parapick.cli.base import Command, CommandError, configure_jax, configure_logging
from parapick.cli.commands import COMMANDS
from parapick.formats import FormatRegistry, default_registry

__all__ = ["build_parser", "main"]

logger = logging.getLogger("parapick")


def _common_options() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    group = common.add_argument_group("general")
    group.add_argument(
        "-v", "--verbose", action="count", default=0, help="More output (repeatable)."
    )
    group.add_argument("-q", "--quiet", action="count", default=0, help="Less output (repeatable).")
    group.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Compute device for JAX (default: auto).",
    )
    group.add_argument(
        "--x64", action="store_true", help="Use double precision (slower, mostly for debugging)."
    )
    return common


def build_parser(
    commands: Sequence[Command] = COMMANDS, registry: FormatRegistry | None = None
) -> argparse.ArgumentParser:
    """Build the argument parser with one sub-parser per command."""
    from parapick import __version__

    registry = registry or default_registry()
    parser = argparse.ArgumentParser(
        prog="parapick",
        description="Tomogram-free particle localization from 2D picks on cryo-ET tilt-series.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(title="commands", dest="command", metavar="COMMAND")
    subparsers.required = True
    common = _common_options()
    for command in commands:
        sub = subparsers.add_parser(
            command.name,
            help=command.help,
            description=command.help,
            parents=[common],
        )
        command.add_arguments(sub, registry)
        sub.set_defaults(_command=command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the exit code."""
    registry = default_registry()
    parser = build_parser(registry=registry)
    args = parser.parse_args(argv)
    configure_logging(args.verbose - args.quiet)
    command: Command = args._command
    try:
        configure_jax(args.device, args.x64)
        return command.run(args, registry)
    except (CommandError, ValueError, KeyError, FileNotFoundError) as error:
        if args.verbose > 0:
            logger.exception("parapick %s failed", command.name)
        else:
            logger.error("%s", error)
        return 1
    except BrokenPipeError:  # pragma: no cover - e.g. `parapick formats | head -1`
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
