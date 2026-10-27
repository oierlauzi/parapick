"""Sub-commands of the ``parapick`` CLI.

To add a command, implement :class:`parapick.cli.base.Command` in a new module and append an
instance to :data:`COMMANDS`; no other code needs to change.
"""

from parapick.cli.base import Command
from parapick.cli.commands.convert_picks import ConvertPicksCommand
from parapick.cli.commands.evaluate import EvaluateCommand
from parapick.cli.commands.filter import FilterCommand
from parapick.cli.commands.formats import FormatsCommand
from parapick.cli.commands.localize import LocalizeCommand
from parapick.cli.commands.project import ProjectCommand
from parapick.cli.commands.simulate import SimulateCommand
from parapick.cli.commands.unstack import UnstackCommand

COMMANDS: tuple[Command, ...] = (
    LocalizeCommand(),
    UnstackCommand(),
    ConvertPicksCommand(),
    ProjectCommand(),
    FilterCommand(),
    EvaluateCommand(),
    SimulateCommand(),
    FormatsCommand(),
)
"""Registered sub-commands, in the order shown by ``parapick --help``."""

__all__ = ["COMMANDS"]
