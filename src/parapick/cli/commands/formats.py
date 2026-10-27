"""``parapick formats``: List available formats."""

from __future__ import annotations

import argparse
import logging

from parapick.formats import FormatRegistry

__all__ = ["FormatsCommand"]

logger = logging.getLogger(__name__)


class FormatsCommand:
    """List the available formats."""

    name = "formats"
    help = "List the available file formats (including installed plug-ins) and their roles."

    def add_arguments(self, parser: argparse.ArgumentParser, registry: FormatRegistry) -> None:
        """No arguments."""

    def run(self, args: argparse.Namespace, registry: FormatRegistry) -> int:
        """Print one line per format."""
        for fmt in registry:
            roles = ", ".join(role.label for role in registry.roles(fmt))
            print(f"{fmt.name:<16} {fmt.description}\n{'':<16} roles: {roles}")
        return 0
