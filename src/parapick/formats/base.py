"""Format plug-in interfaces.

A *format* is a small object with a ``name``, a ``description``, the file ``suffixes`` it
usually uses, and a ``sniff`` method. It implements one or more *roles* by providing the
matching method:

=====================  ==========================================================
Role                   Method
=====================  ==========================================================
``TomogramsReader``    ``read_tomograms(path, options) -> TomogramSet``
``TomogramsWriter``    ``write_tomograms(tomograms, path) -> list[Path]``
``PicksReader``        ``read_picks(path) -> PickTable``
``PicksWriter``        ``write_picks(picks, path) -> list[Path]``
``ParticlesReader``    ``read_particles(path, tomograms) -> Particles``
``ParticlesWriter``    ``write_particles(particles, path, context) -> list[Path]``
=====================  ==========================================================

Formats are registered in :mod:`parapick.formats.registry`. Third-party packages can add
formats without touching ParaPick through the ``parapick.formats`` entry-point group. See
``docs/conventions.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from parapick.tomography import Particles, PickTable, TomogramSet

__all__ = [
    "Format",
    "ParticlesReader",
    "ParticlesWriter",
    "PicksReader",
    "PicksWriter",
    "TomogramsReadOptions",
    "TomogramsReader",
    "TomogramsWriter",
    "WriteContext",
]


@runtime_checkable
class Format(Protocol):
    """Common metadata of every format."""

    name: str
    """Short identifier used on the command line (e.g. ``relion5``)."""
    description: str
    """One-line human-readable description."""
    suffixes: tuple[str, ...]
    """File suffixes usually used by this format (e.g. ``(".star",)``)."""

    def sniff(self, path: Path) -> bool:
        """Cheaply check whether ``path`` looks like this format."""
        ...


@dataclass(frozen=True)
class TomogramsReadOptions:
    """Format-independent hints for reading tilt-series geometry.

    Parameters
    ----------
    tilt_image_size
        ``(width, height)`` of the tilt images in native pixels. Formats that need it and
        cannot otherwise find it (e.g. from image headers) raise an error asking for it.
    """

    tilt_image_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class WriteContext:
    """Information that writers may use to produce cross-referencing metadata.

    Parameters
    ----------
    tomograms
        Geometry the written data refers to.
    tomograms_path
        Path of the geometry file it was read from, if any.
    """

    tomograms: TomogramSet | None = None
    tomograms_path: Path | None = None


@runtime_checkable
class TomogramsReader(Format, Protocol):
    """Reads tilt-series geometry."""

    def read_tomograms(self, path: Path, options: TomogramsReadOptions) -> TomogramSet:
        """Read every tilt-series described by ``path``."""
        ...


@runtime_checkable
class TomogramsWriter(Format, Protocol):
    """Writes tilt-series geometry."""

    def write_tomograms(self, tomograms: TomogramSet, path: Path) -> list[Path]:
        """Write ``tomograms`` to ``path`` and return the files created."""
        ...


@runtime_checkable
class PicksReader(Format, Protocol):
    """Reads 2D picks on tilt images."""

    def read_picks(self, path: Path) -> PickTable:
        """Read every pick described by ``path``."""
        ...


@runtime_checkable
class PicksWriter(Format, Protocol):
    """Writes 2D picks on tilt images."""

    def write_picks(self, picks: PickTable, path: Path) -> list[Path]:
        """Write ``picks`` to ``path`` and return the files created."""
        ...


@runtime_checkable
class ParticlesReader(Format, Protocol):
    """Reads 3D particle coordinates."""

    def read_particles(self, path: Path, tomograms: TomogramSet | None = None) -> Particles:
        """Read particles. ``tomograms`` is available for formats with non-centred coordinates."""
        ...


@runtime_checkable
class ParticlesWriter(Format, Protocol):
    """Writes 3D particle coordinates."""

    def write_particles(
        self, particles: Particles, path: Path, context: WriteContext
    ) -> list[Path]:
        """Write ``particles`` to ``path`` and return the files created."""
        ...
