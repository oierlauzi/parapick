"""RELION 5 STAR conventions: tomograms, picks and particles."""

from __future__ import annotations

from pathlib import Path

from parapick.formats.base import TomogramsReadOptions, WriteContext
from parapick.formats.relion5 import labels
from parapick.formats.relion5 import particles as _particles
from parapick.formats.relion5 import picks as _picks
from parapick.formats.relion5 import tomograms as _tomograms
from parapick.formats.star import peek_labels
from parapick.tomography import Particles, PickTable, TomogramSet

__all__ = ["Relion5Format", "labels"]


class Relion5Format:
    """RELION 5 tomography STAR files.

    * tomograms: ``tomograms.star`` + per tilt-series STAR files (read and write)
    * picks: one ``data_picks`` table with SPA coordinate labels (read and write)
    * particles: ``particles.star`` with ``rlnCenteredCoordinate*Angst`` (read and write), plus
      ``optimisation_set.star``
    """

    name = "relion5"
    description = "RELION 5 STAR files (tomograms.star, picks, particles.star)"
    suffixes = (".star",)

    def sniff(self, path: Path) -> bool:
        """True for the RELION 5 tomogram, picks and particles STAR files."""
        found = peek_labels(path)
        return (
            labels.TILT_SERIES_STAR in found
            or {labels.MICROGRAPH_NAME, labels.COORDINATE_X} <= found
            or labels.CENTERED_COORDINATE[0] in found
            or labels.PARTICLES_FILE in found
            or {labels.TOMO_NAME, labels.COORDINATE_Z} <= found
        )

    def read_tomograms(self, path: Path, options: TomogramsReadOptions) -> TomogramSet:
        """Read ``tomograms.star``."""
        return _tomograms.read_tomograms(path, options)

    def write_tomograms(self, tomograms: TomogramSet, path: Path) -> list[Path]:
        """Write ``tomograms.star`` and ``tilt_series/<name>.star``."""
        return _tomograms.write_tomograms(tomograms, path)

    def read_picks(self, path: Path) -> PickTable:
        """Read a ``data_picks`` STAR file."""
        return _picks.read_picks(path)

    def write_picks(self, picks: PickTable, path: Path) -> list[Path]:
        """Write a ``data_picks`` STAR file."""
        return _picks.write_picks(picks, path)

    def read_particles(self, path: Path, tomograms: TomogramSet | None = None) -> Particles:
        """Read ``particles.star`` (or an ``optimisation_set.star`` pointing to it)."""
        return _particles.read_particles(path, tomograms)

    def write_particles(
        self, particles: Particles, path: Path, context: WriteContext
    ) -> list[Path]:
        """Write ``particles.star`` (+ ``optimisation_set.star``)."""
        return _particles.write_particles(particles, path, context)
