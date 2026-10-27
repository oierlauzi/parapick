"""ParaPick: tomogram-free particle localization from 2D picks on cryo-ET tilt-series.

Quick start::

    import parapick as pp

    tomograms = pp.read_tomograms("tomograms.star")
    picks = pp.read_picks("picks.star")
    result = pp.localize(tomograms, picks, config=pp.LocalizeConfig(alpha=0.5))
    particles = result.particles(pp.Selection(min_score=0.2))
    pp.write_particles(particles, "particles.star", tomograms_path="tomograms.star")
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from parapick.formats import AUTO, TomogramsReadOptions, WriteContext, default_registry
from parapick.gmm import EMFitter, GMMFit, LocalizeConfig
from parapick.tomography import Particles, PickTable, TiltSeries, TomogramSet
from parapick.workflows import (
    LocalizationResult,
    Selection,
    SimulationConfig,
    evaluate,
    localize,
    project_particles,
    select_particles,
    simulate,
)

try:
    from parapick._version import __version__
except ImportError:  # pragma: no cover - only when running from a source tree without a build
    __version__ = "0.0.0"

__all__ = [
    "EMFitter",
    "GMMFit",
    "LocalizationResult",
    "LocalizeConfig",
    "Particles",
    "PickTable",
    "Selection",
    "SimulationConfig",
    "TiltSeries",
    "TomogramSet",
    "__version__",
    "evaluate",
    "localize",
    "project_particles",
    "read_particles",
    "read_picks",
    "read_tomograms",
    "select_particles",
    "simulate",
    "write_particles",
    "write_picks",
]


def read_tomograms(
    path: str | PathLike[str],
    *,
    fmt: str = AUTO,
    tilt_image_size: tuple[int, int] | None = None,
) -> TomogramSet:
    """Read tilt-series geometry with the format ``fmt`` (auto-detected by default)."""
    path = Path(path)
    reader = default_registry().tomograms_reader(fmt, path)
    return reader.read_tomograms(path, TomogramsReadOptions(tilt_image_size=tilt_image_size))


def read_picks(path: str | PathLike[str], *, fmt: str = AUTO) -> PickTable:
    """Read 2D picks with the format ``fmt`` (auto-detected by default)."""
    path = Path(path)
    return default_registry().picks_reader(fmt, path).read_picks(path)


def read_particles(
    path: str | PathLike[str], *, fmt: str = AUTO, tomograms: TomogramSet | None = None
) -> Particles:
    """Read 3D particles with the format ``fmt`` (auto-detected by default)."""
    path = Path(path)
    return default_registry().particles_reader(fmt, path).read_particles(path, tomograms)


def write_picks(picks: PickTable, path: str | PathLike[str], *, fmt: str = "relion5") -> list[Path]:
    """Write 2D picks with the format ``fmt``."""
    return default_registry().picks_writer(fmt).write_picks(picks, Path(path))


def write_particles(
    particles: Particles,
    path: str | PathLike[str],
    *,
    fmt: str = "relion5",
    tomograms_path: str | PathLike[str] | None = None,
) -> list[Path]:
    """Write 3D particles with the format ``fmt``."""
    context = WriteContext(tomograms_path=None if tomograms_path is None else Path(tomograms_path))
    return default_registry().particles_writer(fmt).write_particles(particles, Path(path), context)
