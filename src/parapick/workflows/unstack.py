"""Write every tilt image as its own 2D MRC file, for single-particle pickers.

Most SPA pickers expect one micrograph per file: crYOLO rejects MRC files with more than one
section, and Topaz in 2D mode treats each file as one micrograph keyed by its stem. Tilt-series
are commonly stored as one stack per tilt-series. :func:`unstack` writes one ``.mrc`` per tilt
and returns the same geometry with its tilts pointing at the new files. Picks made on those
files therefore match the tilts when they are localised with the new geometry.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from parapick.formats import AUTO, FormatRegistry, TomogramsReadOptions, default_registry
from parapick.formats.images import ImageReference, read_section, write_mrc
from parapick.tomography import TiltSeries, TomogramSet

__all__ = ["MICROGRAPHS_DIRECTORY", "run_unstack", "unstack", "unstacked_name"]

logger = logging.getLogger(__name__)

MICROGRAPHS_DIRECTORY = "micrographs"


def unstacked_name(tilt_series: str, index: int) -> str:
    """File name of tilt ``index`` (0-based) of a tilt-series: ``<TS>_<NNN>.mrc``, 1-based."""
    return f"{tilt_series}_{index + 1:03d}.mrc"


def _single_mrc_file(reference: ImageReference) -> bool:
    return reference.index is None and Path(reference.path).suffix.lower() == ".mrc"


def _unstack_tilt_series(
    ts: TiltSeries,
    directory: Path,
    anchor: Path | None,
    *,
    link: bool,
    overwrite: bool,
) -> tuple[TiltSeries, list[Path]]:
    names: list[str] = []
    written: list[Path] = []
    for i, image in enumerate(ts.images):
        target = directory / unstacked_name(ts.name, i)
        names.append(str(target))
        if (target.exists() or target.is_symlink()) and not overwrite:
            logger.debug("Keeping existing %s", target)
            continue
        reference = ImageReference.parse(image)
        if link and _single_mrc_file(reference):
            source = reference.resolve(anchor).resolve()
            if target.exists() or target.is_symlink():
                target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
        else:
            write_mrc(target, read_section(reference, anchor), ts.pixel_size, overwrite=True)
        written.append(target)
    unstacked = dataclasses.replace(ts, tilt_ids=tuple(names), images=tuple(names))
    return unstacked, written


def unstack(
    tomograms: TomogramSet,
    output_directory: Path,
    *,
    anchor: Path | None = None,
    link: bool = False,
    overwrite: bool = False,
) -> tuple[TomogramSet, list[Path]]:
    """Write one 2D MRC file per tilt into ``output_directory/micrographs``.

    Parameters
    ----------
    tomograms
        Geometry whose ``images`` reference the tilt images (files or ``N@stack`` sections).
    output_directory
        Destination. Files are named ``<TS>_<NNN>.mrc`` (1-based, in tilt order), so their stems
        are unique across tilt-series.
    anchor
        File that relative image paths are resolved against (e.g. the ``tomograms.star`` they
        were read from), as RELION does.
    link
        Symlink tilts that already are single 2D ``.mrc`` files instead of copying them.
    overwrite
        Replace existing output files (by default they are kept, so runs can resume).

    Returns
    -------
    TomogramSet, list[Path]
        The same geometry with tilt identifiers and images pointing at the new files, and the
        image files created in this call.
    """
    directory = output_directory / MICROGRAPHS_DIRECTORY
    series, written = [], []
    for ts in tomograms:
        unstacked, files = _unstack_tilt_series(
            ts, directory, anchor, link=link, overwrite=overwrite
        )
        series.append(unstacked)
        written += files
        logger.info("%s: %d tilt images -> %s", ts.name, ts.n_tilts, directory)
    return TomogramSet(tuple(series), tomograms.metadata), written


def run_unstack(
    tomograms_path: Path,
    output_directory: Path,
    *,
    tomograms_format: str = AUTO,
    output_format: str = "relion5",
    read_options: TomogramsReadOptions | None = None,
    link: bool = False,
    overwrite: bool = False,
    registry: FormatRegistry | None = None,
) -> tuple[TomogramSet, list[Path]]:
    """File-based :func:`unstack`: also writes the new geometry (e.g. ``tomograms.star``).

    Pick on ``output_directory/micrographs/*.mrc``, then use the written geometry for
    ``convert-picks`` and ``localize``.
    """
    registry = registry or default_registry()
    reader = registry.tomograms_reader(tomograms_format, tomograms_path)
    tomograms = reader.read_tomograms(tomograms_path, read_options or TomogramsReadOptions())
    unstacked, images = unstack(
        tomograms, output_directory, anchor=tomograms_path, link=link, overwrite=overwrite
    )
    writer = registry.tomograms_writer(output_format)
    geometry_path = output_directory / f"tomograms{writer.suffixes[0] if writer.suffixes else ''}"
    written = writer.write_tomograms(unstacked, geometry_path)
    logger.info(
        "Wrote %d tilt images and %s. Pick on %s, then pass --tomograms %s to "
        "convert-picks and localize.",
        len(images),
        geometry_path,
        output_directory / MICROGRAPHS_DIRECTORY,
        geometry_path,
    )
    return unstacked, written + images
