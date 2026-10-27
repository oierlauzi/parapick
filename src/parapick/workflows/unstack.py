"""Write every tilt image as its own 2D MRC file, for single-particle pickers.

The pixel size is not recorded in the MRC headers: the pickers read only the image data and
shape, and ParaPick takes the pixel size from the geometry.

Most SPA pickers expect one micrograph per file: crYOLO rejects MRC files with more than one
section, and Topaz in 2D mode treats each file as one micrograph keyed by its stem. Tilt-series
are commonly stored as one stack per tilt-series. :func:`unstack` writes one ``.mrc`` per tilt
and returns the same geometry with its tilts pointing at the new files. Picks made on those
files therefore match the tilts when they are localised with the new geometry.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from parapick.formats import AUTO, FormatRegistry, TomogramsReadOptions, default_registry
from parapick.formats.images import ImageIO, ImageReference, Pending, default_image_io
from parapick.tomography import TomogramSet

__all__ = ["MICROGRAPHS_DIRECTORY", "run_unstack", "unstack", "unstacked_name"]

logger = logging.getLogger(__name__)

MICROGRAPHS_DIRECTORY = "micrographs"


def unstacked_name(tilt_series: str, index: int) -> str:
    """File name of tilt ``index`` (0-based) of a tilt-series: ``<TS>_<NNN>.mrc``, 1-based."""
    return f"{tilt_series}_{index + 1:03d}.mrc"


def _single_mrc_file(reference: ImageReference) -> bool:
    return reference.index is None and Path(reference.path).suffix.lower() == ".mrc"


@dataclass(frozen=True)
class _Batch:
    """Images of one tilt-series to copy together (all of the same size)."""

    sources: tuple[ImageReference, ...]
    targets: tuple[Path, ...]


def _batches(copies: list[tuple[ImageReference, Path]], batch_size: int) -> Iterator[_Batch]:
    for start in range(0, len(copies), batch_size):
        chunk = copies[start : start + batch_size]
        yield _Batch(tuple(s for s, _ in chunk), tuple(t for _, t in chunk))


def _replace(target: Path) -> None:
    """Remove an existing output (never write through a symlink into its source)."""
    if target.exists() or target.is_symlink():
        target.unlink()


def _copy(batches: list[_Batch], io: ImageIO, max_pending_writes: int) -> list[Path]:
    """Read and write batches as a pipeline.

    The read of batch ``k + 1`` is issued before batch ``k`` is written, and at most
    ``max_pending_writes`` writes are left in flight. With a synchronous :class:`ImageIO` this
    runs in order. With an asynchronous one, reads, writes and the next batch overlap.
    """
    written: list[Path] = []
    writes: deque[Pending[None]] = deque()
    upcoming = io.read_batch(batches[0].sources) if batches else None
    for k, batch in enumerate(batches):
        assert upcoming is not None
        images = upcoming.result()
        upcoming = io.read_batch(batches[k + 1].sources) if k + 1 < len(batches) else None
        for target in batch.targets:
            _replace(target)
        writes.append(io.write_batch(images, batch.targets))
        written += batch.targets
        while len(writes) > max_pending_writes:
            writes.popleft().result()
    while writes:
        writes.popleft().result()
    return written


def unstack(
    tomograms: TomogramSet,
    output_directory: Path,
    *,
    anchor: Path | None = None,
    link: bool = False,
    overwrite: bool = False,
    io: ImageIO | None = None,
    batch_size: int = 8,
    max_pending_writes: int = 2,
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
    io
        Image reader/writer (default: :func:`~parapick.formats.images.default_image_io`).
    batch_size
        Images per read/write batch. It bounds memory: about
        ``(2 + max_pending_writes) × batch_size`` images are held at once.
    max_pending_writes
        Writes allowed in flight while the next batches are read (asynchronous ``io`` only).

    Returns
    -------
    TomogramSet, list[Path]
        The same geometry with tilt identifiers and images pointing at the new files, and the
        image files created in this call.
    """
    if batch_size < 1 or max_pending_writes < 0:
        raise ValueError("batch_size must be >= 1 and max_pending_writes >= 0")
    io = io or default_image_io()
    directory = output_directory / MICROGRAPHS_DIRECTORY
    series, batches, written = [], [], []
    for ts in tomograms:
        names: list[str] = []
        copies: list[tuple[ImageReference, Path]] = []
        for i, image in enumerate(ts.images):
            target = directory / unstacked_name(ts.name, i)
            names.append(str(target))
            if (target.exists() or target.is_symlink()) and not overwrite:
                logger.debug("Keeping existing %s", target)
                continue
            source = ImageReference.parse(image).resolved(anchor)
            if link and _single_mrc_file(source):
                _replace(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(Path(source.path).resolve())
                written.append(target)
            else:
                copies.append((source, target))
        batches += list(_batches(copies, batch_size))
        series.append(dataclasses.replace(ts, tilt_ids=tuple(names), images=tuple(names)))
        logger.info("%s: %d tilt images -> %s", ts.name, ts.n_tilts, directory)
    written += _copy(batches, io, max_pending_writes)
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
    io: ImageIO | None = None,
    batch_size: int = 8,
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
        tomograms,
        output_directory,
        anchor=tomograms_path,
        link=link,
        overwrite=overwrite,
        io=io,
        batch_size=batch_size,
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
